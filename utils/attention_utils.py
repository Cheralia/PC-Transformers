import torch
from torch.amp import autocast
import logging

# Set up logging
logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO 
)

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_AVAILABLE = True
    logging.info("FlashAttention is available and will be used.")
except ImportError:
    FLASH_AVAILABLE = False
    logging.warning("FlashAttention is not installed. Falling back to standard attention.")
    
device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_flash_attention(q, k, v, mask=None):
    """
    Apply FlashAttention if available, else fallback to standard attention.
    Args:
        q, k, v: Query, Key, Value tensors (B, num_heads, T, head_dim)
        mask: Optional mask tensor (B, 1, T, T) or (B, T)
    Returns:
        attn_output: Output tensor after attention
    """
    if not FLASH_AVAILABLE:
        return apply_standard_attention(q, k, v, mask)
    B, num_heads, T, head_dim = q.shape
    # FlashAttention expects [B, T, 3, num_heads, head_dim]
    qkv = torch.stack([q, k, v], dim=2) # [B, T, 3, num_heads, head_dim]
     # Rearrange to [B, T, 3, num_heads, head_dim]
    qkv = qkv.permute(0, 2, 1, 3, 4).contiguous()
    orig_dtype = qkv.dtype
    with autocast(device_type=device, dtype=torch.float16):
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16)
        attn_out = flash_attn_qkvpacked_func(qkv, 0.0, mask, causal=True)
        attn_out = attn_out.to(orig_dtype)
    # Output: [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
    return attn_out.permute(0, 2, 1, 3).contiguous()

def apply_standard_attention(q, k, v, mask=None):
    """
    Standard scaled dot-product attention with masking and mixed precision.
    Args:
        q, k, v: Query, Key, Value tensors (B, num_heads, T, head_dim)
        mask: Optional mask tensor, broadcastable to (B, num_heads, T, T)
    Returns:
        attn_output: Output tensor after attention
    """
    with autocast(device_type=device, dtype=torch.float16):
        # (B, num_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        
        print("[DEBUG] Attention scores BEFORE masking:\n", attn_scores[0,0,:10,:10].detach().cpu())
        print(f"[DEBUG] Scores BEFORE masking: min={attn_scores.min().item():.3f}, "
            f"max={attn_scores.max().item():.3f}, "
            f"mean={attn_scores.mean().item():.3f}")
         
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            print("[DEBUG] Scores AFTER masking (10x10 slice):\n", attn_scores[0,0,:10,:10].detach().cpu())
            print(f"[DEBUG] AFTER masking stats: min={attn_scores.min().item():.3f}, "
              f"max={attn_scores.max().item():.3f}, mean={attn_scores.mean().item():.3f}")
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

    return attn_output



def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Precompute rotary positional embeddings frequencies as complex exponentials.

    Args:
        dim (int): Head dimension (embedding size per attention head)
        seq_len (int): Maximum sequence length
        theta (float): Scaling factor (default 10000.0)

    Returns:
        torch.Tensor: Complex tensor of shape (seq_len, dim) for RoPE
    """
    # Only half of the dimension is needed for interleaving
    dim_half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim_half).float() / dim_half))  # (dim//2,)
    positions = torch.arange(seq_len).float()  # (seq_len,)
    angles = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # (seq_len, dim//2), complex64
    # Interleave to full dim
    freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)  # (seq_len, dim)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape precomputed frequencies for broadcasting with Q/K tensors.

    Args:
        freqs_cis (torch.Tensor): (seq_len, head_dim)
        x (torch.Tensor): Query/Key tensor of shape (B, T, H, head_dim)

    Returns:
        torch.Tensor: Reshaped tensor (1, T, 1, head_dim) for broadcasting
    """
    B, T, H, head_dim = x.shape
    assert freqs_cis.shape == (T, head_dim), f"Expected freqs shape (T, head_dim), got {freqs_cis.shape}"
    # reshape for broadcasting
    return freqs_cis.view(1, T, 1, head_dim)


def apply_rotary_pos_emb(q, k, seq_len, head_dim, device):
    """
    Apply Rotary Positional Embeddings (RoPE) to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor of shape (B, T, num_heads, head_dim)
        k (torch.Tensor): Key tensor of shape (B, T, num_heads, head_dim)
        seq_len (int): Sequence length (T)
        head_dim (int): Dimension of each attention head
        device (torch.device): Device for computations

    Returns:
        tuple: (q_rot, k_rot) tensors after applying RoPE, same shape as inputs
    """
    freqs_cis = precompute_freqs_cis(head_dim, seq_len).to(device)
    freqs_cis = reshape_for_broadcast(freqs_cis, q)

    cos = freqs_cis.real
    sin = freqs_cis.imag

    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x_rot

    q_rot = q * cos + rotate(q) * sin
    k_rot = k * cos + rotate(k) * sin
    return q_rot, k_rot
