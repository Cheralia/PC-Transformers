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
        mask: Optional mask tensor
    Returns:
        attn_output: Output tensor after attention
    """
    if not FLASH_AVAILABLE:
        return apply_standard_attention(q, k, v, mask)
    B, num_heads, T, head_dim = q.shape
    # FlashAttention expects [B, T, 3, num_heads, head_dim]
    qkv = torch.stack([q, k, v], dim=2) # [B, T, 3, num_heads, head_dim]
    orig_dtype = qkv.dtype
    with autocast(device_type=device, dtype=torch.float16):
        if qkv.dtype not in [torch.float16, torch.bfloat16]:
            qkv = qkv.to(torch.float16)
        attn_out = flash_attn_qkvpacked_func(qkv, 0.0, None, causal=True)
        attn_out = attn_out.to(orig_dtype)
    # Output: [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
    return attn_out

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

        if mask is not None:
            # ensure mask is boolean or float {0,1}, and broadcast correctly
            # mask == 0 → block (set to -inf before softmax)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # compute softmax in float32 for numerical stability
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)

        # apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

    return attn_output


def apply_rotary_pos_emb(q, k, seq_len, head_dim, device):
    # Generate rotary position encodings
    theta = 10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('i,j->ij', pos, 1.0 / theta)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x_rot

    q_rot = q * cos + rotate(q) * sin
    k_rot = k * cos + rotate(k) * sin
    return q_rot, k_rot
