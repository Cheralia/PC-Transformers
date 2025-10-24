import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import gc
from typing import Optional, Tuple
from torch.amp import autocast
from contextlib import nullcontext
from utils.attention_utils import apply_flash_attention, apply_standard_attention

def compute_DVL(attn_v, requires_update):
    B, H, T, D = attn_v.shape
    device = attn_v.device
    
    x = attn_v.transpose(0, 1).flatten(2, 3).transpose(0, 1)    # (B, H, T*D)
    x = F.normalize(x, p=2, dim=-1)
    s_m = torch.bmm(x, x.transpose(1, 2))  # (B, H, H)
    s_m = s_m.mean(dim=0)  # (H, H)
    identity = torch.eye(H, device=device)  
    corr = s_m - identity  
    dvl = (corr ** 2).mean()  
    
    dvl_grad = torch.zeros_like(attn_v, device=device)
    if requires_update:
        try:
            dvl_grad = torch.autograd.grad(dvl, attn_v, retain_graph=True)[0]
        except Exception as e:
            warnings.warn(f"Error computing diversity gradient: {e}")
            dvl_grad = torch.zeros_like(attn_v, device=device)
    return dvl_grad

def get_head_similarity(mu_heads):
    """
    Compute per-batch head similarity metric (absolute correlation mean per head).
    Input mu_heads shape: (B, H, T, D)
    Returns: tensor of shape (B,) containing mean head similarity per batch element.
    """
    B, H, T, D = mu_heads.shape
    x = mu_heads.transpose(0, 1).flatten(2, 3)  # (H, B, T*D) 
    x = F.normalize(x, p=2, dim=-1)
    corr = torch.bmm(x, x.transpose(1, 2))   # (H, B, B)
    # compute mean absolute off-diagonal similarity per head then mean across heads
    mask = ~torch.eye(corr.size(1), device=corr.device).bool()
    s_v = corr[:, mask].mean(dim= -1)
    corr = s_v.abs().mean(dim=-1)  
    return corr.detach().cpu()
    
def x_init(batch_size: int, seq_len: int, embedding_size: int, device: torch.device = None) -> torch.Tensor:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return torch.randn(batch_size, seq_len, embedding_size, device = device)

def step_embed(
    t: int,
    T: int,
    target: torch.Tensor,
    layer: dict,
    layer_type: str,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    local_lr: float,
    clamp_value: float,
    energy_fn_name: str,
    requires_update: bool,
    layer_norm: Optional[nn.Module] = None,
    mu_word_cache: Optional[torch.Tensor] = None,
    mu_pos_cache: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predictive coding step for embedding layer.
    Returns (mu, mu_word, mu_pos, error)
    """
    word_layer: nn.Embedding = layer["word"]
    pos_layer: nn.Embedding = layer["pos"]
    
    use_amp = target.is_cuda
    autocast_ctx = autocast('cuda') if use_amp else nullcontext()

    # clip ids
    vocab_size = word_layer.weight.size(0)
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    max_pos = pos_layer.weight.size(0)
    if position_ids.max() >= max_pos:
        position_ids = torch.clamp(position_ids, max=max_pos-1)
        
    with autocast_ctx:
        if requires_update or mu_word_cache is None or mu_pos_cache is None:
            mu_word = word_layer(input_ids)
            mu_pos = pos_layer(position_ids)
        else:
            mu_word = mu_word_cache
            mu_pos = mu_pos_cache
            
        mu = mu_word + mu_pos
        mu_norm=layer_norm(mu) if layer_norm is not None else mu
    
        error = target - mu_norm
        
    if requires_update: 
        with torch.no_grad():
            flat_input_ids = input_ids.reshape(-1)
            flat_update = error.reshape(-1, error.size(-1))
            flat_position_ids = position_ids.reshape(-1)
            
            delta = local_lr * flat_update
            delta = torch.clamp(delta, -0.01, 0.01)
            
            word_layer.weight.data.index_add_(0, flat_input_ids, delta)
            pos_layer.weight.data.index_add_(0, flat_position_ids, delta)
            
    if t == T - 1:
           finalize_step(mu, target, error, t, layer_type, energy_fn_name)
  
    return mu, mu_word, mu_pos, error
    
def step_linear(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    layer: nn.Module,
    W_latents: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    use_lateral: bool,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module], 
   ):
    """
    Predictive coding step for linear-like layers.
    Returns: (updated_x, mu, bu_err)
    """
    device = x.device
    use_amp = target.is_cuda
    autocast_ctx = autocast('cuda') if use_amp else nullcontext()
    
    with autocast_ctx:
        if layer_norm is not None and layer_type == "fc1":
           x = layer_norm(x)
        elif layer_type == "fc2":
           x = F.gelu(x)
           
        mu = layer(x)
        if layer_type == "fc1":
            mu = F.gelu(mu)
        elif layer_norm is not None and layer_type in ["linear_attn", "fc2"]:
              mu = layer_norm(mu)
              
        if layer_type=="linear_output":
          bu_err= target - F.softmax(mu, dim=-1) 
        else:    
          bu_err = target - mu 
          
        # project bottom-up error through weights
        error_proj= bu_err @ layer.weight 
       
        if td_err is not None:
           error= error_proj- td_err
        else:
           error= error_proj     

        if use_lateral and (layer_type in W_latents):
           W_latent = W_latents[layer_type].to(device) 
           x_latent = torch.einsum("bsh,hv->bsv", x, W_latent)
           delta_x = error + x_latent
           x = x + local_lr * delta_x

           if requires_update:
               anti_hebbian_latent = -torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
               # update parametrically in-place to preserve Parameter semantics
               W_latents[layer_type].data.add_(local_lr * anti_hebbian_latent)
               W_latents[layer_type].data = F.normalize(W_latents[layer_type].data, p=2, dim=1)
        else:
          x= x + local_lr * error 
    
        x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))
    
    # parameter updates for the layer
    if requires_update:
        delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x.detach())
        delta_W = torch.clamp(delta_W, -0.01, 0.01)
        layer.weight.data.add_(delta_W)
        if layer.bias is not None and update_bias:
            delta_b = local_lr * bu_err.mean(dim=(0, 1))
            delta_b = torch.clamp(delta_b, -0.01, 0.01)
            layer.bias.data.add_(delta_b)

    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)

    return x, mu, bu_err

def step_attn(
    t: int,
    T: int,
    target: torch.Tensor,
    x: torch.Tensor,
    W_latents: dict,
    proj_layers: dict,
    layer_type: str,
    local_lr: float,
    clamp_value: float,
    use_lateral: bool,
    energy_fn_name: str,
    update_bias: bool,
    requires_update: bool,
    layer_instance: Optional[object],
    num_heads: int,
    n_embed: int,
    la: float,
    td_err: Optional[torch.Tensor],
    layer_norm: Optional[nn.Module],
    flash: bool = False,
    ):
    """
    Predictive coding step for attention. Returns (updated_x, mu, bu_err).
    - proj_layers must contain 'q_proj','k_proj','v_proj' modules
    """
    assert proj_layers is not None, "proj_layers dict is required for attention"
    device = x.device
    x=layer_norm(x) if layer_norm is not None else x
        
    q_proj = proj_layers.get("q_proj")
    k_proj = proj_layers.get("k_proj")
    v_proj = proj_layers.get("v_proj")
    assert q_proj is not None and k_proj is not None and v_proj is not None, "Missing Q/K/V projections" 
        
    use_amp = target.is_cuda
    autocast_ctx = autocast('cuda') if use_amp else nullcontext()
        
    batch_size, seq_len, embed_dim = target.shape
    head_dim = n_embed // num_heads
    la = la * math.sqrt(1.0 / head_dim)

    with autocast_ctx:      
        Q= q_proj(x)
        K= k_proj(x)
        V= v_proj(x)
            
        Q = Q.view(batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, num_heads, seq_len, head_dim)
            
        #create causal mask (1=keep, 0=mask)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        # !! Causal Mask
        if flash:
            # TODO: add support for causal masking in flash attention
            mu_heads = apply_flash_attention(Q, K, V)
        else:
            mu_heads = apply_standard_attention(Q, K, V, mask=causal_mask)

        dvl_grad = compute_DVL(mu_heads, requires_update)
        if dvl_grad is not None:
            dvl_grad = dvl_grad.to(device)
            
        similarity = get_head_similarity(mu_heads)
        mu = mu_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
     
        bu_err = target - mu  # B, T, D
        error = bu_err - td_err if td_err is not None else bu_err  
        
        # incorporate diversity gradient projection if available
        if dvl_grad is not None:
            B, H, T, D = dvl_grad.shape              
            dvl_projected = dvl_grad.permute(0, 2, 1, 3).contiguous().view(B, T, H*D) 
            dvl_projected=dvl_projected.clamp(-1e-3, 1e-3)
            error = error + la * dvl_projected
                
    if layer_instance is not None:
        setattr(layer_instance, '_head_similarity', similarity)
        setattr(layer_instance, '_head_similarity_avg', similarity.mean().item())
        setattr(layer_instance, '_head_similarity_max', similarity.max().item())
        
    if use_lateral and (layer_type in W_latents):
        W_latent = W_latents[layer_type].to(device) 
        x_latent = x @ W_latent
        delta_x = error + x_latent
        x = x + local_lr * delta_x 

        if requires_update:
            anti_hebbian_latent = - torch.einsum("bsh,bsv->hv", x.detach(), x.detach())
            W_latents[layer_type].data.add_(local_lr * anti_hebbian_latent)
            W_latents[layer_type].data = F.normalize(W_latents[layer_type].data, p=2, dim=1)
    else:
        x= x+ local_lr * error

    x = torch.clamp(x, -abs(clamp_value), abs(clamp_value))

    # PC update W_latent
    if requires_update:
        for proj in (q_proj, k_proj, v_proj):
            delta_W = local_lr * torch.einsum("bsv, bsh -> vh", bu_err, x.detach())
            delta_W = torch.clamp(delta_W, -0.01, 0.01)
            proj.weight.data.add_(delta_W)
            if proj.bias is not None and update_bias:
                delta_b = local_lr * bu_err.mean(dim=(0, 1))
                delta_b = torch.clamp(delta_b, -0.01, 0.01)
                delta_b = delta_b.view(-1)
                proj.bias.data.add_(delta_b)
 
    if t == T - 1:
        finalize_step(mu, target, error, t, layer_type,energy_fn_name)
     
    return x, mu, bu_err
    
ENERGY_FUNCTIONS = {
    "pc_e": lambda mu, x: ((mu - x) ** 2) * 0.5,    
    "kld": lambda mu, x: torch.clamp(
        F.kl_div(mu.log_softmax(dim=-1), x, reduction="batchmean"), min=0.0, max=100.0
    ),
}

def energy_fn(mu: torch.Tensor, x: torch.Tensor,energy_fn_name: str) -> torch.Tensor:
    if energy_fn_name not in ENERGY_FUNCTIONS:
        raise ValueError(f"Unknown energy function: {energy_fn_name}. Choose from {list(ENERGY_FUNCTIONS.keys())}")
    return ENERGY_FUNCTIONS[energy_fn_name](mu, x)

def finalize_step(mu: torch.Tensor, target: torch.Tensor, error: torch.Tensor, t: int, layer_type: str, energy_fn_name: str):
    device = mu.device
    target = target.to(device)
    error = error.to(device)
    energy = float(energy_fn(mu, target, energy_fn_name).mean().item())
    errors = [{"step": t, "type": layer_type, "error": error.mean().item()}]
    return energy, errors
    
def ids_to_one_hot(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    device = input_ids.device
    if input_ids.max() >= vocab_size:
        input_ids = torch.clamp(input_ids, max=vocab_size-1)
    return F.one_hot(input_ids, num_classes=vocab_size).float().to(device)

def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()