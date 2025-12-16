import logging
from predictive_coding.config import GPTConfig

logger = logging.getLogger(__name__)

# Fixed hyperparameters 
FIXED_PARAMS = {
    'dropout': 0.1,
    'warmup_steps': 200,
    'batch_size': 8,
    'num_epochs': 10,
    'combined_internal_weight': 0.8,
    'combined_output_weight': 0.2,
    'alpha': 0.5,
    'internal_energy_fn_name': 'pc_e',
    'output_energy_fn_name': 'pc_e',
}

def get_dynamic_model_config(trial, vocab_size, flash=False):
    """Get model configuration with dynamic parameter combinations, including flash attention flag."""
    
    # Tunable parameters
    n_embed = trial.suggest_int("n_embed", 64, 512, step=16)
    block_size = trial.suggest_int("block_size", 64, 512, step=16)
    n_blocks = trial.suggest_int('n_blocks', 1, 12)
    T = trial.suggest_int('T', 1, 14, log=True)
    peak_lr = trial.suggest_float('peak_lr', 1e-5, 1e-2, log=True)
    
    # Derive num_heads from n_embed
    valid_heads = [h for h in range(2, min(32, n_embed // 8) + 1) if n_embed % h == 0 and 8 <= n_embed // h <= 128]
    if not valid_heads:
        logger.warning(f"No valid heads for n_embed={n_embed}, forcing fallback.")
        return None 
    num_heads = valid_heads[trial.suggest_int('head_idx', 0, len(valid_heads) - 1)]
    
    # Derive lr from peak_lr
    lr = peak_lr * 0.1 
    
    return GPTConfig(
        vocab_size=vocab_size,
        # Tuned
        n_embed=n_embed,
        num_heads=num_heads,
        block_size=block_size,
        n_blocks=n_blocks,
        T=T,
        peak_learning_rate=peak_lr,
        lr=lr,
        # Fixed
        use_flash_attention=flash,
        **FIXED_PARAMS
    )

def update_global_config(config):
    """Update global GPTConfig"""
    for key, value in config.__dict__.items():
        if hasattr(GPTConfig, key):
            try:
                setattr(GPTConfig, key, value)
            except Exception as e:
                logger.warning(f"Failed to update config key '{key}': {e}")