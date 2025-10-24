import torch
import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayer(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        """
        Initialize the OutputLayer.

        Args:
            config: Configuration object with n_embed, vocab_size, T, lr, etc.
        """
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        self.pc_layer = PCLayer(
            T=config.T,
            lr=config.lr,
            update_bias = config.update_bias,
            energy_fn_name=config.output_energy_fn_name,
        )
