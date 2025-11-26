import torch
from typing import Optional  # <<< ADD THIS LINE

class PCReplayBuffer:
    def __init__(self):
        self.records = []          # List of dicts: {layer_type, t, energy, x_state}
        self.best_states = {}      # Dict: layer_type -> best_x_state (lowest energy)

    def record_step(self, layer, layer_type: str, t: int, T: int):
        """
         Record state + energy for a layer at step `t`.
         At t == T-1, keeps only the lowest-energy state for each layer_type.
        """
        energy = layer.get_energy()
        x_state = layer.get_x(layer_type)
        if energy is None or x_state is None:
            return

        if layer_type == "embed":
           x_word, x_pos = x_state
           x_state = torch.cat([x_word, x_pos], dim=-1)

        self.records.append({
           "layer_type": layer_type,
           "t": t,
           "energy": energy,
           "x_state": x_state.detach().clone()  # Now safe — always a tensor
        })

    def finalize_recording(self):
       """Call this AFTER all record_step calls at t == T-1."""
       self._reduce_to_best_states() 
    
    def _reduce_to_best_states(self):
        if not self.records:
            return
        layer_groups = {}
        for rec in self.records:
            layer_type = rec["layer_type"]
            if layer_type not in layer_groups:
                layer_groups[layer_type] = []
            layer_groups[layer_type].append(rec)
        # Keep only best (lowest energy) state per layer_type
        self.best_states = {
            layer_type: min(records, key=lambda r: r["energy"])["x_state"]
            for layer_type, records in layer_groups.items()
        }
        # Clear raw records
        self.records = []

    def get_initial_state(self, layer_type: str) -> Optional[torch.Tensor]:
        return self.best_states.get(layer_type, None)

    def clear(self):
        self.records = []
        self.best_states = {}