import torch
import random
from typing import Optional  
class PCReplayBuffer:
    def __init__(self, fifo: bool = True, max_size_per_layer: int = 100):
        """
        Replay buffer supporting two modes:
        
        - fifo=True (default): 
            Stores up to `max_size_per_layer` states per layer (FIFO).
            `get_initial_state()` returns a random sample from the buffer.
            
        - fifo=False:
            Keeps only the lowest-energy state per layer (legacy "best state" mode).
        """
        self.fifo = fifo
        self.max_size_per_layer = max_size_per_layer

        if self.fifo:
            # FIFO mode: map layer_type -> list of (x_state)
            self._fifo_buffer = {}
            self.records = []      
            self.best_states = {} 
        else:
            self.records = []      
            self.best_states = {}  

    def record_step(self, layer, layer_type: str, t: int, T: int):
        """
        Record state for a layer at step `t`.
        In FIFO mode: only records at t == T - 1.
        In best-energy mode: records every step, reduces at t == T - 1.
        """
        x_state = layer.get_x(layer_type)
        if x_state is None:
            return

        if layer_type == "embed":
            x_word, x_pos = x_state
            x_state = torch.cat([x_word, x_pos], dim=-1)
        x_state = x_state.detach().clone()

        if self.fifo:
            # FIFO: only record final step
            if t == T - 1:
                if layer_type not in self._fifo_buffer:
                    self._fifo_buffer[layer_type] = []
                buf = self._fifo_buffer[layer_type]
                buf.append(x_state)
                if len(buf) > self.max_size_per_layer:
                    buf.pop(0)  
        else:
            # Best-energy mode: record every step
            energy = layer.get_energy()
            if energy is None:
                return
            self.records.append({
                "layer_type": layer_type,
                "t": t,
                "energy": energy,
                "x_state": x_state
            })
            
    def finalize_recording(self):
        """No-op in FIFO mode. In best-energy mode, finalize reduction."""
        if not self.fifo:
            self._reduce_to_best_states()

    def _reduce_to_best_states(self):
        """Keep only lowest-energy state per layer (best-energy mode only)."""
        if not self.records:
            return
        layer_groups = {}
        for rec in self.records:
            lt = rec["layer_type"]
            layer_groups.setdefault(lt, []).append(rec)
        self.best_states = {
            lt: min(recs, key=lambda r: r["energy"])["x_state"]
            for lt, recs in layer_groups.items()
        }
        self.records = []

    def get_initial_state(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get initial state: random from FIFO buffer, or best state if not FIFO."""
        if self.fifo:
            buf = self._fifo_buffer.get(layer_type, [])
            return random.choice(buf) if buf else None
        else:
            return self.best_states.get(layer_type, None)

    def clear(self):
        """Clear all stored states."""
        if self.fifo:
            self._fifo_buffer.clear()
        else:
            self.records.clear()
            self.best_states.clear()