import torch
import os
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer, load_model, reset_pc_modules, decode_ids, compute_text_metrics
from utils.config_utils import load_best_config
import torch.nn.functional as F
from Data_preprocessing.dataloader import get_loaders
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils.device_utils import setup_device
import argparse
from contextlib import nullcontext
import tiktoken

"""
This script generates text using the trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.

Usage: torchrun --nproc-per-node=<NUM_GPU> generate_text.py

"""


local_rank, device, use_ddp = setup_device()
def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")
   
    
    best_config = load_best_config()
    
    config = GPTConfig(
        vocab_size =  50259,
        block_size = 256,
        peak_learning_rate = best_config["peak_learning_rate"],
        warmup_steps = 10,
        n_embed = 12,
        dropout = 0.1,
        local_learning_rate = 1e-3,
        T = 3,
        is_holding_error = True,
        num_heads = 6,
        n_blocks = 6,
        num_epochs = 1, 
        update_bias = best_config["update_bias"],
        use_lateral = True,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight=0.7,
        combined_output_weight=0.3,
        use_flash_attention=True  
    )

   
    
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    num_samples = 10
    # Generation setup
    top_k = 200
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ptdtype = torch.float32
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    # encode the beginning of the prompt
    start = "\n"
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    if not dist.is_initialized() or dist.get_rank() == 0:
        # decoded_preds, decoded_targets = text_generation(model, config, device, max_samples=2)
        # if decoded_preds and decoded_targets and local_rank == 0:
        #     compute_text_metrics(decoded_preds, decoded_targets)
        vocab_size = enc.n_vocab
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens=300, temperature=0.8, top_k=top_k)
                    print(decode(y[0].tolist()))
                    print('---------------')
    
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()