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
# def generate_text(model, config, input_ids, max_new_tokens, temperature, device = None):
#     model.eval()
#     input_tensor = input_ids.unsqueeze(0).to(device)

#     for _ in range(max_new_tokens):
#         if input_tensor.size(1) > config.block_size:
#             input_tensor = input_tensor[:, -config.block_size:]
      
#         logits = model(input_tensor, input_tensor)
#         logits = logits[:, -1, :] / temperature
#         probs = F.softmax(logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1)
#         input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
#         reset_pc_modules(model)
#         if next_token.item() == config.eos_token_id:
#             break
                
#     return input_tensor[0] 

# def text_generation(model, config, device = None,  max_samples=2):
#     decoded_preds, decoded_targets = [], []
#     prompt_len = 5
#     total_samples = 0

#     _, _, test_loader = get_loaders(distributed=use_ddp)
#     tokenizer = load_tokenizer()
#     pad_token_id = tokenizer.pad_token_id

#     for batch_idx, batch in enumerate(test_loader):
#         input_ids = batch["input_ids"].to(device) 
#         batch_size = input_ids.size(0)

#         for i in range(batch_size):
#             if total_samples >= max_samples:
#                 break

#             prompt_ids = input_ids[i][:prompt_len]
#             generated_ids = generate_text(model, config, prompt_ids, max_new_tokens= 50, temperature=0.7, device = device)

#             target_continuation = input_ids[i][prompt_len:]
#             target_continuation = target_continuation[target_continuation != pad_token_id].tolist()

#             generated_continuation = generated_ids[prompt_len:].tolist()

#             # Decode all
#             prompt_str = decode_ids(tokenizer, prompt_ids.tolist())
#             target_str = decode_ids(tokenizer, target_continuation, stop_at_eos=True)
#             generated_str = decode_ids(tokenizer, generated_continuation, stop_at_eos=True)

#             decoded_preds.append(generated_str)
#             decoded_targets.append(target_str)

            
#             if not dist.is_initialized() or dist.get_rank() == 0:
#                 print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
#                 print(f"[PROMPT ]: {prompt_str}")
#                 print(f"[TARGET ]: {target_str}")
#                 print(f"[PREDICT]: {generated_str}")
            
#             total_samples += 1

#         if total_samples >= max_samples:
#             break

#     return decoded_preds, decoded_targets
# best_config = load_best_config()
# config = GPTConfig(
#         vocab_size = 50340,
#         block_size = best_config["block_size"],
#         n_embed = best_config["n_embed"],
#         dropout = best_config["dropout"],
#         local_learning_rate = best_config["peak_learning_rate"],
#         T = best_config["T"],
#         is_holding_error = True,
#         num_heads = best_config["num_heads"],
#         n_blocks = best_config["n_blocks"],
#         num_epochs = 1,
#         internal_energy_fn_name="pc_e",
#         output_energy_fn_name="pc_e",
#         update_bias = best_config["update_bias"],
      
#     )
best_config = load_best_config()
config = GPTConfig(
        vocab_size =  50340,

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

# @torch.no_grad()
# def generate(self,idx, max_new_tokens, temperature=1.0, top_k=None):
#     """
#     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
#     the sequence max_new_tokens times, feeding the predictions back into the model each time.
#     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
#     """
#     for _ in range(max_new_tokens):
#         # if the sequence context is growing too long we must crop it at block_size
#         idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
#         # forward the model to get the logits for the index in the sequence
#         logits, _ = self(idx_cond)
#         # pluck the logits at the final step and scale by desired temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop the logits to only the top k options
#         if top_k is not None:
#             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#             logits[logits < v[:, [-1]]] = -float('Inf')
#         # apply softmax to convert logits to (normalized) probabilities
#         probs = F.softmax(logits, dim=-1)
#         # sample from the distribution
#         idx_next = torch.multinomial(probs, num_samples=1)
#         # append sampled index to the running sequence and continue
#         idx = torch.cat((idx, idx_next), dim=1)

#     return idx


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    
    best_config = load_best_config()

    # config = GPTConfig(
    #     vocab_size = vocab_size,
    #     block_size = best_config["block_size"],
    #     n_embed = best_config["n_embed"],
    #     dropout = best_config["dropout"],
    #     local_learning_rate = best_config["peak_learning_rate"],
    #     T = best_config["T"],
    #     is_holding_error = True,
    #     num_heads = best_config["num_heads"],
    #     n_blocks = best_config["n_blocks"],
    #     num_epochs = 1,
    #     internal_energy_fn_name="pc_e",
    #     output_energy_fn_name="pc_e",
    #     update_bias = best_config["update_bias"],
    #     eos_token_id = tokenizer.eos_token_id
    # )
    
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