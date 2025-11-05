import torch
from tokenizers import Tokenizer
from predictive_coding.config import GPTConfig
from utils.model_utils import load_model, reset_pc_modules, decode_ids, compute_text_metrics
from utils.config_utils import load_best_config
import torch.nn.functional as F
from data_preparation.dataloader import get_loaders
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.device_utils import setup_device
import argparse
from data_preparation.config import vocab_size

"""
This script generates text using the trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.

Usage: torchrun --nproc-per-node=<NUM_GPU> generate_text.py

"""


local_rank, device, use_ddp = setup_device()
def generate_text(model, config, input_ids, max_new_tokens, temperature, device = None, use_cache=True):
    model.eval()
    input_tensor = input_ids.unsqueeze(0).to(device)

    # Clear KV cache at the start
    if use_cache:
        for module in model.modules():
            if hasattr(module, 'clear_kv_cache'):
                module.clear_kv_cache()
    
    generated_tokens = []
    
    for step in range(max_new_tokens):
        # For first token or without cache, use full sequence
        # For subsequent tokens with cache, only pass the last token
        if use_cache and step > 0:
            current_input = input_tensor[:, -1:]  
        else:
            current_input = input_tensor
            
        if current_input.size(1) > config.block_size:
            current_input = current_input[:, -config.block_size:]
      
        logits = model(current_input, current_input, use_kv_cache=use_cache)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens.append(next_token.item())
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        if not use_cache:
            reset_pc_modules(model)
        if next_token.item() == getattr(config, 'eos_token_id', None):
            break
    
    # Reset PC modules and clear cache after generation
    reset_pc_modules(model)
    if use_cache:
        for module in model.modules():
            if hasattr(module, 'clear_kv_cache'):
                module.clear_kv_cache()   
                      
    return input_tensor[0] 

def text_generation(model, config, device = None,  max_samples=2, use_cache = True):
    decoded_preds, decoded_targets = [], []
    prompt_len = 5
    total_samples = 0

    _, _, test_loader = get_loaders(distributed=use_ddp)

    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device) 
        batch_size = input_ids.size(0)

        for i in range(batch_size):
            if total_samples >= max_samples:
                break

            prompt_ids = input_ids[i][:prompt_len]
            generated_ids = generate_text(model, config, prompt_ids, max_new_tokens= 50, temperature=0.7, device = device, use_cache = use_cache)

            target_continuation = input_ids[i][prompt_len:]
            target_continuation = target_continuation[target_continuation != 0].tolist()

            generated_continuation = generated_ids[prompt_len:].tolist()

            tokenizer = Tokenizer.from_file("data_preparation/tokenizer.json")

            # Decode all
            prompt_str = decode_ids(tokenizer, prompt_ids.tolist())
            target_str = decode_ids(tokenizer, target_continuation, stop_at_eos=True)
            generated_str = decode_ids(tokenizer, generated_continuation, stop_at_eos=True)

            decoded_preds.append(generated_str)
            decoded_targets.append(target_str)

            
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n[Batch {batch_idx + 1}, Sample {i + 1}]")
                print(f"[PROMPT ]: {prompt_str}")
                print(f"[TARGET ]: {target_str}")
                print(f"[PREDICT]: {generated_str}")
                if use_cache:
                    print("[Using KV Cache]")
            
            total_samples += 1

        if total_samples >= max_samples:
            break

    return decoded_preds, decoded_targets

def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    args = parser.parse_args()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")
    
    best_config = load_best_config()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = best_config["block_size"],
        lr = best_config["peak_learning_rate"],
        peak_learning_rate = best_config["peak_learning_rate"],
        warmup_steps = best_config["warmup_steps"],
        n_embed = best_config["n_embed"],
        dropout = best_config["dropout"],
        T = best_config["T"],
        num_heads = best_config["num_heads"],
        n_blocks = best_config["n_blocks"],
        batch_size = 8,
        num_epochs = 1,
        update_bias = best_config["update_bias"],
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight=0.7,
        combined_output_weight=0.3,
        use_flash_attention=False,
        alpha = best_config["alpha"]    
    )
    
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if not dist.is_initialized() or dist.get_rank() == 0:
        decoded_preds, decoded_targets = text_generation(model, config, device, max_samples=2, use_cache=True)
        if decoded_preds and decoded_targets and local_rank == 0:
            compute_text_metrics(decoded_preds, decoded_targets)
    
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()