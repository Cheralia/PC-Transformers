import torch
import os
import torch.nn as nn
import math
import time
import torch.nn.functional as F
import torch.distributed as dist
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from Data_preprocessing.dataloader import get_loaders
from utils.model_utils import load_tokenizer, reset_pc_modules
from utils.config_utils import load_best_config
from utils.pc_utils import cleanup_memory
from eval import evaluate
from visualization import plot_metrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.device_utils import setup_device, cleanup_memory
import json
import logging
import tiktoken
from contextlib import nullcontext
import requests
import numpy as np
best_config = load_best_config()  
config = GPTConfig(
        # vocab_size = 50340,

        # block_size = 256,
        # peak_learning_rate = best_config["peak_learning_rate"],
        # warmup_steps = 100,
        # n_embed = 12,
        # dropout = 0.1,
        # local_learning_rate = 2e-3,
        # T = 1,
        # is_holding_error = True,
        # num_heads = 3,
        # n_blocks = 2,
        # num_epochs = 5, 
        # update_bias = best_config["update_bias"],
        # use_lateral = True,
        # internal_energy_fn_name="pc_e",
        # output_energy_fn_name="pc_e",
        # combined_internal_weight=0.7,
        # combined_output_weight=0.3,
        # use_flash_attention=True  
        
        vocab_size = 50259,
        block_size =  176,
        local_learning_rate =  1e-05,
        peak_learning_rate = 1.1479716638455936e-05,
        warmup_steps = 166,
        la =  0.5,
        n_embed = 64,
        dropout =  0.1854371025507931,
        T = 6,
        is_holding_error =  True,
        update_bias = True,
        num_heads = 4,
        n_blocks = 5,
        batch_size = 8,
        num_epochs = 3,
        use_lateral =  True,
        internal_energy_fn_name = "pc_e",
        output_energy_fn_name = "pc_e",
        eos_token_id =  None,
        use_flash_attention = False,
        combined_internal_weight = 0.3,
        combined_output_weight = 0.7,
    )
"""
This script trains the predictive coding transformer model on the provided dataset.
It tracks and plots the average predictive coding energy per epoch and saves the trained model.

Usage: torchrun --nproc-per-node=<NUM_GPU> training.py

"""
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files in 'data' directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def train(model, config, global_step, device, logger,num_batches=100):
    model.train()
    total_ce_loss = 0.0
    total_energy = 0.0
    batch_count = 0
    # pad_token_id = tokenizer.pad_token_id
    # vocab_size = len(tokenizer)

    base_model = model.module if hasattr(model, 'module') else model
    output_pc_layer = base_model.output.pc_layer

    alpha = getattr(config, 'combined_internal_weight', 0.3)
    beta = getattr(config, 'combined_output_weight', 0.7)
    

    # for batch_idx, batch in enumerate(dataloader):
    #     input_ids = batch["input_ids"].to(device)
    #     target_ids = batch["target_ids"].to(device)
    for batch_idx in range(num_batches):
        xb, yb = get_batch('train')
        logits,ce_loss = model(xb, yb)

        # if target_ids.max() >= vocab_size:
        #     target_ids = torch.clamp(target_ids, max=vocab_size - 1)

        if global_step < config.warmup_steps:
            lr = config.local_learning_rate + global_step / config.warmup_steps * (
                config.peak_learning_rate - config.local_learning_rate)
        else:
            lr = config.peak_learning_rate

        for module in model.modules():
            if hasattr(module, 'local_lr'):
                module.set_learning_rate(lr)
                
        global_step += 1
        # if target_ids.max() >= config.vocab_size:
        #     target_ids = torch.clamp(target_ids, max=vocab_size-1)
            
            
        # logits = model(target_ids, input_ids)
        # ce_loss = F.cross_entropy(
        #     logits.view(-1, logits.size(-1)),
        #     target_ids.view(-1),
        #     ignore_index=-
        # )
        total_ce_loss += ce_loss.item()

        internal_energies = []
        output_energy = None

        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is None or (isinstance(energy, float) and math.isnan(energy)):
                    continue

                if hasattr(module, 'layer_type') and module.layer_type == 'linear_output':
                    if getattr(module, 'energy_fn_name', None) == "kld":
                        output_energy = energy
                    else:
                        internal_energies.append(energy)
                else:
                    internal_energies.append(energy)

                if hasattr(module, "_head_similarity_avg"):
                    _ = module._head_similarity_avg
                if hasattr(module, "_head_similarity_max"):
                    _ = module._head_similarity_max

        avg_internal_energy = sum(internal_energies) / len(internal_energies) if internal_energies else ce_loss.item()
                
        if output_energy is not None:
           avg_output_energy = output_energy
           batch_energy = alpha * avg_internal_energy + beta* avg_output_energy 
        else:
            batch_energy = avg_internal_energy
        total_energy += batch_energy
        batch_count += 1

        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            if logger:
                logger.info(f"  Batch {batch_idx + 1}/{num_batches} | Batch Energy: {batch_energy:.4f} | Batch Loss: {ce_loss.item():.4f}  | Perplexity: {perplexity:.4f}")
            else:
                print(f"  Batch {batch_idx + 1}/{num_batches} | Batch Energy: {batch_energy:.4f} | Batch Loss: {ce_loss.item():.4f} | Perplexity: {perplexity:.4f}")

        reset_pc_modules(model)
        cleanup_memory()
    

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")
    return avg_energy, avg_perplexity, global_step


def main():
    local_rank, device, use_ddp = setup_device()
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    rank = dist.get_rank() if dist.is_initialized() else 0

    best_config = load_best_config()   
    # Configure logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # build handlers and remove existing ones
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    root_logger.addHandler(stream_h)

    if rank == 0:
        file_h = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="a")
        file_h.setFormatter(fmt)
        root_logger.addHandler(file_h)

    logger = logging.getLogger(__name__)
   
    # config = GPTConfig(
    #     vocab_size = vocab_size,

    #     block_size = best_config["block_size"],
    #     peak_learning_rate = best_config["peak_learning_rate"],
    #     warmup_steps = best_config["warmup_steps"],
    #     n_embed = best_config["n_embed"],
    #     dropout = best_config["dropout"],
    #     local_learning_rate = 1e-5,
    #     T = best_config["T"],
    #     is_holding_error = True,
    #     num_heads = best_config["num_heads"],
    #     n_blocks = best_config["n_blocks"],
    #     num_epochs = 20, 
    #     update_bias = best_config["update_bias"],
    #     use_lateral = True,
    #     internal_energy_fn_name="pc_e",
    #     output_energy_fn_name="pc_e",
    #     eos_token_id=tokenizer.eos_token_id,
    #     combined_internal_weight=0.7,
    #     combined_output_weight=0.3,
    #     use_flash_attention=True  
    # )
    #cpu
   
    # Create a separate logger for hyperparameters
    param_logger = logging.getLogger('param_logger')
    param_logger.setLevel(logging.INFO)
    if rank == 0 and root_logger.handlers:
        param_logger.addHandler(root_logger.handlers[1])
        param_logger.propagate = False

    if rank == 0:
        param_logger.info(f"\n{'#' * 120}") 
        logger.info(f"Using device: {device} (local rank {local_rank})")
        try:
            cfg = config.__dict__
        except Exception:
            cfg = {k: getattr(config, k) for k in dir(config) if not k.startswith("_") and not callable(getattr(config, k))}
        config_json = json.dumps(cfg, indent=6, default=str)
        param_logger.info("Saving the hyperparameters configurations:")
        param_logger.info(config_json)

    model = PCTransformer(config).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], 
                    output_device=local_rank, 
                    find_unused_parameters=True)

        model.module.register_all_lateral_weights()

    #train_loader, valid_loader, _ = get_loaders(distributed=use_ddp)
    
    global_step = 0
    train_energies = []
    val_energies = []
    start_time = time.time()
    if rank == 0:
        logger.info("========== Training started ==========") 
        logger.info(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

    for epoch in range(config.num_epochs):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        num_batches=100
        model.train()
        train_energy, train_perplexity, global_step = train(
            model, config, global_step, device, logger,num_batches)
        train_energies.append(train_energy)

        # model.eval()
        # val_energy, val_perplexity = evaluate(
        #     model, valid_loader, tokenizer, max_batches=None, device=device
        # )
        
        # val_energies.append(val_energy)

        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} | "
                  f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
                os.makedirs("checkpoints", exist_ok=True)
                # Get the underlying model (handle both DDP and non-DDP cases)
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'train_energy': train_energy,
                    # 'val_energy': val_energy,
                    'train_perplexity': train_perplexity,
                    # 'val_perplexity': val_perplexity
                }
                checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    if rank == 0:
        plot_metrics(train_energies, val_energies)
        os.makedirs("checkpoints", exist_ok=True)
        # Get the underlying model (handle both DDP and non-DDP cases)
        model_to_save = model.module if hasattr(model, 'module') else model
        final_checkpoint = {
            'epoch': config.num_epochs,
            'model_state_dict': model_to_save.state_dict(),
            'train_energy': train_energy,
            # 'val_energy': val_energy,
            'train_perplexity': train_perplexity,
            # 'val_perplexity': val_perplexity
        }
        torch.save(final_checkpoint, 'checkpoints/final_model.pt')
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info("Final model saved to: checkpoints/final_model.pt")
        logger.info("========== Training completed ==========")

    # dist.destroy_process_group()
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()