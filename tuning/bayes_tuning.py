import torch
import logging
import optuna
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tuning.trial_objective import objective
from tuning.tuning_logs import initialize_logs, write_final_results
import torch.distributed as dist
from utils.device_utils import setup_device
from utils.model_utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

"""
This script performs Bayesian hyperparameter tuning for the model using Optuna. 
It supports energy-based evaluation, model configuration, and training setup,
and is compatible with multi-GPU execution using DDP.

Usage:  torchrun --nproc-per-node=<NUM_GPU> tuning/bayes_tuning.py 

"""

def log_best_trial(trial, prefix=""):
    """Helper to log best trial info"""
    energy = trial.user_attrs.get("energy", "N/A")
    perplexity = trial.user_attrs.get("perplexity", "N/A")
    combined = trial.user_attrs.get("combined_loss", "N/A")
    logger.info(f"{prefix}Trial {trial.number} | Combined: {combined:.5f} | Energy: {energy:.4f} | Perplexity: {perplexity:.4f}")

def run_tuning(n_trials, study_name, local_rank, device, flash=False, enable_batch_logging=False):
    """Run clean dynamic hyperparameter tuning"""
    storage_url = f"sqlite:///tuning/{study_name}.db"
    
    # Create or load study (only rank 0 creates)
    if local_rank == 0:
        _ = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
        )
        initialize_logs(study_name)
        logger.info(f"Starting tuning: {n_trials} trials")
        
    if dist.is_initialized():
        dist.barrier()

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    def callback(study, trial):
        if local_rank == 0:
            log_best_trial(study.best_trial, prefix="\nBest so far: ")

    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, device, flash, enable_batch_logging),
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=(local_rank == 0)
        )
        
        if local_rank == 0:
            logger.info("Tuning completed!")
            log_best_trial(study.best_trial, prefix="\nFinal best: ")
            write_final_results(f"tuning/{study_name}_results.txt", study.best_trial)
    
    except KeyboardInterrupt:
        logger.warning(f"[Rank {local_rank}] Tuning interrupted")
      
    if dist.is_initialized():
        dist.barrier()
          
    return study

if __name__ == "__main__":
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Tuning with Predictive Coding Transformer")
    parser.add_argument('--flash', '--flash_attention', action='store_true', help='Enable FlashAttention for attention layers')
    parser.add_argument('--log_batches', action='store_true', help='Enable batch-level logging during tuning')
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank, device, use_ddp = setup_device()
    
    if use_ddp and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    if use_ddp:
        dist.barrier()
    
    study = run_tuning(n_trials= 70, study_name="bayesian_tuning", local_rank=local_rank, device=device, flash=args.flash, enable_batch_logging=args.log_batches)

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()