"""
RL-100 Training Script
======================
Main entry point for training RL-100 on 3D-Diffusion-Policy tasks.

Usage:
    python train_rl100.py task=metaworld_push

This script orchestrates the complete RL-100 pipeline:
1. Load configuration
2. Initialize RL100Trainer
3. Run training pipeline (IL -> Offline RL -> Online RL)
4. Save checkpoints and logs

The script uses Hydra for configuration management, allowing easy
experimentation with different hyperparameters and tasks.
"""

import os
import sys
import pathlib

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import random
import wandb
from termcolor import cprint

from diffusion_policy_3d.rl100_trainer import RL100Trainer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner


# Register custom resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config')),
    config_name='rl100'
)
def main(cfg: OmegaConf):
    """
    Main training function.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    cprint("\n" + "="*80, "cyan")
    cprint(" "*30 + "RL-100 TRAINING", "cyan")
    cprint("="*80, "cyan")
    cprint("\nConfiguration:", "yellow")
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    seed = cfg.training.seed
    set_seed(seed)
    cprint(f"\n[Setup] Random seed: {seed}", "green")

    # Get output directory
    output_dir = os.getcwd()  # Hydra changes cwd to output dir
    cfg.output_dir = output_dir
    cprint(f"[Setup] Output directory: {output_dir}", "green")

    # Initialize WandB
    if cfg.logging.use_wandb:
        wandb_run = wandb.init(
            project=cfg.logging.project,
            name=cfg.logging.name,
            group=cfg.logging.group,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=output_dir,
            mode=cfg.logging.mode
        )
        cprint("[Setup] WandB initialized", "green")

    # Load dataset
    cprint("\n[Setup] Loading dataset...", "cyan")
    dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
    cprint(f"[Setup] Dataset loaded: {len(dataset)} episodes", "green")

    # Initialize environment runner
    cprint("[Setup] Initializing environment runner...", "cyan")
    env_runner: BaseRunner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )
    cprint("[Setup] Environment runner initialized", "green")

    # Initialize RL100Trainer
    cprint("\n[Setup] Initializing RL100Trainer...", "cyan")
    trainer = RL100Trainer(cfg)
    cprint("[Setup] RL100Trainer initialized", "green")

    # Optionally load checkpoint
    if cfg.training.resume and cfg.training.resume_path:
        cprint(f"\n[Setup] Resuming from checkpoint: {cfg.training.resume_path}", "yellow")
        trainer.load_checkpoint(cfg.training.resume_path)

    # Run training pipeline
    cprint("\n[Training] Starting RL-100 pipeline...", "cyan")
    try:
        trainer.run_pipeline(
            initial_dataset=dataset,
            env_runner=env_runner,
            num_offline_iterations=cfg.training.num_offline_iterations
        )
    except KeyboardInterrupt:
        cprint("\n[Training] Interrupted by user. Saving checkpoint...", "yellow")
        trainer.save_checkpoint(tag='interrupted')
    except Exception as e:
        cprint(f"\n[Training] Error occurred: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(tag='error')
        raise

    # Final evaluation
    cprint("\n[Evaluation] Running final evaluation...", "cyan")
    eval_policy = trainer.ema_policy if trainer.ema_policy else trainer.policy
    eval_policy.eval()
    with torch.no_grad():
        final_metrics = env_runner.run(eval_policy, num_episodes=50)

    cprint("\n" + "="*80, "green")
    cprint(" "*25 + "FINAL RESULTS", "green")
    cprint("="*80, "green")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            cprint(f"{key}: {value:.4f}", "green")

    if cfg.logging.use_wandb:
        wandb.log({f'final/{k}': v for k, v in final_metrics.items()})
        wandb.finish()

    cprint("\n[Training] Complete! Checkpoints saved to:", "magenta")
    cprint(f"  {os.path.join(output_dir, 'checkpoints')}", "magenta")


if __name__ == "__main__":
    main()
