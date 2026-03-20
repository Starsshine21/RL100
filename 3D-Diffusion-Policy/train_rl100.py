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
    if torch.cuda.is_available():
        visible_gpus = torch.cuda.device_count()
        if visible_gpus > 1:
            cprint(
                f"[Setup] {visible_gpus} CUDA devices are visible, but the current RL100 trainer runs on a single device ({cfg.training.device}).",
                "yellow",
            )

    # Get output directory
    output_dir = os.getcwd()  # Hydra changes cwd to output dir
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
    dataset_samples = len(dataset)
    dataset_episodes = None
    if hasattr(dataset, 'replay_buffer'):
        try:
            dataset_episodes = int(dataset.replay_buffer.n_episodes)
        except Exception:
            dataset_episodes = None
    if dataset_episodes is not None:
        cprint(f"[Setup] Dataset loaded: {dataset_samples} samples across {dataset_episodes} episodes", "green")
    else:
        cprint(f"[Setup] Dataset loaded: {dataset_samples} samples", "green")
    if hasattr(dataset, 'has_rl_data'):
        cprint(f"[Setup] Dataset has reward/done labels: {bool(dataset.has_rl_data)}", "green")

    dataset_num_points = None
    if hasattr(dataset, 'replay_buffer'):
        try:
            dataset_num_points = int(dataset.replay_buffer['point_cloud'].shape[1])
            cprint(f"[Setup] Dataset point_cloud points: {dataset_num_points}", "green")
        except Exception:
            dataset_num_points = None

    # Initialize environment runner
    cprint("[Setup] Initializing environment runner...", "cyan")
    env_runner_kwargs = {'output_dir': output_dir}
    if dataset_num_points is not None:
        configured_points = int(cfg.task.env_runner.get('num_points', dataset_num_points))
        if configured_points != dataset_num_points:
            cprint(
                f"[Setup] env_runner.num_points={configured_points} differs from dataset={dataset_num_points}. "
                f"Override to dataset value to avoid merge shape mismatch.",
                "yellow"
            )
        env_runner_kwargs['num_points'] = dataset_num_points

    env_runner: BaseRunner = hydra.utils.instantiate(cfg.task.env_runner, **env_runner_kwargs)
    cprint("[Setup] Environment runner initialized", "green")

    # Initialize RL100Trainer
    cprint("\n[Setup] Initializing RL100Trainer...", "cyan")
    trainer = RL100Trainer(cfg, output_dir=output_dir)
    cprint("[Setup] RL100Trainer initialized", "green")

    # Optionally load checkpoint
    skip_il = False
    if cfg.training.resume and cfg.training.resume_path:
        cprint(f"\n[Setup] Resuming from checkpoint: {cfg.training.resume_path}", "yellow")
        trainer.load_checkpoint(cfg.training.resume_path)
        skip_il = True
        cprint("[Setup] IL phase will be skipped — starting directly from offline RL.", "yellow")

    # Run training pipeline
    cprint("\n[Training] Starting RL-100 pipeline...", "cyan")
    try:
        trainer.run_pipeline(
            initial_dataset=dataset,
            env_runner=env_runner,
            num_offline_iterations=cfg.training.num_offline_iterations,
            skip_il=skip_il,
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
    final_eval_policies = list(getattr(cfg.runtime, 'final_eval_policies', ['ddim']))
    final_eval_use_ema = bool(getattr(cfg.runtime, 'final_eval_use_ema', False))
    final_results = {}
    for policy_mode in final_eval_policies:
        eval_policy = trainer.get_runtime_policy(mode=policy_mode, use_ema=final_eval_use_ema)
        eval_policy.eval()
        with torch.no_grad():
            metrics = env_runner.run(eval_policy)
        final_results[policy_mode] = metrics

    cprint("\n" + "="*80, "green")
    cprint(" "*25 + "FINAL RESULTS", "green")
    cprint("="*80, "green")
    for policy_mode, metrics in final_results.items():
        cprint(f"[{policy_mode}]", "green")
        for key, value in metrics.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", "green")

    if cfg.logging.use_wandb:
        for policy_mode, metrics in final_results.items():
            wandb.log({f'final/{policy_mode}/{k}': v for k, v in metrics.items()})
        wandb.finish()

    cprint("\n[Training] Complete! Checkpoints saved to:", "magenta")
    cprint(f"  {os.path.join(output_dir, 'checkpoints')}", "magenta")


if __name__ == "__main__":
    main()
