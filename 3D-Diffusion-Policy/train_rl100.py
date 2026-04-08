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

    dataset_point_cloud_shape = None
    if hasattr(dataset, 'replay_buffer'):
        try:
            dataset_point_cloud_shape = tuple(int(x) for x in dataset.replay_buffer['point_cloud'].shape[1:])
            cprint(f"[Setup] Dataset point_cloud shape: {dataset_point_cloud_shape}", "green")
        except Exception:
            dataset_point_cloud_shape = None

    # Initialize environment runner
    cprint("[Setup] Initializing environment runner...", "cyan")
    configured_points = cfg.task.env_runner.get('num_points', None)
    configured_pc_shape = None
    try:
        configured_pc_shape = tuple(int(x) for x in cfg.task.shape_meta.obs.point_cloud.shape)
    except Exception:
        configured_pc_shape = None

    if (
        dataset_point_cloud_shape is not None
        and configured_pc_shape is not None
        and dataset_point_cloud_shape != configured_pc_shape
    ):
        cprint(
            f"[Setup] Dataset point_cloud shape {dataset_point_cloud_shape} differs from "
            f"configured shape {configured_pc_shape}. Re-collect demos to fully align training data; "
            f"new rollouts will follow the configured shape and be auto-aligned only when merged.",
            "yellow"
        )
    elif dataset_point_cloud_shape is not None and configured_points is not None and int(configured_points) != dataset_point_cloud_shape[0]:
        cprint(
            f"[Setup] Using configured env_runner.num_points={int(configured_points)} "
            f"while dataset point_cloud has {dataset_point_cloud_shape[0]} points. "
            f"Collected point clouds will be auto-aligned on merge.",
            "yellow"
        )

    env_runner: BaseRunner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        seed=seed,
    )
    cprint("[Setup] Environment runner initialized", "green")

    # Initialize RL100Trainer
    cprint("\n[Setup] Initializing RL100Trainer...", "cyan")
    trainer = RL100Trainer(cfg, output_dir=output_dir)
    cprint("[Setup] RL100Trainer initialized", "green")

    # Optionally load checkpoint
    skip_il = False
    if cfg.training.resume and cfg.training.resume_path:
        load_rl_state = bool(getattr(cfg.training, 'resume_load_rl_state', True))
        cprint(f"\n[Setup] Resuming from checkpoint: {cfg.training.resume_path}", "yellow")
        trainer.load_checkpoint(cfg.training.resume_path, load_rl_state=load_rl_state)
        skip_il = True
        if load_rl_state:
            cprint("[Setup] IL phase will be skipped — resuming RL state from checkpoint.", "yellow")
        else:
            cprint("[Setup] IL phase will be skipped — starting offline RL from restored IL policy with fresh RL heads.", "yellow")

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
