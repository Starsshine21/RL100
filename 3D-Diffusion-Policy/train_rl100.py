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
from diffusion_policy_3d.common.config_util import (
    get_final_eval_policies,
    get_task_execution_mode,
    is_amq_enabled,
    is_cm_policy_enabled,
    is_eval_enabled,
)
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
    execution_mode = get_task_execution_mode(cfg)
    eval_enabled = is_eval_enabled(cfg)
    amq_enabled = is_amq_enabled(cfg)
    cm_enabled = is_cm_policy_enabled(cfg)
    cprint(
        f"[Setup] task.execution mode={execution_mode}, "
        f"enable_eval={eval_enabled}, "
        f"enable_amq={amq_enabled}, "
        f"enable_cm_policy={cm_enabled}",
        "green",
    )
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
    env_runner = None
    cprint("[Setup] Initializing environment runner...", "cyan")
    env_runner_cfg = cfg.task.get('env_runner', None)
    offline_collection_requested = (
        int(getattr(cfg.training, 'num_offline_iterations', 0)) > 0
        and bool(getattr(cfg.training, 'offline_collect_new_data', True))
        and int(getattr(cfg.training, 'collection_episodes', 0)) > 0
    )
    requires_rollout_env = (
        offline_collection_requested
        or bool(getattr(cfg.training, 'run_online_rl', False))
    )
    needs_env_runner = requires_rollout_env or eval_enabled
    configured_points = env_runner_cfg.get('num_points', None) if env_runner_cfg is not None else None
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

    if env_runner_cfg is not None and needs_env_runner:
        env_runner = hydra.utils.instantiate(
            env_runner_cfg,
            output_dir=output_dir,
            seed=seed,
        )
        cprint("[Setup] Environment runner initialized", "green")
    elif env_runner_cfg is not None:
        cprint(
            "[Setup] task.env_runner is configured but not needed because both rollout and eval are disabled.",
            "yellow",
        )
    else:
        cprint("[Setup] task.env_runner is null; skipping online collection/evaluation runner init.", "yellow")

    if env_runner is None:
        if needs_env_runner:
            raise ValueError(
                "This config requires task.env_runner, because rollout and/or evaluation are enabled. "
                "Set task.env_runner for sim or real-robot execution, or disable the corresponding stages."
            )
    elif (
        getattr(env_runner, 'env', object()) is None
        and needs_env_runner
    ):
        raise ValueError(
            "task.env_runner is instantiated but task.env_runner.env is null. "
            "Provide a real/sim env via task.env_runner.env._target_=... for rollout/evaluation, "
            "or disable the corresponding stages."
        )

    # Initialize RL100Trainer
    cprint("\n[Setup] Initializing RL100Trainer...", "cyan")
    trainer = RL100Trainer(cfg, output_dir=output_dir)
    cprint("[Setup] RL100Trainer initialized", "green")

    # Optionally load checkpoint
    skip_il = False
    resume_il = False
    if cfg.training.resume and cfg.training.resume_path:
        load_rl_state = bool(getattr(cfg.training, 'resume_load_rl_state', True))
        resume_il = bool(getattr(cfg.training, 'resume_il', False))
        if resume_il and not load_rl_state:
            raise ValueError("training.resume_il=true requires training.resume_load_rl_state=true.")
        cprint(f"\n[Setup] Resuming from checkpoint: {cfg.training.resume_path}", "yellow")
        trainer.load_checkpoint(cfg.training.resume_path, load_rl_state=load_rl_state)
        skip_il = not resume_il
        if resume_il:
            cprint("[Setup] IL phase will resume from checkpoint state.", "yellow")
        elif load_rl_state:
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
            resume_il=resume_il,
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
    final_results = {}
    final_eval_policies = get_final_eval_policies(cfg, default=['ddim'])
    if env_runner is not None and eval_enabled and final_eval_policies:
        cprint("\n[Evaluation] Running final evaluation...", "cyan")
        final_eval_use_ema = bool(getattr(cfg.runtime, 'final_eval_use_ema', False))
        for policy_mode in final_eval_policies:
            eval_policy = trainer.get_runtime_policy(mode=policy_mode, use_ema=final_eval_use_ema)
            eval_policy.eval()
            with torch.no_grad():
                metrics = env_runner.run(eval_policy)
            final_results[policy_mode] = metrics
    elif not eval_enabled:
        cprint("\n[Evaluation] Skipped final evaluation because task.execution.enable_eval=false.", "yellow")
    elif not final_eval_policies:
        cprint("\n[Evaluation] Skipped final evaluation because no runtime policies are enabled.", "yellow")
    else:
        cprint("\n[Evaluation] Skipped final evaluation because no env_runner is configured.", "yellow")

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
