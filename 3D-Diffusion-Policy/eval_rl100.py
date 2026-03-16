"""
RL-100 Evaluation Script
========================
Evaluate a trained RL-100 checkpoint on a given task.

Usage:
    python eval_rl100.py \
        task=metaworld_dial-turn \
        checkpoint_path=/path/to/final.ckpt \
        task.env_runner.eval_episodes=100 \
        hydra.run.dir=data/outputs/eval_rl100
"""

import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from termcolor import cprint

from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.model.rl.consistency_model import ConsistencyModel
from diffusion_policy_3d.policy.consistency_policy import ConsistencyPolicyWrapper

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config')),
    config_name='rl100'
)
def main(cfg: DictConfig):
    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None:
        raise ValueError("Must specify checkpoint_path=... on the command line.")
    policy_mode = str(getattr(cfg.runtime, 'eval_policy_mode', 'ddim')).lower()
    eval_use_ema = bool(getattr(cfg.runtime, 'eval_use_ema', False))

    device = torch.device(cfg.training.device)
    cprint(f"[Eval] Loading checkpoint: {checkpoint_path}", "cyan")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Instantiate policy only (no critics / optimizers needed)
    cprint("[Eval] Instantiating RL100Policy...", "cyan")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)

    if eval_use_ema and 'ema_policy' in checkpoint:
        cprint("[Eval] Loading ema_policy weights.", "green")
        policy.load_state_dict(checkpoint['ema_policy'])
    else:
        if eval_use_ema and 'ema_policy' not in checkpoint:
            cprint("[Eval] ema_policy not found, falling back to policy weights.", "yellow")
        else:
            cprint("[Eval] Loading policy weights.", "green")
        policy.load_state_dict(checkpoint['policy'])

    runtime_policy = policy
    if policy_mode == 'cm':
        cprint("[Eval] Building 1-step consistency policy wrapper.", "cyan")
        consistency_model = ConsistencyModel(
            input_dim=policy.action_dim,
            global_cond_dim=(
                policy.obs_feature_dim
                if "cross_attention" in policy.condition_type
                else policy.obs_feature_dim * policy.n_obs_steps
            ),
            diffusion_step_embed_dim=cfg.policy.diffusion_step_embed_dim,
            down_dims=cfg.policy.down_dims,
            kernel_size=cfg.policy.kernel_size,
            n_groups=cfg.policy.n_groups,
            condition_type=cfg.policy.condition_type
        )
        consistency_model.to(device)
        consistency_model.load_state_dict(checkpoint['consistency_model'])
        runtime_policy = ConsistencyPolicyWrapper(policy, consistency_model)
        runtime_policy.to(device)
    elif policy_mode != 'ddim':
        raise ValueError(f"Unsupported eval policy mode: {policy_mode}")

    runtime_policy.eval()

    # Instantiate env runner
    output_dir = os.getcwd()
    cprint("[Eval] Instantiating env runner...", "cyan")
    env_runner: BaseRunner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir
    )

    cprint(f"[Eval] Running evaluation with {policy_mode} policy ({cfg.task.env_runner.eval_episodes} episodes)...", "cyan")
    with torch.no_grad():
        metrics = env_runner.run(runtime_policy)

    cprint("\n" + "="*60, "green")
    cprint(" "*20 + "EVAL RESULTS", "green")
    cprint("="*60, "green")
    for key, value in metrics.items():
        if isinstance(value, float):
            cprint(f"  {key}: {value:.4f}", "green")


if __name__ == "__main__":
    main()
