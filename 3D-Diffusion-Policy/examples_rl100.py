"""
RL-100 Usage Examples
=====================
Demonstrates how to use RL-100 components independently.

This script provides minimal examples for:
1. IQL Critics training
2. Consistency distillation
3. K-step PPO loss computation
4. Complete RL-100 pipeline

Usage:
    python examples/rl100_examples.py --example [critics|consistency|ppo|pipeline]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


# ============================================
# Example 1: Using IQL Critics
# ============================================

def example_iql_critics():
    """Example: Training IQL critics on a batch of data."""
    print("\n" + "="*60)
    print(" "*15 + "Example 1: IQL Critics")
    print("="*60 + "\n")

    from diffusion_policy_3d.model.rl.iql_critics import IQLCritics

    # Initialize critics
    obs_dim = 64  # PointNet encoder output
    action_dim = 4  # Metaworld push action dim
    critics = IQLCritics(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=0.99,
        tau=0.7
    )

    # Create dummy data (in practice, from dataset)
    batch_size = 256
    obs_features = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    rewards = torch.randn(batch_size, 1)
    next_obs_features = torch.randn(batch_size, obs_dim)
    dones = torch.zeros(batch_size, 1)

    # Create optimizers
    v_optimizer = torch.optim.AdamW(critics.v_network.parameters(), lr=3e-4)
    q_optimizer = torch.optim.AdamW(critics.q_network.parameters(), lr=3e-4)

    print("Training IQL Critics...")

    # Training loop
    for step in range(10):
        # 1. Update V network
        v_loss, v_info = critics.compute_v_loss(obs_features, actions)
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        # 2. Update Q network
        q_loss, q_info = critics.compute_q_loss(
            obs_features, actions, rewards, next_obs_features, dones
        )
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # 3. Update target network
        critics.update_target_network(tau=0.005)

        if step % 5 == 0:
            print(f"Step {step}: V Loss={v_loss.item():.4f}, Q Loss={q_loss.item():.4f}")

    # Compute advantage
    advantages = critics.compute_advantage(obs_features, actions)
    print(f"\nAdvantage stats:")
    print(f"  Mean: {advantages.mean().item():.4f}")
    print(f"  Std:  {advantages.std().item():.4f}")
    print(f"  Min:  {advantages.min().item():.4f}")
    print(f"  Max:  {advantages.max().item():.4f}")

    print("\n✓ IQL Critics example complete!")


# ============================================
# Example 2: Consistency Distillation
# ============================================

def example_consistency_distillation():
    """Example: Distilling a teacher policy into a consistency model."""
    print("\n" + "="*60)
    print(" "*10 + "Example 2: Consistency Distillation")
    print("="*60 + "\n")

    from diffusion_policy_3d.model.rl.consistency_model import ConsistencyModel

    # Initialize models
    action_dim = 4
    obs_feature_dim = 64
    n_obs_steps = 2
    horizon = 16

    # Student (consistency model)
    student = ConsistencyModel(
        input_dim=action_dim,
        global_cond_dim=obs_feature_dim * n_obs_steps,
        diffusion_step_embed_dim=128,
        down_dims=(512, 1024, 2048)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    print("Training consistency model...")

    # Simulate distillation
    batch_size = 16
    for step in range(10):
        # Sample data
        noise = torch.randn(batch_size, horizon, action_dim)
        global_cond = torch.randn(batch_size, obs_feature_dim * n_obs_steps)

        # Teacher output (simulated - in practice, use DP3.conditional_sample)
        teacher_output = torch.randn(batch_size, horizon, action_dim)

        # Student prediction
        student_output = student(noise, global_cond)

        # Distillation loss
        loss = nn.functional.mse_loss(student_output, teacher_output)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}: Distillation Loss={loss.item():.4f}")

    # Fast inference test
    print("\nTesting 1-step generation...")
    with torch.no_grad():
        fast_action = student.predict_action(
            batch_size=4,
            horizon=horizon,
            global_cond=torch.randn(4, obs_feature_dim * n_obs_steps),
            device=torch.device('cpu')
        )
    print(f"Generated action shape: {fast_action.shape}")
    print(f"Generated action range: [{fast_action.min().item():.2f}, {fast_action.max().item():.2f}]")

    print("\n✓ Consistency distillation example complete!")


# ============================================
# Example 3: K-step PPO Loss Computation
# ============================================

def example_ppo_loss():
    """Example: Computing K-step PPO loss."""
    print("\n" + "="*60)
    print(" "*10 + "Example 3: K-step PPO Loss")
    print("="*60 + "\n")

    # Simulate denoising trajectory
    K = 10  # Number of denoising steps
    batch_size = 256
    horizon = 16
    action_dim = 4

    print(f"Simulating {K}-step denoising trajectory...")

    # Old trajectory (from old policy)
    trajectory_old = [torch.randn(batch_size, horizon, action_dim) for _ in range(K + 1)]

    # Old log probabilities for each step
    log_probs_old = [torch.randn(batch_size) for _ in range(K)]

    # New log probabilities (from new policy)
    # In practice, recompute these by running new policy
    log_probs_new = [torch.randn(batch_size) for _ in range(K)]

    # Advantages from IQL critics
    advantages = torch.randn(batch_size, 1)

    print(f"\nComputing PPO loss over {K} steps...")

    # Compute PPO loss
    ppo_clip_eps = 0.2
    total_loss = 0
    ratios = []

    for k in range(K):
        # Probability ratio
        ratio = torch.exp(log_probs_new[k] - log_probs_old[k])
        ratios.append(ratio.mean().item())

        # Clipped ratio
        ratio_clipped = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps)

        # PPO objective
        adv = advantages.squeeze()
        surr1 = ratio * adv
        surr2 = ratio_clipped * adv
        loss_k = -torch.min(surr1, surr2).mean()

        total_loss += loss_k

        if k % 3 == 0:
            print(f"  Step {k}: Ratio={ratio.mean().item():.4f}, Loss={loss_k.item():.4f}")

    print(f"\nTotal PPO Loss: {total_loss.item():.4f}")
    print(f"Ratio statistics:")
    print(f"  Mean: {np.mean(ratios):.4f}")
    print(f"  Min:  {np.min(ratios):.4f}")
    print(f"  Max:  {np.max(ratios):.4f}")

    # Check if ratios are in safe range
    if all(0.5 < r < 2.0 for r in ratios):
        print("✓ All ratios in safe range [0.5, 2.0]")
    else:
        print("⚠ Some ratios outside safe range - may need to adjust learning rate")

    print("\n✓ K-step PPO loss example complete!")


# ============================================
# Example 4: Complete RL-100 Pipeline (Conceptual)
# ============================================

def example_pipeline():
    """Example: High-level RL-100 pipeline structure."""
    print("\n" + "="*60)
    print(" "*10 + "Example 4: Complete RL-100 Pipeline")
    print("="*60 + "\n")

    print("This is a conceptual overview of the RL-100 pipeline.\n")
    print("See train_rl100.py and RL100Trainer for actual implementation.\n")

    print("="*60)
    print("Phase 1: Imitation Learning")
    print("="*60)
    print("""
    for epoch in range(IL_EPOCHS):
        for batch in dataloader:
            # Standard diffusion BC training
            loss = policy.compute_loss(batch)
            optimizer.step()

        # Evaluate periodically
        if epoch % EVAL_EVERY == 0:
            metrics = env_runner.run(policy)
            print(f"Success Rate: {metrics['success_rate']}")
    """)

    print("="*60)
    print("Phase 2: Offline RL Loop")
    print("="*60)
    print("""
    for iteration in range(M):
        # 2a. Train IQL Critics
        for epoch in range(CRITIC_EPOCHS):
            for batch in dataloader:
                # Update V
                v_loss = critics.compute_v_loss(obs, action)
                v_optimizer.step()

                # Update Q
                q_loss = critics.compute_q_loss(obs, action, reward, next_obs, done)
                q_optimizer.step()

        # 2b. Optimize Policy
        for epoch in range(PPO_EPOCHS):
            for batch in dataloader:
                # Compute advantage
                advantages = critics.compute_advantage(obs, action)

                # PPO loss over K steps
                ppo_loss = policy.compute_rl_loss(obs, advantages)
                policy_optimizer.step()

                # Consistency distillation
                if step % CD_EVERY == 0:
                    cd_loss = consistency.train_step(obs)

        # 2c. Collect New Data
        new_data = collect_rollouts(env, policy, num_episodes=50)
        dataset.add_data(new_data)

        # 2d. Retrain IL
        train_imitation_learning(dataset, epochs=IL_RETRAIN_EPOCHS)
    """)

    print("="*60)
    print("Phase 3: Online RL (Optional)")
    print("="*60)
    print("""
    for iteration in range(ONLINE_ITERATIONS):
        # Collect fresh data
        online_data = collect_rollouts(env, policy, num_episodes=20)

        # Continue RL optimization
        train_critics(online_data)
        optimize_policy(online_data)
    """)

    print("\n" + "="*60)
    print("Key Points:")
    print("="*60)
    print("""
    1. IQL Critics provide advantage: A = Q(s,a) - V(s)
    2. PPO optimizes over K denoising steps with shared A
    3. Consistency distillation enables fast 1-step generation
    4. Iterative data collection improves dataset quality
    5. Optional online fine-tuning for final performance boost
    """)

    print("\n✓ Pipeline overview complete!")
    print("\nFor actual implementation, see:")
    print("  - diffusion_policy_3d/rl100_trainer.py")
    print("  - train_rl100.py")
    print("  - RL100_IMPLEMENTATION.md")


# ============================================
# Main
# ============================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RL-100 Usage Examples")
    parser.add_argument(
        '--example',
        type=str,
        choices=['critics', 'consistency', 'ppo', 'pipeline', 'all'],
        default='all',
        help='Which example to run'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print(" "*15 + "RL-100 USAGE EXAMPLES")
    print("="*60)

    if args.example == 'critics' or args.example == 'all':
        example_iql_critics()

    if args.example == 'consistency' or args.example == 'all':
        example_consistency_distillation()

    if args.example == 'ppo' or args.example == 'all':
        example_ppo_loss()

    if args.example == 'pipeline' or args.example == 'all':
        example_pipeline()

    print("\n" + "="*60)
    print(" "*20 + "ALL EXAMPLES COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Read RL100_IMPLEMENTATION.md for details")
    print("  2. Run: python train_rl100.py task=metaworld_push")
    print("  3. Monitor training in WandB")
    print("\n")


if __name__ == "__main__":
    main()
