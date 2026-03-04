#!/usr/bin/env python3
"""
RL-100 Implementation Verification Script
==========================================
Quick verification that all RL-100 components are correctly installed.

Usage:
    python verify_rl100.py

This script checks:
1. File existence
2. Module imports
3. Component instantiation
4. Basic functionality tests
"""

import sys
import os
from pathlib import Path
from termcolor import cprint


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(filepath).exists():
        cprint(f"✓ {description}", "green")
        return True
    else:
        cprint(f"✗ {description} - NOT FOUND: {filepath}", "red")
        return False


def check_import(module_path: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        exec(f"from {module_path} import *")
        cprint(f"✓ {description}", "green")
        return True
    except Exception as e:
        cprint(f"✗ {description} - IMPORT ERROR: {str(e)}", "red")
        return False


def main():
    cprint("\n" + "="*60, "cyan")
    cprint(" "*15 + "RL-100 VERIFICATION", "cyan")
    cprint("="*60 + "\n", "cyan")

    all_checks = []

    # ======================================
    # 1. File Existence Checks
    # ======================================
    cprint("[1/4] Checking file existence...\n", "yellow")

    files_to_check = [
        ("train_rl100.py", "Training script"),
        ("diffusion_policy_3d/policy/rl100_policy.py", "RL100Policy"),
        ("diffusion_policy_3d/model/rl/__init__.py", "RL module init"),
        ("diffusion_policy_3d/model/rl/iql_critics.py", "IQL Critics"),
        ("diffusion_policy_3d/model/rl/consistency_model.py", "Consistency Model"),
        ("diffusion_policy_3d/rl100_trainer.py", "RL100Trainer"),
        ("diffusion_policy_3d/config/rl100.yaml", "RL100 config"),
        ("RL100_IMPLEMENTATION.md", "Implementation guide"),
        ("RL100_README.md", "README"),
    ]

    for filepath, description in files_to_check:
        all_checks.append(check_file_exists(filepath, description))

    # ======================================
    # 2. Module Import Checks
    # ======================================
    cprint("\n[2/4] Checking module imports...\n", "yellow")

    imports_to_check = [
        ("diffusion_policy_3d.policy.rl100_policy", "RL100Policy import"),
        ("diffusion_policy_3d.model.rl.iql_critics", "IQLCritics import"),
        ("diffusion_policy_3d.model.rl.consistency_model", "ConsistencyModel import"),
        ("diffusion_policy_3d.rl100_trainer", "RL100Trainer import"),
        ("diffusion_policy_3d.model.rl", "RL module import"),
    ]

    for module_path, description in imports_to_check:
        all_checks.append(check_import(module_path, description))

    # ======================================
    # 3. Component Instantiation Tests
    # ======================================
    cprint("\n[3/4] Testing component instantiation...\n", "yellow")

    try:
        # Test IQL Critics
        from diffusion_policy_3d.model.rl.iql_critics import IQLCritics
        critics = IQLCritics(obs_dim=64, action_dim=4)
        cprint("✓ IQL Critics instantiation", "green")
        all_checks.append(True)
    except Exception as e:
        cprint(f"✗ IQL Critics instantiation - ERROR: {str(e)}", "red")
        all_checks.append(False)

    try:
        # Test Consistency Model
        from diffusion_policy_3d.model.rl.consistency_model import ConsistencyModel
        cm = ConsistencyModel(input_dim=4, global_cond_dim=128)
        cprint("✓ Consistency Model instantiation", "green")
        all_checks.append(True)
    except Exception as e:
        cprint(f"✗ Consistency Model instantiation - ERROR: {str(e)}", "red")
        all_checks.append(False)

    # ======================================
    # 4. Basic Functionality Tests
    # ======================================
    cprint("\n[4/4] Testing basic functionality...\n", "yellow")

    try:
        import torch
        from diffusion_policy_3d.model.rl.iql_critics import IQLCritics

        # Test IQL forward pass
        critics = IQLCritics(obs_dim=64, action_dim=4)
        obs = torch.randn(16, 64)
        action = torch.randn(16, 4)

        # Test V
        v_value = critics.get_value(obs)
        assert v_value.shape == (16, 1), "V output shape mismatch"

        # Test Q
        q_value = critics.get_q_value(obs, action)
        assert q_value.shape == (16, 1), "Q output shape mismatch"

        # Test advantage
        advantage = critics.compute_advantage(obs, action)
        assert advantage.shape == (16, 1), "Advantage shape mismatch"

        cprint("✓ IQL Critics forward pass", "green")
        all_checks.append(True)
    except Exception as e:
        cprint(f"✗ IQL Critics forward pass - ERROR: {str(e)}", "red")
        all_checks.append(False)

    try:
        import torch
        from diffusion_policy_3d.model.rl.consistency_model import ConsistencyModel

        # Test Consistency Model forward pass
        cm = ConsistencyModel(input_dim=4, global_cond_dim=128)
        noisy_action = torch.randn(16, 16, 4)  # [B, T, Da]
        global_cond = torch.randn(16, 128)

        clean_action = cm(noisy_action, global_cond)
        assert clean_action.shape == noisy_action.shape, "CM output shape mismatch"

        cprint("✓ Consistency Model forward pass", "green")
        all_checks.append(True)
    except Exception as e:
        cprint(f"✗ Consistency Model forward pass - ERROR: {str(e)}", "red")
        all_checks.append(False)

    # ======================================
    # Summary
    # ======================================
    cprint("\n" + "="*60, "cyan")
    cprint(" "*20 + "SUMMARY", "cyan")
    cprint("="*60 + "\n", "cyan")

    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100

    if passed == total:
        cprint(f"✓ All checks passed! ({passed}/{total})", "green")
        cprint("\n✨ RL-100 implementation is ready to use!", "green")
        cprint("\nNext steps:", "yellow")
        cprint("  1. Read RL100_IMPLEMENTATION.md for detailed guide", "white")
        cprint("  2. Prepare dataset with rewards and next_obs", "white")
        cprint("  3. Run: python train_rl100.py task=metaworld_push", "white")
        return 0
    else:
        cprint(f"⚠ {total - passed} checks failed! ({passed}/{total} passed, {percentage:.1f}%)", "red")
        cprint("\nPlease fix the issues above before using RL-100.", "red")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        cprint("\n\nVerification interrupted by user.", "yellow")
        sys.exit(1)
    except Exception as e:
        cprint(f"\n\nUnexpected error: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)
