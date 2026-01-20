#!/usr/bin/env python3
"""Verify that all robust configurations can be imported and are valid."""

import sys
from pathlib import Path

# Add holosoma to path
holosoma_dir = Path(__file__).parent.parent / "third_party" / "holosoma" / "src"
sys.path.insert(0, str(holosoma_dir))

def verify_imports():
    """Verify all robust configurations can be imported."""
    errors = []
    
    try:
        from holosoma.config_values.loco.g1.command import g1_29dof_command_robust
        print("✓ Command config imported successfully")
    except Exception as e:
        errors.append(f"Command config import failed: {e}")
        print(f"✗ Command config import failed: {e}")
    
    try:
        from holosoma.config_values.loco.g1.reward import g1_29dof_loco_robust
        print("✓ Reward config imported successfully")
    except Exception as e:
        errors.append(f"Reward config import failed: {e}")
        print(f"✗ Reward config import failed: {e}")
    
    try:
        from holosoma.config_values.loco.g1.randomization import g1_29dof_randomization_robust
        print("✓ Randomization config imported successfully")
    except Exception as e:
        errors.append(f"Randomization config import failed: {e}")
        print(f"✗ Randomization config import failed: {e}")
    
    try:
        from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel_robust
        print("✓ Observation config imported successfully")
    except Exception as e:
        errors.append(f"Observation config import failed: {e}")
        print(f"✗ Observation config import failed: {e}")
    
    try:
        from holosoma.config_values.loco.g1.termination import g1_29dof_termination_robust
        print("✓ Termination config imported successfully")
    except Exception as e:
        errors.append(f"Termination config import failed: {e}")
        print(f"✗ Termination config import failed: {e}")
    
    try:
        from holosoma.config_values.loco.g1.experiment import g1_29dof_robust
        print("✓ Experiment config imported successfully")
    except Exception as e:
        errors.append(f"Experiment config import failed: {e}")
        print(f"✗ Experiment config import failed: {e}")
    
    # Verify reward functions
    try:
        from holosoma.managers.reward.terms.locomotion import (
            flat_orientation_l2,
            feet_air_time_positive_biped,
            both_feet_air,
            feet_slide,
            joint_torques_l2,
            joint_acc_l2,
            joint_pos_limits,
            joint_deviation_l1,
        )
        print("✓ All new reward functions imported successfully")
    except Exception as e:
        errors.append(f"Reward functions import failed: {e}")
        print(f"✗ Reward functions import failed: {e}")
    
    if errors:
        print(f"\n✗ Verification failed with {len(errors)} error(s)")
        return False
    else:
        print("\n✓ All configurations verified successfully!")
        return True

if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
