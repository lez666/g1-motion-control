#!/bin/bash
source /home/wasabi/g1-motion-control/third_party/holosoma/scripts/source_isaacsim_setup.sh
python /home/wasabi/g1-motion-control/third_party/holosoma/src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  --training.num_envs=9400 \
  --command.setup_terms.motion_command.params.motion_config.motion_file="/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz"
