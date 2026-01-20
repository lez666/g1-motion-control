#!/usr/bin/env python3
"""
Isaac Sim 验证脚本
验证 Isaac Sim 安装和 Holosoma 集成是否正常工作
"""

import sys
import time

print("=" * 60)
print("Isaac Sim 安装验证")
print("=" * 60)

# 1. 验证核心包导入
print("\n[1/5] 验证核心包导入...")
try:
    import isaacsim
    import isaaclab
    import isaaclab_rl
    print("✓ isaacsim, isaaclab, isaaclab_rl 导入成功")
except Exception as e:
    print(f"✗ 核心包导入失败: {e}")
    sys.exit(1)

# 2. 验证 Holosoma 集成
print("\n[2/5] 验证 Holosoma 集成...")
try:
    from holosoma.utils.simulator_config import SimulatorType
    from holosoma.utils.sim_utils import setup_simulator_imports, setup_isaaclab_launcher
    print("✓ Holosoma Isaac Sim 工具导入成功")
except Exception as e:
    print(f"✗ Holosoma 集成失败: {e}")
    sys.exit(1)

# 3. 验证配置系统
print("\n[3/5] 验证配置系统...")
try:
    from holosoma.config_types.experiment import ExperimentConfig
    import holosoma.config_values.experiment as exp_configs
    import holosoma.config_values.simulator as sim_configs
    
    # 获取配置（使用 dataclasses.replace）
    import dataclasses
    config = exp_configs.g1_29dof
    config = dataclasses.replace(
        config,
        simulator=sim_configs.isaacsim,
        training=dataclasses.replace(
            config.training,
            num_envs=1,
            headless=True,
        )
    )
    print("✓ 配置系统正常")
except Exception as e:
    print(f"✗ 配置系统失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 验证模拟器导入
print("\n[4/5] 验证模拟器导入...")
try:
    setup_simulator_imports(config)
    print("✓ 模拟器导入成功")
except Exception as e:
    print(f"✗ 模拟器导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 验证 Isaac Sim 启动（简短测试）
print("\n[5/5] 验证 Isaac Sim 启动...")
print("   (这可能需要 10-15 秒...)")
try:
    simulation_app = setup_isaaclab_launcher(config, device='cpu')
    if simulation_app is not None:
        print("✓ Isaac Sim launcher 初始化成功")
        print("\n" + "=" * 60)
        print("✓ 所有验证通过！Isaac Sim 已准备就绪！")
        print("=" * 60)
        print("\n可以开始使用 Isaac Sim 进行训练和仿真了！")
    else:
        print("✗ Launcher 初始化失败")
        sys.exit(1)
except KeyboardInterrupt:
    print("\n用户中断")
    sys.exit(0)
except Exception as e:
    print(f"✗ Isaac Sim 启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
