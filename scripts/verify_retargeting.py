#!/usr/bin/env python3
"""
验证 holosoma_retargeting 安装是否成功

此脚本检查：
1. 基础包导入
2. 配置类型模块
3. 配置值模块
4. 核心功能模块
5. 依赖项版本
6. 基本功能测试
"""

from __future__ import annotations

import sys
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def check_import(module_name: str, description: str) -> bool:
    """检查模块导入"""
    try:
        __import__(module_name)
        print_success(f"{description}: {module_name}")
        return True
    except ImportError as e:
        print_error(f"{description}: {module_name} - {e}")
        return False
    except Exception as e:
        print_error(f"{description}: {module_name} - Unexpected error: {e}")
        return False


def check_from_import(module_name: str, item_name: str, description: str) -> bool:
    """检查从模块导入特定项"""
    try:
        module = __import__(module_name, fromlist=[item_name])
        getattr(module, item_name)
        print_success(f"{description}: {module_name}.{item_name}")
        return True
    except (ImportError, AttributeError) as e:
        print_error(f"{description}: {module_name}.{item_name} - {e}")
        return False
    except Exception as e:
        print_error(f"{description}: {module_name}.{item_name} - Unexpected error: {e}")
        return False


def check_dependency(package_name: str, min_version: str | None = None) -> bool:
    """检查依赖项"""
    try:
        import importlib.metadata
        version = importlib.metadata.version(package_name)
        if min_version:
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print_error(f"{package_name}: {version} (required: >={min_version})")
                return False
        print_success(f"{package_name}: {version}")
        return True
    except importlib.metadata.PackageNotFoundError:
        print_error(f"{package_name}: not installed")
        return False
    except Exception as e:
        print_error(f"{package_name}: error checking version - {e}")
        return False


def main() -> int:
    """主验证函数"""
    print_section("Holosoma Retargeting 安装验证")
    
    all_passed = True
    
    # 1. 检查基础包导入
    print_section("1. 基础包导入")
    all_passed &= check_import("holosoma_retargeting", "主包")
    
    # 检查包路径
    try:
        import holosoma_retargeting
        print_info(f"包路径: {holosoma_retargeting.__file__}")
    except Exception as e:
        print_error(f"无法获取包路径: {e}")
        all_passed = False
    
    # 2. 检查配置类型模块
    print_section("2. 配置类型模块 (config_types)")
    config_types = [
        ("holosoma_retargeting.config_types", "data_type", "MotionDataConfig"),
        ("holosoma_retargeting.config_types", "robot", "RobotConfig"),
        ("holosoma_retargeting.config_types", "retargeter", "RetargeterConfig"),
        ("holosoma_retargeting.config_types", "retargeting", "RetargetingConfig"),
        ("holosoma_retargeting.config_types", "retargeting", "ParallelRetargetingConfig"),
        ("holosoma_retargeting.config_types", "task", "TaskConfig"),
        ("holosoma_retargeting.config_types", "data_conversion", "DataConversionConfig"),
        ("holosoma_retargeting.config_types", "viser", "ViserConfig"),
    ]
    
    for module, submodule, class_name in config_types:
        full_module = f"{module}.{submodule}"
        all_passed &= check_from_import(full_module, class_name, f"config_types.{submodule}.{class_name}")
    
    # 检查 __init__ 导出
    all_passed &= check_from_import("holosoma_retargeting.config_types", "MotionDataConfig", "config_types.__init__")
    all_passed &= check_from_import("holosoma_retargeting.config_types", "RobotConfig", "config_types.__init__")
    all_passed &= check_from_import("holosoma_retargeting.config_types", "RetargetingConfig", "config_types.__init__")
    
    # 3. 检查配置值模块
    print_section("3. 配置值模块 (config_values)")
    config_values = [
        ("holosoma_retargeting.config_values", "robot", "get_default_robot_config"),
        ("holosoma_retargeting.config_values", "data_type", "get_default_motion_data_config"),
        ("holosoma_retargeting.config_values", "data_conversion", "get_default_data_conversion_config"),
        ("holosoma_retargeting.config_values", "viser", "get_default_viser_config"),
    ]
    
    for module, submodule, func_name in config_values:
        full_module = f"{module}.{submodule}"
        all_passed &= check_from_import(full_module, func_name, f"config_values.{submodule}.{func_name}")
    
    # 4. 检查核心功能模块
    print_section("4. 核心功能模块 (src)")
    core_modules = [
        ("holosoma_retargeting.src", "interaction_mesh_retargeter", "InteractionMeshRetargeter"),
        ("holosoma_retargeting.src", "utils", "calculate_scale_factor"),
        ("holosoma_retargeting.src", "utils", "preprocess_motion_data"),
        ("holosoma_retargeting.src", "mujoco_utils", "_world_mesh_from_geom"),
        ("holosoma_retargeting.src", "viser_utils", "create_motion_control_sliders"),
    ]
    
    for module, submodule, item_name in core_modules:
        full_module = f"{module}.{submodule}"
        all_passed &= check_from_import(full_module, item_name, f"src.{submodule}.{item_name}")
    
    # 5. 检查示例模块
    print_section("5. 示例模块 (examples)")
    all_passed &= check_import("holosoma_retargeting.examples", "examples 模块")
    all_passed &= check_import("holosoma_retargeting.examples.robot_retarget", "robot_retarget 示例")
    all_passed &= check_import("holosoma_retargeting.examples.parallel_robot_retarget", "parallel_robot_retarget 示例")
    
    # 6. 检查依赖项
    print_section("6. 依赖项检查")
    dependencies = [
        ("numpy", None),
        ("torch", None),
        ("scipy", None),
        ("mujoco", None),
        ("trimesh", None),
        ("tyro", None),
        ("cvxpy", None),
        ("viser", None),
    ]
    
    for dep, min_ver in dependencies:
        all_passed &= check_dependency(dep, min_ver)
    
    # 7. 基本功能测试
    print_section("7. 基本功能测试")
    
    # 测试配置创建
    try:
        from holosoma_retargeting.config_values.robot import get_default_robot_config
        from holosoma_retargeting.config_types.robot import RobotConfig
        
        robot_config = get_default_robot_config("g1")
        print_success("创建默认 G1 机器人配置")
        
        # 验证配置类型
        assert isinstance(robot_config, RobotConfig), "配置类型不正确"
        print_success("配置类型验证通过")
        
    except Exception as e:
        print_error(f"配置创建测试失败: {e}")
        all_passed = False
    
    # 测试数据配置
    try:
        from holosoma_retargeting.config_values.data_type import get_default_motion_data_config
        from holosoma_retargeting.config_types.data_type import MotionDataConfig
        
        motion_config = get_default_motion_data_config("smplh")
        print_success("创建默认 SMPLH 运动数据配置")
        
        assert isinstance(motion_config, MotionDataConfig), "配置类型不正确"
        print_success("运动数据配置类型验证通过")
        
    except Exception as e:
        print_error(f"运动数据配置测试失败: {e}")
        all_passed = False
    
    # 测试工具函数
    try:
        import numpy as np
        from holosoma_retargeting.src.utils import calculate_scale_factor
        
        # 简单测试
        scale = calculate_scale_factor("test_task", 1.0)
        print_success("工具函数 calculate_scale_factor 可用")
        
    except Exception as e:
        print_error(f"工具函数测试失败: {e}")
        all_passed = False
    
    # 8. 检查数据格式注册表
    print_section("8. 数据格式注册表")
    try:
        from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY
        
        print_success(f"数据格式注册表包含 {len(DEMO_JOINTS_REGISTRY)} 个格式")
        for fmt in DEMO_JOINTS_REGISTRY.keys():
            print_info(f"  - {fmt}")
            
    except Exception as e:
        print_error(f"数据格式注册表检查失败: {e}")
        all_passed = False
    
    # 总结
    print_section("验证总结")
    if all_passed:
        print_success("所有检查通过！holosoma_retargeting 安装成功。")
        print_info("你现在可以使用 retargeting 功能了。")
        return 0
    else:
        print_error("部分检查失败，请检查上述错误信息。")
        print_info("请确保：")
        print_info("  1. 已运行 setup_retargeting.sh 或 source_retargeting_setup.sh")
        print_info("  2. 已正确设置 PYTHONPATH 或 .pth 文件")
        print_info("  3. 所有依赖项已正确安装")
        return 1


if __name__ == "__main__":
    sys.exit(main())
