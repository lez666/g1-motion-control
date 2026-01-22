#!/usr/bin/env python3
"""
自动运行WBT sim2sim仿真并收集数据进行分析
"""
import sys
import time
import numpy as np
from pathlib import Path

# 添加holosoma路径
script_dir = Path(__file__).parent
holosoma_root = script_dir.parent / "third_party" / "holosoma"
sys.path.insert(0, str(holosoma_root / "src"))

import torch
from loguru import logger
from holosoma_inference.config.config_values import inference
from holosoma_inference.policies.whole_body_tracking import WholeBodyTrackingPolicy
from dataclasses import replace

def analyze_motion_data(motion_file: str):
    """分析motion数据文件"""
    logger.info(f"分析motion文件: {motion_file}")
    data = np.load(motion_file)
    
    # 检查数据字段
    logger.info(f"数据字段: {list(data.keys())}")
    
    if "joint_vel" in data:
        joint_vel = data["joint_vel"]
        logger.info(f"joint_vel shape: {joint_vel.shape}")
        logger.info(f"joint_vel 统计:")
        logger.info(f"  均值: {np.mean(joint_vel, axis=0)}")
        logger.info(f"  标准差: {np.std(joint_vel, axis=0)}")
        logger.info(f"  最大值: {np.max(joint_vel, axis=0)}")
        logger.info(f"  最小值: {np.min(joint_vel, axis=0)}")
        
        # 检查速度范围是否合理
        max_vel = np.max(np.abs(joint_vel))
        logger.info(f"  最大绝对速度: {max_vel:.4f} rad/s")
        if max_vel > 20.0:
            logger.warning(f"⚠️  速度过大！最大速度 {max_vel:.4f} rad/s 可能不合理")
    
    if "body_lin_vel_w" in data:
        body_lin_vel = data["body_lin_vel_w"]
        logger.info(f"body_lin_vel_w shape: {body_lin_vel.shape}")
        max_lin_vel = np.max(np.linalg.norm(body_lin_vel, axis=-1))
        logger.info(f"  最大线性速度: {max_lin_vel:.4f} m/s")
        if max_lin_vel > 5.0:
            logger.warning(f"⚠️  线性速度过大！最大速度 {max_lin_vel:.4f} m/s 可能不合理")
    
    if "body_ang_vel_w" in data:
        body_ang_vel = data["body_ang_vel_w"]
        logger.info(f"body_ang_vel_w shape: {body_ang_vel.shape}")
        max_ang_vel = np.max(np.linalg.norm(body_ang_vel, axis=-1))
        logger.info(f"  最大角速度: {max_ang_vel:.4f} rad/s")
        if max_ang_vel > 10.0:
            logger.warning(f"⚠️  角速度过大！最大速度 {max_ang_vel:.4f} rad/s 可能不合理")
    
    if "fps" in data:
        logger.info(f"FPS: {data['fps']}")
    
    return data

def run_sim2sim_analysis(model_path: str, motion_file: str, duration: float = 10.0):
    """运行sim2sim并收集数据"""
    logger.info("=" * 80)
    logger.info("开始WBT Sim2Sim分析")
    logger.info("=" * 80)
    
    # 1. 分析motion数据
    logger.info("\n[步骤1] 分析motion数据文件")
    motion_data = analyze_motion_data(motion_file)
    
    # 2. 创建policy配置
    logger.info("\n[步骤2] 初始化WBT Policy")
    config = replace(
        inference.g1_29dof_wbt,
        task=replace(
            inference.g1_29dof_wbt.task,
            model_path=model_path,
            interface="lo",
            use_sim_time=True,
            rl_rate=50,
        )
    )
    
    try:
        policy = WholeBodyTrackingPolicy(config=config)
        logger.info("✅ Policy初始化成功")
        
        # 3. 收集仿真数据
        logger.info(f"\n[步骤3] 运行仿真 {duration} 秒并收集数据...")
        logger.info("注意: 这需要MuJoCo仿真环境在另一个终端运行")
        logger.info("请先运行: cd third_party/holosoma && source scripts/source_mujoco_setup.sh")
        logger.info("然后运行: python src/holosoma/holosoma/run_sim.py robot:g1-29dof")
        
        # 等待用户确认
        input("\n按Enter键开始收集数据（确保MuJoCo已启动）...")
        
        # 收集数据
        collected_data = {
            "joint_pos": [],
            "joint_vel": [],
            "joint_pos_ref": [],
            "joint_vel_ref": [],
            "body_pos": [],
            "body_vel": [],
            "body_pos_ref": [],
            "body_vel_ref": [],
            "errors": {
                "joint_pos_error": [],
                "joint_vel_error": [],
                "body_pos_error": [],
                "body_vel_error": [],
            }
        }
        
        start_time = time.time()
        step_count = 0
        
        # 启动policy（这会连接到MuJoCo）
        logger.info("启动policy...")
        # 注意: 这里我们需要修改policy的run方法来收集数据
        # 由于policy.run()是阻塞的，我们需要一个不同的方法
        
        logger.info("⚠️  由于policy.run()是阻塞的，我们需要手动收集数据")
        logger.info("请在policy运行时观察仿真行为")
        
        # 运行policy一段时间
        try:
            policy.run()
        except KeyboardInterrupt:
            logger.info("仿真被用户中断")
        
        logger.info("\n[步骤4] 分析收集的数据")
        if len(collected_data["joint_pos"]) > 0:
            analyze_collected_data(collected_data)
        else:
            logger.warning("⚠️  未收集到数据，请检查MuJoCo连接")
        
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def analyze_collected_data(data: dict):
    """分析收集的仿真数据"""
    logger.info("分析仿真数据...")
    
    if len(data["joint_pos"]) == 0:
        logger.warning("没有数据可分析")
        return
    
    joint_pos = np.array(data["joint_pos"])
    joint_vel = np.array(data["joint_vel"])
    joint_pos_ref = np.array(data["joint_pos_ref"])
    joint_vel_ref = np.array(data["joint_vel_ref"])
    
    # 计算误差
    joint_pos_error = np.linalg.norm(joint_pos - joint_pos_ref, axis=-1)
    joint_vel_error = np.linalg.norm(joint_vel - joint_vel_ref, axis=-1)
    
    logger.info(f"\n关节位置误差:")
    logger.info(f"  均值: {np.mean(joint_pos_error):.4f} rad")
    logger.info(f"  最大值: {np.max(joint_pos_error):.4f} rad")
    logger.info(f"  标准差: {np.std(joint_pos_error):.4f} rad")
    
    logger.info(f"\n关节速度误差:")
    logger.info(f"  均值: {np.mean(joint_vel_error):.4f} rad/s")
    logger.info(f"  最大值: {np.max(joint_vel_error):.4f} rad/s")
    logger.info(f"  标准差: {np.std(joint_vel_error):.4f} rad/s")
    
    # 检查是否有异常大的误差
    if np.max(joint_pos_error) > 1.0:
        logger.warning(f"⚠️  关节位置误差过大！最大误差 {np.max(joint_pos_error):.4f} rad")
    
    if np.max(joint_vel_error) > 10.0:
        logger.warning(f"⚠️  关节速度误差过大！最大误差 {np.max(joint_vel_error):.4f} rad/s")

def main():
    """主函数"""
    # 使用最新的checkpoint
    model_path = "/home/wasabi/g1-motion-control/third_party/holosoma/src/holosoma/logs/WholeBodyTracking/20260121_121616-g1_29dof_wbt_manager-locomotion/model_20000.onnx"
    motion_file = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz"
    
    if not Path(model_path).exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    if not Path(motion_file).exists():
        logger.error(f"Motion文件不存在: {motion_file}")
        return
    
    run_sim2sim_analysis(model_path, motion_file, duration=10.0)

if __name__ == "__main__":
    main()
