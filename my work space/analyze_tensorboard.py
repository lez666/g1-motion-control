#!/usr/bin/env python3
"""分析TensorBoard训练日志，诊断机器人无法站起来的问题"""

import sys
from pathlib import Path

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("警告: TensorFlow未安装，尝试使用tensorboard包")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("警告: tensorboard包未安装")

def analyze_tensorboard(log_dir):
    """分析TensorBoard日志"""
    if not TB_AVAILABLE:
        print("错误: 无法读取TensorBoard日志，请安装: pip install tensorboard")
        return
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"错误: 日志目录不存在: {log_dir}")
        return
    
    print(f"正在读取TensorBoard日志: {log_dir}")
    print("=" * 80)
    
    try:
        ea = EventAccumulator(str(log_path))
        ea.Reload()
        
        # 获取所有标量标签
        scalar_tags = ea.Tags().get('scalars', [])
        
        print(f"\n找到 {len(scalar_tags)} 个标量指标\n")
        
        # 关键指标
        key_metrics = [
            'Mean reward',
            'Mean episode length',
            'Value',
            'Surrogate',
            'Entropy',
            'KL',
            'actor_loss',
            'critic_loss',
            'Episode/rew_tracking_lin_vel',
            'Episode/rew_tracking_ang_vel',
            'Episode/rew_penalty_orientation',
            'Episode/rew_alive',
            'Episode/rew_feet_phase',
            'Episode/rew_penalty_action_rate',
            'Episode/rew_joint_deviation_all',
            'Policy/mean_noise_std',
        ]
        
        print("=" * 80)
        print("关键训练指标趋势分析")
        print("=" * 80)
        
        for metric in key_metrics:
            # 尝试匹配标签
            matching_tags = [tag for tag in scalar_tags if metric.lower() in tag.lower()]
            if not matching_tags:
                continue
            
            tag = matching_tags[0]
            scalar_events = ea.Scalars(tag)
            
            if not scalar_events:
                continue
            
            # 获取最新值和初始值
            initial_value = scalar_events[0].value
            latest_value = scalar_events[-1].value
            latest_step = scalar_events[-1].step
            
            # 计算趋势
            if len(scalar_events) > 10:
                recent_values = [e.value for e in scalar_events[-10:]]
                avg_recent = sum(recent_values) / len(recent_values)
                trend = "↑" if latest_value > avg_recent else "↓"
            else:
                trend = "?"
            
            print(f"\n{tag}:")
            print(f"  最新值: {latest_value:.4f} (步骤 {latest_step})")
            print(f"  初始值: {initial_value:.4f}")
            print(f"  变化: {latest_value - initial_value:.4f} {trend}")
        
        # 分析奖励组件
        print("\n" + "=" * 80)
        print("奖励组件详细分析")
        print("=" * 80)
        
        reward_tags = [tag for tag in scalar_tags if 'Episode/rew_' in tag or 'Episode/raw_rew_' in tag]
        reward_tags.sort()
        
        for tag in reward_tags:
            scalar_events = ea.Scalars(tag)
            if not scalar_events:
                continue
            
            latest = scalar_events[-1]
            if len(scalar_events) > 10:
                recent_avg = sum([e.value for e in scalar_events[-10:]]) / 10
            else:
                recent_avg = latest.value
            
            print(f"{tag:50s} 最新: {latest.value:8.4f}  平均(最近10): {recent_avg:8.4f}")
        
        # 分析训练稳定性
        print("\n" + "=" * 80)
        print("训练稳定性分析")
        print("=" * 80)
        
        kl_tag = None
        for tag in scalar_tags:
            if 'KL' in tag or 'kl' in tag:
                kl_tag = tag
                break
        
        if kl_tag:
            kl_events = ea.Scalars(kl_tag)
            if kl_events:
                latest_kl = kl_events[-1].value
                print(f"KL散度: {latest_kl:.6f}")
                if latest_kl > 0.05:
                    print("  ⚠️  警告: KL散度过高，策略更新可能不稳定")
                elif latest_kl < 0.001:
                    print("  ⚠️  警告: KL散度过低，策略可能更新不足")
                else:
                    print("  ✓ KL散度正常")
        
        # 分析存活率
        alive_tag = None
        for tag in scalar_tags:
            if 'rew_alive' in tag:
                alive_tag = tag
                break
        
        if alive_tag:
            alive_events = ea.Scalars(alive_tag)
            if alive_events:
                latest_alive = alive_events[-1].value
                print(f"\n存活奖励: {latest_alive:.4f}")
                if latest_alive < 0.3:
                    print("  ⚠️  警告: 存活奖励过低，机器人可能经常摔倒")
                elif latest_alive > 0.7:
                    print("  ✓ 存活奖励良好")
        
        # 分析姿态惩罚
        orientation_tag = None
        for tag in scalar_tags:
            if 'penalty_orientation' in tag:
                orientation_tag = tag
                break
        
        if orientation_tag:
            orientation_events = ea.Scalars(orientation_tag)
            if orientation_events:
                latest_penalty = orientation_events[-1].value
                print(f"\n姿态惩罚: {latest_penalty:.4f}")
                if latest_penalty < -1.0:
                    print("  ⚠️  警告: 姿态惩罚过高，机器人姿态控制不佳")
        
        print("\n" + "=" * 80)
        print("诊断建议")
        print("=" * 80)
        
        # 基于数据给出建议
        suggestions = []
        
        if alive_tag:
            alive_events = ea.Scalars(alive_tag)
            if alive_events and alive_events[-1].value < 0.3:
                suggestions.append("1. 存活奖励过低 - 检查初始化状态和随机化配置")
                suggestions.append("2. 增加存活奖励权重或调整初始化高度")
        
        if orientation_tag:
            orientation_events = ea.Scalars(orientation_tag)
            if orientation_events and orientation_events[-1].value < -1.0:
                suggestions.append("3. 姿态惩罚过高 - 机器人可能无法保持直立")
                suggestions.append("4. 检查初始姿态随机化范围是否过大")
        
        # 检查episode长度
        length_tag = None
        for tag in scalar_tags:
            if 'episode_length' in tag.lower() or 'ep_len' in tag.lower():
                length_tag = tag
                break
        
        if length_tag:
            length_events = ea.Scalars(length_tag)
            if length_events:
                latest_length = length_events[-1].value
                if latest_length < 100:
                    suggestions.append("5. Episode长度过短 - 机器人可能立即摔倒")
                    suggestions.append("6. 检查终止条件和接触力阈值")
        
        if suggestions:
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        else:
            print("  训练指标看起来正常，但机器人仍无法站起来")
            print("  建议检查:")
            print("    - 仿真环境初始化配置")
            print("    - 机器人URDF模型是否正确")
            print("    - 控制频率和物理参数")
        
    except Exception as e:
        print(f"错误: 无法读取TensorBoard数据: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 默认日志目录
    default_log_dir = "/home/wasabi/g1-motion-control/third_party/holosoma/scripts/logs/hv-g1-manager/20260113_075416-g1_robust_8192envs_20260113_155409-locomotion"
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = default_log_dir
    
    analyze_tensorboard(log_dir)
