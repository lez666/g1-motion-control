#!/usr/bin/env python3
"""
Multi-Policy Sim2Sim Deployment Script
支持在站立、走路模式之间切换（使用Locomotion策略）

状态机设计：
- 站立（数字键1）：Locomotion策略，stand模式（stand_command=0）
- 走路（数字键2）：Locomotion策略，walk模式（stand_command=1）

键盘控制（按下时运动，松开时停止）：
- 方向键上：前进
- 方向键下：后退
- 方向键左：左平移
- 方向键右：右平移
- q：左转
- e：右转

使用方法：
1. 先启动MuJoCo仿真环境
2. 运行此脚本
3. 使用数字键1-2切换站立/走路模式
4. 使用方向键和q/e控制运动
"""

import sys
import os
from pathlib import Path
from dataclasses import replace

# 添加holosoma路径
script_dir = Path(__file__).parent
holosoma_root = script_dir.parent / "third_party" / "holosoma"
sys.path.insert(0, str(holosoma_root / "src"))

from loguru import logger  # noqa: E402
from holosoma_inference.config.config_values import (  # noqa: E402
    inference,
)
from holosoma_inference.policies.locomotion import (  # noqa: E402
    LocomotionPolicy,
)
from holosoma_inference.utils.misc import (  # noqa: E402
    restore_terminal_settings,
)
from termcolor import colored  # noqa: E402

# 尝试导入pynput库
try:
    from pynput import keyboard as pynput_keyboard  # noqa: E402
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    pynput_keyboard = None


class KeyboardStateTracker:
    """使用pynput跟踪键盘按键状态（按下/释放）"""

    def __init__(self):
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput库不可用。请安装: pip install pynput")
        
        # 按键状态字典
        self._key_states = {
            'up': False,      # 方向键上 - 前进
            'down': False,    # 方向键下 - 后退
            'left': False,    # 方向键左 - 左平移
            'right': False,   # 方向键右 - 右平移
            'q': False,       # Q - 左转
            'e': False,       # E - 右转
            '1': False,       # 数字键1 - 站立模式
            '2': False,       # 数字键2 - 走路模式
        }

        # 按键映射函数

        def normalize_key(key):
            if isinstance(key, pynput_keyboard.KeyCode):
                try:
                    return key.char.lower() if key.char else None
                except AttributeError:
                    return None
            elif isinstance(key, pynput_keyboard.Key):
                key_mapping = {
                    pynput_keyboard.Key.up: 'up',
                    pynput_keyboard.Key.down: 'down',
                    pynput_keyboard.Key.left: 'left',
                    pynput_keyboard.Key.right: 'right',
                }
                return key_mapping.get(key, None)
            return None
        
        # 键盘监听器
        def on_press(key):
            key_str = normalize_key(key)
            if key_str and key_str in self._key_states:
                self._key_states[key_str] = True
        
        def on_release(key):
            key_str = normalize_key(key)
            if key_str and key_str in self._key_states:
                self._key_states[key_str] = False
        
        self._listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            suppress=False
        )
        self._listener.start()

    def get_key_state(self, key):
        """获取按键状态"""
        return self._key_states.get(key, False)

    def cleanup(self):
        """停止键盘监听器"""
        if hasattr(self, '_listener'):
            self._listener.stop()


class LocomotionPolicyWithKeyboardControl(LocomotionPolicy):
    """扩展Locomotion策略，支持方向键和q/e实时控制"""

    def __init__(self, config):
        super().__init__(config)
        self.current_mode = 1  # 1=站立, 2=走路

        # 初始化键盘状态跟踪器
        if PYNPUT_AVAILABLE:
            try:
                self.keyboard_tracker = KeyboardStateTracker()
                logger.info("✅ 键盘状态跟踪器已启动")
            except Exception as e:
                logger.warning(f"⚠️  无法启动键盘状态跟踪器: {e}")
                self.keyboard_tracker = None
        else:
            logger.warning("⚠️  pynput库不可用，将使用默认的键盘控制方式")
            self.keyboard_tracker = None

        # 速度参数
        self.max_lin_vel = 1.0  # 最大线性速度 (m/s)
        self.max_lat_vel = 0.5  # 最大横向速度 (m/s)
        self.max_ang_vel = 0.8  # 最大角速度 (rad/s)

        # 用于边缘检测的前一状态
        self._prev_1_state = False
        self._prev_2_state = False

    def handle_keyboard_button(self, keycode):
        """处理键盘按键事件（用于模式切换等）"""
        # 处理模式切换（如果使用默认键盘输入）
        if keycode == "1":
            self._switch_to_stand()
            return
        elif keycode == "2":
            self._switch_to_walk()
            return
        
        # 其他按键交给父类处理
        super().handle_keyboard_button(keycode)

    def update_velocity_from_keyboard(self):
        """根据当前键盘状态更新速度命令（在策略循环中调用）"""
        if self.keyboard_tracker is None:
            return

        # 检查模式切换（数字键1和2）
        key_1_pressed = self.keyboard_tracker.get_key_state('1')
        key_2_pressed = self.keyboard_tracker.get_key_state('2')

        if key_1_pressed and not self._prev_1_state:
            self._switch_to_stand()
        elif key_2_pressed and not self._prev_2_state:
            self._switch_to_walk()

        self._prev_1_state = key_1_pressed
        self._prev_2_state = key_2_pressed

        # 只在走路模式下响应速度控制
        if not self.stand_command[0, 0]:
            # 站立模式：清零所有速度
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.ang_vel_command[0, 0] = 0.0
            return

        # 根据按键状态设置速度命令（按下时运动，松开时停止）
        lin_vel_x = 0.0
        lin_vel_y = 0.0
        ang_vel = 0.0

        # 方向键上：前进
        if self.keyboard_tracker.get_key_state('up'):
            lin_vel_x = self.max_lin_vel
        # 方向键下：后退
        elif self.keyboard_tracker.get_key_state('down'):
            lin_vel_x = -self.max_lin_vel

        # 方向键左：左平移
        if self.keyboard_tracker.get_key_state('left'):
            lin_vel_y = self.max_lat_vel
        # 方向键右：右平移
        elif self.keyboard_tracker.get_key_state('right'):
            lin_vel_y = -self.max_lat_vel

        # q：左转
        if self.keyboard_tracker.get_key_state('q'):
            ang_vel = self.max_ang_vel
        # e：右转
        elif self.keyboard_tracker.get_key_state('e'):
            ang_vel = -self.max_ang_vel

        # 更新速度命令
        self.lin_vel_command[0, 0] = lin_vel_x
        self.lin_vel_command[0, 1] = lin_vel_y
        self.ang_vel_command[0, 0] = ang_vel

    def update_phase_time(self):
        """重写update_phase_time，在更新前先处理键盘输入"""
        # 更新速度命令
        self.update_velocity_from_keyboard()
        # 调用父类方法
        super().update_phase_time()

    def _switch_to_stand(self):
        """切换到站立模式"""
        if self.current_mode != 1:
            self.current_mode = 1
            self.stand_command[0, 0] = 0  # 0表示站立
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.ang_vel_command[0, 0] = 0.0
            self.logger.info(colored("切换到模式1: 站立 (Stand)", "green"))

    def _switch_to_walk(self):
        """切换到走路模式"""
        if self.current_mode != 2:
            self.current_mode = 2
            self.stand_command[0, 0] = 1  # 1表示走路
            self.base_height_command[0, 0] = self.desired_base_height
            self.logger.info(colored("切换到模式2: 走路 (Walk)", "green"))

    def cleanup(self):
        """清理资源"""
        if (hasattr(self, 'keyboard_tracker') and
                self.keyboard_tracker is not None):
            self.keyboard_tracker.cleanup()


def main():
    """主函数"""
    # 模型路径 - 支持通过环境变量或命令行参数指定
    if len(sys.argv) > 1:
        # 命令行参数优先
        model_path = Path(sys.argv[1])
        logger.info(f"📁 使用命令行参数指定的模型: {model_path}")
    elif os.getenv("ONNX_MODEL_PATH"):
        # 环境变量次之
        model_path = Path(os.getenv("ONNX_MODEL_PATH"))
        logger.info(f"📁 使用环境变量指定的模型: {model_path}")
    else:
        # 默认路径
        model_path = (
            holosoma_root /
            "src/holosoma_inference/holosoma_inference/models/loco/"
            "g1_29dof/fastsac_g1_29dof.onnx"
        )
        logger.info(f"📁 使用默认模型: {model_path}")
    
    logger.info("=" * 80)
    msg = "🚀 多模式Sim2Sim部署 - Locomotion策略（改进键盘控制）"
    logger.info(msg)
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"📁 模型路径: {model_path}")
    logger.info("")
    logger.info("模式切换（数字键）：")
    logger.info("  1 - 站立 (Stand)")
    logger.info("  2 - 走路 (Walk)")
    logger.info("")
    logger.info("运动控制（按下时运动，松开时停止）：")
    logger.info("  ↑ - 前进")
    logger.info("  ↓ - 后退")
    logger.info("  ← - 左平移")
    logger.info("  → - 右平移")
    logger.info("  Q - 左转")
    logger.info("  E - 右转")
    logger.info("")
    logger.info("其他控制：")
    logger.info("  ] - 启动策略")
    logger.info("  o - 停止策略")
    logger.info("  i - 恢复到默认姿态")
    logger.info("  = - 切换站立/走路模式（也可以使用数字键1/2）")
    logger.info("  z - 清零所有速度")
    logger.info("")
    logger.info("⚠️  注意：WBT策略（跳舞/爬行）需要单独运行，")
    msg2 = "   因为WBT和Locomotion策略架构不同，无法在同一进程中切换。"
    logger.info(msg2)
    logger.info("=" * 80)
    logger.info("")
    
    policy = None
    try:
        # 创建配置
        config = replace(
            inference.g1_29dof_loco,
            task=replace(
                inference.g1_29dof_loco.task,
                model_path=str(model_path),
                interface="lo",
            )
        )
        
        # 创建策略实例
        policy = LocomotionPolicyWithKeyboardControl(config=config)
        
        logger.info("✅ 策略初始化成功！")
        logger.info("")

        # 运行策略
        policy.run()

    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理资源
        if policy is not None and hasattr(policy, 'cleanup'):
            policy.cleanup()
        restore_terminal_settings()
        logger.info("✅ 程序退出")


if __name__ == "__main__":
    main()
