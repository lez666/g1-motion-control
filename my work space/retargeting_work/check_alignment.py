import numpy as np
import viser
import time
from pathlib import Path
import yourdfpy
import yourdfpy.urdf
from viser.extras import ViserUrdf

# --- Monkey Patch yourdfpy ---
_original_fk_joint = yourdfpy.urdf.URDF._forward_kinematics_joint
def _patched_fk_joint(self, joint, q=None):
    try:
        return _original_fk_joint(self, joint, q)
    except TypeError as e:
        if "only 0-dimensional arrays can be converted to Python scalars" in str(e):
            if q is None:
                q_val = self.cfg[self.actuated_dof_indices[self.actuated_joint_names.index(joint.name)]]
                q = float(np.asarray(q_val).item())
            else:
                q = float(np.asarray(q).item())
            return _original_fk_joint(self, joint, q)
        raise e
yourdfpy.urdf.URDF._forward_kinematics_joint = _patched_fk_joint

def main():
    server = viser.ViserServer(port=8081) # 使用 8081 避免冲突
    
    # 1. 加载 G1 机器人 (Reference)
    # 显式指定 mesh_dir 确保所有部件（肢体）都能加载出来
    urdf_dir = "/home/wasabi/g1-motion-control/third_party/holosoma/src/holosoma_retargeting/models/g1"
    urdf_path = f"{urdf_dir}/g1_29dof.urdf"
    
    robot_urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=f"{urdf_dir}/")
    viser_robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
    
    # 设置一个自然的初始姿态（避免肢体卡在身体里）
    # 这里的 key 是 URDF 里的 joint name
    default_angles = {
        "left_hip_pitch_joint": -0.4,
        "right_hip_pitch_joint": -0.4,
        "left_knee_joint": 0.8,
        "right_knee_joint": 0.8,
        "left_ankle_pitch_joint": -0.4,
        "right_ankle_pitch_joint": -0.4,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
    # 应用初始姿态
    current_config = robot_urdf.zero_cfg
    for name, val in default_angles.items():
        if name in robot_urdf.actuated_joint_names:
            idx = robot_urdf.actuated_joint_names.index(name)
            current_config[idx] = val
    viser_robot.update_cfg(current_config)

    # 2. 加载 NPY 数据
    npy_path = "/home/wasabi/g1-motion-control/my work space/retargeting_work/input/New Session-014.npy"
    human_points = np.load(npy_path) 
    
    joint_names = [
        "Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "Spine", "Spine1", "Spine2", "Neck", "Head",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"
    ]

    # 3. GUI Controls
    with server.gui.add_folder("Controls"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(human_points)-1, step=1, initial_value=0)
        show_labels = server.gui.add_checkbox("Show Joint Names", initial_value=True)
        # 增加透明度控制，方便看清内部的点
        robot_alpha = server.gui.add_slider("Robot Opacity", min=0.0, max=1.0, step=0.1, initial_value=0.5)

    # 4. Spheres and Labels
    spheres = []
    labels = []
    for i, name in enumerate(joint_names):
        color = (255, 0, 0) if "Left" in name else (0, 255, 0)
        if any(x in name for x in ["Spine", "Hips", "Neck", "Head"]): color = (255, 255, 0)
        
        s = server.scene.add_icosphere(f"/human/{name}", radius=0.03, color=color)
        spheres.append(s)
        l = server.scene.add_label(f"/labels/{name}", text=name)
        labels.append(l)

    def update_scene(_):
        f = frame_slider.value
        points = human_points[f]
        for i, name in enumerate(joint_names):
            if i < len(points):
                pos = points[i]
                spheres[i].position = pos
                labels[i].position = pos
                labels[i].visible = show_labels.value
        # 这里 ViserUrdf 暂时不支持直接设透明度，但我们可以通过 visibility 开关模拟或者控制 labels

    frame_slider.on_update(update_scene)
    show_labels.on_update(update_scene)
    update_scene(None)

    print("\nCheck Tool: http://localhost:8081")
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
