import numpy as np
from scipy.spatial.transform import Rotation as R
import re

def parse_bvh(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    joints, stack, motion_data, parsing_hierarchy = [], [], [], True
    joint_pattern = re.compile(r'(ROOT|JOINT|END SITE)\s+(\w+)?')
    offset_pattern = re.compile(r'OFFSET\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)')
    current_joint = None
    for line in lines:
        line = line.strip()
        if not line or line == "MOTION":
            if line == "MOTION": parsing_hierarchy = False
            continue
        if parsing_hierarchy:
            joint_match = joint_pattern.match(line)
            if joint_match:
                name = joint_match.group(2) if joint_match.group(2) else "EndSite"
                current_joint = {'name': name, 'parent': stack[-1] if stack else None, 'offset': None}
                joints.append(current_joint)
                continue
            if line == "{": stack.append(current_joint); continue
            if line == "}": stack.pop(); continue
            offset_match = offset_pattern.match(line)
            if offset_match and current_joint:
                current_joint['offset'] = np.array([float(x) for x in offset_match.groups()])
        else:
            if any(line.startswith(s) for s in ["Frames:", "Frame Time:"]): continue
            motion_data.append([float(x) for x in line.split()])
    return [j for j in joints if j['name'] != "EndSite"], np.array(motion_data)

def compute_fk(joints, motion_data):
    num_frames, num_joints = motion_data.shape[0], len(joints)
    positions = np.zeros((num_frames, num_joints, 3))
    for f in range(num_frames):
        data_idx, joint_transforms = 0, {}
        for i, joint in enumerate(joints):
            if joint['parent'] is None:
                pos, rot_data = motion_data[f, 0:3], motion_data[f, 3:6]
                data_idx = 6
            else:
                pos, rot_data = joint['offset'], motion_data[f, data_idx:data_idx+3]
                data_idx += 3
            r = R.from_euler('YXZ', rot_data, degrees=True)
            local_m = np.eye(4)
            local_m[:3, :3], local_m[:3, 3] = r.as_matrix(), pos
            global_m = local_m if joint['parent'] is None else joint_transforms[joint['parent']['name']] @ local_m
            joint_transforms[joint['name']], positions[f, i] = global_m, global_m[:3, 3]
    # Z-up: Holo_X = BVH_Z, Holo_Y = -BVH_X, Holo_Z = BVH_Y
    res = np.zeros_like(positions)
    res[:, :, 0], res[:, :, 1], res[:, :, 2] = positions[:, :, 2], -positions[:, :, 0], positions[:, :, 1]
    # Scale: Hips height -> 0.75m
    scale = 0.75 / np.mean(res[:, 0, 2])
    return res * scale

if __name__ == "__main__":
    bvh_path = "/home/wasabi/g1-motion-control/my work space/prompt/bvhfile/New Session-014.bvh"
    output_path = "/home/wasabi/g1-motion-control/my work space/retargeting_work/input/New Session-014.npy"
    joints, motion = parse_bvh(bvh_path)
    # 30 FPS, 30 Seconds
    motion = motion[::8][:900]
    pos = compute_fk(joints, motion)
    
    # 官方 LAFAN 22 关节顺序
    lafan_order = [
        "Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "Spine", "Spine1", "Spine2", "Neck", "Head",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"
    ]
    
    # 核心修正：按照 G1 URDF 驱动逻辑进行“错位映射”
    mapping = {
        "Hips": "Hips",          # 用于朝向计算
        "Spine": "Chest",        # 用于朝向计算
        "Spine1": "Hips",        # 重要：驱动机器人盆骨 (Pelvis)
        "Spine2": "Chest2",      # 驱动机器人腰部
        "RightUpLeg": "RightHip", "RightLeg": "RightKnee", "RightFoot": "RightAnkle", "RightToeBase": "RightToe",
        "LeftUpLeg": "LeftHip", "LeftLeg": "LeftKnee", "LeftFoot": "LeftAnkle", "LeftToeBase": "LeftToe",
        "Neck": "Neck", "Head": "Head",
        "RightShoulder": "RightCollar", # 仅作占位
        "RightArm": "RightShoulder",    # 重要：驱动机器人肩膀
        "RightForeArm": "RightElbow",   # 重要：驱动机器人肘部
        "RightHand": "RightWrist",      # 驱动机器人手部
        "LeftShoulder": "LeftCollar",
        "LeftArm": "LeftShoulder",
        "LeftForeArm": "LeftElbow",
        "LeftHand": "LeftWrist"
    }
    
    joint_map = {j['name']: i for i, j in enumerate(joints)}
    filtered_pos = np.zeros((pos.shape[0], 22, 3))
    for i, target_name in enumerate(lafan_order):
        bvh_name = mapping.get(target_name, "Hips")
        if bvh_name in joint_map:
            filtered_pos[:, i] = pos[:, joint_map[bvh_name]]
            
    np.save(output_path, filtered_pos)
    print("LAFAN disguise complete. Ready for official pipeline.")
