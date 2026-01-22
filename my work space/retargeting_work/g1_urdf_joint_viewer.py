import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import viser

try:
    import yourdfpy
    import yourdfpy.urdf
    from viser.extras import ViserUrdf
except Exception:
    yourdfpy = None
    ViserUrdf = None


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def make_transform(xyz, rpy):
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rpy_to_matrix(rpy)
    mat[:3, 3] = xyz
    return mat


def quat_from_two_vectors(v0, v1):
    v0 = v0 / (np.linalg.norm(v0) + 1e-8)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    c = np.cross(v0, v1)
    d = float(np.dot(v0, v1))
    if d < -0.999999:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v0[0]) > abs(v0[1]):
            axis = np.array([0.0, 1.0, 0.0])
        c = np.cross(v0, axis)
        c = c / (np.linalg.norm(c) + 1e-8)
        return np.array([0.0, c[0], c[1], c[2]])
    s = math.sqrt((1.0 + d) * 2.0)
    invs = 1.0 / s
    return np.array([s * 0.5, c[0] * invs, c[1] * invs, c[2] * invs])


def parse_urdf_joints(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    link_children = set()
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        jtype = joint.attrib.get("type")
        parent = joint.find("parent").attrib.get("link")
        child = joint.find("child").attrib.get("link")
        origin = joint.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin is not None:
            if "xyz" in origin.attrib:
                xyz = np.array([float(x) for x in origin.attrib["xyz"].split()])
            if "rpy" in origin.attrib:
                rpy = np.array([float(x) for x in origin.attrib["rpy"].split()])
        axis = joint.find("axis")
        axis_xyz = np.array([1.0, 0.0, 0.0])
        if axis is not None and "xyz" in axis.attrib:
            axis_xyz = np.array([float(x) for x in axis.attrib["xyz"].split()])
        joints.append(
            {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis_xyz,
            }
        )
        link_children.add(child)

    # Base link: link that never appears as a child
    links = {l.attrib["name"] for l in root.findall("link")}
    base_links = list(links - link_children)
    base = base_links[0] if base_links else None
    return joints, base


def build_link_transforms(joints, base_link):
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)

    link_tf = {base_link: np.eye(4)}
    joint_tf = {}

    stack = [base_link]
    while stack:
        parent = stack.pop()
        parent_tf = link_tf[parent]
        for j in children.get(parent, []):
            origin_tf = make_transform(j["xyz"], j["rpy"])
            joint_world = parent_tf @ origin_tf
            joint_tf[j["name"]] = joint_world
            link_tf[j["child"]] = joint_world
            stack.append(j["child"])

    return joint_tf


def main():
    urdf_dir = "/home/wasabi/g1-motion-control/third_party/holosoma/src/holosoma_retargeting/models/g1"
    urdf_path = f"{urdf_dir}/g1_29dof.urdf"

    server = viser.ViserServer(port=8084)
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))
    server.scene.add_frame("/origin", axes_length=0.2, axes_radius=0.01, position=(0.0, 0.0, 0.0))

    # Load URDF mesh if possible
    if yourdfpy and ViserUrdf:
        try:
            # Monkey patch for numpy scalar issue
            _orig_fk = yourdfpy.urdf.URDF._forward_kinematics_joint

            def _patched_fk(self, joint, q=None):
                try:
                    return _orig_fk(self, joint, q)
                except TypeError as e:
                    if "only 0-dimensional arrays can be converted to Python scalars" in str(e):
                        if q is None:
                            q_val = self.cfg[
                                self.actuated_dof_indices[
                                    self.actuated_joint_names.index(joint.name)
                                ]
                            ]
                            q = float(np.asarray(q_val).item())
                        else:
                            q = float(np.asarray(q).item())
                        return _orig_fk(self, joint, q)
                    raise e

            yourdfpy.urdf.URDF._forward_kinematics_joint = _patched_fk
            urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=f"{urdf_dir}/")
            ViserUrdf(server, urdf_or_path=urdf, root_node_name="/robot")
        except Exception as e:
            print(f"Warning: URDF mesh load failed: {e}")

    joints, base = parse_urdf_joints(urdf_path)
    joint_tf = build_link_transforms(joints, base)

    # Draw labels and axis for each joint
    for idx, j in enumerate(joints):
        tf = joint_tf.get(j["name"], np.eye(4))
        pos = tf[:3, 3]
        axis_local = j["axis"]
        axis_world = tf[:3, :3] @ axis_local
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-8)

        # label
        label = f"{idx}:{j['name']}"
        server.scene.add_label(f"/labels/{j['name']}", text=label, position=pos)

        # small axis: align X-axis to joint axis
        q = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), axis_world)
        server.scene.add_frame(
            f"/axes/{j['name']}",
            axes_length=0.08,
            axes_radius=0.006,
            position=pos,
            wxyz=q,
        )

        # Add a small marker sphere at joint for visibility
        server.scene.add_icosphere(
            f"/markers/{j['name']}",
            radius=0.01,
            color=(255, 200, 0),
            position=pos,
        )

    print("G1 URDF joint viewer at http://localhost:8084")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
