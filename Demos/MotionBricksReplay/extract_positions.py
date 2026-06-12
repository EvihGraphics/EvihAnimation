import os
import numpy as np
import mujoco

def get_mujoco_joint_positions(qpos_seq, xml_path):
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    
    num_frames = qpos_seq.shape[0]
    num_bodies = m.nbody
    
    positions = np.zeros((num_frames, num_bodies, 3))
    parents = m.body_parentid
    
    for i in range(num_frames):
        d.qpos[:] = qpos_seq[i]
        mujoco.mj_forward(m, d)
        positions[i] = d.xpos.copy()
        
    return positions, parents

def get_mujoco_geom_transforms(qpos_seq, xml_path):
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    
    num_frames = qpos_seq.shape[0]
    num_geoms = m.ngeom
    
    positions = np.zeros((num_frames, num_geoms, 3))
    rotations = np.zeros((num_frames, num_geoms, 3, 3))
    
    geom_mesh_names = []
    for i in range(num_geoms):
        mesh_id = m.geom_dataid[i]
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            geom_mesh_names.append(mesh_name)
        else:
            geom_mesh_names.append("")
            
    for i in range(num_frames):
        d.qpos[:] = qpos_seq[i]
        mujoco.mj_forward(m, d)
        positions[i] = d.geom_xpos.copy()
        rotations[i] = d.geom_xmat.copy().reshape(-1, 3, 3)
        
    return positions, rotations, geom_mesh_names

def transform_mujoco_to_evih(positions):
    evih_positions = np.zeros_like(positions)
    evih_positions[:, :, 0] = positions[:, :, 1]   # X_e = Y_m
    evih_positions[:, :, 1] = positions[:, :, 2]   # Y_e = Z_m
    evih_positions[:, :, 2] = positions[:, :, 0]   # Z_e = X_m
    return evih_positions

def transform_geom_to_evih(positions, rotations):
    # T = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    evih_positions = np.zeros_like(positions)
    evih_positions[:, :, 0] = positions[:, :, 1]
    evih_positions[:, :, 1] = positions[:, :, 2]
    evih_positions[:, :, 2] = positions[:, :, 0]
    
    T = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    T_inv = T.T
    
    # R_e = T * R_m * T_inv
    # rotations shape: (frames, geoms, 3, 3)
    evih_rotations = np.zeros_like(rotations)
    for i in range(rotations.shape[0]):
        for j in range(rotations.shape[1]):
            R_m = rotations[i, j]
            R_e = T @ R_m @ T_inv
            evih_rotations[i, j] = R_e
            
    return evih_positions, evih_rotations

def main():
    import sys
    import json
    base_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl"
    export_dir = os.path.join(base_dir, "output/motionbricks_lite/exports/demo_run_01")
    qpos_file = os.path.join(export_dir, "qpos.npy")
    
    qpos_seq = np.load(qpos_file)
    xml_path = os.path.join(base_dir, "motionbricks/assets/skeletons/g1/scene_29dof.xml")
    
    # Still extract joints for the skeletal view if needed
    mj_positions, parents = get_mujoco_joint_positions(qpos_seq, xml_path)
    evih_positions = transform_mujoco_to_evih(mj_positions)
    
    # Extract geoms for the mesh view
    geom_pos, geom_rot, geom_mesh_names = get_mujoco_geom_transforms(qpos_seq, xml_path)
    evih_geom_pos, evih_geom_rot = transform_geom_to_evih(geom_pos, geom_rot)
    
    output_dir = r"D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay"
    np.save(os.path.join(output_dir, "evih_positions.npy"), evih_positions)
    np.save(os.path.join(output_dir, "parents.npy"), parents)
    np.save(os.path.join(output_dir, "evih_geom_pos.npy"), evih_geom_pos)
    np.save(os.path.join(output_dir, "evih_geom_rot.npy"), evih_geom_rot)
    with open(os.path.join(output_dir, "geom_mesh_names.json"), "w") as f:
        json.dump(geom_mesh_names, f)
        
    print("Done generating Evih array formats, including mesh geoms.")

if __name__ == "__main__":
    main()
