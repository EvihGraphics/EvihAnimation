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

def transform_mujoco_to_evih(positions):
    evih_positions = np.zeros_like(positions)
    evih_positions[:, :, 0] = positions[:, :, 1]   # X_e = Y_m
    evih_positions[:, :, 1] = positions[:, :, 2]   # Y_e = Z_m
    evih_positions[:, :, 2] = positions[:, :, 0]   # Z_e = X_m
    return evih_positions

def main():
    import sys
    base_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl"
    export_dir = os.path.join(base_dir, "output/motionbricks_lite/exports/demo_run_01")
    qpos_file = os.path.join(export_dir, "qpos.npy")
    
    qpos_seq = np.load(qpos_file)
    xml_path = os.path.join(base_dir, "motionbricks/assets/skeletons/g1/scene_29dof.xml")
    
    mj_positions, parents = get_mujoco_joint_positions(qpos_seq, xml_path)
    evih_positions = transform_mujoco_to_evih(mj_positions)
    
    output_dir = r"D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay"
    np.save(os.path.join(output_dir, "evih_positions.npy"), evih_positions)
    np.save(os.path.join(output_dir, "parents.npy"), parents)
    print("Done generating Evih array formats.")

if __name__ == "__main__":
    main()
