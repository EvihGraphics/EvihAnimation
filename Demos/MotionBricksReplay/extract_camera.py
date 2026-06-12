import os
import numpy as np
import mujoco

def main():
    base_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl"
    export_dir = os.path.join(base_dir, "output/motionbricks_lite/exports/demo_run_01")
    qpos_file = os.path.join(export_dir, "qpos.npy")
    
    qpos_seq = np.load(qpos_file)
    num_frames = qpos_seq.shape[0]
    
    xml_path = os.path.join(base_dir, "motionbricks/assets/skeletons/g1/scene_29dof.xml")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance = 3.0
    cam.elevation = -20
    cam.azimuth = 90
    print("Camera FOVY:", m.vis.global_.fovy if hasattr(m.vis, 'global_') else "unknown")
    
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id == -1: pelvis_id = 1
    
    vcam = mujoco.MjvCamera()
    vopt = mujoco.MjvOption()
    scn = mujoco.MjvScene(m, maxgeom=1000)

    cam_pos = np.zeros((num_frames, 3))
    cam_lookat = np.zeros((num_frames, 3))
    cam_up = np.zeros((num_frames, 3))
    
    for i in range(num_frames):
        d.qpos[:] = qpos_seq[i]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.subtree_com[1]
        # To get the exact global position of the camera, we update the scene
        mujoco.mjv_updateScene(m, d, vopt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        
        # scn.camera[0] or we can just read scn.camera
        # Wait, scn.camera is an array of 2 cameras. index 0 is the one used.
        c = scn.camera[0]
        cam_pos[i] = c.pos
        cam_lookat[i] = cam.lookat[:]
        cam_up[i] = c.up
        
    # Now transform these coordinates to Evih space
    def transform_to_evih(points):
        evih_points = np.zeros_like(points)
        evih_points[:, 0] = points[:, 1]  # X_e = Y_m
        evih_points[:, 1] = points[:, 2]  # Y_e = Z_m
        evih_points[:, 2] = points[:, 0]  # Z_e = X_m
        return evih_points
        
    cam_pos = transform_to_evih(cam_pos)
    cam_lookat = transform_to_evih(cam_lookat)
    cam_up = transform_to_evih(cam_up)

    out_dir = r"D:\AnimationTech-learning\EvihAnimation-motionbricks-replay\Demos\MotionBricksReplay"
    np.save(os.path.join(out_dir, "evih_cam_pos.npy"), cam_pos)
    np.save(os.path.join(out_dir, "evih_cam_lookat.npy"), cam_lookat)
    np.save(os.path.join(out_dir, "evih_cam_up.npy"), cam_up)
    print("Exported Evih camera trajectory.")

if __name__ == "__main__":
    main()
