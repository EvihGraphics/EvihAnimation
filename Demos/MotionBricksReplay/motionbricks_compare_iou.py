import cv2
import mujoco
import numpy as np
import os

def extract_silhouette_evih(frame):
    """
    Extract silhouette from EvihAnimation mask video.
    The robot mesh is greenish [130, 220, 106] due to PyRay ambient lighting, 
    while the background is a gray gradient.
    """
    frame_int = frame.astype(np.int32)
    b, g, r = frame_int[:, :, 0], frame_int[:, :, 1], frame_int[:, :, 2]
    mask = ((g - r > 20) & (g - b > 20)).astype(np.uint8) * 255
    return mask

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1 > 128, mask2 > 128)
    union = np.logical_or(mask1 > 128, mask2 > 128)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)

def main():
    evih_video_path = "evihanimation_mask_replay.mp4"
    
    if not os.path.exists(evih_video_path):
        print(f"Video {evih_video_path} not found.")
        return
        
    xml_path = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml"
    evih_video_path = "evihanimation_mask_replay.mp4"
    cap_evih = cv2.VideoCapture(evih_video_path)
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.vis.global_.fovy = 45.0
    d = mujoco.MjData(m)
    
    qpos_seq = np.load(r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\qpos.npy")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance = 3.0
    cam.elevation = -20
    cam.azimuth = 90
    
    renderer = mujoco.Renderer(m, 480, 640)
    renderer.enable_segmentation_rendering()
    
    ious = []
    frame_idx = 0
    
    os.makedirs("comparison_overlays", exist_ok=True)
    
    # Find perfect FOVY and Z offset using frame 50
    cap_evih.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret_e, frame_e = cap_evih.read()
    mask_e = extract_silhouette_evih(frame_e)
    
    d.qpos[:] = qpos_seq[50]
    mujoco.mj_forward(m, d)
    
    best_iou = 0.0
    best_fovy = 0.0
    best_offset = 0.0
    
    base_lookat = d.subtree_com[1].copy()
    
    for fovy in np.arange(25.0, 50.0, 1.0):
        m.vis.global_.fovy = fovy
        for offset in np.arange(-0.15, 0.15, 0.005):
            mujoco.mj_forward(m, d)
            cam.lookat[:] = base_lookat
            cam.lookat[2] += offset
            mujoco.mj_forward(m, d)
            
            renderer.update_scene(d, camera=cam)
            seg = renderer.render()
            geom_ids = seg[:, :, 0]
            mask_m = (geom_ids > 0).astype(np.uint8) * 255
            
            iou = calculate_iou(mask_e, mask_m)
            if iou > best_iou:
                best_iou = iou
                best_fovy = fovy
                best_offset = offset
            
    print(f"Best FOVY: {best_fovy:.2f}, Best Offset: {best_offset:.4f} with IoU: {best_iou:.4f}")
    
    # Save the best mask
    m.vis.global_.fovy = best_fovy
    mujoco.mj_forward(m, d)
    cam.lookat[:] = base_lookat
    cam.lookat[2] += best_offset
    mujoco.mj_forward(m, d)
    
    renderer.update_scene(d, camera=cam)
    seg = renderer.render()
    geom_ids = seg[:, :, 0]
    mask_m = (geom_ids > 0).astype(np.uint8) * 255
    
    overlay = np.zeros_like(frame_e)
    overlay[:, :, 2] = mask_e # Red
    overlay[:, :, 0] = mask_m # Blue
    cv2.imwrite("comparison_overlays/best_match_overlay.png", overlay)
    
if __name__ == "__main__":
    main()
