import cv2
import numpy as np
import os
import json

def extract_silhouette_evih(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    _, mask = cv2.threshold(v_channel, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def extract_silhouette_mujoco(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_sky = np.array([90, 40, 40])
    upper_sky = np.array([150, 255, 255])
    sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
    lower_floor = np.array([0, 0, 150])
    upper_floor = np.array([180, 50, 255])
    floor_mask = cv2.inRange(hsv, lower_floor, upper_floor)
    bg_mask = cv2.bitwise_or(sky_mask, floor_mask)
    robot_mask = cv2.bitwise_not(bg_mask)
    kernel = np.ones((3,3), np.uint8)
    robot_mask = cv2.morphologyEx(robot_mask, cv2.MORPH_OPEN, kernel)
    robot_mask = cv2.dilate(robot_mask, kernel, iterations=1)
    return robot_mask

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou

def main():
    evih_video_path = "evihanimation_true_mesh_replay.mp4"
    mujoco_video_path = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\output\motionbricks_lite\exports\demo_run_01\motionbricks_evih_replay.mp4"
    
    cap_evih = cv2.VideoCapture(evih_video_path)
    cap_mujoco = cv2.VideoCapture(mujoco_video_path)
    
    ious = []
    frame_idx = 0
    
    os.makedirs("comparison_overlays", exist_ok=True)
    
    while True:
        ret_e, frame_e = cap_evih.read()
        ret_m, frame_m = cap_mujoco.read()
        
        if not ret_e or not ret_m:
            break
            
        mask_e = extract_silhouette_evih(frame_e)
        mask_m = extract_silhouette_mujoco(frame_m)
        
        iou = compute_iou(mask_e, mask_m)
        ious.append(iou)
        
        if frame_idx % 50 == 0:
            overlay = np.zeros_like(frame_e)
            overlay[mask_e > 0] = [0, 0, 255]
            overlay[mask_m > 0] = [255, 0, 0]
            intersection = np.logical_and(mask_e, mask_m)
            overlay[intersection] = [0, 255, 0]
            cv2.imwrite(f"comparison_overlays/overlay_{frame_idx:04d}_iou_{iou:.2f}.png", overlay)
            
        frame_idx += 1

    print(f"Processed {len(ious)} frames. Mean IoU: {np.mean(ious):.4f}")

if __name__ == "__main__":
    main()
