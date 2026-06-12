import cv2
import numpy as np
import os
import json

def extract_silhouette(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The background is likely a solid color (e.g., raylib gray or mujoco skybox).
    # However, mujoco skybox has gradient. We can use a simple threshold or background subtraction.
    # We will assume pixels with high saturation or low value are foreground, 
    # but a simpler way is to compare against background frame or just threshold if background is plain.
    # Actually, we can use Canny edge + morphological close, or simple thresholding if it's mostly gray/white.
    
    # Since Evih renders on a plain background, let's just do a basic binary threshold
    # assuming background is very light or very dark. 
    # Let's use adaptive thresholding or Otsu's.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def main():
    evih_video_path = "evihanimation_true_mesh_replay.mp4"
    mujoco_video_path = "C:/Users/l3d/.gemini/antigravity/brain/15989d56-c4aa-4e43-b36a-5a49b647bd13/motionbricks_evih_replay.mp4"
    
    cap_evih = cv2.VideoCapture(evih_video_path)
    cap_mujoco = cv2.VideoCapture(mujoco_video_path)
    
    if not cap_evih.isOpened() or not cap_mujoco.isOpened():
        print("Error: Could not open videos.")
        return

    out_dir = "comparison_overlays"
    os.makedirs(out_dir, exist_ok=True)
    
    ious = []
    frame_idx = 0
    
    while True:
        ret1, frame_evih = cap_evih.read()
        ret2, frame_mujoco = cap_mujoco.read()
        
        if not ret1 or not ret2:
            break
            
        # Ensure sizes match
        if frame_evih.shape != frame_mujoco.shape:
            frame_mujoco = cv2.resize(frame_mujoco, (frame_evih.shape[1], frame_evih.shape[0]))
            
        # Wait, the camera angles might be slightly different. We might just overlay them for visual comparison.
        # IoU might be low if camera is not perfectly matched (FOV, distance, center).
        # We will extract silhouettes and compute IoU anyway.
        sil_evih = extract_silhouette(frame_evih)
        sil_mujoco = extract_silhouette(frame_mujoco)
        
        iou = calculate_iou(sil_evih, sil_mujoco)
        ious.append(iou)
        
        # Create overlay image: Red for Evih, Blue for MuJoCo
        overlay = np.zeros_like(frame_evih)
        overlay[sil_evih > 0] = [0, 0, 255] # Red Evih
        # For MuJoCo, set blue channel to 255, leaving red intact to create Magenta on overlap
        overlay[sil_mujoco > 0, 0] = 255 # Blue channel is index 0
        
        # Save overlay for every 50th frame
        if frame_idx % 50 == 0:
            cv2.imwrite(os.path.join(out_dir, f"overlay_{frame_idx:04d}_iou_{iou:.2f}.png"), overlay)
            
        frame_idx += 1
        
    cap_evih.release()
    cap_mujoco.release()
    
    mean_iou = float(np.mean(ious))
    print(f"Processed {frame_idx} frames. Mean IoU: {mean_iou:.4f}")
    
    report = {
        "mean_silhouette_iou": mean_iou,
        "frames_processed": frame_idx,
        "parity_pass": mean_iou >= 0.90
    }
    
    with open("visual_metric_report.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    main()
