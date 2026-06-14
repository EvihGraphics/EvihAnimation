import cv2
import numpy as np

img1 = cv2.imread('/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/Demos/MimicKitReplay/results/white_knight_full_v2/silhouettes/frame_000005.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/runs/view_motion_humanoid_sword_shield_args/RL_Avatar_Atk_2xCombo01_Motion/render/silhouettes/frame_000005.png', cv2.IMREAD_GRAYSCALE)

print('img1 shape:', img1.shape)
print('img2 shape:', img2.shape)

mask1 = img1 > 127
mask2 = img2 > 127

intersection = np.logical_and(mask1, mask2).sum()
union = np.logical_or(mask1, mask2).sum()
iou = intersection / union if union > 0 else 0

print(f'Intersection: {intersection}')
print(f'Union: {union}')
print(f'IoU: {iou}')
