from PIL import Image
import numpy as np

def get_bbox(path):
    try:
        img = Image.open(path).convert('L')
        arr = np.array(img)
        rows = np.any(arr > 127, axis=1)
        cols = np.any(arr > 127, axis=0)
        if not np.any(rows):
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (int(cmin), int(rmin), int(cmax), int(rmax))
    except Exception as e:
        return str(e)

for i in [0, 5, 10, 15, 20]:
    mimic_path = f'/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/runs/view_motion_humanoid_sword_shield_args/RL_Avatar_Atk_2xCombo01_Motion/render/silhouettes/frame_{i:06d}.png'
    evih_path = f'/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/Demos/MimicKitReplay/results/white_knight_full_v2/silhouettes/frame_{i:06d}.png'
    print(f'Frame {i}: Mimic={get_bbox(mimic_path)} Evih={get_bbox(evih_path)}')
