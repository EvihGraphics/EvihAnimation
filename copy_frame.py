from PIL import Image

evih_path = 'Demos/MimicKitReplay/results/white_knight_full_v2/frames/frame_000005.png'
mimic_path = '/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/rgb_frames/frame_000005.png'
# Wait, mimic_path should be the RGB frame from Isaac Sim. Is it in rgb_frames?
# I'll just use the ones I copied to the scratch folder in a previous session!
import shutil
evih_path = '/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/Demos/MimicKitReplay/results/white_knight_full_v2/frames/frame_000005.png'
shutil.copy(evih_path, '/mnt/c/Users/l3d/.gemini/antigravity/brain/d6effeda-a943-49d0-93bb-759a62df1a4a/artifacts/evih_frame_5_new.png')
