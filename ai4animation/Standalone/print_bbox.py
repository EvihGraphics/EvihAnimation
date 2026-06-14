import json
from PIL import Image

manifest = json.load(open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260608_bridge_smoke_v3/mesh_reference_manifest.json'))
silhouettes = manifest['silhouettes']
for f in [0, 5]:
    path = silhouettes.get(str(f))
    if not path: continue
    img = Image.open(path).convert('L')
    bbox = img.getbbox()
    if bbox is not None:
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        print(f'Reference {f}: w={w}, h={h}, area={w*h}')

evih_silhouettes = {
    0: '/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/Demos/MimicKitReplay/results/white_knight_mesh_replay_v3_smoke/silhouettes/frame_000000.png',
    5: '/mnt/d/AnimationTech-learning/EvihAnimation-mimickit-bridge/Demos/MimicKitReplay/results/white_knight_mesh_replay_v3_smoke/silhouettes/frame_000005.png'
}
for f in [0, 5]:
    path = evih_silhouettes[f]
    img = Image.open(path).convert('L')
    bbox = img.getbbox()
    if bbox is not None:
        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        print(f'Evih {f}: w={w}, h={h}, area={w*h}')
