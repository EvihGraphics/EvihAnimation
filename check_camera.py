import json
with open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference/scene_contract_v3.json') as f:
    contract = json.load(f)
if 'camera' in contract:
    print(contract['camera'])
if 'camera_samples' in contract:
    print(contract['camera_samples'][0])
