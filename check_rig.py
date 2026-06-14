import json
data = json.load(open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference/mimickit_source_rig_asset_spec.json'))
print(data['flat_bodies'][0].keys())
print(data['flat_bodies'][1])
