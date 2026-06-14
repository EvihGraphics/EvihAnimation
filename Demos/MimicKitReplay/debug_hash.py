import json, hashlib
data = json.load(open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference/scene_contract_v3.json'))
data.pop('scene_contract_sha256', None)
payload = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
print('hash:', hashlib.sha256(payload.encode('utf-8')).hexdigest())
