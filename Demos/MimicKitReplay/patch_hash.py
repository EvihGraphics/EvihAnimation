import json, hashlib
path = '/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference/scene_contract_v3.json'
data = json.load(open(path))
data.pop('scene_contract_sha256', None)
payload = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
actual_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
data['scene_contract_sha256'] = actual_hash
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print('Patched', path, 'with hash', actual_hash)
