import json, hashlib
manifest = json.load(open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/mesh_reference_manifest.json'))
contract = manifest.get('scene_contract_v3')
contract.pop('scene_contract_sha256', None)
payload = json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
print('hash:', hashlib.sha256(payload.encode('utf-8')).hexdigest())
