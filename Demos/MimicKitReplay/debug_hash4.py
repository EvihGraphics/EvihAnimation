import json, hashlib
manifest = json.load(open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/mesh_reference_manifest.json'))
contract = manifest.get('scene_contract_v3')
print(contract.get('scene_contract_sha256'))
