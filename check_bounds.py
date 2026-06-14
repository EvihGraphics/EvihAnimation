from pathlib import Path
import math

package_dir = Path('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference')
mesh_asset = Path('/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/assets/humanoid_sword_shield.glb')

import json
with open(mesh_asset, 'rb') as f:
    magic = f.read(4)
    version = int.from_bytes(f.read(4), 'little')
    length = int.from_bytes(f.read(4), 'little')
    json_chunk_length = int.from_bytes(f.read(4), 'little')
    json_chunk_type = f.read(4)
    json_data = f.read(json_chunk_length)
    doc = json.loads(json_data.decode('utf-8'))

for acc in doc['accessors']:
    if acc['type'] == 'VEC3':
        print(f"Max: {acc.get('max')}, Min: {acc.get('min')}")
