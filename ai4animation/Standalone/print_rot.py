import json
data = json.loads(open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260608_bridge_smoke_v3/ue_export_mesh_reference/obs_fixture.jsonl').readline())
print('Initial rot:', data['root_rot_xyzw'])
