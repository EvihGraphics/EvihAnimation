import json

with open('/root/Project/MimicKit/output/train/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/ue_export_mesh_reference/visual_replay/pose_dof_replay.jsonl') as f:
    line = f.readline()
    data = json.loads(line)
    print(data['dof_pos'][:5])
    print(sum(data['dof_pos']))
