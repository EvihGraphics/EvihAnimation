import json

with open(r'\\wsl.localhost\Ubuntu-20.04\root\Project\MimicKit\output\train\tmp_white_knight_mesh_reference_20260607_bridge_full_v2\ue_export_mesh_reference\visual_replay\pose_dof_replay.jsonl') as f:
    lines = f.readlines()
    data = json.loads(lines[5])
    print(data.keys())
    for key in data:
        if isinstance(data[key], list) and len(data[key]) > 0:
            print(f"{key}: {data[key][:3]} (len: {len(data[key])})")
