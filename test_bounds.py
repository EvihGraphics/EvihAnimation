import trimesh
mesh = trimesh.load('/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260607_bridge_full_v2/assets/humanoid_sword_shield.glb', force='mesh')
print('Bounds:', mesh.bounds)
