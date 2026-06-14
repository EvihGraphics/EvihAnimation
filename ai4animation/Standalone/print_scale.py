import pygltflib
glb = pygltflib.GLTF2().load('/root/Project/MimicKit/output/img/tmp_white_knight_mesh_reference_20260608_bridge_smoke_v3/assets/humanoid_sword_shield.glb')
for n in glb.nodes:
    if n.scale is not None and n.scale != [1.0, 1.0, 1.0]:
        print(n.name, n.scale)
