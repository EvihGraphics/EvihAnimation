import os
import sys
import numpy as np
import mujoco
import trimesh
import pygltflib
import json
from scipy.spatial.transform import Rotation as R

# Determine paths
import platform
if platform.system() == 'Windows':
    MOTIONBRICKS_ROOT = r'\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks'
else:
    MOTIONBRICKS_ROOT = '/root/Project/GR00T-WholeBodyControl/motionbricks'

model_path = f"{MOTIONBRICKS_ROOT}/assets/skeletons/g1/scene_29dof.xml"
m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

mujoco.mj_forward(m, d)
mujoco.mj_kinematics(m, d)

# We want to transform from MuJoCo (Z-up) to EvihAnimation (Y-up)
# Evih T_m_to_e: x->y, y->z, z->x. Wait, previously in InteractiveMain:
# [0, 1, 0]
# [0, 0, 1]
# [1, 0, 0]
# Meaning new_x = old_y, new_y = old_z, new_z = old_x.
# Wait, standard Y-up conversion is: x=x, y=z, z=-y.
# But Evih's T_m_to_e is a pure permutation. Let's stick exactly to T_m_to_e used in traces so everything matches perfectly.
T_m_to_e = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=np.float32)

# Inverse (to transform stuff back if needed)
T_e_to_m = T_m_to_e.T

# We need to compute global bind poses for each body in MuJoCo space, then convert to Evih space.
# In MuJoCo, bind pose is when qpos=0. 
# We just called mj_forward with qpos=0.
# The body's local pos/quat relative to parent is in m.body_pos and m.body_quat.

num_nodes = m.nbody

nodes = []
global_bind_matrices_m = np.zeros((num_nodes, 4, 4), dtype=np.float32)

# Build MuJoCo global bind matrices
for i in range(num_nodes):
    mat = np.eye(4, dtype=np.float32)
    pos = d.xpos[i]
    mat[:3, :3] = d.xmat[i].reshape(3, 3)
    mat[:3, 3] = pos
    global_bind_matrices_m[i] = mat

# Calculate global bind matrices in Evih space
global_bind_matrices_e = np.zeros((num_nodes, 4, 4), dtype=np.float32)
inverse_bind_matrices_e = np.zeros((num_nodes, 4, 4), dtype=np.float32)

for i in range(num_nodes):
    g_m = global_bind_matrices_m[i]
    # Transform matrix to Evih space
    R_m = g_m[:3, :3]
    p_m = g_m[:3, 3]
    
    R_e = T_m_to_e @ R_m @ T_m_to_e.T
    p_e = T_m_to_e @ p_m
    
    g_e = np.eye(4, dtype=np.float32)
    g_e[:3, :3] = R_e
    g_e[:3, 3] = p_e
    
    global_bind_matrices_e[i] = g_e
    inverse_bind_matrices_e[i] = np.linalg.inv(g_e)

# Now calculate LOCAL transforms in Evih space to build the gltf nodes array
# For node i, its local transform is Inv(Global(Parent)) * Global(i)
local_bind_matrices_e = np.zeros((num_nodes, 4, 4), dtype=np.float32)
for i in range(num_nodes):
    parent_id = m.body_parentid[i]
    if i == 0 or parent_id == 0 and i != 0: 
        # Usually body 0 is world, body 1 is root.
        # MuJoCo body 0 is worldbody.
        # Wait, if parent_id == 0, its local transform is just its global transform.
        local_bind_matrices_e[i] = global_bind_matrices_e[i]
    else:
        local_bind_matrices_e[i] = inverse_bind_matrices_e[parent_id] @ global_bind_matrices_e[i]

# Create pygltflib nodes
for i in range(num_nodes):
    # Determine children
    children = [j for j in range(num_nodes) if m.body_parentid[j] == i and j != i]
    
    # Extract translation and rotation from local_bind_matrices_e
    mat = local_bind_matrices_e[i]
    trans = mat[:3, 3].tolist()
    
    rot_mat = mat[:3, :3]
    # GLTF uses x,y,z,w
    quat = R.from_matrix(rot_mat).as_quat().tolist()
    
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    if not name:
        name = f"Body_{i}"
        
    node = pygltflib.Node(
        name=name,
        translation=trans,
        rotation=quat,
        children=children if children else None
    )
    nodes.append(node)

# We need to assemble the meshes
all_vertices = []
all_normals = []
all_faces = []
all_joints = []
all_weights = []
vertex_offset = 0

# Use m.geom_group

for g in range(m.ngeom):
    if m.geom_group[g] != 1:
        continue # Only visual
    
    geom_path = f"meshes_mujoco/Geom_{g}.glb"
    if not os.path.exists(geom_path):
        continue
        
    mesh = trimesh.load(geom_path, force='mesh')
    
    body_id = m.geom_bodyid[g]
    
    # The vertices in Geom_{g}.glb are in the geom's local space (if exported from STL as-is)
    # BUT wait! We need to know if ExportInteractiveMeshes or whatever tool exported Geom_{g}.glb
    # exported them in world space or geom space.
    # Standard behavior: The GLB file meshes_mujoco/Geom_{g}.glb usually contains 1 mesh.
    # If it was exported directly from the MuJoCo VFS or STLs, it's in Geom space.
    
    # In interactive_main, the meshes are placed using:
    # p_m = d.geom_xpos[i], R_m = d.geom_xmat[i]
    # This proves the vertices in Geom_{g}.glb are in Geom Local Space!
    
    # We want to transform vertices from Geom Local Space -> Body Local Space -> Global Space (Bind) -> Evih Space
    
    # geom_pos and geom_quat are the transform from Geom -> Body
    pos_gb = m.geom_pos[g]
    quat_gb_mujoco = m.geom_quat[g] # w,x,y,z
    rot_gb = np.zeros((3,3))
    mujoco.mju_quat2Mat(rot_gb.reshape(-1), quat_gb_mujoco)
    
    T_geom_to_body_m = np.eye(4)
    T_geom_to_body_m[:3, :3] = rot_gb
    T_geom_to_body_m[:3, 3] = pos_gb
    
    T_body_to_world_m = global_bind_matrices_m[body_id]
    
    T_geom_to_world_m = T_body_to_world_m @ T_geom_to_body_m
    
    # We want the vertices to be in the Evih Global Bind Space.
    # Note: the vertices in Geom_{g}.glb were ALREADY rotated by T_m_to_e during export.
    # So v_mesh = T_m_to_e * v_geom_mujoco.
    # We want: v_world_e = T_m_to_e * (T_geom_to_world_m * v_geom_mujoco)
    # v_world_e = T_m_to_e * (R_geom_m * (T_m_to_e.T * v_mesh) + p_geom_m)
    # v_world_e = (T_m_to_e * R_geom_m * T_m_to_e.T) * v_mesh + T_m_to_e * p_geom_m
    
    T_mesh_to_world_e = np.eye(4)
    T_mesh_to_world_e[:3, :3] = T_m_to_e @ T_geom_to_world_m[:3, :3] @ T_m_to_e.T
    T_mesh_to_world_e[:3, 3] = T_m_to_e @ T_geom_to_world_m[:3, 3]
    
    mesh.apply_transform(T_mesh_to_world_e)
    
    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    norms = np.array(mesh.vertex_normals, dtype=np.float32)
    
    num_v = len(verts)
    joints = np.full((num_v, 4), [body_id, 0, 0, 0], dtype=np.uint16)
    weights = np.full((num_v, 4), [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    all_vertices.append(verts)
    all_normals.append(norms)
    all_faces.append(faces + vertex_offset)
    all_joints.append(joints)
    all_weights.append(weights)
    
    vertex_offset += num_v

if not all_vertices:
    print("No visual geoms found.")
    sys.exit(0)

all_vertices = np.vstack(all_vertices)
all_normals = np.vstack(all_normals)
all_faces = np.vstack(all_faces)
all_joints = np.vstack(all_joints)
all_weights = np.vstack(all_weights)

print(f"Total vertices: {len(all_vertices)}")
print(f"Total faces: {len(all_faces)}")

# Now we construct the GLTF using pygltflib
import struct

# Data to binary
points_bin = all_vertices.tobytes()
normals_bin = all_normals.tobytes()
faces_bin = all_faces.tobytes()
joints_bin = all_joints.tobytes()
weights_bin = all_weights.tobytes()

# GLTF specifies column-major layout for inverseBindMatrices.
# numpy is row-major by default, so we transpose the last two dimensions.
inv_bind_bin = inverse_bind_matrices_e.transpose(0, 2, 1).copy().tobytes()

buffer_data = points_bin + normals_bin + faces_bin + joints_bin + weights_bin + inv_bind_bin
buffer = pygltflib.Buffer(byteLength=len(buffer_data))

# Buffer Views
offset = 0
points_bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(points_bin), target=pygltflib.ARRAY_BUFFER)
offset += len(points_bin)

normals_bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(normals_bin), target=pygltflib.ARRAY_BUFFER)
offset += len(normals_bin)

faces_bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(faces_bin), target=pygltflib.ELEMENT_ARRAY_BUFFER)
offset += len(faces_bin)

joints_bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(joints_bin), target=pygltflib.ARRAY_BUFFER)
offset += len(joints_bin)

weights_bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(weights_bin), target=pygltflib.ARRAY_BUFFER)
offset += len(weights_bin)

inv_bind_bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=len(inv_bind_bin))
offset += len(inv_bind_bin)

# Accessors
points_acc = pygltflib.Accessor(
    bufferView=0, componentType=pygltflib.FLOAT, count=len(all_vertices), type=pygltflib.VEC3,
    max=all_vertices.max(axis=0).tolist(), min=all_vertices.min(axis=0).tolist()
)
normals_acc = pygltflib.Accessor(
    bufferView=1, componentType=pygltflib.FLOAT, count=len(all_normals), type=pygltflib.VEC3
)
faces_acc = pygltflib.Accessor(
    bufferView=2, componentType=pygltflib.UNSIGNED_INT, count=len(all_faces)*3, type=pygltflib.SCALAR,
    max=[int(all_faces.max())], min=[int(all_faces.min())]
)
joints_acc = pygltflib.Accessor(
    bufferView=3, componentType=pygltflib.UNSIGNED_SHORT, count=len(all_joints), type=pygltflib.VEC4
)
weights_acc = pygltflib.Accessor(
    bufferView=4, componentType=pygltflib.FLOAT, count=len(all_weights), type=pygltflib.VEC4
)
inv_bind_acc = pygltflib.Accessor(
    bufferView=5, componentType=pygltflib.FLOAT, count=num_nodes, type=pygltflib.MAT4
)

# Primitive & Mesh
primitive = pygltflib.Primitive(
    attributes=pygltflib.Attributes(
        POSITION=0, NORMAL=1, JOINTS_0=3, WEIGHTS_0=4
    ),
    indices=2,
    material=0
)
mesh_gltf = pygltflib.Mesh(name="G1_Mesh", primitives=[primitive])

# Skin
skin = pygltflib.Skin(
    name="G1_Skin",
    inverseBindMatrices=5,
    joints=list(range(num_nodes))
)

# Material
material = pygltflib.Material(
    name="DefaultMaterial",
    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.8
    )
)

# Add mesh to the root node (node 0 or a new node)
# We will create a new node for the mesh, which is parented to nothing, but applies the skin.
mesh_node_idx = len(nodes)
mesh_node = pygltflib.Node(
    name="G1_Skinned_Mesh_Node",
    mesh=0,
    skin=0
)
nodes.append(mesh_node)

# Scene
scene = pygltflib.Scene(nodes=[0, mesh_node_idx])

gltf = pygltflib.GLTF2(
    scene=0,
    scenes=[scene],
    nodes=nodes,
    meshes=[mesh_gltf],
    skins=[skin],
    materials=[material],
    accessors=[points_acc, normals_acc, faces_acc, joints_acc, weights_acc, inv_bind_acc],
    bufferViews=[points_bv, normals_bv, faces_bv, joints_bv, weights_bv, inv_bind_bv],
    buffers=[buffer]
)

gltf.set_binary_blob(buffer_data)
gltf.save("g1_skinned.glb")

print("Successfully exported g1_skinned.glb with EvihAnimation Y-up coordinate system!")
