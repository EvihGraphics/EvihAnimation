import mujoco
import numpy as np
import json
import os

model_path = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\scene_29dof.xml"
try:
    m = mujoco.MjModel.from_xml_path(model_path)
except Exception as e:
    print("Error loading:", e)
    with open(model_path, 'r') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('meshdir="../meshes/g1/"', 'meshdir="\\\\wsl.localhost\\Ubuntu-20.04\\root\\Project\\GR00T-WholeBodyControl\\motionbricks\\assets\\skeletons\\g1\\meshes"')
    
    with open('scene_29dof_patched.xml', 'w') as f:
        f.write(xml_content)
    
    with open(r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\g1_29dof.xml", 'r') as f:
        g1_xml = f.read()
    g1_xml = g1_xml.replace('meshdir="../meshes/g1/"', 'meshdir="\\\\wsl.localhost\\Ubuntu-20.04\\root\\Project\\GR00T-WholeBodyControl\\motionbricks\\assets\\skeletons\\g1\\meshes"')
    with open('g1_29dof_patched.xml', 'w') as f:
        f.write(g1_xml)
        
    xml_content = xml_content.replace('g1_29dof.xml', 'g1_29dof_patched.xml')
    with open('scene_29dof_patched.xml', 'w') as f:
        f.write(xml_content)

    m = mujoco.MjModel.from_xml_path('scene_29dof_patched.xml')

with open('geom_mesh_names.json', 'r') as f:
    evih_mesh_names = json.load(f)
    
evih_types = np.load('evih_geom_types.npy')
evih_sizes = np.load('evih_geom_sizes.npy')

missing = []
total_visual = 0
for i in range(m.ngeom):
    if m.geom_group[i] == 1: # Visual group
        total_visual += 1
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        geom_type = m.geom_type[i]
        
        # How does Evih handle it?
        mesh_name = evih_mesh_names[i]
        evih_type = evih_types[i]
        
        if mesh_name == "" and evih_type not in [2, 5]: # Neither mesh, nor sphere, nor cylinder
            missing.append(f"Geom {i} (Name: {name}, Type: {geom_type}) is visual but not handled in Evih! Size: {m.geom_size[i]}")

print(f"Total Visual Geoms: {total_visual}")
if len(missing) == 0:
    print("All visual geoms are handled in Evih!")
else:
    print("MISSING GEOMS:")
    for m in missing:
        print(m)
