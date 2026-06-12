import numpy as np
import mujoco
import trimesh
import os

def main():
    base_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl"
    xml_path = os.path.join(base_dir, "motionbricks/assets/skeletons/g1/scene_29dof.xml")
    qpos_file = os.path.join(base_dir, "output/motionbricks_lite/exports/demo_run_01/qpos.npy")
    qpos_seq = np.load(qpos_file)
    
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    
    d.qpos[:] = qpos_seq[0]
    mujoco.mj_forward(m, d)
    
    combined_mesh = trimesh.Scene()
    
    for i in range(m.ngeom):
        mesh_id = m.geom_dataid[i]
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            stl_path = os.path.join(base_dir, "motionbricks/assets/skeletons/g1/meshes", f"{mesh_name}.STL")
            if not os.path.exists(stl_path): continue
            
            mesh = trimesh.load(stl_path)
            
            p = d.geom_xpos[i]
            R = d.geom_xmat[i].reshape(3, 3)
            
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = p
            
            mesh.apply_transform(transform)
            combined_mesh.add_geometry(mesh)
            
    combined_mesh.export('mujoco_frame0_combined.obj')
    print("Exported mujoco_frame0_combined.obj")

if __name__ == "__main__":
    main()
