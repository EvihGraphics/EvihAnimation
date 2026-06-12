import mujoco

xml_path = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\g1.xml"
m = mujoco.MjModel.from_xml_path(xml_path)

for i in range(m.ngeom):
    g_type = m.geom_type[i]
    if g_type != mujoco.mjtGeom.mjGEOM_MESH:
        group = m.geom_group[i]
        # In MuJoCo, default visual group is 0, collision is 3, or visual is 1, collision is 0.
        # Let's print the name, type, group, and size.
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        size = m.geom_size[i]
        print(f"Geom {i}: Name='{name}', Type={g_type}, Group={group}, Size={size}")
