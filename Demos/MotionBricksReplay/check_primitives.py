import mujoco
import numpy as np

xml_path = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl\motionbricks\assets\skeletons\g1\g1.xml"
m = mujoco.MjModel.from_xml_path(xml_path)

geom_types = {
    0: "PLANE",
    1: "HFIELD",
    2: "SPHERE",
    3: "CAPSULE",
    4: "ELLIPSOID",
    5: "CYLINDER",
    6: "BOX",
    7: "MESH",
    8: "SDF"
}

print(f"Total Geoms: {m.ngeom}")
for i in range(m.ngeom):
    g_type = m.geom_type[i]
    if g_type != mujoco.mjtGeom.mjGEOM_MESH:
        size = m.geom_size[i]
        type_str = geom_types.get(g_type, str(g_type))
        # Print group (usually visual=1, collision=0 or 3)
        group = m.geom_group[i]
        print(f"Geom {i}: Type {type_str}, Size {size}, Group {group}")
