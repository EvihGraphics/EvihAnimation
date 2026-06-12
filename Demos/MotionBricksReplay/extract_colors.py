import os
import json
import mujoco

def main():
    base_dir = r"\\wsl.localhost\Ubuntu-20.04\root\Project\GR00T-WholeBodyControl"
    xml_path = os.path.join(base_dir, "motionbricks/assets/skeletons/g1/scene_29dof.xml")
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    
    geom_colors = []
    
    for i in range(m.ngeom):
        # type 7 is mjGEOM_MESH
        rgba = m.geom_rgba[i]
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        a = int(rgba[3] * 255)
        geom_colors.append([r, g, b, a])
            
    with open("geom_colors.json", "w") as f:
        json.dump(geom_colors, f)
        
    print(f"Exported {len(geom_colors)} colors.")

if __name__ == "__main__":
    main()
