import numpy as np
import pyvista as pv
from tqdm import tqdm
import os
import io
from PIL import Image

class OfflineMeshRenderer:
    def __init__(self, glb_importer, width=800, height=800):
        self.glb = glb_importer
        self.mesh_data = self.glb.SkinnedMesh
        self.skin_data = self.glb.Skin
        
        self.vertices = self.mesh_data.Vertices
        self.triangles = self.mesh_data.Triangles
        self.skin_indices = self.mesh_data.SkinIndices.astype(np.int32)
        self.skin_weights = self.mesh_data.SkinWeights
        
        self.inv_bind = np.transpose(self.skin_data.Inverse_bind_matrices, (0, 2, 1))
        self.joint_mapping = self.skin_data.Joints
        
        # Prepare homogeneous vertices
        self.v_homo = np.ones((self.vertices.shape[0], 4), dtype=np.float32)
        self.v_homo[:, :3] = self.vertices
        self.v_homo_col = self.v_homo.reshape(-1, 4, 1)
        
        # PyVista mesh setup
        faces = np.empty((self.triangles.shape[0], 4), dtype=np.int32)
        faces[:, 0] = 3
        faces[:, 1:] = self.triangles.astype(np.int32)
        self.pv_mesh = pv.PolyData(self.vertices, faces)
        
        # Extract texture if available
        self.texture = None
        if hasattr(self.mesh_data, 'Image') and self.mesh_data.Image is not None:
            if isinstance(self.mesh_data.Image, Image.Image):
                if self.mesh_data.Image.size[0] > 1:
                    self.texture = pv.Texture(np.array(self.mesh_data.Image))
                    if hasattr(self.mesh_data, 'Texcoords') and self.mesh_data.Texcoords is not None:
                        tex_coords = self.mesh_data.Texcoords.copy()
                        tex_coords[:, 1] = 1.0 - tex_coords[:, 1] # Flip V coordinate for PyVista
                        self.pv_mesh.active_t_coords = tex_coords
                        
        self.width = width
        self.height = height

    def compute_lbs(self, global_matrices):
        """
        global_matrices: [NumNodes, 4, 4]
        Returns: [NumVertices, 3] deformed vertices
        """
        current_joint_transforms = global_matrices[self.joint_mapping]
        skin_transforms = current_joint_transforms @ self.inv_bind
        
        vertex_transforms = skin_transforms[self.skin_indices]
        weights = self.skin_weights.reshape(-1, 4, 1, 1)
        weighted_transforms = np.sum(vertex_transforms * weights, axis=1)
        
        v_new = (weighted_transforms @ self.v_homo_col).reshape(-1, 4)[:, :3]
        return v_new
        
    def create_checkerboard(self):
        grid = np.zeros((1024, 1024, 3), dtype=np.uint8)
        cell_size = 128
        for i in range(1024):
            for j in range(1024):
                if (i // cell_size + j // cell_size) % 2 == 0:
                    grid[i, j] = [150, 150, 150]
                else:
                    grid[i, j] = [210, 210, 210]
        return pv.numpy_to_texture(grid)

    def render_animation(self, global_matrices_seq, output_path, fps=30):
        """
        global_matrices_seq: [Frames, NumNodes, 4, 4]
        """
        num_frames = len(global_matrices_seq)
        
        plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        plotter.set_background("white")
        
        # Add checkerboard ground plane
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=20, j_size=20)
        plane.texture_map_to_plane(inplace=True)
        checker_tex = self.create_checkerboard()
        plotter.add_mesh(plane, texture=checker_tex, ambient=0.3, diffuse=0.7, lighting=True)
        
        # Add character mesh
        if self.texture is not None:
            actor = plotter.add_mesh(self.pv_mesh, texture=self.texture, smooth_shading=True, specular=0.1, diffuse=0.9, ambient=0.2)
        else:
            actor = plotter.add_mesh(self.pv_mesh, color="white", smooth_shading=True, specular=0.1, diffuse=0.9, ambient=0.2)
            
        # Optional skeletal lines/axes equivalent
        plotter.add_axes()
        
        # Setup lighting equivalent to AnimLab
        plotter.enable_shadows()
        light = pv.Light(position=(3, 8, 5), focal_point=(0, 0, 0), color='white', intensity=0.9)
        light.positional = True
        light.cone_angle = 60
        plotter.add_light(light)
        
        plotter.open_movie(output_path, framerate=fps)
        
        print(f"Rendering {num_frames} frames to {output_path}...")
        for i in tqdm(range(num_frames)):
            new_verts = self.compute_lbs(global_matrices_seq[i])
            self.pv_mesh.points = new_verts
            
            # Follow cam maintaining an isometric angle like the original notebook viewer
            root_pos = global_matrices_seq[i][0, :3, 3]
            plotter.camera.focal_point = (root_pos[0], 0.8, root_pos[2])
            plotter.camera.position = (root_pos[0] + 3.0, 3.0, root_pos[2] + 4.0)
            
            plotter.write_frame()
            
        plotter.close()
        print("Rendering complete!")
