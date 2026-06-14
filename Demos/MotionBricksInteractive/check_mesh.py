import numpy as np
import sys
sys.path.append('../../ai4animation/x64/Release')
import _AI4Animation
model = _AI4Animation.Model("g1_skinned.glb")
for i, mesh in enumerate(model.Meshes):
    if mesh.HasSkinning:
        print(f"Mesh {i} SkinIndices:")
        print(mesh.SkinIndices[:10])
        print("Max index:", np.max(mesh.SkinIndices))
        print("Min index:", np.min(mesh.SkinIndices))
