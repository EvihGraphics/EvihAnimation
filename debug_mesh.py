import sys
import os
sys.path.insert(0, '.')
from ai4animation.Import.GLBImporter import GLB
from ai4animation.Standalone.OfflineMeshRenderer import OfflineMeshRenderer
import numpy as np

glb = GLB.Create('Demos/_ASSETS_/Geno/Model.glb')
renderer = OfflineMeshRenderer(glb)
renderer.inv_bind = np.transpose(renderer.inv_bind, (0, 2, 1))

glb_global = np.zeros((len(glb._nodes), 4, 4), dtype=np.float32)
for i, node in enumerate(glb._nodes):
    if node.Parent is None:
        glb_global[i] = node.LocalMatrix
    else:
        glb_global[i] = glb_global[node.Parent] @ node.LocalMatrix

verts = renderer.compute_lbs(glb_global)
print('Num new verts:', len(verts))
print('New vertex min:', np.min(verts, axis=0))
print('New vertex max:', np.max(verts, axis=0))
