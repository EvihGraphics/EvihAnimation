import numpy as np
import matplotlib.pyplot as plt
import trimesh

m = trimesh.load('evih_frame0_combined.obj')
v = m.vertices

fig = plt.figure(figsize=(15, 5))

# Front view (XY)
ax1 = fig.add_subplot(131)
ax1.scatter(v[:, 0], v[:, 1], s=0.1, alpha=0.1)
ax1.set_title('Front View (XY)')
ax1.set_xlabel('X (Right/Left)')
ax1.set_ylabel('Y (Up/Down)')
ax1.axis('equal')

# Side view (ZY)
ax2 = fig.add_subplot(132)
ax2.scatter(v[:, 2], v[:, 1], s=0.1, alpha=0.1)
ax2.set_title('Side View (ZY)')
ax2.set_xlabel('Z (Forward/Back)')
ax2.set_ylabel('Y (Up/Down)')
ax2.axis('equal')

# Top view (XZ)
ax3 = fig.add_subplot(133)
ax3.scatter(v[:, 0], v[:, 2], s=0.1, alpha=0.1)
ax3.set_title('Top View (XZ)')
ax3.set_xlabel('X (Right/Left)')
ax3.set_ylabel('Z (Forward/Back)')
ax3.axis('equal')

plt.tight_layout()
plt.savefig('evih_frame0_plot.png')
print("Plot saved to evih_frame0_plot.png")
