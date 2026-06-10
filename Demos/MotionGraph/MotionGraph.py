import numpy as np
import scipy.sparse.csgraph as csgraph
import scipy.ndimage as ndimage

from ai4animation import Motion, Transform, Rotation, Vector3, Tensor
from collections import defaultdict


class PointCloudBuilder:
    def __init__(self, motion: Motion, window_size: int = 15):
        self.motion = motion
        self.window_size = window_size
        self.num_frames = motion.NumFrames
        self.num_joints = motion.NumJoints
        
        # We will extract local joint positions and weights. For simplicity, 
        # we weight all joints equally except root.
        self.weights = np.ones(self.num_joints)
        self.weights[0] = 0.0 # root is usually ignored in point cloud distance 
        self.weights = self.weights / np.sum(self.weights)
        
        # [num_frames, num_joints, 3]
        self.positions = motion.GetBonePositions()

    def get_window(self, frame_idx):
        """Returns the point cloud window for a given frame."""
        start = max(0, frame_idx - self.window_size)
        end = min(self.num_frames, frame_idx + self.window_size + 1)
        
        # Create a fixed size window by padding if necessary
        window = np.zeros((2 * self.window_size + 1, self.num_joints, 3))
        
        actual_start = self.window_size - (frame_idx - start)
        actual_end = self.window_size + (end - frame_idx)
        
        window[actual_start:actual_end] = self.positions[start:end]
        
        # If padded at the beginning, duplicate first frame
        if start == 0 and actual_start > 0:
            window[:actual_start] = self.positions[0]
        # If padded at the end, duplicate last frame
        if end == self.num_frames and actual_end < window.shape[0]:
            window[actual_end:] = self.positions[-1]
            
        return window


def align_point_clouds_2d(pc_i, pc_j, weights):
    """
    Aligns pc_j to pc_i along the XZ plane.
    pc_i, pc_j: [window, joints, 3]
    weights: [joints]
    
    Returns:
        distance: float
        transformation: (theta, tx, tz)
    """
    window_len = pc_i.shape[0]
    
    # Flatten window and joints
    w = np.tile(weights[np.newaxis, :], (window_len, 1)).flatten() # [window * joints]
    w = w / np.sum(w)
    
    pts_i = pc_i.reshape(-1, 3) # [N, 3]
    pts_j = pc_j.reshape(-1, 3)
    
    # Compute weighted center of mass (only X and Z)
    cm_i = np.average(pts_i[:, [0, 2]], axis=0, weights=w)
    cm_j = np.average(pts_j[:, [0, 2]], axis=0, weights=w)
    
    # Center the points
    pts_i_centered = pts_i.copy()
    pts_i_centered[:, [0, 2]] -= cm_i
    
    pts_j_centered = pts_j.copy()
    pts_j_centered[:, [0, 2]] -= cm_j
    
    # Compute optimal rotation theta around Y axis
    # theta = arctan2(sum(w * (x_i z_j - z_i x_j)), sum(w * (x_i x_j + z_i z_j)))
    x_i, z_i = pts_i_centered[:, 0], pts_i_centered[:, 2]
    x_j, z_j = pts_j_centered[:, 0], pts_j_centered[:, 2]
    
    top = np.sum(w * (x_i * z_j - z_i * x_j))
    bot = np.sum(w * (x_i * x_j + z_i * z_j))
    theta = np.arctan2(top, bot)
    
    # Apply rotation to centered j points
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rot_x_j = x_j * cos_theta + z_j * sin_theta
    rot_z_j = -x_j * sin_theta + z_j * cos_theta
    
    pts_j_aligned = pts_j_centered.copy()
    pts_j_aligned[:, 0] = rot_x_j
    pts_j_aligned[:, 2] = rot_z_j
    
    # Distance is weighted sum of squared differences
    diff = pts_i_centered - pts_j_aligned
    sq_dist = np.sum(diff**2, axis=1)
    distance = np.sum(w * sq_dist)
    
    # Transformation to move j to i
    tx = cm_i[0] - (cm_j[0] * cos_theta + cm_j[1] * sin_theta)
    tz = cm_i[1] - (-cm_j[0] * sin_theta + cm_j[1] * cos_theta)
    
    return distance, (theta, tx, tz)


def compute_distance_matrix(motion: Motion, window_size: int = 15):
    builder = PointCloudBuilder(motion, window_size)
    num_frames = motion.NumFrames
    
    dist_matrix = np.zeros((num_frames, num_frames))
    transform_matrix = np.zeros((num_frames, num_frames, 3)) # theta, tx, tz
    
    print("Pre-computing windows...")
    windows = np.array([builder.get_window(i) for i in range(num_frames)])
    weights = builder.weights
    
    print(f"Computing distance matrix {num_frames}x{num_frames}...")
    import time
    start_time = time.time()
    
    # We can optimize this with broadcasting in numpy, but doing it loop-wise is okay for ~300 frames.
    # To be reasonably fast, we vectorize over the J dimension.
    
    # windows shape: [F, W, J, 3]
    # We want to compare all pairs (i, j). We can do this row by row.
    
    w_expanded = np.tile(weights[np.newaxis, :], (2*window_size+1, 1)).flatten()
    w_expanded = w_expanded / np.sum(w_expanded)
    
    pts = windows.reshape(num_frames, -1, 3) # [F, W*J, 3]
    cm = np.average(pts[:, :, [0, 2]], axis=1, weights=w_expanded) # [F, 2]
    
    pts_centered = pts.copy()
    pts_centered[:, :, 0] -= cm[:, 0, np.newaxis]
    pts_centered[:, :, 2] -= cm[:, 1, np.newaxis]
    
    x = pts_centered[:, :, 0] # [F, W*J]
    z = pts_centered[:, :, 2] # [F, W*J]
    
    for i in range(num_frames):
        x_i = x[i:i+1, :]
        z_i = z[i:i+1, :]
        
        top = np.sum(w_expanded * (x_i * z - z_i * x), axis=1) # [F]
        bot = np.sum(w_expanded * (x_i * x + z_i * z), axis=1) # [F]
        
        theta = np.arctan2(top, bot) # [F]
        
        cos_theta = np.cos(theta) # [F]
        sin_theta = np.sin(theta) # [F]
        
        rot_x = x * cos_theta[:, np.newaxis] + z * sin_theta[:, np.newaxis] # [F, W*J]
        rot_z = -x * sin_theta[:, np.newaxis] + z * cos_theta[:, np.newaxis] # [F, W*J]
        
        pts_aligned = pts_centered.copy()
        pts_aligned[:, :, 0] = rot_x
        pts_aligned[:, :, 2] = rot_z
        
        diff = pts_centered[i:i+1, :, :] - pts_aligned # [F, W*J, 3]
        sq_dist = np.sum(diff**2, axis=2) # [F, W*J]
        distance = np.sum(w_expanded * sq_dist, axis=1) # [F]
        
        tx = cm[i, 0] - (cm[:, 0] * cos_theta + cm[:, 1] * sin_theta)
        tz = cm[i, 1] - (-cm[:, 0] * sin_theta + cm[:, 1] * cos_theta)
        
        dist_matrix[i, :] = distance
        transform_matrix[i, :, 0] = theta
        transform_matrix[i, :, 1] = tx
        transform_matrix[i, :, 2] = tz
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{num_frames} frames")
            
    print(f"Distance matrix computed in {time.time() - start_time:.2f}s")
    return dist_matrix, transform_matrix


def extract_local_minima(dist_matrix, threshold=0.1, local_window=10):
    num_frames = dist_matrix.shape[0]
    minima = []
    
    # Filter out transitions that are too close to identity (i.e. self transitions)
    # We want to jump at least local_window frames away
    mask = np.ones_like(dist_matrix)
    for i in range(num_frames):
        start = max(0, i - local_window)
        end = min(num_frames, i + local_window + 1)
        mask[i, start:end] = np.inf
        
    masked_dist = dist_matrix * mask
    masked_dist[masked_dist == 0] = np.inf # Ignore exact 0
    
    # 2D local minima
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(masked_dist, size=local_window)
    is_local_min = (masked_dist == local_min) & (masked_dist < threshold)
    
    for i in range(num_frames):
        for j in range(num_frames):
            if is_local_min[i, j]:
                minima.append((i, j, masked_dist[i, j]))
                
    print(f"Found {len(minima)} valid local minima (threshold={threshold})")
    return minima


class MotionGraph:
    def __init__(self, motion: Motion):
        self.motion = motion
        self.nodes = [] # List of clip indices
        self.edges = defaultdict(list) # node_idx -> list of (target_node_idx, transition_frame, target_frame, transform)
        self.scc_nodes = set()
        
    def build(self, dist_matrix, transform_matrix, minima):
        # In a simple motion graph, nodes are just frames.
        # Edges are natural progression (i -> i+1) OR jumps (minima).
        
        num_frames = self.motion.NumFrames
        
        # Build adjacency matrix for Tarjan
        adj = defaultdict(list)
        
        for i in range(num_frames - 1):
            adj[i].append((i + 1, 1.0)) # Normal progression
            
        for (i, j, dist) in minima:
            # We want to jump from frame i to frame j
            adj[i].append((j, dist))
            
        # Extract Strongly Connected Components
        graph_matrix = np.zeros((num_frames, num_frames))
        for u in adj:
            for v, _ in adj[u]:
                graph_matrix[u, v] = 1
                
        n_components, labels = csgraph.connected_components(csgraph=graph_matrix, directed=True, connection='strong')
        
        # Find the largest component
        unique, counts = np.unique(labels, return_counts=True)
        largest_comp_label = unique[np.argmax(counts)]
        
        self.scc_nodes = set(np.where(labels == largest_comp_label)[0])
        print(f"Largest Strongly Connected Component has {len(self.scc_nodes)} nodes out of {num_frames}")
        
        # Build final edges only within SCC
        for i in range(num_frames - 1):
            if i in self.scc_nodes and (i+1) in self.scc_nodes:
                self.edges[i].append({
                    'target': i + 1,
                    'type': 'seq',
                    'transform': None
                })
                
        for (i, j, dist) in minima:
            if i in self.scc_nodes and j in self.scc_nodes:
                self.edges[i].append({
                    'target': j,
                    'type': 'jump',
                    'transform': transform_matrix[i, j] # (theta, tx, tz)
                })

    def get_next_state(self, current_frame):
        if current_frame not in self.scc_nodes or current_frame not in self.edges or len(self.edges[current_frame]) == 0:
            # Fallback
            if (current_frame + 1) < self.motion.NumFrames:
                return current_frame + 1, 'seq', None
            else:
                return 0, 'jump', None # Hard reset
                
        choices = self.edges[current_frame]
        
        # Strategy: favor sequence, but occasionally jump if available
        jumps = [c for c in choices if c['type'] == 'jump']
        seqs = [c for c in choices if c['type'] == 'seq']
        
        if len(jumps) > 0 and np.random.rand() < 0.2: # 20% chance to take a jump if available
            choice = np.random.choice(jumps)
            return choice['target'], choice['type'], choice['transform']
        elif len(seqs) > 0:
            return seqs[0]['target'], seqs[0]['type'], None
        else:
            choice = np.random.choice(choices)
            return choice['target'], choice['type'], choice['transform']
