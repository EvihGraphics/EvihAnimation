import numpy as np
from scipy.signal import savgol_filter
from ai4animation import Math
import scipy.spatial

LN2 = np.log(2.0)

def fast_negexp(x):
    """Fast approximation of exp(-x) using rational function"""
    return 1.0 / (1.0 + x + 0.48*x*x + 0.235*x*x*x)

def halflife_to_damping(halflife, eps=1e-5):
    return (4.0 * LN2) / (halflife + eps)

def quat_abs(q):
    sign = np.where(q[..., 0] < 0, -1.0, 1.0)
    return q * sign[..., None]

def quat_log(x, eps=1e-8):
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,np.newaxis]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / (length+eps))
    return halfangle * x[...,1:]

def quat_exp(x, eps=1e-8):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)

def quat_to_scaled_angle_axis(x, eps=1e-8):
    return 2.0 * quat_log(x, eps)

def quat_from_scaled_angle_axis(x, eps=1e-8):
    return quat_exp(x / 2.0, eps)

def quat_inv(q):
    # [w, x, y, z] -> [w, -x, -y, -z]
    inv = q.copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def quat_mul_vec(q, v):
    q_v = np.zeros(list(v.shape)[:-1] + [4], dtype=v.dtype)
    q_v[..., 1:] = v
    res = quat_mul(quat_mul(q, q_v), quat_inv(q))
    return res[..., 1:]

def qp_inv(q, p):
    qi = quat_inv(q)
    pi = -quat_mul_vec(qi, p)
    return qi, pi

def qp_mul(qp1, qp2):
    q1, p1 = qp1
    q2, p2 = qp2
    q_out = quat_mul(q1, q2)
    p_out = quat_mul_vec(q1, p2) + p1
    return q_out, p_out

def quat_fk(local_q, local_p, parents):
    global_q = np.empty_like(local_q)
    global_p = np.empty_like(local_p)
    for j, parent in enumerate(parents):
        if parent == -1:
            global_q[..., j, :] = local_q[..., j, :]
            global_p[..., j, :] = local_p[..., j, :]
        else:
            gq, gp = qp_mul((global_q[..., parent, :], global_p[..., parent, :]), 
                            (local_q[..., j, :], local_p[..., j, :]))
            global_q[..., j, :] = gq
            global_p[..., j, :] = gp
    return global_q, global_p

def quat_normalize(q):
    norm = np.sqrt(np.sum(q*q, axis=-1, keepdims=True))
    return q / np.maximum(norm, 1e-8)

def simple_spring_damper_exact(x, v, x_goal, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j0 = x - x_goal
    j1 = v + j0 * y
    eydt = fast_negexp(y * dt)

    x_next = eydt * (j0 + j1 * dt) + x_goal
    v_next = eydt * (v - j1 * y * dt)
    return x_next, v_next

def simple_spring_damper_exact_quat(x, v, x_goal, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    eydt = fast_negexp(y * dt)

    q_rel = quat_abs(quat_mul(x, quat_inv(x_goal)))
    j0 = quat_to_scaled_angle_axis(q_rel)
    j1 = v + j0 * y

    x_next = quat_mul(quat_from_scaled_angle_axis(eydt * (j0 + j1 * dt)), x_goal)
    v_next = eydt * (v - j1 * y * dt)
    return quat_normalize(x_next), v_next

def spring_character_update(x, v, a, v_goal, halflife, dt):
    y = halflife_to_damping(halflife) / 2.0
    j0 = v - v_goal
    j1 = a + j0 * y
    eydt = fast_negexp(y * dt)

    x_next = (
        eydt * ((-j1) / (y * y) + (-j0 - j1 * dt) / y)
        + (j1 / (y * y))
        + (j0 / y)
        + v_goal * dt
        + x
    )
    v_next = eydt * (j0 + j1 * dt) + v_goal
    a_next = eydt * (a - j1 * y * dt)

    return x_next, v_next, a_next

def inertialize_transition_vec3(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x = (src_x + off_x) - dst_x
    off_v = (src_v + off_v) - dst_v
    return off_x, off_v

def inertialize_update_vec3(in_x, in_v, off_x, off_v, halflife, dt):
    off_x, off_v = simple_spring_damper_exact(off_x, off_v, np.zeros_like(off_x), halflife, dt)
    out_x = in_x + off_x
    out_v = in_v + off_v
    return out_x, out_v, off_x, off_v

def inertialize_transition_quat(off_x, off_v, src_x, src_v, dst_x, dst_v):
    off_x = quat_abs(quat_mul(quat_mul(off_x, src_x), quat_inv(dst_x)))
    off_v = (off_v + src_v) - dst_v
    return off_x, off_v

def inertialize_update_quat(in_x, in_v, off_x, off_v, halflife, dt):
    goal = np.zeros_like(off_x)
    goal[..., 0] = 1.0 # identity
    off_x, off_v = simple_spring_damper_exact_quat(off_x, off_v, goal, halflife, dt)
    out_x = quat_mul(off_x, in_x)
    out_v = off_v + quat_mul_vec(off_x, in_v)
    return out_x, out_v, off_x, off_v

# ----------------- Database Builder -----------------
class MotionDatabase:
    def __init__(self):
        self.bone_positions = []
        self.bone_quats = []
        self.bone_velocities = []
        self.bone_angular_velocities = []
        
        self.features = []
        self.features_normalized = None
        self.features_mean = None
        self.features_std = None
        
        self.frame_count = 0
        self.kdtree = None
        
    def build_from_motions(self, motions):
        """
        Builds the motion matching database from a list of EvihAnimation Motion objects.
        Requires global bone positions and orientations.
        """
        all_features = []
        all_pos = []
        all_quat = []
        all_vel = []
        all_avel = []
        
        # We need bones to find feet and hips
        if len(motions) == 0: return
        motion = motions[0]
        
        # Try to find common joint indices
        hips_idx = 0  # Assuming Root is 0
        lfoot_idx = -1
        rfoot_idx = -1
        for i, name in enumerate(motion.Hierarchy.BoneNames):
            if "LeftFoot" in name or "LFoot" in name: lfoot_idx = i
            if "RightFoot" in name or "RFoot" in name: rfoot_idx = i
        
        # If still -1, use arbitrary indices for testing if standard names not found
        if lfoot_idx == -1: lfoot_idx = motion.NumJoints - 2
        if rfoot_idx == -1: rfoot_idx = motion.NumJoints - 1
            
        print(f"Using Hips: {hips_idx}, LFoot: {lfoot_idx}, RFoot: {rfoot_idx}")

        for m_idx, motion in enumerate(motions):
            print(f"Processing motion {m_idx} ({motion.Name})...")
            
            frames = motion.Frames # [F, J, 4, 4] local matrices
            F, J = frames.shape[0], frames.shape[1]
            parents = motion.Hierarchy.ParentIndices
            
            # Extract local pos and quat
            local_pos = frames[:, :, :3, 3]
            
            rot_matrices = frames[:, :, :3, :3]
            r11, r12, r13 = rot_matrices[..., 0, 0], rot_matrices[..., 0, 1], rot_matrices[..., 0, 2]
            r21, r22, r23 = rot_matrices[..., 1, 0], rot_matrices[..., 1, 1], rot_matrices[..., 1, 2]
            r31, r32, r33 = rot_matrices[..., 2, 0], rot_matrices[..., 2, 1], rot_matrices[..., 2, 2]
            
            qw = 0.5 * np.sqrt(np.maximum(1 + r11 + r22 + r33, 0))
            qx = 0.5 * np.sqrt(np.maximum(1 + r11 - r22 - r33, 0)) * np.sign(r32 - r23)
            qy = 0.5 * np.sqrt(np.maximum(1 - r11 + r22 - r33, 0)) * np.sign(r13 - r31)
            qz = 0.5 * np.sqrt(np.maximum(1 - r11 - r22 + r33, 0)) * np.sign(r21 - r12)
            local_quats = np.stack((qw, qx, qy, qz), axis=-1)
            
            # Compute global poses via FK for feature extraction
            global_quats, global_pos = quat_fk(local_quats, local_pos, parents)
            
            # Save local poses to database!
            all_pos.append(local_pos)
            all_quat.append(local_quats)
            
            # Feature extraction relies on global positions (to get root velocity and foot heights correctly)
            pos = global_pos
            quats = global_quats
            root_x = savgol_filter(pos[:, 0, 0], 60, 4)
            root_y = savgol_filter(pos[:, 0, 1], 60, 4)
            root_z = savgol_filter(pos[:, 0, 2], 60, 4)
            root_p = np.stack([root_x, root_y, root_z], axis=1)
            
            # Realign data to filtered root trajectory
            # For simplicity, we just use the raw global positions for velocity and features
            # (In production, the trajectory should be completely relative to the root frame)
            
            # Compute velocities via central difference
            velocities = np.empty_like(pos)
            velocities[1:-1] = 0.5 * (pos[2:] - pos[:-2]) * 30.0
            velocities[0] = velocities[1] - (velocities[2] - velocities[1])
            velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
            
            angular_velocities = np.zeros_like(pos)
            angular_velocities[1:-1] = 0.5 * quat_to_scaled_angle_axis(quat_abs(quat_mul(quats[2:], quat_inv(quats[:-2])))) * 30.0
            angular_velocities[0] = angular_velocities[1] - (angular_velocities[2] - angular_velocities[1])
            angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])
            
            all_pos.append(pos)
            all_quat.append(quats)
            all_vel.append(velocities)
            all_avel.append(angular_velocities)
            
            # Build features (33-dimensional)
            features = np.zeros([F, 33], np.float32)
            for f in range(F):
                f_10 = min(f + 10, F - 1)
                f_20 = min(f + 20, F - 1)
                f_30 = min(f + 30, F - 1)
                
                # We need all features in the LOCAL space of the character at frame f
                # Root transform at f
                f_pos = root_p[f]
                f_q = quats[f, 0]
                
                # Hips velocity in local space
                hips_v_local = quat_mul_vec(quat_inv(f_q), velocities[f, hips_idx])
                features[f, 0:3] = hips_v_local
                
                # Left/Right Foot Position relative to root
                features[f, 3:6] = quat_mul_vec(quat_inv(f_q), pos[f, lfoot_idx] - f_pos)
                features[f, 6:9] = quat_mul_vec(quat_inv(f_q), pos[f, rfoot_idx] - f_pos)
                
                # Left/Right Foot Velocity in local space
                features[f, 9:12] = quat_mul_vec(quat_inv(f_q), velocities[f, lfoot_idx])
                features[f, 12:15] = quat_mul_vec(quat_inv(f_q), velocities[f, rfoot_idx])
                
                # Future trajectories (Pos + Dir)
                features[f, 15:18] = quat_mul_vec(quat_inv(f_q), root_p[f_10] - f_pos)
                features[f, 18:21] = quat_mul_vec(quat_inv(f_q), root_p[f_20] - f_pos)
                features[f, 21:24] = quat_mul_vec(quat_inv(f_q), root_p[f_30] - f_pos)
                
                # Future directions (forward vector [0,0,1] transformed by future root rot, then back to local)
                dir_10 = quat_mul_vec(quats[f_10, 0], np.array([0,0,1], dtype=np.float32))
                dir_20 = quat_mul_vec(quats[f_20, 0], np.array([0,0,1], dtype=np.float32))
                dir_30 = quat_mul_vec(quats[f_30, 0], np.array([0,0,1], dtype=np.float32))
                
                features[f, 24:27] = quat_mul_vec(quat_inv(f_q), dir_10)
                features[f, 27:30] = quat_mul_vec(quat_inv(f_q), dir_20)
                features[f, 30:33] = quat_mul_vec(quat_inv(f_q), dir_30)
                
            all_features.append(features)
            
        self.bone_positions = np.concatenate(all_pos)
        self.bone_quats = np.concatenate(all_quat)
        self.bone_velocities = np.concatenate(all_vel)
        self.bone_angular_velocities = np.concatenate(all_avel)
        self.features = np.concatenate(all_features)
        
        self.frame_count = self.features.shape[0]
        
        # Feature Normalization and Weighting
        self.features_mean = self.features.mean(axis=0)
        self.features_std = self.features.std(axis=0) + 1e-8
        
        feature_weights = np.array([
            1, # Hips velocity
            .75, # Left foot position
            .75, # Right foot position
            1, # Left foot velocity
            1, # right foot velocity
            1 * .99**10, # future trajectory position 10
            1 * .99**20, # future trajectory position 20
            1 * .99**30, # future trajectory position 30
            1.5 * .99**10, # future trajectory direction 10
            1.5 * .99**20, # future trajectory direction 20
            1.5 * .99**30, # future trajectory direction 30
        ]).repeat(3)
        
        self.features_std /= feature_weights
        self.features_normalized = (self.features - self.features_mean) / self.features_std
        
        print(f"Database built. Total frames: {self.frame_count}")
        print("Building KD-Tree for fast search...")
        self.kdtree = scipy.spatial.cKDTree(self.features_normalized)
        print("KD-Tree built successfully.")

    def query(self, query_feature_normalized):
        # Return closest frame index
        dist, idx = self.kdtree.query(query_feature_normalized, k=1)
        return idx


class Player:
    def __init__(self, db: MotionDatabase):
        self.db = db
        self.frame = 0
        self.bone_count = db.bone_positions.shape[1]
        
        self.q = db.bone_quats[0].copy()
        self.p = db.bone_positions[0].copy()
        self.q[0] = np.array([1,0,0,0], dtype=np.float32)
        self.p[0] = np.zeros(3, dtype=np.float32)

        self.off_x = np.zeros([self.bone_count-1, 3], dtype=np.float32)
        self.off_v = np.zeros([self.bone_count-1, 3], dtype=np.float32)
        self.off_qx = np.zeros([self.bone_count-1, 4], dtype=np.float32)
        self.off_qx[:, 0] = 1
        self.off_qv = np.zeros([self.bone_count-1, 3], dtype=np.float32)
        
        self.accum_root_q = np.array([1,0,0,0], dtype=np.float32)
        self.accum_root_p = np.zeros(3, dtype=np.float32)

    def set_next_frame(self, target_frame, inertialize=True):
        if target_frame >= self.db.frame_count: target_frame = 0
            
        q = self.db.bone_quats[target_frame].copy()
        p = self.db.bone_positions[target_frame].copy()
        
        # Determine displacement from previous frame
        prv_frame = max(0, target_frame - 1)
        
        # Calculate local root displacement delta between prv and target frame
        prv_q = self.db.bone_quats[prv_frame, 0]
        prv_p = self.db.bone_positions[prv_frame, 0]
        rq, rp = qp_mul(qp_inv(prv_q, prv_p), (q[0], p[0]))
        
        # Add delta to our current active root
        q[0], p[0] = qp_mul((self.q[0], self.p[0]), (rq, rp))
        
        # Apply inertialize blending
        if inertialize and target_frame != self.frame + 1:
            self.off_x, self.off_v = inertialize_transition_vec3(
                self.off_x, self.off_v, 
                self.db.bone_positions[self.frame, 1:], self.db.bone_velocities[self.frame, 1:], 
                self.db.bone_positions[target_frame, 1:], self.db.bone_velocities[target_frame, 1:])
            self.off_qx, self.off_qv = inertialize_transition_quat(
                self.off_qx, self.off_qv, 
                self.db.bone_quats[self.frame, 1:], self.db.bone_angular_velocities[self.frame, 1:], 
                self.db.bone_quats[target_frame, 1:], self.db.bone_angular_velocities[target_frame, 1:])

        if inertialize:
            p[1:], _, self.off_x, self.off_v = inertialize_update_vec3(
                self.db.bone_positions[target_frame, 1:], self.db.bone_velocities[target_frame, 1:], 
                self.off_x, self.off_v, .09, 1.0/30.)
            q[1:], _, self.off_qx, self.off_qv = inertialize_update_quat(
                self.db.bone_quats[target_frame, 1:], self.db.bone_angular_velocities[target_frame, 1:], 
                self.off_qx, self.off_qv, .09, 1.0/30.)

        self.frame = target_frame
        self.q = q
        self.p = p
