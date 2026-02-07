"""
MediaPipe → Genesis 9 Full-Body Animation v6.1
===============================================
BONE-LENGTH CONSTRAINED 3D + DIRECT CAMERA→G9 MAPPING

Key insights:
  1. Human bones have fixed lengths → depth = sqrt(L² - l_2d²)
  2. Each bone is defined by its 2 MP landmark endpoints
  3. Fixed camera→G9 mapping eliminates torso frame errors
  4. Parent bones left at identity → no double-rotation bug
  5. Limb directions naturally encode body rotation from the landmarks

Camera → G9 mapping (person roughly faces camera):
  G9_X = +cam_dx   (image right = person's left)
  G9_Y = -cam_dz   (into screen = person's backward)
  G9_Z = -cam_dy   (image down = world down)

Depth sign: MP Z used as directional hint (not magnitude).

Face: proven relative-metric approach from v4/v5.
Run in Blender — paste Part 1 then Part 2.
"""

import bpy, json, math, os
from mathutils import Vector, Euler, Quaternion, Matrix

# ============================================================================
# CONFIG
# ============================================================================

LANDMARKS_FILE = r"D:\blend_output\landmarks_output.json"
ARMATURE_NAME  = "Genesis9"
SOURCE_FPS     = 30
START_FRAME    = 1
SMOOTHING_WINDOW = 3
MASTER_SCALE   = 1.0
DEPTH_WEIGHT   = 0.7   # 0 = pure 2D, 1 = full depth

BODY_SENS = {
    "l_upperarm": 1.0, "r_upperarm": 1.0,
    "l_forearm": 1.0,  "r_forearm": 1.0,
    "l_hand": 0.6,     "r_hand": 0.6,
    "l_thigh": 0.8,    "r_thigh": 0.8,
    "l_shin": 1.0,     "r_shin": 1.0,
    "l_foot": 0.5,     "r_foot": 0.5,
    "fingers": 0.8,
}

FACE_SENS = {
    "head_pitch": 0.8, "head_yaw": 0.9, "head_roll": 0.7,
    "neck_share": 0.35, "jaw": 1.2, "eye_gaze": 1.0,
    "eyelid_upper": 1.0, "eyelid_lower": 0.5,
    "brow_inner": 1.2, "brow_outer": 1.0, "brow_center": 0.9,
    "lip_upper": 1.0, "lip_lower": 1.0,
    "lip_corner": 1.2, "lip_middle": 0.8,
    "cheek": 0.7, "cheek_lower": 0.6,
    "squint": 0.6, "nostril": 0.7,
    "infraorbital": 0.4, "chin": 0.6,
}

# ============================================================================
# LANDMARK INDICES
# ============================================================================

LM = {
    "lip_top": 13, "lip_bottom": 14,
    "mouth_left": 61, "mouth_right": 291,
    "lip_upper_left": 40, "lip_upper_right": 270,
    "lip_lower_left": 88, "lip_lower_right": 318, "lip_lower_mid": 17,
    "l_eye_upper": 159, "l_eye_lower": 145,
    "l_eye_inner": 133, "l_eye_outer": 33, "l_eye_iris": 468,
    "r_eye_upper": 386, "r_eye_lower": 374,
    "r_eye_inner": 362, "r_eye_outer": 263, "r_eye_iris": 473,
    "l_brow_inner": 107, "l_brow_mid": 70, "l_brow_outer": 46,
    "r_brow_inner": 336, "r_brow_mid": 300, "r_brow_outer": 276,
    "nose_tip": 1, "l_nostril": 129, "r_nostril": 358,
    "l_cheek": 123, "r_cheek": 352,
    "l_cheek_lower": 215, "r_cheek_lower": 435,
    "chin": 152, "forehead": 10,
    "l_temple": 234, "r_temple": 454,
    "l_squint": 111, "r_squint": 340,
    "l_infraorbital": 116, "r_infraorbital": 345,
    "face_center": 4, "l_lip_corner": 61, "r_lip_corner": 291,
}

POSE = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky_pose": 17, "r_pinky_pose": 18,
    "l_index_pose": 19, "r_index_pose": 20,
    "l_thumb_pose": 21, "r_thumb_pose": 22,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_foot_index": 31, "r_foot_index": 32,
}

HAND_LM = {
    "wrist": 0,
    "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
    "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
    "mid_mcp": 9, "mid_pip": 10, "mid_dip": 11, "mid_tip": 12,
    "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
    "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
}

# ============================================================================
# HELPERS
# ============================================================================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def remap(v, i0, i1, o0, o1):
    if abs(i1 - i0) < 1e-10: return (o0 + o1) * 0.5
    return o0 + clamp((v - i0) / (i1 - i0), 0, 1) * (o1 - o0)

def rad(d): return d * math.pi / 180.0

def mp_vec(lm_list, idx):
    lm = lm_list[idx]
    return Vector((lm['x'], lm['y'], lm.get('z', 0.0)))

def smooth_arr(a, w):
    if w <= 1 or len(a) < 2: return a[:]
    hw = w // 2
    return [sum(a[max(0, i-hw):min(len(a), i+hw+1)]) /
            len(a[max(0, i-hw):min(len(a), i+hw+1)]) for i in range(len(a))]

def dist3d(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 +
                     (a.get('z', 0)-b.get('z', 0))**2)


# ============================================================================
# BONE LENGTH CALIBRATION
# ============================================================================

MP_BONE_SEGMENTS = {
    "l_upperarm": (11, 13), "r_upperarm": (12, 14),
    "l_forearm":  (13, 15), "r_forearm":  (14, 16),
    "l_hand":     (15, 19), "r_hand":     (16, 20),
    "l_thigh":    (23, 25), "r_thigh":    (24, 26),
    "l_shin":     (25, 27), "r_shin":     (26, 28),
    "l_foot":     (27, 31), "r_foot":     (28, 32),
}


def calibrate_bone_lengths(frames):
    """Scan all frames. Max observed 2D length ≈ true 3D length."""
    nf = len(frames)
    bl = {}
    for seg, (i1, i2) in MP_BONE_SEGMENTS.items():
        mx = 0.0
        for fi in range(nf):
            pl = frames[fi]['pose_landmarks']
            dx = pl[i2]['x'] - pl[i1]['x']
            dy = pl[i2]['y'] - pl[i1]['y']
            mx = max(mx, math.sqrt(dx*dx + dy*dy))
        bl[seg] = mx

    print(f"[Calibration] Bone lengths from {nf} frames:")
    for k in sorted(bl):
        print(f"  {k:16s}: {bl[k]:.5f}")
    return bl


# ============================================================================
# BODY DRIVER — Direct cam→G9 mapping, no torso frame
# ============================================================================

class BodyDriver:
    """
    Drives limb bones via:
      1. Bone endpoint direction from MP landmarks (2D + depth from bone length)
      2. Fixed camera→G9 mapping: G9 = (cam_dx, -cam_dz, -cam_dy)
      3. inv(matrix_local) to convert world→bone-local
      4. rotation_difference((0,1,0), local_dir) → quaternion
    
    Parent bones (spine, hip, pelvis) are NOT rotated.
    This means pose_bone.rotation_quaternion IS effectively world-space,
    eliminating the double-rotation bug.
    """

    # G9 bone → (MP parent idx, MP child idx, calibration key)
    BONE_MAP = {
        "l_upperarm": (11, 13, "l_upperarm"),
        "l_forearm":  (13, 15, "l_forearm"),
        "l_hand":     (15, 19, "l_hand"),
        "r_upperarm": (12, 14, "r_upperarm"),
        "r_forearm":  (14, 16, "r_forearm"),
        "r_hand":     (16, 20, "r_hand"),
        "l_thigh":    (23, 25, "l_thigh"),
        "l_shin":     (25, 27, "l_shin"),
        "l_foot":     (27, 31, "l_foot"),
        "r_thigh":    (24, 26, "r_thigh"),
        "r_shin":     (26, 28, "r_shin"),
        "r_foot":     (28, 32, "r_foot"),
    }

    def __init__(self, armature, bone_lengths):
        self.arm = armature
        self.pb = armature.pose.bones
        self.bl = bone_lengths

        # Pre-compute inverse rest matrix for each bone
        self.inv_rest = {}
        for bn in self.BONE_MAP:
            bone = armature.data.bones.get(bn)
            if bone:
                self.inv_rest[bn] = bone.matrix_local.to_3x3().inverted()

        # Temporal depth signs
        self.prev_depth = {bn: 1.0 for bn in self.BONE_MAP}

        print(f"[BodyDriver] {len(self.inv_rest)} bones ready")

    def _cam_to_g9(self, cam_dx, cam_dy, cam_dz):
        """
        Fixed camera-space to G9-world-space mapping.
        Camera: X=right, Y=down, Z=into_screen
        G9:     X=person_left, Y=forward, Z=up
        """
        return Vector((cam_dx, -cam_dz, -cam_dy))

    def apply_frame(self, pose_lm, frame):
        if pose_lm is None:
            return

        for bn, (pidx, cidx, bl_key) in self.BONE_MAP.items():
            if bn not in self.inv_rest:
                continue

            # 2D displacement in camera space
            cam_dx = pose_lm[cidx]['x'] - pose_lm[pidx]['x']
            cam_dy = pose_lm[cidx]['y'] - pose_lm[pidx]['y']
            l2d = math.sqrt(cam_dx * cam_dx + cam_dy * cam_dy)

            # Depth from bone length constraint
            L3d = self.bl.get(bl_key, l2d)
            l2d_c = min(l2d, L3d)
            depth = math.sqrt(max(0, L3d * L3d - l2d_c * l2d_c))

            # Depth sign from MP Z hint + temporal smoothing
            pz = pose_lm[pidx].get('z', 0)
            cz = pose_lm[cidx].get('z', 0)
            mp_sign = 1.0 if cz < pz else -1.0  # MP: negative Z = closer = forward
            prev_sign = self.prev_depth[bn]
            sign = 1.0 if (mp_sign * 0.7 + prev_sign * 0.3) > 0 else -1.0
            self.prev_depth[bn] = sign

            cam_dz = depth * sign * DEPTH_WEIGHT

            # Camera → G9 world direction
            g9_dir = self._cam_to_g9(cam_dx, cam_dy, cam_dz)
            if g9_dir.length < 1e-8:
                continue
            g9_dir.normalize()

            # World → bone-local
            obs_local = (self.inv_rest[bn] @ g9_dir).normalized()

            # Rotation from rest (0,1,0) to observed
            quat = Vector((0, 1, 0)).rotation_difference(obs_local)

            # Scale and apply
            s = BODY_SENS.get(bn, 1.0) * MASTER_SCALE
            scaled = Quaternion().slerp(quat, clamp(s, 0, 2))

            pb = self.pb[bn]
            if pb.rotation_mode == 'QUATERNION':
                pb.rotation_quaternion = scaled
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            else:
                pb.rotation_euler = scaled.to_euler(pb.rotation_mode)
                pb.keyframe_insert(data_path="rotation_euler", frame=frame)

    def apply_fingers(self, hand_lm, side, frame):
        """Finger curl from hand landmarks."""
        if hand_lm is None:
            return

        wrist = mp_vec(hand_lm, 0)
        mid_mcp = mp_vec(hand_lm, 9)
        index_mcp = mp_vec(hand_lm, 5)
        pinky_mcp = mp_vec(hand_lm, 17)

        hand_fwd = (mid_mcp - wrist).normalized()
        hand_lat = (index_mcp - pinky_mcp).normalized()
        hand_norm = hand_fwd.cross(hand_lat).normalized()
        hand_lat = hand_norm.cross(hand_fwd).normalized()

        prefix = 'l' if side == 'left' else 'r'
        s = BODY_SENS.get("fingers", 0.8) * MASTER_SCALE

        chains = {
            'thumb': [(1,2),(2,3),(3,4)],
            'index': [(5,6),(6,7),(7,8)],
            'mid':   [(9,10),(10,11),(11,12)],
            'ring':  [(13,14),(14,15),(15,16)],
            'pinky': [(17,18),(18,19),(19,20)],
        }

        for fname, joints in chains.items():
            for ji, (jf, jt) in enumerate(joints):
                bn = f"{prefix}_{fname}{ji+1}"
                if bn not in self.pb:
                    continue

                seg = (mp_vec(hand_lm, jt) - mp_vec(hand_lm, jf)).normalized()
                if ji == 0:
                    prev = hand_fwd
                else:
                    pf, pt_ = joints[ji - 1]
                    prev = (mp_vec(hand_lm, pt_) - mp_vec(hand_lm, pf)).normalized()

                curl = prev.angle(seg, 0)
                pb = self.pb[bn]
                if pb.rotation_mode == 'QUATERNION':
                    pb.rotation_quaternion = Quaternion(Vector((1,0,0)), curl * s)
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                else:
                    pb.rotation_euler = Euler((curl * s, 0, 0), pb.rotation_mode)
                    pb.keyframe_insert(data_path="rotation_euler", frame=frame)


# ============================================================================
# FACE (proven from v4/v5 — unchanged)
# ============================================================================

class Calibration:
    KEYS = [
        "mouth_open", "mouth_width",
        "l_eye_open", "r_eye_open",
        "l_brow_inner_h", "l_brow_outer_h", "l_brow_mid_h",
        "r_brow_inner_h", "r_brow_outer_h", "r_brow_mid_h",
        "l_cheek_h", "r_cheek_h", "l_cheek_lower_h", "r_cheek_lower_h",
        "l_squint_h", "r_squint_h",
        "l_nostril_spread", "r_nostril_spread",
        "lip_upper_l_h", "lip_upper_r_h",
        "lip_lower_l_h", "lip_lower_r_h", "lip_lower_mid_h",
        "l_lip_corner_h", "r_lip_corner_h", "chin_h",
    ]

    def __init__(self, frames, n=8):
        ff = [f for f in frames if 'face_landmarks' in f][:n]
        if not ff: raise RuntimeError("No face frames!")
        acc = {k: [] for k in self.KEYS}
        fh_list = []
        for fr in ff:
            fl = fr['face_landmarks']
            m = self._measure(fl)
            for k in self.KEYS: acc[k].append(m[k])
            fh_list.append(dist3d(fl[LM["forehead"]], fl[LM["chin"]]))
        self.neutral = {k: sum(v)/len(v) for k, v in acc.items()}
        self.face_h = sum(fh_list) / len(fh_list)
        print(f"[Calibration] {len(ff)} face frames, face_h={self.face_h:.5f}")

    def _measure(self, fl):
        return {
            "mouth_open": dist3d(fl[LM["lip_top"]], fl[LM["lip_bottom"]]),
            "mouth_width": dist3d(fl[LM["mouth_left"]], fl[LM["mouth_right"]]),
            "l_eye_open": dist3d(fl[LM["l_eye_upper"]], fl[LM["l_eye_lower"]]),
            "r_eye_open": dist3d(fl[LM["r_eye_upper"]], fl[LM["r_eye_lower"]]),
            "l_brow_inner_h": fl[LM["l_brow_inner"]]['y'] - fl[LM["l_eye_upper"]]['y'],
            "l_brow_outer_h": fl[LM["l_brow_outer"]]['y'] - fl[LM["l_eye_outer"]]['y'],
            "l_brow_mid_h":   fl[LM["l_brow_mid"]]['y']   - fl[LM["l_eye_upper"]]['y'],
            "r_brow_inner_h": fl[LM["r_brow_inner"]]['y'] - fl[LM["r_eye_upper"]]['y'],
            "r_brow_outer_h": fl[LM["r_brow_outer"]]['y'] - fl[LM["r_eye_outer"]]['y'],
            "r_brow_mid_h":   fl[LM["r_brow_mid"]]['y']   - fl[LM["r_eye_upper"]]['y'],
            "l_cheek_h": dist3d(fl[LM["l_cheek"]], fl[LM["l_eye_lower"]]),
            "r_cheek_h": dist3d(fl[LM["r_cheek"]], fl[LM["r_eye_lower"]]),
            "l_cheek_lower_h": fl[LM["l_cheek_lower"]]['y'] - fl[LM["l_cheek"]]['y'],
            "r_cheek_lower_h": fl[LM["r_cheek_lower"]]['y'] - fl[LM["r_cheek"]]['y'],
            "l_squint_h": dist3d(fl[LM["l_squint"]], fl[LM["l_eye_lower"]]),
            "r_squint_h": dist3d(fl[LM["r_squint"]], fl[LM["r_eye_lower"]]),
            "l_nostril_spread": fl[LM["l_nostril"]]['x'] - fl[LM["nose_tip"]]['x'],
            "r_nostril_spread": fl[LM["nose_tip"]]['x']  - fl[LM["r_nostril"]]['x'],
            "lip_upper_l_h": dist3d(fl[LM["lip_upper_left"]], fl[LM["lip_top"]]),
            "lip_upper_r_h": dist3d(fl[LM["lip_upper_right"]], fl[LM["lip_top"]]),
            "lip_lower_l_h": dist3d(fl[LM["lip_lower_left"]], fl[LM["lip_bottom"]]),
            "lip_lower_r_h": dist3d(fl[LM["lip_lower_right"]], fl[LM["lip_bottom"]]),
            "lip_lower_mid_h": dist3d(fl[LM["lip_lower_mid"]], fl[LM["lip_bottom"]]),
            "l_lip_corner_h": fl[LM["l_lip_corner"]]['y'] - fl[LM["lip_top"]]['y'],
            "r_lip_corner_h": fl[LM["r_lip_corner"]]['y'] - fl[LM["lip_top"]]['y'],
            "chin_h": fl[LM["chin"]]['y'] - fl[LM["lip_bottom"]]['y'],
        }

    def delta(self, fl):
        m = self._measure(fl)
        return {k: (m[k] - self.neutral[k]) / max(self.face_h, 0.001) for k in self.KEYS}


def head_pose(fl):
    nose  = mp_vec(fl, LM["face_center"])
    top   = mp_vec(fl, LM["forehead"])
    bot   = mp_vec(fl, LM["chin"])
    left  = mp_vec(fl, LM["l_temple"])
    right = mp_vec(fl, LM["r_temple"])
    mx = (left.x + right.x) * 0.5
    my = (top.y + bot.y) * 0.5
    fw = (left - right).length
    fh = (top - bot).length
    yaw   = math.asin(clamp((nose.x - mx) / max(fw*0.5, 0.001), -1, 1)) if fw > 0.001 else 0
    pitch = math.asin(clamp((nose.y - my) / max(fh*0.35, 0.001), -1, 1)) if fh > 0.001 else 0
    roll  = math.atan2(right.y - left.y, right.x - left.x)
    return pitch, yaw, roll


def eye_gaze(fl, side):
    if side == 'left':
        iris, inner, outer = mp_vec(fl, LM["l_eye_iris"]), mp_vec(fl, LM["l_eye_inner"]), mp_vec(fl, LM["l_eye_outer"])
        upper, lower = mp_vec(fl, LM["l_eye_upper"]), mp_vec(fl, LM["l_eye_lower"])
    else:
        iris, inner, outer = mp_vec(fl, LM["r_eye_iris"]), mp_vec(fl, LM["r_eye_inner"]), mp_vec(fl, LM["r_eye_outer"])
        upper, lower = mp_vec(fl, LM["r_eye_upper"]), mp_vec(fl, LM["r_eye_lower"])
    center = (inner + outer) * 0.5
    ew = (inner - outer).length; eh = (upper - lower).length
    h = (iris.x - center.x) / max(ew * 0.5, 0.0001)
    v = (iris.y - center.y) / max(eh * 0.5, 0.0001)
    return clamp(h, -1, 1), clamp(v, -1, 1)


class FaceDriver:
    def __init__(self, armature):
        self.pb = armature.pose.bones

    def _key(self, bn, x=0, y=0, z=0, frame=1):
        if bn not in self.pb: return
        pb = self.pb[bn]
        if pb.rotation_mode == 'QUATERNION':
            pb.rotation_quaternion = Euler((x, y, z), 'YZX').to_quaternion()
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        else:
            if pb.rotation_mode not in ('YZX','XYZ','ZXY','XZY','YXZ','ZYX'):
                pb.rotation_mode = 'YZX'
            pb.rotation_euler = Euler((x, y, z), pb.rotation_mode)
            pb.keyframe_insert(data_path="rotation_euler", frame=frame)

    def apply_frame(self, hp, dd, gl, gr, frame):
        s = FACE_SENS; ms = MASTER_SCALE
        nf = s["neck_share"]; hf = 1.0 - nf

        self._key('head',  hp[0]*hf*s["head_pitch"]*ms, -hp[1]*hf*s["head_yaw"]*ms, -hp[2]*hf*s["head_roll"]*ms, frame)
        self._key('neck2', hp[0]*nf*0.6*s["head_pitch"]*ms, -hp[1]*nf*0.6*s["head_yaw"]*ms, -hp[2]*nf*0.6*s["head_roll"]*ms, frame)
        self._key('neck1', hp[0]*nf*0.4*s["head_pitch"]*ms, -hp[1]*nf*0.4*s["head_yaw"]*ms, -hp[2]*nf*0.4*s["head_roll"]*ms, frame)
        self._key('upperteeth', 0, 0, 0, frame)
        self._key('upperfacerig', 0, 0, 0, frame)

        angle = remap(dd.get("mouth_open",0), 0, 0.025, 0, rad(25)) * s["jaw"] * ms
        self._key('lowerjaw', -angle, 0, 0, frame)
        self._key('lowerfacerig', 0, 0, 0, frame)
        self._key('lowerteeth', 0, 0, 0, frame)

        mh=rad(20); mv=rad(12); sg=s["eye_gaze"]*ms
        self._key('l_eye', -gl[1]*mv*sg, -gl[0]*mh*sg, 0, frame)
        self._key('r_eye', -gr[1]*mv*sg, -gr[0]*mh*sg, 0, frame)

        su=s["eyelid_upper"]*ms; sl=s["eyelid_lower"]*ms; bm=rad(35)
        ld=dd.get("l_eye_open",0); rd=dd.get("r_eye_open",0)
        self._key('l_eyelidupper', remap(ld,-0.025,0.01,-bm,bm*0.2)*su, 0, 0, frame)
        self._key('r_eyelidupper', remap(rd,-0.025,0.01,-bm,bm*0.2)*su, 0, 0, frame)
        self._key('l_eyelidlower', remap(ld,-0.03,0.005,rad(10),rad(-5))*sl, 0, 0, frame)
        self._key('r_eyelidlower', remap(rd,-0.03,0.005,rad(10),rad(-5))*sl, 0, 0, frame)

        si=s["brow_inner"]*ms; so=s["brow_outer"]*ms; sc=s["brow_center"]*ms; bmx=rad(18); bs=0.015
        li=-dd.get("l_brow_inner_h",0); ri=-dd.get("r_brow_inner_h",0)
        lo=-dd.get("l_brow_outer_h",0); ro=-dd.get("r_brow_outer_h",0)
        self._key('l_browinner', clamp(li/bs,-1,1)*bmx*si, 0, 0, frame)
        self._key('r_browinner', clamp(ri/bs,-1,1)*bmx*si, 0, 0, frame)
        self._key('l_browouter', clamp(lo/bs,-1,1)*bmx*0.7*so, 0, 0, frame)
        self._key('r_browouter', clamp(ro/bs,-1,1)*bmx*0.7*so, 0, 0, frame)
        self._key('centerbrow',  clamp((li+ri)*0.5/bs,-1,1)*bmx*0.6*sc, 0, 0, frame)

        su2=s["lip_upper"]*ms; sl2=s["lip_lower"]*ms; slc=s["lip_corner"]*ms; sm=s["lip_middle"]*ms; mx2=rad(10)
        mw = clamp(dd.get("mouth_width",0)/0.015, -1, 1)
        ul=dd.get("lip_upper_l_h",0); ur=dd.get("lip_upper_r_h",0)
        self._key('l_lipupper', (clamp(ul/0.01,-1,1)*0.5+mw*0.5)*mx2*su2, 0, 0, frame)
        self._key('r_lipupper', (clamp(ur/0.01,-1,1)*0.5+mw*0.5)*mx2*su2, 0, 0, frame)
        self._key('lipuppermiddle', clamp((ul+ur)*0.5/0.01,-1,1)*mx2*0.7*sm, 0, 0, frame)
        ll=dd.get("lip_lower_l_h",0); lr=dd.get("lip_lower_r_h",0); lm_=dd.get("lip_lower_mid_h",0)
        self._key('l_liplower', clamp(ll/0.01,-1,1)*mx2*0.6*sl2, 0, 0, frame)
        self._key('r_liplower', clamp(lr/0.01,-1,1)*mx2*0.6*sl2, 0, 0, frame)
        self._key('liplowermiddle', clamp(lm_/0.01,-1,1)*mx2*0.5*sm, 0, 0, frame)
        lcr=dd.get("l_lip_corner_h",0); rcr=dd.get("r_lip_corner_h",0)
        self._key('l_lipcorner', clamp(-lcr/0.012,-1,1)*mx2*1.2*slc, 0, 0, frame)
        self._key('r_lipcorner', clamp(-rcr/0.012,-1,1)*mx2*1.2*slc, 0, 0, frame)

        cs=s["cheek"]*ms; cls=s["cheek_lower"]*ms; cmx=rad(8)
        self._key('l_cheek', clamp(-dd.get("l_cheek_h",0)/0.012,-0.3,1)*cmx*cs, 0, 0, frame)
        self._key('r_cheek', clamp(-dd.get("r_cheek_h",0)/0.012,-0.3,1)*cmx*cs, 0, 0, frame)
        self._key('l_cheeklower', clamp(-dd.get("l_cheek_lower_h",0)/0.01,-0.3,1)*rad(5)*cls, 0, 0, frame)
        self._key('r_cheeklower', clamp(-dd.get("r_cheek_lower_h",0)/0.01,-0.3,1)*rad(5)*cls, 0, 0, frame)

        sqs=s["squint"]*ms; sqx=rad(6)
        self._key('l_squint', clamp(-dd.get("l_squint_h",0)/0.01,-0.3,1)*sqx*sqs, 0, 0, frame)
        self._key('r_squint', clamp(-dd.get("r_squint_h",0)/0.01,-0.3,1)*sqx*sqs, 0, 0, frame)

        ns=s["nostril"]*ms; nmx=rad(5)
        self._key('l_nostril', 0, 0, clamp(dd.get("l_nostril_spread",0)/0.006,-0.5,1)*nmx*ns, frame)
        self._key('r_nostril', 0, 0, clamp(-dd.get("r_nostril_spread",0)/0.006,-1,0.5)*nmx*ns, frame)

        ios=s["infraorbital"]*ms; iox=rad(4)
        self._key('l_infraorbital', clamp(-dd.get("l_cheek_h",0)/0.015,-0.2,1)*iox*ios, 0, 0, frame)
        self._key('r_infraorbital', clamp(-dd.get("r_cheek_h",0)/0.015,-0.2,1)*iox*ios, 0, 0, frame)

        chs=s["chin"]*ms
        self._key('chin', clamp(dd.get("chin_h",0)/0.01,-1,1)*rad(6)*chs, 0, 0, frame)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MediaPipe → Genesis 9 Animation v6.1")
    print("Bone-length constrained 3D, direct cam→G9, no double-rotation")
    print("=" * 70)

    print(f"\nLoading: {LANDMARKS_FILE}")
    with open(LANDMARKS_FILE, 'r') as f:
        frames = json.load(f)
    nf = len(frames)
    fi_face = [i for i, f in enumerate(frames) if 'face_landmarks' in f]
    fi_lh = [i for i, f in enumerate(frames) if 'left_hand_landmarks' in f]
    fi_rh = [i for i, f in enumerate(frames) if 'right_hand_landmarks' in f]
    print(f"  Frames: {nf}, Face: {len(fi_face)}, L-hand: {len(fi_lh)}, R-hand: {len(fi_rh)}")

    bone_lengths = calibrate_bone_lengths(frames)

    arm = bpy.data.objects.get(ARMATURE_NAME)
    if arm is None:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and any(t in obj.name.lower() for t in ['genesis9', 'g9']):
                arm = obj; break
    if arm is None:
        raise RuntimeError("Armature not found!")
    print(f"  Armature: {arm.name}")

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')

    print("\n  Clearing animation...")
    if arm.animation_data:
        if arm.animation_data.nla_tracks:
            for t in list(arm.animation_data.nla_tracks):
                arm.animation_data.nla_tracks.remove(t)
        if arm.animation_data.action:
            old = arm.animation_data.action
            arm.animation_data.action = None
            if old.users == 0: bpy.data.actions.remove(old)
        arm.animation_data_clear()
    for a in list(bpy.data.actions):
        if "MediaPipe" in a.name and a.users == 0:
            bpy.data.actions.remove(a)
    for pb in arm.pose.bones:
        pb.location = (0,0,0)
        pb.rotation_euler = (0,0,0)
        pb.rotation_quaternion = (1,0,0,0)
        pb.scale = (1,1,1)

    arm.animation_data_create()
    action = bpy.data.actions.new(name="MediaPipe_Anim_v6.1")
    arm.animation_data.action = action

    cal = Calibration(frames)
    body_drv = BodyDriver(arm, bone_lengths)
    face_drv = FaceDriver(arm)

    # Pass 1: Extract + smooth face
    print("\n[Pass 1] Extracting face data...")
    raw_hp, raw_dd, raw_gl, raw_gr = [], [], [], []
    has_face = []
    for fi, fr in enumerate(frames):
        hf = 'face_landmarks' in fr
        has_face.append(hf)
        if hf:
            fl = fr['face_landmarks']
            raw_hp.append(head_pose(fl))
            raw_dd.append(cal.delta(fl))
            raw_gl.append(eye_gaze(fl, 'left'))
            raw_gr.append(eye_gaze(fl, 'right'))
        else:
            raw_hp.append(None); raw_dd.append(None)
            raw_gl.append(None); raw_gr.append(None)

    W = SMOOTHING_WINDOW
    if W > 1 and fi_face:
        print("[Pass 1.5] Smoothing face...")
        for axis in range(3):
            vals = [raw_hp[i][axis] for i in fi_face]
            sm = smooth_arr(vals, W)
            for idx, i in enumerate(fi_face):
                h = list(raw_hp[i]); h[axis] = sm[idx]; raw_hp[i] = tuple(h)
        for src in [raw_gl, raw_gr]:
            for axis in range(2):
                vals = [src[i][axis] for i in fi_face]
                sm = smooth_arr(vals, W)
                for idx, i in enumerate(fi_face):
                    g = list(src[i]); g[axis] = sm[idx]; src[i] = tuple(g)
        dk = list(raw_dd[fi_face[0]].keys())
        for k in dk:
            vals = [raw_dd[i][k] for i in fi_face]
            sm = smooth_arr(vals, W)
            for idx, i in enumerate(fi_face):
                raw_dd[i][k] = sm[idx]

    # Pass 2: Keyframe
    print(f"\n[Pass 2] Keyframing {nf} frames...")
    last_face = None

    for fi in range(nf):
        bf = START_FRAME + fi

        if has_face[fi]:
            hp, dd, gl, gr = raw_hp[fi], raw_dd[fi], raw_gl[fi], raw_gr[fi]
            last_face = (hp, dd, gl, gr)
        elif last_face:
            hp, dd, gl, gr = last_face
        else:
            hp = dd = gl = gr = None

        if hp is not None:
            face_drv.apply_frame(hp, dd, gl, gr, bf)

        body_drv.apply_frame(frames[fi].get('pose_landmarks'), bf)
        body_drv.apply_fingers(frames[fi].get('left_hand_landmarks'), 'left', bf)
        body_drv.apply_fingers(frames[fi].get('right_hand_landmarks'), 'right', bf)

        if fi % 50 == 0:
            print(f"  {fi}/{nf}...")

    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + nf - 1
    bpy.context.scene.frame_current = START_FRAME
    bpy.context.scene.render.fps = SOURCE_FPS

    fa = arm.animation_data.action
    if fa:
        try:
            for fc in fa.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.handle_left_type = 'AUTO_CLAMPED'
                    kp.handle_right_type = 'AUTO_CLAMPED'
            print(f"\n  Bezier on {len(fa.fcurves)} F-curves")
        except Exception as e:
            print(f"\n  Interpolation: {e}")

    bpy.context.view_layer.update()

    print(f"\n{'=' * 70}")
    print("DONE!")
    print(f"  Frames: {START_FRAME}–{START_FRAME + nf - 1} @ {SOURCE_FPS}fps")
    print(f"  Method: bone-length depth + direct cam→G9 + rotation_difference")
    print(f"  Body: 12 limb bones (no spine rotation — eliminates double-rotation bug)")
    print(f"  Face: {len(fi_face)} frames, L-hand: {len(fi_lh)}, R-hand: {len(fi_rh)}")
    print(f"  Spine/torso animation: to be added after limb verification")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
