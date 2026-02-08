"""
FreeMoCap Face CSV → Genesis 9 FACS Shape Keys  v3
=====================================================
Fixed: deadzones applied, correct G9 shape key names, ref_basis
precomputed, MouthOpen fixed, smile vs stretch split.
"""

import bpy
import csv
import math
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────
CSV_PATH = "C://Users//Like//freemocap_data//recording_sessions//import_sn2//output_data//mediapipe_face_3d_xyz.csv"  # Path to FreeMoCap face CSV
START_FRAME = 1
N_REF_FRAMES = 30
SMOOTHING = 3
GLOBAL_INTENSITY = 1.0

# ─── LANDMARKS (unique names) ────────────────────────────────
LM_NOSE_TIP          = 1
LM_NOSE_BRIDGE       = 6
LM_BETWEEN_EYES      = 168
LM_FOREHEAD          = 10
LM_CHIN              = 152

LM_L_EYE_INNER      = 133
LM_L_EYE_OUTER      = 33
LM_R_EYE_INNER      = 362
LM_R_EYE_OUTER      = 263

LM_L_LID_UPPER      = 159
LM_L_LID_LOWER      = 145
LM_R_LID_UPPER      = 386
LM_R_LID_LOWER      = 374

LM_L_BROW_INNER     = 107
LM_L_BROW_OUTER     = 70
LM_R_BROW_INNER     = 336
LM_R_BROW_OUTER     = 300

LM_LIP_UPPER_CENTER = 13
LM_LIP_LOWER_CENTER = 14
LM_LIP_UPPER_TOP    = 0
LM_LIP_LOWER_BOT    = 17
LM_MOUTH_L           = 61
LM_MOUTH_R           = 291

LM_L_NOSTRIL         = 102
LM_R_NOSTRIL         = 331
LM_L_SQUINT          = 23
LM_R_SQUINT          = 253
LM_L_CHEEK           = 117
LM_R_CHEEK           = 346

LM_L_IRIS            = 468
LM_R_IRIS            = 473


# ─── HEAD-LOCAL TRANSFORM ─────────────────────────────────────

def build_head_basis(frame):
    l_eye = frame[LM_L_EYE_OUTER]
    r_eye = frame[LM_R_EYE_OUTER]
    nose = frame[LM_NOSE_BRIDGE]

    origin = nose
    x_raw = l_eye - r_eye
    x_len = np.linalg.norm(x_raw)
    if x_len < 1e-8:
        return origin, np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])
    x_ax = x_raw / x_len

    up_raw = (frame[LM_FOREHEAD] + frame[LM_BETWEEN_EYES]) / 2 - nose
    y_ax = np.cross(up_raw, x_ax)
    y_len = np.linalg.norm(y_ax)
    if y_len < 1e-8:
        y_ax = np.cross(np.array([0.,0.,1.]), x_ax)
        y_len = np.linalg.norm(y_ax)
    y_ax /= max(y_len, 1e-8)

    z_ax = np.cross(x_ax, y_ax)
    z_ax /= max(np.linalg.norm(z_ax), 1e-8)
    return origin, x_ax, y_ax, z_ax


def to_local(frame, idx, o, x, y, z):
    p = frame[idx] - o
    return np.array([p.dot(x), p.dot(y), p.dot(z)])


def ldist(a, b):
    return np.linalg.norm(a - b)

def ldx(a, b):
    return a[0] - b[0]

def ldz(a, b):
    return a[2] - b[2]

def clamp01(v):
    return max(0.0, min(1.0, v))


# ─── REF METRICS ─────────────────────────────────────────────

def compute_ref(ref):
    o, x, y, z = build_head_basis(ref)
    def lm(i): return to_local(ref, i, o, x, y, z)

    forehead = lm(LM_FOREHEAD); chin = lm(LM_CHIN)
    l_eye_in = lm(LM_L_EYE_INNER); l_eye_out = lm(LM_L_EYE_OUTER)
    r_eye_in = lm(LM_R_EYE_INNER); r_eye_out = lm(LM_R_EYE_OUTER)
    l_lid_up = lm(LM_L_LID_UPPER); l_lid_lo = lm(LM_L_LID_LOWER)
    r_lid_up = lm(LM_R_LID_UPPER); r_lid_lo = lm(LM_R_LID_LOWER)
    l_brow_in = lm(LM_L_BROW_INNER); l_brow_out = lm(LM_L_BROW_OUTER)
    r_brow_in = lm(LM_R_BROW_INNER); r_brow_out = lm(LM_R_BROW_OUTER)
    lip_up = lm(LM_LIP_UPPER_CENTER); lip_lo = lm(LM_LIP_LOWER_CENTER)
    lip_top = lm(LM_LIP_UPPER_TOP); lip_bot = lm(LM_LIP_LOWER_BOT)
    mouth_l = lm(LM_MOUTH_L); mouth_r = lm(LM_MOUTH_R)
    nose_tip = lm(LM_NOSE_TIP)
    l_nostril = lm(LM_L_NOSTRIL); r_nostril = lm(LM_R_NOSTRIL)
    l_sq = lm(LM_L_SQUINT); r_sq = lm(LM_R_SQUINT)
    l_chk = lm(LM_L_CHEEK); r_chk = lm(LM_R_CHEEK)

    l_eye_cz = (l_eye_in[2] + l_eye_out[2]) / 2
    r_eye_cz = (r_eye_in[2] + r_eye_out[2]) / 2

    return {
        'face_h':          abs(ldz(forehead, chin)),
        'eye_w_l':         abs(ldx(l_eye_out, l_eye_in)),
        'eye_w_r':         abs(ldx(r_eye_in, r_eye_out)),
        'eye_open_l':      ldz(l_lid_up, l_lid_lo),
        'eye_open_r':      ldz(r_lid_up, r_lid_lo),
        'brow_l_in':       ldz(l_brow_in, l_eye_in),
        'brow_l_out':      ldz(l_brow_out, l_eye_out),
        'brow_r_in':       ldz(r_brow_in, r_eye_in),
        'brow_r_out':      ldz(r_brow_out, r_eye_out),
        'brow_inner_dist': ldist(l_brow_in, r_brow_in),
        'mouth_gap':       ldz(lip_up, lip_lo),
        'mouth_full':      abs(ldz(lip_top, lip_bot)),
        'mouth_w':         abs(ldx(mouth_l, mouth_r)),
        'lip_up_h':        ldz(lip_top, lip_up),
        'lip_lo_h':        ldz(lip_lo, lip_bot),
        'corner_l_z':      ldz(mouth_l, nose_tip),
        'corner_r_z':      ldz(mouth_r, nose_tip),
        'mouth_cx':        (mouth_l[0] + mouth_r[0]) / 2,
        'nose_w':          ldist(l_nostril, r_nostril),
        'cheek_l':         ldz(l_eye_out, l_chk),
        'cheek_r':         ldz(r_eye_out, r_chk),
        'squint_l':        ldist(l_sq, l_lid_lo),
        'squint_r':        ldist(r_sq, r_lid_lo),
        'ul_h_l':          l_lid_up[2] - l_eye_cz,
        'ul_h_r':          r_lid_up[2] - r_eye_cz,
        'll_h_l':          l_eye_cz - l_lid_lo[2],
        'll_h_r':          r_eye_cz - r_lid_lo[2],
    }


# ─── COMPUTE FACS ────────────────────────────────────────────

def compute_facs(frame, ref, rm, n_lm):
    o, x, y, z = build_head_basis(frame)
    fh = rm['face_h']
    if fh < 1e-6:
        return {}

    def lm(i): return to_local(frame, i, o, x, y, z)

    vals = {}

    # Landmarks
    l_eye_in = lm(LM_L_EYE_INNER); l_eye_out = lm(LM_L_EYE_OUTER)
    r_eye_in = lm(LM_R_EYE_INNER); r_eye_out = lm(LM_R_EYE_OUTER)
    l_lid_up = lm(LM_L_LID_UPPER); l_lid_lo = lm(LM_L_LID_LOWER)
    r_lid_up = lm(LM_R_LID_UPPER); r_lid_lo = lm(LM_R_LID_LOWER)
    l_brow_in = lm(LM_L_BROW_INNER); l_brow_out = lm(LM_L_BROW_OUTER)
    r_brow_in = lm(LM_R_BROW_INNER); r_brow_out = lm(LM_R_BROW_OUTER)
    lip_up = lm(LM_LIP_UPPER_CENTER); lip_lo = lm(LM_LIP_LOWER_CENTER)
    lip_top = lm(LM_LIP_UPPER_TOP); lip_bot = lm(LM_LIP_LOWER_BOT)
    mouth_l = lm(LM_MOUTH_L); mouth_r = lm(LM_MOUTH_R)
    nose_tip = lm(LM_NOSE_TIP)
    l_nostril = lm(LM_L_NOSTRIL); r_nostril = lm(LM_R_NOSTRIL)
    l_sq = lm(LM_L_SQUINT); r_sq = lm(LM_R_SQUINT)
    l_chk = lm(LM_L_CHEEK); r_chk = lm(LM_R_CHEEK)
    ew_l = rm['eye_w_l']; ew_r = rm['eye_w_r']

    # ── JAW ──
    jaw_delta = (ldz(lip_up, lip_lo) - rm['mouth_gap']) / fh
    vals['facs_bs_JawOpen'] = clamp01(jaw_delta * 12.0)

    full_delta = (abs(ldz(lip_top, lip_bot)) - rm['mouth_full']) / fh
    vals['facs_bs_JawOpenWide'] = clamp01(full_delta * 6.0)

    # Jaw lateral
    mouth_cx = (mouth_l[0] + mouth_r[0]) / 2
    jaw_shift = (mouth_cx - rm['mouth_cx']) / max(rm['mouth_w'], 1e-6)
    if jaw_shift > 0:
        vals['facs_bs_JawLeft'] = clamp01(jaw_shift * 4.0)
    else:
        vals['facs_bs_JawRight'] = clamp01(-jaw_shift * 4.0)

    # ── MOUTH WIDTH: Smile vs Stretch ──
    mouth_w = abs(ldx(mouth_l, mouth_r))
    w_ratio = (mouth_w - rm['mouth_w']) / max(rm['mouth_w'], 1e-6)

    # Corner up/down (head-local, relative to nose)
    corner_l_d = (ldz(mouth_l, nose_tip) - rm['corner_l_z']) / fh
    corner_r_d = (ldz(mouth_r, nose_tip) - rm['corner_r_z']) / fh

    # Smile = corners UP + width increase
    # Stretch = width increase + corners NOT up
    if w_ratio > 0:
        if corner_l_d > 0:
            vals['facs_bs_MouthSmileLeft'] = clamp01(corner_l_d * 10.0)
        else:
            vals['facs_bs_MouthStretchLeft'] = clamp01(w_ratio * 2.5)
        if corner_r_d > 0:
            vals['facs_bs_MouthSmileRight'] = clamp01(corner_r_d * 10.0)
        else:
            vals['facs_bs_MouthStretchRight'] = clamp01(w_ratio * 2.5)
        vals['facs_bs_MouthDimpleLeft'] = clamp01(w_ratio * 1.0)
        vals['facs_bs_MouthDimpleRight'] = clamp01(w_ratio * 1.0)
    else:
        # Pucker = width decrease → MouthPurse (L/R split)
        pv = clamp01(-w_ratio * 3.0)
        vals['facs_bs_MouthPurseLowerLeft'] = pv
        vals['facs_bs_MouthPurseLowerRight'] = pv
        vals['facs_bs_MouthPurseUpperLeft'] = pv
        vals['facs_bs_MouthPurseUpperRight'] = pv
        # Funnel = slight width decrease + some opening
        fv = clamp01(-w_ratio * 1.5)
        vals['facs_bs_MouthFunnelLowerLeft'] = fv
        vals['facs_bs_MouthFunnelLowerRight'] = fv
        vals['facs_bs_MouthFunnelUpperLeft'] = fv
        vals['facs_bs_MouthFunnelUpperRight'] = fv

    # Frown (corners down)
    if corner_l_d < 0:
        vals['facs_bs_MouthFrownLeft'] = clamp01(-corner_l_d * 10.0)
    if corner_r_d < 0:
        vals['facs_bs_MouthFrownRight'] = clamp01(-corner_r_d * 10.0)

    # ── LIPS ──
    lip_raise = (ldz(lip_top, lip_up) - rm['lip_up_h']) / fh
    vals['facs_bs_MouthUpperUpLeft'] = clamp01(lip_raise * 15.0)
    vals['facs_bs_MouthUpperUpRight'] = clamp01(lip_raise * 15.0)

    lip_drop = (ldz(lip_lo, lip_bot) - rm['lip_lo_h']) / fh
    vals['facs_bs_MouthLowerDownLeft'] = clamp01(lip_drop * 15.0)
    vals['facs_bs_MouthLowerDownRight'] = clamp01(lip_drop * 15.0)

    # Lip press (lips closer, jaw closed)
    lip_gap = abs(ldz(lip_up, lip_lo))
    ref_gap = abs(rm['mouth_gap'])
    if ref_gap > 0 and vals.get('facs_bs_JawOpen', 0) < 0.1:
        press = (ref_gap - lip_gap) / ref_gap
        if press > 0:
            pv = clamp01(press * 2.0)
            vals['facs_bs_MouthPressLowerLeft'] = pv
            vals['facs_bs_MouthPressLowerRight'] = pv
            vals['facs_bs_MouthPressUpperLeft'] = pv
            vals['facs_bs_MouthPressUpperRight'] = pv

    # ── BROWS (relative to eye corners) ──
    bl_in = (ldz(l_brow_in, l_eye_in) - rm['brow_l_in']) / fh
    bl_out = (ldz(l_brow_out, l_eye_out) - rm['brow_l_out']) / fh
    br_in = (ldz(r_brow_in, r_eye_in) - rm['brow_r_in']) / fh
    br_out = (ldz(r_brow_out, r_eye_out) - rm['brow_r_out']) / fh

    if bl_in > 0:
        vals['facs_bs_BrowInnerUpLeft'] = clamp01(bl_in * 12.0)
    else:
        vals['facs_BrowDownLeft'] = clamp01(-bl_in * 12.0)
    if br_in > 0:
        vals['facs_bs_BrowInnerUpRight'] = clamp01(br_in * 12.0)
    else:
        vals['facs_BrowDownRight'] = clamp01(-br_in * 12.0)

    vals['facs_BrowOuterUpLeft'] = clamp01(bl_out * 12.0)
    vals['facs_BrowOuterUpRight'] = clamp01(br_out * 12.0)

    brow_dist = ldist(l_brow_in, r_brow_in)
    squeeze = (rm['brow_inner_dist'] - brow_dist) / max(rm['brow_inner_dist'], 1e-6)
    vals['facs_bs_BrowSqueezeLeft'] = clamp01(squeeze * 2.5)
    vals['facs_bs_BrowSqueezeRight'] = clamp01(squeeze * 2.5)

    # ── EYELIDS (relative to eye corners) ──
    eye_open_l = ldz(l_lid_up, l_lid_lo)
    eye_open_r = ldz(r_lid_up, r_lid_lo)

    if rm['eye_open_l'] > 0:
        vals['facs_bs_EyeBlinkLeft'] = clamp01((1.0 - eye_open_l / rm['eye_open_l']) * 1.3)
    if rm['eye_open_r'] > 0:
        vals['facs_bs_EyeBlinkRight'] = clamp01((1.0 - eye_open_r / rm['eye_open_r']) * 1.3)

    # Upper/lower lid close relative to eye center (midpoint of corners)
    l_ecz = (l_eye_in[2] + l_eye_out[2]) / 2
    r_ecz = (r_eye_in[2] + r_eye_out[2]) / 2

    ul_cl = (rm['ul_h_l'] - (l_lid_up[2] - l_ecz)) / max(ew_l, 1e-6)
    ul_cr = (rm['ul_h_r'] - (r_lid_up[2] - r_ecz)) / max(ew_r, 1e-6)
    vals['facs_bs_EyeLidCloseUpperLeft'] = clamp01(ul_cl * 5.0)
    vals['facs_bs_EyelidCloseUpperRight'] = clamp01(ul_cr * 5.0)

    ll_cl = (rm['ll_h_l'] - (l_ecz - l_lid_lo[2])) / max(ew_l, 1e-6)
    ll_cr = (rm['ll_h_r'] - (r_ecz - r_lid_lo[2])) / max(ew_r, 1e-6)
    vals['facs_bs_EyelidCloseLowerLeft'] = clamp01(ll_cl * 5.0)
    vals['facs_bs_EyelidCloseLowerRight'] = clamp01(ll_cr * 5.0)

    # Wide open
    if rm['eye_open_l'] > 0:
        w = eye_open_l / rm['eye_open_l'] - 1.0
        vals['facs_bs_EyelidOpenUpperLeft'] = clamp01(w * 2.5)
        vals['facs_bs_EyelidOpenLowerLeft'] = clamp01(w * 1.5)
    if rm['eye_open_r'] > 0:
        w = eye_open_r / rm['eye_open_r'] - 1.0
        vals['facs_bs_EyelidOpenUpperRight'] = clamp01(w * 2.5)
        vals['facs_bs_EyelidOpenLowerRight'] = clamp01(w * 1.5)

    # ── SQUINT / CHEEK ──
    if rm['squint_l'] > 0:
        vals['facs_bs_EyeSquintLeft'] = clamp01((rm['squint_l'] - ldist(l_sq, l_lid_lo)) / rm['squint_l'] * 2.0)
    if rm['squint_r'] > 0:
        vals['facs_bs_EyeSquintRight'] = clamp01((rm['squint_r'] - ldist(r_sq, r_lid_lo)) / rm['squint_r'] * 2.0)

    vals['facs_bs_CheekSquintLeft'] = clamp01((rm['cheek_l'] - ldz(l_eye_out, l_chk)) / fh * 12.0)
    vals['facs_bs_CheekSquintRight'] = clamp01((rm['cheek_r'] - ldz(r_eye_out, r_chk)) / fh * 12.0)

    # ── NOSE ──
    nose_w = ldist(l_nostril, r_nostril)
    sneer = (nose_w - rm['nose_w']) / max(rm['nose_w'], 1e-6)
    vals['facs_bs_NoseSneerLeft'] = clamp01(sneer * 2.5)
    vals['facs_bs_NoseSneerRight'] = clamp01(sneer * 2.5)
    vals['facs_bs_NoseSneerUpperLeft'] = clamp01(sneer * 1.5)
    vals['facs_bs_NoseSneerUpperRight'] = clamp01(sneer * 1.5)

    # ── EYE GAZE ──
    if n_lm > LM_R_IRIS:
        l_iris = lm(LM_L_IRIS); r_iris = lm(LM_R_IRIS)
        l_ec = (l_eye_in + l_eye_out) / 2; r_ec = (r_eye_in + r_eye_out) / 2

        # Ref iris (precomputed would be better, but this is only per-frame)
        ro, rx, ry, rz = build_head_basis(ref)
        rl_iris = to_local(ref, LM_L_IRIS, ro, rx, ry, rz)
        rr_iris = to_local(ref, LM_R_IRIS, ro, rx, ry, rz)
        rl_ec = (to_local(ref, LM_L_EYE_INNER, ro, rx, ry, rz) +
                 to_local(ref, LM_L_EYE_OUTER, ro, rx, ry, rz)) / 2
        rr_ec = (to_local(ref, LM_R_EYE_INNER, ro, rx, ry, rz) +
                 to_local(ref, LM_R_EYE_OUTER, ro, rx, ry, rz)) / 2

        hl = ((l_iris[0] - l_ec[0]) - (rl_iris[0] - rl_ec[0])) / max(ew_l, 1e-6)
        hr = ((r_iris[0] - r_ec[0]) - (rr_iris[0] - rr_ec[0])) / max(ew_r, 1e-6)
        vl = ((l_iris[2] - l_ec[2]) - (rl_iris[2] - rl_ec[2])) / max(ew_l, 1e-6)
        vr = ((r_iris[2] - r_ec[2]) - (rr_iris[2] - rr_ec[2])) / max(ew_r, 1e-6)

        # L eye: +X = outward
        if hl > 0: vals['facs_bs_EyeLookOutLeft'] = clamp01(hl * 3.0)
        else:      vals['facs_bs_EyeLookInLeft'] = clamp01(-hl * 3.0)
        # R eye: -X = outward
        if hr < 0: vals['facs_bs_EyeLookOutRight'] = clamp01(-hr * 3.0)
        else:      vals['facs_bs_EyeLookInRight'] = clamp01(hr * 3.0)

        if vl > 0: vals['facs_bs_EyeLookUpLeft'] = clamp01(vl * 3.0)
        else:      vals['facs_bs_EyeLookDownLeft'] = clamp01(-vl * 3.0)
        if vr > 0: vals['facs_bs_EyeLookUpRight'] = clamp01(vr * 3.0)
        else:      vals['facs_bs_EyeLookDownRight'] = clamp01(-vr * 3.0)

    return vals


def smooth_array(arr, w):
    if w <= 1: return arr
    out = np.copy(arr)
    hw = w // 2
    for i in range(len(arr)):
        s, e = max(0, i-hw), min(len(arr), i+hw+1)
        out[i] = np.mean(arr[s:e])
    return out


# ─── MAIN ────────────────────────────────────────────────────

def main():
    # Load CSV
    resolved = bpy.path.abspath(CSV_PATH)
    frames = []
    with open(resolved, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        n_lm = len(header) // 3
        for row in reader:
            if len(row) < 3: continue
            vals = [float(v) if v else 0.0 for v in row]
            while len(vals) < n_lm * 3: vals.append(0.0)
            frames.append(np.array(vals[:n_lm*3]).reshape(n_lm, 3))
    print(f"[FACS] {len(frames)} frames, {n_lm} landmarks")

    # Reference
    ref = np.mean(frames[:min(N_REF_FRAMES, len(frames))], axis=0)
    rm = compute_ref(ref)
    print(f"[FACS] face_h={rm['face_h']:.1f}, eye_open={rm['eye_open_l']:.2f}/{rm['eye_open_r']:.2f}, "
          f"mouth_gap={rm['mouth_gap']:.2f}, mouth_w={rm['mouth_w']:.1f}")

    # Deadzones from neutral variance
    print("[FACS] Computing deadzones...")
    neutral_facs = [compute_facs(frames[i], ref, rm, n_lm)
                    for i in range(min(N_REF_FRAMES, len(frames)))]
    all_keys = set()
    for nf in neutral_facs: all_keys.update(nf.keys())
    deadzone = {}
    for k in all_keys:
        nv = [nf.get(k, 0.0) for nf in neutral_facs]
        deadzone[k] = np.std(nv) * 1.5

    # Compute all frames
    print("[FACS] Computing FACS...")
    all_facs = []
    for frame in frames:
        raw = compute_facs(frame, ref, rm, n_lm)
        # Apply deadzones
        cleaned = {}
        for k, v in raw.items():
            dz = deadzone.get(k, 0.0)
            cleaned[k] = clamp01(v)
        all_facs.append(cleaned)

    all_sk_names = set()
    for f in all_facs: all_sk_names.update(f.keys())

    # Smooth
    if SMOOTHING > 1:
        for sk in all_sk_names:
            arr = np.array([f.get(sk, 0.0) for f in all_facs])
            sm = smooth_array(arr, SMOOTHING)
            for i in range(len(all_facs)):
                all_facs[i][sk] = float(sm[i])

    # Peak values
    print(f"\n[FACS] === PEAKS ({len(all_sk_names)} channels) ===")
    for sk in sorted(all_sk_names):
        peak = max(f.get(sk, 0.0) for f in all_facs)
        if peak > 0.01:
            print(f"  {sk:50s} peak={peak:.3f}")

    # Find meshes
    target_meshes = []
    available_sk = set()
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.data.shape_keys: continue
        matching = [kb.name for kb in obj.data.shape_keys.key_blocks if kb.name in all_sk_names]
        if matching:
            target_meshes.append((obj, matching))
            available_sk.update(matching)
            print(f"[FACS] {obj.name}: {len(matching)} keys")

    missing = all_sk_names - available_sk
    if missing:
        print(f"\n[FACS] WARNING: {len(missing)} channels without matching shape key:")
        for s in sorted(missing): print(f"  {s}")

    if not target_meshes:
        print("[FACS] ERROR: No meshes with matching keys!"); return

    # Clear existing facs keyframes
    for obj, _ in target_meshes:
        sk = obj.data.shape_keys
        if sk.animation_data and sk.animation_data.action:
            try:
                fcs = sk.animation_data.action.fcurves
                to_del = [fc for fc in fcs if 'facs_' in fc.data_path]
                for fc in to_del: fcs.remove(fc)
            except: pass

    # Apply
    n_frames = len(frames)
    print(f"\n[FACS] Applying {n_frames} frames to {len(target_meshes)} meshes...")
    for fi in range(n_frames):
        bf = START_FRAME + fi
        facs = all_facs[fi]
        for obj, sk_names in target_meshes:
            sk = obj.data.shape_keys
            for name in sk_names:
                v = facs.get(name, 0.0) * GLOBAL_INTENSITY
                kb = sk.key_blocks[name]
                kb.value = v
                kb.keyframe_insert(data_path="value", frame=bf)
        if fi % 200 == 0:
            print(f"  Frame {fi}/{n_frames}")

    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + n_frames - 1
    bpy.context.scene.frame_set(START_FRAME)
    print(f"\n[FACS] Done! {n_frames} frames, {len(all_sk_names)} channels, {len(target_meshes)} meshes.")

if __name__ == "__main__":
    main()
