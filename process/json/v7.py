"""
MediaPipe Skeleton Rig Builder + Animator v7
=============================================
Creates a proper Blender armature from MediaPipe pose/hand landmarks,
animates it frame-by-frame, then allows retargeting to any standard rig.

Pipeline:
  1. Load landmarks_output.json
  2. Build armature with bones matching MediaPipe skeleton topology
     (same lines as shown in the MediaPipe overlay visualization)
  3. Calibrate rest pose from first N frames
  4. Animate all bones per-frame using landmark positions → bone rotations
  5. Result: a fully animated "MP_Skeleton" rig ready for retargeting

Run in Blender Text Editor (Alt+P).
"""

import bpy, json, math, os
from mathutils import Vector, Matrix, Quaternion, Euler

# ============================================================================
# CONFIG
# ============================================================================

LANDMARKS_FILE = r"D:\VideoAIs\mp\process\json\landmarks_output.json"
ARMATURE_NAME  = "MP_Skeleton"
SOURCE_FPS     = 30
START_FRAME    = 1
SMOOTHING_WINDOW = 3
HUMAN_HEIGHT   = 1.70        # meters — scales the rig to realistic size
REST_CALIBRATION_FRAMES = 8  # number of frames to average for rest pose

# ============================================================================
# MEDIAPIPE POSE LANDMARK INDICES (33 total)
# ============================================================================

MP = {
    "nose": 0,
    "l_eye_inner": 1, "l_eye": 2, "l_eye_outer": 3,
    "r_eye_inner": 4, "r_eye": 5, "r_eye_outer": 6,
    "l_ear": 7, "r_ear": 8,
    "mouth_l": 9, "mouth_r": 10,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20,
    "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_foot_index": 31, "r_foot_index": 32,
}

# ============================================================================
# SKELETON DEFINITION
# Bones defined as (bone_name, parent_landmark_idx, child_landmark_idx, parent_bone_name)
# This matches the MediaPipe connection lines shown in the visualization.
# ============================================================================

# Virtual landmarks: computed midpoints (will be filled at runtime)
VIRTUAL_HIP_CENTER      = 100  # midpoint of l_hip + r_hip
VIRTUAL_SHOULDER_CENTER = 101  # midpoint of l_shoulder + r_shoulder
VIRTUAL_HEAD_TOP        = 102  # above nose

SKELETON_BONES = [
    # --- TORSO ---
    ("spine",          VIRTUAL_HIP_CENTER,      VIRTUAL_SHOULDER_CENTER, None),
    ("neck",           VIRTUAL_SHOULDER_CENTER,  MP["nose"],              "spine"),
    ("head",           MP["nose"],               VIRTUAL_HEAD_TOP,        "neck"),

    # --- LEFT ARM ---
    ("l_clavicle",     VIRTUAL_SHOULDER_CENTER,  MP["l_shoulder"],        "spine"),
    ("l_upperarm",     MP["l_shoulder"],          MP["l_elbow"],          "l_clavicle"),
    ("l_forearm",      MP["l_elbow"],             MP["l_wrist"],          "l_upperarm"),
    ("l_hand",         MP["l_wrist"],             MP["l_index"],          "l_forearm"),

    # --- RIGHT ARM ---
    ("r_clavicle",     VIRTUAL_SHOULDER_CENTER,  MP["r_shoulder"],        "spine"),
    ("r_upperarm",     MP["r_shoulder"],          MP["r_elbow"],          "r_clavicle"),
    ("r_forearm",      MP["r_elbow"],             MP["r_wrist"],          "r_upperarm"),
    ("r_hand",         MP["r_wrist"],             MP["r_index"],          "r_forearm"),

    # --- LEFT LEG ---
    ("l_hip_bone",     VIRTUAL_HIP_CENTER,       MP["l_hip"],            "spine"),
    ("l_thigh",        MP["l_hip"],               MP["l_knee"],          "l_hip_bone"),
    ("l_shin",         MP["l_knee"],              MP["l_ankle"],          "l_thigh"),
    ("l_foot",         MP["l_ankle"],             MP["l_foot_index"],     "l_shin"),

    # --- RIGHT LEG ---
    ("r_hip_bone",     VIRTUAL_HIP_CENTER,       MP["r_hip"],            "spine"),
    ("r_thigh",        MP["r_hip"],               MP["r_knee"],          "r_hip_bone"),
    ("r_shin",         MP["r_knee"],              MP["r_ankle"],          "r_thigh"),
    ("r_foot",         MP["r_ankle"],             MP["r_foot_index"],     "r_shin"),
]

# --- FINGER BONES (added if hand landmarks are present) ---
HAND_FINGER_CHAINS = {
    "thumb":  [(1, 2), (2, 3), (3, 4)],
    "index":  [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring":   [(13, 14), (14, 15), (15, 16)],
    "pinky":  [(17, 18), (18, 19), (19, 20)],
}


# ============================================================================
# HELPERS
# ============================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_values(values, window):
    """Simple moving average smoother."""
    if window <= 1 or len(values) < 2:
        return values[:]
    hw = window // 2
    out = []
    for i in range(len(values)):
        sl = values[max(0, i - hw):min(len(values), i + hw + 1)]
        out.append(sum(sl) / len(sl))
    return out


def mp_to_blender(lm, scale=1.0, offset=Vector((0, 0, 0))):
    """
    Convert MediaPipe landmark dict to Blender world coords.
    MP: x=right(0..1), y=down(0..1), z=towards_camera(neg=closer)
    Blender: x=right, y=forward(depth), z=up
    """
    x = (lm['x'] - 0.5) * scale + offset.x
    y = -lm.get('z', 0.0) * scale + offset.y
    z = -(lm['y'] - 0.5) * scale + offset.z
    return Vector((x, y, z))


def get_landmark_pos(pose_lm, idx, scale=1.0, offset=Vector((0, 0, 0))):
    """Get 3D position for a landmark index, including virtual landmarks."""
    if idx == VIRTUAL_HIP_CENTER:
        a = mp_to_blender(pose_lm[MP["l_hip"]], scale, offset)
        b = mp_to_blender(pose_lm[MP["r_hip"]], scale, offset)
        return (a + b) * 0.5
    elif idx == VIRTUAL_SHOULDER_CENTER:
        a = mp_to_blender(pose_lm[MP["l_shoulder"]], scale, offset)
        b = mp_to_blender(pose_lm[MP["r_shoulder"]], scale, offset)
        return (a + b) * 0.5
    elif idx == VIRTUAL_HEAD_TOP:
        nose = mp_to_blender(pose_lm[MP["nose"]], scale, offset)
        # Estimate head top: above nose by ~12% of body height
        return nose + Vector((0, 0, scale * 0.08))
    else:
        return mp_to_blender(pose_lm[idx], scale, offset)


def compute_scale(frames, n_cal=8):
    """Estimate scale factor so the skeleton is HUMAN_HEIGHT meters tall."""
    heights = []
    for fi in range(min(n_cal, len(frames))):
        pl = frames[fi].get('pose_landmarks')
        if pl is None:
            continue
        # Measure: ankle to nose vertical span
        nose_y = pl[MP["nose"]]['y']
        l_ankle_y = pl[MP["l_ankle"]]['y']
        r_ankle_y = pl[MP["r_ankle"]]['y']
        ankle_y = (l_ankle_y + r_ankle_y) * 0.5
        span = abs(ankle_y - nose_y)
        if span > 0.01:
            heights.append(span)
    if not heights:
        return HUMAN_HEIGHT
    avg_span = sum(heights) / len(heights)
    # nose-to-ankle ≈ 90% of full height
    return HUMAN_HEIGHT / (avg_span * 0.9) if avg_span > 0.01 else HUMAN_HEIGHT


def compute_rest_positions(frames, scale, n_cal=8):
    """Average landmark positions over first N frames for rest pose."""
    accum = {}
    count = 0
    for fi in range(min(n_cal, len(frames))):
        pl = frames[fi].get('pose_landmarks')
        if pl is None:
            continue
        for bone_name, head_idx, tail_idx, _ in SKELETON_BONES:
            h = get_landmark_pos(pl, head_idx, scale)
            t = get_landmark_pos(pl, tail_idx, scale)
            key = bone_name
            if key not in accum:
                accum[key] = {'head': Vector((0, 0, 0)), 'tail': Vector((0, 0, 0))}
            accum[key]['head'] += h
            accum[key]['tail'] += t
        count += 1

    if count == 0:
        raise RuntimeError("No pose landmarks found in calibration frames!")

    rest = {}
    for key in accum:
        rest[key] = {
            'head': accum[key]['head'] / count,
            'tail': accum[key]['tail'] / count,
        }
    return rest


# ============================================================================
# ARMATURE BUILDER
# ============================================================================

def create_armature(rest_positions):
    """Create the Blender armature with bones at rest positions."""

    # Clean up existing
    old = bpy.data.objects.get(ARMATURE_NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    old_arm = bpy.data.armatures.get(ARMATURE_NAME)
    if old_arm:
        bpy.data.armatures.remove(old_arm)

    # Create armature
    arm_data = bpy.data.armatures.new(ARMATURE_NAME)
    arm_data.display_type = 'STICK'
    arm_obj = bpy.data.objects.new(ARMATURE_NAME, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones
    bone_refs = {}

    for bone_name, head_idx, tail_idx, parent_name in SKELETON_BONES:
        rp = rest_positions.get(bone_name)
        if rp is None:
            continue

        eb = edit_bones.new(bone_name)
        eb.head = rp['head']
        eb.tail = rp['tail']

        # Ensure minimum bone length
        if (eb.tail - eb.head).length < 0.001:
            eb.tail = eb.head + Vector((0, 0, 0.02))

        if parent_name and parent_name in bone_refs:
            eb.parent = bone_refs[parent_name]
            eb.use_connect = False  # connected only if head matches parent tail

        bone_refs[bone_name] = eb

    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[Armature] Created '{ARMATURE_NAME}' with {len(bone_refs)} bones")
    return arm_obj


# ============================================================================
# FINGER BONE BUILDER (added in edit mode if hand data exists)
# ============================================================================

def add_finger_bones(arm_obj, frames, scale, n_cal=8):
    """Add finger bones from hand landmarks to the armature."""

    # Check if any hand data exists
    has_left = any('left_hand_landmarks' in f for f in frames[:n_cal])
    has_right = any('right_hand_landmarks' in f for f in frames[:n_cal])

    if not has_left and not has_right:
        print("[Fingers] No hand landmark data found, skipping fingers")
        return

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm_obj.data.edit_bones

    sides = []
    if has_left:
        sides.append(('l', 'left_hand_landmarks', 'l_hand'))
    if has_right:
        sides.append(('r', 'right_hand_landmarks', 'r_hand'))

    finger_count = 0
    for prefix, lm_key, parent_bone_name in sides:
        # Average finger positions over calibration frames
        cal_frames = []
        for fi in range(min(n_cal, len(frames))):
            hlm = frames[fi].get(lm_key)
            if hlm is not None:
                cal_frames.append(hlm)
        if not cal_frames:
            continue

        parent_eb = edit_bones.get(parent_bone_name)

        for finger_name, joints in HAND_FINGER_CHAINS.items():
            prev_eb = parent_eb
            for ji, (jfrom, jto) in enumerate(joints):
                bone_name = f"{prefix}_{finger_name}{ji + 1}"

                # Average positions
                heads, tails = [], []
                for hlm in cal_frames:
                    heads.append(mp_to_blender(hlm[jfrom], scale))
                    tails.append(mp_to_blender(hlm[jto], scale))

                avg_head = sum(heads, Vector((0, 0, 0))) / len(heads)
                avg_tail = sum(tails, Vector((0, 0, 0))) / len(tails)

                eb = edit_bones.new(bone_name)
                eb.head = avg_head
                eb.tail = avg_tail

                if (eb.tail - eb.head).length < 0.0005:
                    eb.tail = eb.head + Vector((0, 0, 0.005))

                if prev_eb:
                    eb.parent = prev_eb
                    eb.use_connect = False

                prev_eb = eb
                finger_count += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Fingers] Added {finger_count} finger bones")


# ============================================================================
# ANIMATION ENGINE
# ============================================================================

class SkeletonAnimator:
    """Drives the MP_Skeleton armature from landmark data."""

    def __init__(self, arm_obj, scale):
        self.arm = arm_obj
        self.scale = scale
        self.pb = arm_obj.pose.bones

        # Cache inverse rest matrices for rotation computation
        self.rest_dirs = {}
        self.inv_rest = {}
        for bone_name, head_idx, tail_idx, parent_name in SKELETON_BONES:
            bone = arm_obj.data.bones.get(bone_name)
            if bone:
                rest_dir = (bone.tail_local - bone.head_local).normalized()
                self.rest_dirs[bone_name] = rest_dir
                self.inv_rest[bone_name] = bone.matrix_local.to_3x3().inverted()

        # Store bone definitions for quick lookup
        self.bone_defs = {b[0]: b for b in SKELETON_BONES}

        print(f"[Animator] Ready: {len(self.rest_dirs)} bones")

    def apply_frame(self, pose_lm, frame):
        """Apply one frame of pose landmarks to the skeleton."""
        if pose_lm is None:
            return

        sc = self.scale
        offset = Vector((0, 0, 0))

        # --- Root bone (spine) location ---
        root_pb = self.pb.get("spine")
        if root_pb:
            hip_center = get_landmark_pos(pose_lm, VIRTUAL_HIP_CENTER, sc, offset)
            # Keyframe root location (spine head = hip center)
            root_pb.location = self.arm.matrix_world.inverted() @ hip_center - root_pb.bone.head_local
            root_pb.keyframe_insert(data_path="location", frame=frame)

        # --- Rotate all bones to match landmark directions ---
        for bone_name, head_idx, tail_idx, parent_name in SKELETON_BONES:
            pb = self.pb.get(bone_name)
            if pb is None or bone_name not in self.inv_rest:
                continue

            # Current observed direction
            h = get_landmark_pos(pose_lm, head_idx, sc, offset)
            t = get_landmark_pos(pose_lm, tail_idx, sc, offset)
            obs_dir = t - h
            if obs_dir.length < 1e-7:
                continue
            obs_dir.normalize()

            # Convert to bone-local space
            obs_local = (self.inv_rest[bone_name] @ obs_dir).normalized()

            # Rotation from rest Y-axis to observed direction
            rest_axis = Vector((0, 1, 0))
            quat = rest_axis.rotation_difference(obs_local)

            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = quat
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    def apply_fingers(self, hand_lm, side, frame):
        """Animate finger bones from hand landmarks."""
        if hand_lm is None:
            return

        prefix = 'l' if side == 'left' else 'r'
        sc = self.scale

        for finger_name, joints in HAND_FINGER_CHAINS.items():
            for ji, (jfrom, jto) in enumerate(joints):
                bone_name = f"{prefix}_{finger_name}{ji + 1}"
                pb = self.pb.get(bone_name)
                if pb is None or bone_name not in self.inv_rest:
                    # Finger bones may not have inv_rest cached; compute on the fly
                    bone = self.arm.data.bones.get(bone_name)
                    if bone is None:
                        continue
                    inv = bone.matrix_local.to_3x3().inverted()
                    self.inv_rest[bone_name] = inv

                inv = self.inv_rest[bone_name]

                h = mp_to_blender(hand_lm[jfrom], sc)
                t = mp_to_blender(hand_lm[jto], sc)
                obs_dir = t - h
                if obs_dir.length < 1e-7:
                    continue
                obs_dir.normalize()

                obs_local = (inv @ obs_dir).normalized()
                quat = Vector((0, 1, 0)).rotation_difference(obs_local)

                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = quat
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)


# ============================================================================
# SMOOTHING PASS
# ============================================================================

def smooth_fcurves(arm_obj, window=3):
    """Apply moving-average smoothing to all animated F-curves."""
    if window <= 1:
        return

    act = arm_obj.animation_data.action if arm_obj.animation_data else None
    if act is None:
        return

    count = 0
    try:
        for fc in act.fcurves:
            kps = fc.keyframe_points
            n = len(kps)
            if n < 3:
                continue

            # Extract values
            vals = [kp.co[1] for kp in kps]
            smoothed = smooth_values(vals, window)

            # Write back
            for i, kp in enumerate(kps):
                kp.co[1] = smoothed[i]

            # Update handles
            kps.handles_recalc()
            count += 1
    except Exception as e:
        print(f"[Smooth] Warning: {e}")

    print(f"[Smooth] Smoothed {count} F-curves (window={window})")


# ============================================================================
# BEZIER INTERPOLATION
# ============================================================================

def apply_bezier(arm_obj):
    """Set all keyframes to Bezier with auto-clamped handles."""
    act = arm_obj.animation_data.action if arm_obj.animation_data else None
    if act is None:
        return

    count = 0
    try:
        for fc in act.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = 'AUTO_CLAMPED'
                kp.handle_right_type = 'AUTO_CLAMPED'
            count += 1
    except Exception as e:
        print(f"[Bezier] Warning: {e}")

    print(f"[Bezier] Applied to {count} F-curves")


# ============================================================================
# VISUAL ENHANCEMENTS — bone colors and custom shapes
# ============================================================================

def setup_bone_groups(arm_obj):
    """Color-code bone groups for easy visual identification."""
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    groups = {
        "Torso":    (["spine", "neck", "head"], (0.2, 0.6, 1.0)),
        "L_Arm":    (["l_clavicle", "l_upperarm", "l_forearm", "l_hand"], (1.0, 0.3, 0.3)),
        "R_Arm":    (["r_clavicle", "r_upperarm", "r_forearm", "r_hand"], (0.3, 1.0, 0.3)),
        "L_Leg":    (["l_hip_bone", "l_thigh", "l_shin", "l_foot"], (1.0, 0.6, 0.2)),
        "R_Leg":    (["r_hip_bone", "r_thigh", "r_shin", "r_foot"], (0.6, 0.2, 1.0)),
    }

    # Add finger bone names
    for prefix in ['l', 'r']:
        key = f"{'L' if prefix == 'l' else 'R'}_Fingers"
        names = []
        for finger_name in HAND_FINGER_CHAINS:
            for ji in range(3):
                names.append(f"{prefix}_{finger_name}{ji + 1}")
        color = (1.0, 0.8, 0.4) if prefix == 'l' else (0.4, 0.8, 1.0)
        groups[key] = (names, color)

    # Use bone color API (Blender 4.0+) or fall back
    try:
        for group_name, (bone_names, color) in groups.items():
            for bn in bone_names:
                pb = arm_obj.pose.bones.get(bn)
                if pb:
                    pb.color.palette = 'CUSTOM'
                    pb.color.custom.normal = (*color,)
    except Exception:
        pass  # Older Blender version

    bpy.ops.object.mode_set(mode='OBJECT')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(" MediaPipe Skeleton Rig Builder + Animator v7")
    print(" Creates animated skeleton armature from landmarks JSON")
    print("=" * 70)

    # --- Load data ---
    print(f"\n[Load] {LANDMARKS_FILE}")
    if not os.path.exists(LANDMARKS_FILE):
        raise FileNotFoundError(f"File not found: {LANDMARKS_FILE}")

    with open(LANDMARKS_FILE, 'r') as f:
        frames = json.load(f)

    nf = len(frames)
    n_pose = sum(1 for f in frames if 'pose_landmarks' in f)
    n_face = sum(1 for f in frames if 'face_landmarks' in f)
    n_lh = sum(1 for f in frames if 'left_hand_landmarks' in f)
    n_rh = sum(1 for f in frames if 'right_hand_landmarks' in f)
    print(f"  Frames: {nf}")
    print(f"  Pose: {n_pose}, Face: {n_face}, L-Hand: {n_lh}, R-Hand: {n_rh}")

    if n_pose == 0:
        raise RuntimeError("No pose landmark data found!")

    # --- Compute scale ---
    scale = compute_scale(frames, REST_CALIBRATION_FRAMES)
    print(f"\n[Scale] {scale:.3f} (targeting {HUMAN_HEIGHT}m height)")

    # --- Compute rest pose ---
    print(f"[Rest] Averaging {REST_CALIBRATION_FRAMES} frames for rest pose...")
    rest_positions = compute_rest_positions(frames, scale, REST_CALIBRATION_FRAMES)

    # --- Build armature ---
    print("\n[Build] Creating armature...")
    arm_obj = create_armature(rest_positions)

    # --- Add finger bones ---
    add_finger_bones(arm_obj, frames, scale, REST_CALIBRATION_FRAMES)

    # --- Setup visual groups ---
    setup_bone_groups(arm_obj)

    # --- Prepare animation ---
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # Clear any existing animation
    if arm_obj.animation_data:
        arm_obj.animation_data_clear()
    arm_obj.animation_data_create()
    action = bpy.data.actions.new(name="MP_Skeleton_Anim")
    arm_obj.animation_data.action = action

    # Reset pose
    for pb in arm_obj.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)

    # --- Animate ---
    print(f"\n[Animate] Keyframing {nf} frames...")
    animator = SkeletonAnimator(arm_obj, scale)

    for fi in range(nf):
        bf = START_FRAME + fi
        fr = frames[fi]

        # Pose (body skeleton)
        pose_lm = fr.get('pose_landmarks')
        animator.apply_frame(pose_lm, bf)

        # Fingers
        animator.apply_fingers(fr.get('left_hand_landmarks'), 'left', bf)
        animator.apply_fingers(fr.get('right_hand_landmarks'), 'right', bf)

        if fi % 100 == 0:
            print(f"  {fi}/{nf}...")

    print(f"  {nf}/{nf} — done")

    # --- Smooth ---
    print(f"\n[Post] Smoothing (window={SMOOTHING_WINDOW})...")
    smooth_fcurves(arm_obj, SMOOTHING_WINDOW)

    # --- Bezier ---
    print("[Post] Applying Bezier interpolation...")
    apply_bezier(arm_obj)

    # --- Scene settings ---
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + nf - 1
    bpy.context.scene.frame_current = START_FRAME
    bpy.context.scene.render.fps = SOURCE_FPS

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    # --- Summary ---
    total_bones = len(arm_obj.data.bones)
    print(f"\n{'=' * 70}")
    print(f" DONE — '{ARMATURE_NAME}' created and animated!")
    print(f"  Bones:  {total_bones}")
    print(f"  Frames: {START_FRAME}–{START_FRAME + nf - 1} @ {SOURCE_FPS}fps")
    print(f"  Scale:  {scale:.3f} ({HUMAN_HEIGHT}m target height)")
    print(f"{'=' * 70}")
    print(f"\n NEXT STEPS:")
    print(f"  1. Import/create your target rig (e.g. Rigify, Genesis, Mixamo)")
    print(f"  2. Use a retarget add-on (e.g. Rokoko, Auto-Rig Pro Remap)")
    print(f"     to transfer animation from '{ARMATURE_NAME}' to your target rig")
    print(f"  3. Or manually add 'Copy Rotation' constraints per bone")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
