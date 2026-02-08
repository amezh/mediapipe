"""
ARKit JSON → Diffeomorphic G9 FACS
====================================
Select the target object (the one with facs props), set JSON_PATH, run.
Keyframes custom properties on the selected object.
"""

import bpy
import json

JSON_PATH = "D://VideoAIs//mp//process//json////blendshapes_output.json"
START_FRAME = 1
GLOBAL_INTENSITY = 1.0

ARKIT_TO_G9 = {
    "browDownLeft":       ["facs_BrowDownLeft"],
    "browDownRight":      ["facs_BrowDownRight"],
    "browInnerUp":        ["facs_bs_BrowInnerUpLeft", "facs_bs_BrowInnerUpRight"],
    "browOuterUpLeft":    ["facs_BrowOuterUpLeft"],
    "browOuterUpRight":   ["facs_BrowOuterUpRight"],
    "cheekPuff":          ["facs_bs_CheekPuffLeft", "facs_bs_CheekPuffRight"],
    "cheekSquintLeft":    ["facs_bs_CheekSquintLeft"],
    "cheekSquintRight":   ["facs_bs_CheekSquintRight"],
    "eyeBlinkLeft":       ["facs_bs_EyeBlinkLeft"],
    "eyeBlinkRight":      ["facs_bs_EyeBlinkRight"],
    "eyeLookDownLeft":    ["facs_bs_EyeLookDownLeft"],
    "eyeLookDownRight":   ["facs_bs_EyeLookDownRight"],
    "eyeLookInLeft":      ["facs_bs_EyeLookInLeft"],
    "eyeLookInRight":     ["facs_bs_EyeLookInRight"],
    "eyeLookOutLeft":     ["facs_bs_EyeLookOutLeft"],
    "eyeLookOutRight":    ["facs_bs_EyeLookOutRight"],
    "eyeLookUpLeft":      ["facs_bs_EyeLookUpLeft"],
    "eyeLookUpRight":     ["facs_bs_EyeLookUpRight"],
    "eyeSquintLeft":      ["facs_bs_EyeSquintLeft"],
    "eyeSquintRight":     ["facs_bs_EyeSquintRight"],
    "eyeWideLeft":        ["facs_bs_EyelidOpenUpperLeft", "facs_bs_EyelidOpenLowerLeft"],
    "eyeWideRight":       ["facs_bs_EyelidOpenUpperRight", "facs_bs_EyelidOpenLowerRight"],
    "jawForward":         ["facs_bs_JawRecess"],
    "jawLeft":            ["facs_bs_JawLeft"],
    "jawOpen":            ["facs_bs_JawOpen"],
    "jawRight":           ["facs_bs_JawRight"],
    "mouthClose":         ["facs_bs_MouthCloseLowerLeft", "facs_bs_MouthCloseLowerRight",
                           "facs_bs_MouthCloseUpperLeft", "facs_bs_MouthCloseUpperRight"],
    "mouthDimpleLeft":    ["facs_bs_MouthDimpleLeft"],
    "mouthDimpleRight":   ["facs_bs_MouthDimpleRight"],
    "mouthFrownLeft":     ["facs_bs_MouthFrownLeft"],
    "mouthFrownRight":    ["facs_bs_MouthFrownRight"],
    "mouthFunnel":        ["facs_bs_MouthFunnelLowerLeft", "facs_bs_MouthFunnelLowerRight",
                           "facs_bs_MouthFunnelUpperLeft", "facs_bs_MouthFunnelUpperRight"],
    "mouthLeft":          ["facs_bs_MouthLeft"],
    "mouthLowerDownLeft": ["facs_bs_MouthLowerDownLeft"],
    "mouthLowerDownRight":["facs_bs_MouthLowerDownRight"],
    "mouthPressLeft":     ["facs_bs_MouthPressLowerLeft", "facs_bs_MouthPressUpperLeft"],
    "mouthPressRight":    ["facs_bs_MouthPressLowerRight", "facs_bs_MouthPressUpperRight"],
    "mouthPucker":        ["facs_bs_MouthPurseLowerLeft", "facs_bs_MouthPurseLowerRight",
                           "facs_bs_MouthPurseUpperLeft", "facs_bs_MouthPurseUpperRight"],
    "mouthRight":         ["facs_bs_MouthRight"],
    "mouthRollLower":     ["facs_bs_MouthRollLowerLeft", "facs_bs_MouthRollLowerRight"],
    "mouthRollUpper":     ["facs_bs_MouthRollUpperLeft", "facs_bs_MouthRollUpperRight"],
    "mouthShrugLower":    ["facs_bs_MouthShrugLowerLeft", "facs_bs_MouthShrugLowerRight"],
    "mouthShrugUpper":    ["facs_bs_MouthShrugUpperLeft", "facs_bs_MouthShrugUpperRight"],
    "mouthSmileLeft":     ["facs_bs_MouthSmileLeft"],
    "mouthSmileRight":    ["facs_bs_MouthSmileRight"],
    "mouthStretchLeft":   ["facs_bs_MouthStretchLeft"],
    "mouthStretchRight":  ["facs_bs_MouthStretchRight"],
    "mouthUpperUpLeft":   ["facs_bs_MouthUpperUpLeft"],
    "mouthUpperUpRight":  ["facs_bs_MouthUpperUpRight"],
    "noseSneerLeft":      ["facs_bs_NoseSneerLeft"],
    "noseSneerRight":     ["facs_bs_NoseSneerRight"],
}

def main():
    obj = bpy.context.active_object
    if not obj:
        print("[FACS] ERROR: No active object!"); return

    print(f"[FACS] Target: {obj.name} (type={obj.type})")

    # ── Detect property style: try (fin) first, then plain facs_ ──
    all_props = set(obj.keys())
    fin_props = {k for k in all_props if '(fin)' in k and 'facs_' in k}
    facs_props = {k for k in all_props if 'facs_' in k and '(fin)' not in k}

    if fin_props:
        use_fin = True
        print(f"[FACS] Found {len(fin_props)} (fin) properties")
    elif facs_props:
        use_fin = False
        print(f"[FACS] Found {len(facs_props)} plain facs properties")
    else:
        print(f"[FACS] ERROR: No facs properties on {obj.name}!"); return

    # ── Build map: ARKit channel → property names ──
    arkit_to_prop = {}
    for arkit_name, g9_list in ARKIT_TO_G9.items():
        for g9_name in g9_list:
            prop_name = g9_name + '(fin)' if use_fin else g9_name
            if prop_name in all_props:
                arkit_to_prop.setdefault(arkit_name, []).append(prop_name)

    mapped = sum(len(v) for v in arkit_to_prop.values())
    print(f"[FACS] {len(arkit_to_prop)} ARKit channels → {mapped} properties")

    # ── Load JSON ──
    resolved = bpy.path.abspath(JSON_PATH)
    with open(resolved, 'r') as f:
        data = json.load(f)

    n_total = len(data)
    n_valid = sum(1 for d in data if d is not None)
    print(f"[FACS] {n_total} frames ({n_valid} valid)")

    # ── Clear old keyframes ──
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        try:
            fcs = action.fcurves
        except AttributeError:
            try:
                fcs = action.curves
            except AttributeError:
                fcs = None
        if fcs:
            to_del = [fc for fc in fcs if 'facs_' in fc.data_path]
            for fc in to_del:
                fcs.remove(fc)
            print(f"[FACS] Cleared {len(to_del)} old fcurves")

    # ── Keyframe ──
    kf_count = 0
    for fi in range(n_total):
        if data[fi] is None:
            continue
        bf = START_FRAME + fi
        for arkit_name, value in data[fi].items():
            for prop_name in arkit_to_prop.get(arkit_name, []):
                obj[prop_name] = value * GLOBAL_INTENSITY
                obj.keyframe_insert(data_path=f'["{prop_name}"]', frame=bf)
                kf_count += 1
        if fi % 200 == 0:
            print(f"  Frame {fi}/{n_total}")

    # ── Verify mid-frame ──
    mid_frame = START_FRAME + n_total // 2
    bpy.context.scene.frame_set(mid_frame)
    print(f"\n[FACS] === VERIFY @ frame {mid_frame} ===")
    for prop_list in list(arkit_to_prop.values())[:5]:
        for p in prop_list:
            print(f"  {p} = {obj[p]:.6f}")

    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + n_total - 1
    bpy.context.scene.frame_set(START_FRAME)
    print(f"\n[FACS] Done! {kf_count} keyframes on {obj.name}")

if __name__ == "__main__":
    main()
