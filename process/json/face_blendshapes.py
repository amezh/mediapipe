import json
import numpy as np
import onnxruntime as ort
import os

# --- Config ---
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
LANDMARKS_JSON = os.path.join(os.path.dirname(__file__), "landmarks_output.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "face_blendshapes.onnx")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "blendshapes_output.json")

# 146 landmark indices (subset of 478)
SUBSET_IDXS = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54,
    55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95,
    103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152,
    153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178,
    181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282,
    283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374,
    375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397,
    398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474,
    475, 476, 477
]

# 52 blendshape names (index 0 = _neutral, skip it)
BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

def main():
    print(f"Loading landmarks from {LANDMARKS_JSON}")
    with open(LANDMARKS_JSON, "r") as f:
        frames = json.load(f)

    print(f"Loading model from {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    results = []

    # Track min/max per blendshape
    bs_min = {}
    bs_max = {}
    for name in BLENDSHAPE_NAMES[1:]:
        bs_min[name] = float('inf')
        bs_max[name] = float('-inf')

    for i, frame in enumerate(frames):
        face_lm = frame.get("face_landmarks")
        if not face_lm or len(face_lm) < 478:
            results.append(None)
            continue

        # Extract 146 subset, denormalize x/y by image dimensions
        pts = np.zeros((1, 146, 2), dtype=np.float32)
        for j, idx in enumerate(SUBSET_IDXS):
            pts[0, j, 0] = face_lm[idx]["x"] * IMAGE_WIDTH
            pts[0, j, 1] = face_lm[idx]["y"] * IMAGE_HEIGHT

        output = session.run(None, {input_name: pts})[0]  # shape (52,)

        # Build dict of blendshape name -> weight, skip _neutral
        bs = {}
        for k in range(1, 52):
            val = round(float(output[k]), 6)
            name = BLENDSHAPE_NAMES[k]
            bs[name] = val
            if val < bs_min[name]:
                bs_min[name] = val
            if val > bs_max[name]:
                bs_max[name] = val
        results.append(bs)

        if i % 100 == 0:
            print(f"  frame {i}/{len(frames)}")

    print(f"Writing {len(results)} frames to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Stats
    valid = sum(1 for r in results if r is not None)
    print(f"Done. {valid} frames with blendshapes, {len(results) - valid} frames skipped (no face).")

    # Print min/max summary
    print(f"\n{'Blendshape':<25} {'Min':>10} {'Max':>10}")
    print("-" * 47)
    for name in BLENDSHAPE_NAMES[1:]:
        if bs_min[name] == float('inf'):
            continue
        print(f"{name:<25} {bs_min[name]:>10.6f} {bs_max[name]:>10.6f}")

if __name__ == "__main__":
    main()
