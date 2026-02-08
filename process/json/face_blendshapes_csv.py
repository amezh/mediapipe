import csv
import json
import numpy as np
import onnxruntime as ort
import os

# --- Config ---
CSV_PATH = os.path.join(os.path.dirname(__file__), "mediapipe_face_3d_xyz.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "face_blendshapes.onnx")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "blendshapes_output.json")
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280

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
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut"
]

def main():
    print(f"Loading CSV from {CSV_PATH}")
    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(v) for v in row])
    print(f"  {len(rows)} frames, {len(header)} columns")

    # Each landmark has 3 columns: x, y, z. Total = 478 * 3 = 1434
    # Column layout: face_0000_x, face_0000_y, face_0000_z, face_0001_x, ...
    # For landmark i: x = row[i*3], y = row[i*3+1], z = row[i*3+2]

    print(f"Loading model from {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    # Track min/max per blendshape
    bs_min = {}
    bs_max = {}
    for name in BLENDSHAPE_NAMES[1:]:
        bs_min[name] = float('inf')
        bs_max[name] = float('-inf')

    results = []
    for i, row in enumerate(rows):
        # Check if row has valid data (not all zeros)
        if all(v == 0.0 for v in row[:6]):
            results.append(None)
            continue

        # Extract all 478 landmarks: x, -z (FreeMoCap: y=0, z=inverted vertical)
        # Fix iris landmarks 468-477 which are always zero in FreeMoCap data:
        # estimate from surrounding eye landmarks
        lm_x = [row[idx * 3] for idx in range(478)]
        lm_y = [-row[idx * 3 + 2] for idx in range(478)]

        # Left iris (468-472): estimate from left eye corners (33, 133)
        left_cx = (lm_x[33] + lm_x[133]) / 2
        left_cy = (lm_y[33] + lm_y[133]) / 2
        left_rx = abs(lm_x[33] - lm_x[133]) * 0.15  # iris radius ~15% of eye width
        lm_x[468], lm_y[468] = left_cx, left_cy
        lm_x[469], lm_y[469] = left_cx + left_rx, left_cy
        lm_x[470], lm_y[470] = left_cx, left_cy - left_rx
        lm_x[471], lm_y[471] = left_cx - left_rx, left_cy
        lm_x[472], lm_y[472] = left_cx, left_cy + left_rx

        # Right iris (473-477): estimate from right eye corners (263, 362)
        right_cx = (lm_x[263] + lm_x[362]) / 2
        right_cy = (lm_y[263] + lm_y[362]) / 2
        right_rx = abs(lm_x[263] - lm_x[362]) * 0.15
        lm_x[473], lm_y[473] = right_cx, right_cy
        lm_x[474], lm_y[474] = right_cx - right_rx, right_cy
        lm_x[475], lm_y[475] = right_cx, right_cy - right_rx
        lm_x[476], lm_y[476] = right_cx + right_rx, right_cy
        lm_x[477], lm_y[477] = right_cx, right_cy + right_rx

        # Extract 146 subset
        pts = np.zeros((1, 146, 2), dtype=np.float32)
        for j, idx in enumerate(SUBSET_IDXS):
            pts[0, j, 0] = lm_x[idx]
            pts[0, j, 1] = lm_y[idx]

        output = session.run(None, {input_name: pts})[0]

        bs = {}
        for k in range(1, min(53, len(BLENDSHAPE_NAMES))):
            val = round(float(output[k]), 6)
            name = BLENDSHAPE_NAMES[k]
            bs[name] = val
            if val < bs_min[name]:
                bs_min[name] = val
            if val > bs_max[name]:
                bs_max[name] = val
        results.append(bs)

        if i % 100 == 0:
            print(f"  frame {i}/{len(rows)}")

    print(f"Writing {len(results)} frames to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    valid = sum(1 for r in results if r is not None)
    print(f"Done. {valid} frames with blendshapes, {len(results) - valid} skipped.")

    # Print min/max summary
    print(f"\n{'Blendshape':<25} {'Min':>10} {'Max':>10}")
    print("-" * 47)
    for name in BLENDSHAPE_NAMES[1:]:
        if bs_min[name] == float('inf'):
            continue
        print(f"{name:<25} {bs_min[name]:>10.6f} {bs_max[name]:>10.6f}")

if __name__ == "__main__":
    main()
