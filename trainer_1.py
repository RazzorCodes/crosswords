import json
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(entry):
    """
    entry: dict with 'label' and 'strokes'
    strokes: list of lists of {'x', 'y', 't'}
    """
    strokes = entry["strokes"]

    # Flatten all points
    all_pts = [pt for stroke in strokes for pt in stroke]
    xs = np.array([p["x"] for p in all_pts])
    ys = np.array([p["y"] for p in all_pts])
    ts = np.array([p["t"] for p in all_pts])

    # Normalize spatial coords to [0, 1]
    x_range = xs.max() - xs.min() + 1e-6
    y_range = ys.max() - ys.min() + 1e-6
    xn = (xs - xs.min()) / x_range
    yn = (ys - ys.min()) / y_range

    feats = []

    # ── 1. Global shape ──────────────────────────────────────────────
    feats.append(len(strokes))  # stroke count
    feats.append(y_range / x_range)  # aspect ratio (tall letters vs wide)
    feats.append(x_range / (x_range + y_range))  # relative width

    # ── 2. Direction histogram (8 bins) ─────────────────────────────
    angles = []
    for stroke in strokes:
        pts = np.array([[p["x"], p["y"]] for p in stroke])
        if len(pts) < 2:
            continue
        d = np.diff(pts, axis=0)
        angles.extend(np.arctan2(d[:, 1], d[:, 0]))

    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    feats.extend(hist / (len(angles) + 1e-6))

    # ── 3. Curvature (direction change rate) ─────────────────────────
    all_angles = np.array(angles)
    if len(all_angles) > 1:
        diffs = np.diff(all_angles)
        # Wrap to [-pi, pi]
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
        feats.append(np.mean(np.abs(diffs)))  # mean curvature
        feats.append(np.std(diffs))  # curvature variance
    else:
        feats.extend([0.0, 0.0])

    # ── 4. Start / end points (normalized) ───────────────────────────
    first_pt = strokes[0][0]
    last_pt = strokes[-1][-1]
    feats.extend(
        [
            (first_pt["x"] - xs.min()) / x_range,
            (first_pt["y"] - ys.min()) / y_range,
            (last_pt["x"] - xs.min()) / x_range,
            (last_pt["y"] - ys.min()) / y_range,
        ]
    )

    # ── 5. Total path length (normalized) ────────────────────────────
    total_len = 0
    for stroke in strokes:
        pts = np.array([[p["x"], p["y"]] for p in stroke])
        if len(pts) > 1:
            total_len += np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    feats.append(total_len / (x_range + y_range))

    # ── 6. Velocity features (you have timestamps!) ───────────────────
    speeds = []
    pauses = []  # time gaps between strokes — catches pen lifts

    prev_end_t = None
    for stroke in strokes:
        if prev_end_t is not None:
            pauses.append(stroke[0]["t"] - prev_end_t)
        prev_end_t = stroke[-1]["t"]

        for i in range(1, len(stroke)):
            dx = stroke[i]["x"] - stroke[i - 1]["x"]
            dy = stroke[i]["y"] - stroke[i - 1]["y"]
            dt = stroke[i]["t"] - stroke[i - 1]["t"] + 1e-6
            dist = np.sqrt(dx**2 + dy**2)
            speeds.append(dist / dt)

    speeds = np.array(speeds) if speeds else np.array([0.0])
    feats.append(np.mean(speeds))  # avg speed
    feats.append(np.std(speeds))  # speed variance
    feats.append(np.percentile(speeds, 90))  # fast segments

    if pauses:
        feats.append(np.mean(pauses))  # avg pen-lift duration
        feats.append(np.max(pauses))  # longest pause
    else:
        feats.extend([0.0, 0.0])

    # ── 7. Per-stroke: relative position of stroke centroid ──────────
    # Captures where in the letter each stroke lives (top, bottom, left, right)
    # Pad / truncate to 3 strokes (handles A=3, B=2, E=4, etc.)
    for i in range(3):
        if i < len(strokes):
            sx = np.mean([p["x"] for p in strokes[i]])
            sy = np.mean([p["y"] for p in strokes[i]])
            feats.append((sx - xs.min()) / x_range)
            feats.append((sy - ys.min()) / y_range)
        else:
            feats.extend([-1.0, -1.0])  # sentinel for missing stroke

    # ── 8. Crossing count (crude: vertical line at x=0.5) ────────────
    # Great for letters like A, X, H, T vs C, L, I
    crossings = 0
    mid_x = xs.min() + x_range * 0.5
    for stroke in strokes:
        pts = np.array([[p["x"], p["y"]] for p in stroke])
        if len(pts) < 2:
            continue
        for j in range(len(pts) - 1):
            x0, x1 = pts[j, 0], pts[j + 1, 0]
            if (x0 < mid_x) != (x1 < mid_x):
                crossings += 1
    feats.append(crossings)

    return np.array(feats, dtype=np.float32)


# ── Training ─────────────────────────────────────────────────────────


def train(data):
    """data: list of {'label': 'A', 'strokes': [...]}"""
    X = np.array([extract_features(e) for e in data])
    y = [e["label"].upper() for e in data]

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]
    )
    clf.fit(X, y)
    return clf


def predict(clf, entry):
    """Returns (predicted_label, confidence_dict)"""
    feats = extract_features(entry).reshape(1, -1)
    label = clf.predict(feats)[0]
    probs = clf.predict_proba(feats)[0]
    return label, dict(zip(clf.classes_, probs))


# ── Usage ─────────────────────────────────────────────────────────────
# with open("my_data.json") as f:
#     data = json.load(f)
#
# clf = train(data)
# pickle.dump(clf, open("letter_clf.pkl", "wb"))
#
# # Inference
# clf = pickle.load(open("letter_clf.pkl", "rb"))
# label, probs = predict(clf, new_entry)
# top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
