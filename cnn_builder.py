import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def render_stroke_entry(entry, size=64, pad=8, line_width=3):
    """
    Renders a stroke entry to a square grayscale PIL image.
    Returns a numpy array of shape (size, size), dtype uint8.
    """
    strokes = entry["strokes"]
    all_pts = [pt for stroke in strokes for pt in stroke]

    xs = [p["x"] for p in all_pts]
    ys = [p["y"] for p in all_pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    # Scale to square with padding, preserving aspect ratio
    draw_size = size - 2 * pad
    scale = draw_size / max(x_range, y_range)  # uniform scale, no stretch

    x_offset = pad + (draw_size - x_range * scale) / 2
    y_offset = pad + (draw_size - y_range * scale) / 2

    def to_px(x, y):
        return (
            int((x - x_min) * scale + x_offset),
            int((y - y_min) * scale + y_offset),
        )

    img = Image.new("L", (size, size), color=255)  # white background
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        points = [to_px(p["x"], p["y"]) for p in stroke]
        if len(points) == 1:
            x, y = points[0]
            r = line_width // 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        else:
            draw.line(points, fill=0, width=line_width, joint="curve")

    return np.array(img)


def build_dataset(entries, out_dir="cnn_data", size=64):
    """
    Saves each entry as a PNG under out_dir/<LABEL>/<index>.png
    Ready for torchvision.datasets.ImageFolder or keras flow_from_directory
    """
    out_dir = Path(out_dir)
    counts = {}

    for entry in entries:
        label = entry["label"].upper()
        folder = out_dir / label
        folder.mkdir(parents=True, exist_ok=True)

        idx = counts.get(label, 0)
        img_array = render_stroke_entry(entry, size=size)
        Image.fromarray(img_array).save(folder / f"{idx:04d}.png")
        counts[label] = idx + 1

    print("Saved:")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n} images")
