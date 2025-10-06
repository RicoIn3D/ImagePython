#!/usr/bin/env python3
"""
Annotate Qwen-1000 bounding boxes on an image, with an option to export YOLO-normalized labels.

Qwen-1000 format (used by Qwen3-VL examples):
  - Boxes are CORNERS: [x1, y1, x2, y2]
  - All coordinates are in the fixed range 0..1000 and DO NOT depend on resize
  - Optional class: [class, x1, y1, x2, y2]

What this script does
  ✓ Load an image from --file or --url
  ✓ Read boxes from --data (JSON) or --labels-qwen (text file; one line per box)
  ✓ Draw the boxes on the image
  ✓ Optional: convert Qwen-1000 boxes to YOLO-normalized [xc, yc, w, h] and
    print them or save to a file via --export-yolo

Dependencies: pillow, requests
    pip install pillow requests

Usage examples
  # 1) From JSON (coords in 0..1000)
  python annotate_bboxes_from_url.py --file "C:/img.jpg" \
    --data '{"boxes":[{"bbox_2d":[250,250,500,500],"description":"crack"}]}'

  # 2) From a Qwen label file (each line: cls x1 y1 x2 y2  # all 0..1000)
  python annotate_bboxes_from_url.py --file "C:/img.jpg" \
    --labels-qwen "C:/img.qwen.txt"

  # 3) Export YOLO-normalized labels (xc,yc,w,h in 0..1) to a file
  python annotate_bboxes_from_url.py --file "C:/img.jpg" \
    --labels-qwen "C:/img.qwen.txt" --export-yolo "C:/img.yolo.txt"

Note: On Windows, wrap paths in quotes.
"""

from __future__ import annotations
import argparse
import io
import json
import sys
import os
from typing import List, Dict, Any, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

# --- Defaults (edit if you like) ---
DEFAULT_IMAGE_URL = (
    "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/"
    "Testpulje_small/Folder%202/DJI_0942.JPG"
)

# Minimal demo structure if --data/--labels-qwen not provided
DEFAULT_DATA = {
    "boxes": [
        {
      "bbox_2d": [356, 555, 432, 581],
      "description": "horizontal hairline crack in mortar joint near center of gable"
    },
    {
      "bbox_2d": [275, 466, 308, 480],
      "description": "faint horizontal mortar erosion near upper left gable"
    },
    {
      "bbox_2d": [354, 448, 389, 466],
      "description": "slight vertical mortar separation near upper center"
    },
    {
      "bbox_2d": [176, 542, 206, 557],
      "description": "minor horizontal mortar gap near lower left gable"
    },
    {
      "bbox_2d": [527, 322, 548, 354],
      "description": "circular hole in brickwork near apex, possible vent or defect"
    },
    {
      "bbox_2d": [730, 33, 757, 60],
      "description": "spalled brick at gable apex, edge damage visible"
    },
    {
      "bbox_2d": [285, 407, 320, 422],
      "description": "thin horizontal mortar line irregularity near upper left"
    },
    {
      "bbox_2d": [511, 675, 747, 698],
      "description": "horizontal mortar joint erosion at roofline flashing interface"
    }
    ]
}

# ---------------- core helpers ----------------

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def load_image_from_file(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def qwen1000_corners_to_pixels(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    """Map Qwen-1000 corners (0..1000) -> pixel corners for image size."""
    sx, sy = img_w / 1000.0, img_h / 1000.0
    X1 = int(round(x1 * sx))
    Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx))
    Y2 = int(round(y2 * sy))
    X1 = max(0, min(img_w - 1, X1))
    Y1 = max(0, min(img_h - 1, Y1))
    X2 = max(0, min(img_w - 1, X2))
    Y2 = max(0, min(img_h - 1, Y2))
    return X1, Y1, X2, Y2


def qwen1000_to_yolo_norm(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert Qwen-1000 corners (0..1000) -> YOLO-normalized [xc,yc,w,h] in 0..1."""
    xc = (x1 + x2) / 2.0 / 1000.0
    yc = (y1 + y2) / 2.0 / 1000.0
    w  = (x2 - x1) / 1000.0
    h  = (y2 - y1) / 1000.0
    return xc, yc, w, h


def read_qwen_labels(path: str) -> List[Dict[str, Any]]:
    """Read Qwen-1000 labels: each line: cls x1 y1 x2 y2 (all 0..1000)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x1, y1, x2, y2 = parts
            out.append({
                "bbox_2d": [float(x1), float(y1), float(x2), float(y2)],
                "class": int(float(cls)),
                "description": "label"
            })
    return out


def parse_items_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find objects containing a 'bbox_2d' with Qwen-1000 corners."""
    items: List[Dict[str, Any]] = []
    for k, v in data.items():
        if isinstance(v, list):
            for el in v:
                if isinstance(el, dict) and "bbox_2d" in el and isinstance(el["bbox_2d"], (list, tuple)) and len(el["bbox_2d"]) >= 4:
                    items.append(el)
    return items


def draw_qwen_boxes(img: Image.Image, items: List[Dict[str, Any]], box_width: int = 3,
                    box_color=(255, 0, 0), text_bg=(255, 255, 255), text_color=(0, 0, 0)) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    img_w, img_h = img.size

    for idx, item in enumerate(items, start=1):
        arr = item.get("bbox_2d")
        if not arr or len(arr) < 4:
            continue
        # accept either [x1,y1,x2,y2] or [cls,x1,y1,x2,y2]
        if len(arr) >= 5:
            x1, y1, x2, y2 = map(float, arr[1:5])
            cls = int(float(arr[0])) if isinstance(arr[0], (int, float, str)) else None
        else:
            x1, y1, x2, y2 = map(float, arr[0:4])
            cls = item.get("class")

        X1, Y1, X2, Y2 = qwen1000_corners_to_pixels(x1, y1, x2, y2, img_w, img_h)
        for o in range(box_width):
            draw.rectangle([X1 - o, Y1 - o, X2 + o, Y2 + o], outline=box_color)

        desc = item.get("description") or "bbox"
        label = f"{idx}: {desc}"
        if cls is not None:
            label = f"{idx}: cls={cls} {desc}"
        tw, th = draw.textlength(label, font=font), font.size + 6
        lx, ly = X1, max(0, Y1 - th - 2)
        draw.rectangle([lx, ly, lx + tw + 8, ly + th], fill=text_bg)
        draw.text((lx + 4, ly + (th - font.size) // 2 - 1), label, fill=text_color, font=font)

    return img


def export_yolo(items: List[Dict[str, Any]], out_path: str) -> None:
    """Write YOLO-normalized labels (xc yc w h), including class if available.
    If class is missing, 0 is used.
    """
    lines: List[str] = []
    for it in items:
        arr = it.get("bbox_2d")
        if not arr or len(arr) < 4:
            continue
        if len(arr) >= 5:
            x1, y1, x2, y2 = map(float, arr[1:5])
            cls = int(float(arr[0]))
        else:
            x1, y1, x2, y2 = map(float, arr[0:4])
            cls = int(it.get("class", 0))
        xc, yc, w, h = qwen1000_to_yolo_norm(x1, y1, x2, y2)
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser(description="Annotate Qwen-1000 boxes; optional YOLO export")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--url", help="Image URL to download")
    group.add_argument("--file", help="Local image file path")
    p.add_argument("--labels-qwen", help="Qwen-1000 label file (each line: cls x1 y1 x2 y2; coords in 0..1000)")
    p.add_argument("--data", help="JSON with objects containing 'bbox_2d' in Qwen-1000 format")
    p.add_argument("--export-yolo", help="Save YOLO-normalized labels to this file")
    p.add_argument("--out", default="annotated_output.jpg", help="Output annotated image filename")
    args = p.parse_args()

    # Load image
    try:
        if args.file:
            if not os.path.exists(args.file):
                sys.stderr.write(f"File does not exist: {args.file}\n")
                sys.exit(4)
            print(f"Loading local file: {args.file}")
            img = load_image_from_file(args.file)
        elif args.url:
            print(f"Downloading image from URL: {args.url}")
            img = load_image_from_url(args.url)
        else:
            print(f"Downloading default image from URL: {DEFAULT_IMAGE_URL}")
            img = load_image_from_url(DEFAULT_IMAGE_URL)
    except Exception as e:
        sys.stderr.write(f"Failed to load image: {e}\n")
        sys.exit(3)

    # Get items (labels take precedence)
    items: List[Dict[str, Any]] = []
    if args.labels_qwen:
        if not os.path.exists(args.labels_qwen):
            sys.stderr.write(f"Label file does not exist: {args.labels_qwen}\n")
            sys.exit(5)
        print(f"Reading Qwen labels from: {args.labels_qwen}")
        items = read_qwen_labels(args.labels_qwen)
    else:
        if args.data:
            try:
                data = json.loads(args.data)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Failed to parse --data JSON: {e}\n")
                sys.exit(2)
        else:
            data = DEFAULT_DATA
        items = parse_items_from_json(data)
        if not items:
            sys.stderr.write("No bbox items found. Expected Qwen-1000 'bbox_2d' lists.\n")
            sys.exit(1)

    # Draw & save
    annotated = draw_qwen_boxes(img, items)
    annotated.save(args.out, quality=95)
    print(f"Saved: {args.out} | size: {annotated.size[0]}x{annotated.size[1]}")

    # Optional YOLO export
    if args.export_yolo:
        export_yolo(items, args.export_yolo)
        print(f"Exported YOLO-normalized labels -> {args.export_yolo}")


if __name__ == "__main__":
    main()
