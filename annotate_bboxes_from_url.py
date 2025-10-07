#!/usr/bin/env python3
"""
Annotate bounding boxes on an image from multiple formats (Qwen-1000, YOLO, JSON).

Supported formats:
  - Qwen-1000: [x1, y1, x2, y2] in range 0..1000 (corners)
  - YOLO: [xc, yc, w, h] normalized 0..1 (center + dimensions)
  - JSON: {"boxes":[{"bbox_2d":[...], "description":"..."}]}

Dependencies: pillow, requests
    pip install pillow requests

Usage examples:
  # From Qwen-1000 label file
  python annotate_bboxes_multi_format.py --file "img.jpg" --labels-qwen "labels.qwen.txt"

  # From YOLO label file
  python annotate_bboxes_multi_format.py --file "img.jpg" --labels-yolo "labels.yolo.txt"

  # Convert YOLO -> Qwen-1000
  python annotate_bboxes_multi_format.py --file "img.jpg" --labels-yolo "labels.yolo.txt" --export-qwen "output.qwen.txt"

  # Convert Qwen-1000 -> YOLO
  python annotate_bboxes_multi_format.py --file "img.jpg" --labels-qwen "labels.qwen.txt" --export-yolo "output.yolo.txt"
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

# --- Defaults ---
DEFAULT_IMAGE_URL = (
    "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/"
    "Testpulje_small/Folder%202/DJI_0942.JPG"
)

DEFAULT_DATA = {
    "boxes": [
         {
      "bbox_2d": [
        0,
        0.371,
        0.457,
        0.035,
        0.018
      ],
      "description": "slight vertical mortar separation"
    },
    {
      "bbox_2d": [
        0,
        0.356,
        0.568,
        0.076,
        0.026
      ],
      "description": "horizontal hairline crack in mortar joint"
    }
    ]
}

# ---------------- Image Loading ----------------

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def load_image_from_file(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------- Coordinate Conversions ----------------

def qwen1000_corners_to_pixels(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
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


def yolo_norm_to_pixels(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized [xc, yc, w, h] (0..1) -> pixel corners [x1, y1, x2, y2]."""
    X_center = xc * img_w
    Y_center = yc * img_h
    W = w * img_w
    H = h * img_h
    
    X1 = int(round(X_center - W / 2.0))
    Y1 = int(round(Y_center - H / 2.0))
    X2 = int(round(X_center + W / 2.0))
    Y2 = int(round(Y_center + H / 2.0))
    
    X1 = max(0, min(img_w - 1, X1))
    Y1 = max(0, min(img_h - 1, Y1))
    X2 = max(0, min(img_w - 1, X2))
    Y2 = max(0, min(img_h - 1, Y2))
    return X1, Y1, X2, Y2


def qwen1000_to_yolo_norm(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert Qwen-1000 corners (0..1000) -> YOLO-normalized [xc, yc, w, h] in 0..1."""
    xc = (x1 + x2) / 2.0 / 1000.0
    yc = (y1 + y2) / 2.0 / 1000.0
    w = (x2 - x1) / 1000.0
    h = (y2 - y1) / 1000.0
    return xc, yc, w, h


def yolo_norm_to_qwen1000(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert YOLO-normalized [xc, yc, w, h] (0..1) -> Qwen-1000 corners (0..1000)."""
    x1 = (xc - w / 2.0) * 1000.0
    y1 = (yc - h / 2.0) * 1000.0
    x2 = (xc + w / 2.0) * 1000.0
    y2 = (yc + h / 2.0) * 1000.0
    return x1, y1, x2, y2


# ---------------- Label File Readers ----------------

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
                "description": "qwen_label",
                "format": "qwen"
            })
    return out


def read_yolo_labels(path: str) -> List[Dict[str, Any]]:
    """Read YOLO labels: each line: cls xc yc w h (all normalized 0..1)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            out.append({
                "yolo": [float(xc), float(yc), float(w), float(h)],
                "class": int(float(cls)),
                "description": "yolo_label",
                "format": "yolo"
            })
    return out


def parse_items_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find objects containing a 'bbox_2d' with Qwen-1000 corners."""
    items: List[Dict[str, Any]] = []
    for k, v in data.items():
        if isinstance(v, list):
            for el in v:
                if isinstance(el, dict) and "bbox_2d" in el and isinstance(el["bbox_2d"], (list, tuple)) and len(el["bbox_2d"]) >= 4:
                    el["format"] = "qwen"
                    items.append(el)
    return items


# ---------------- Drawing ----------------

def draw_boxes(img: Image.Image, items: List[Dict[str, Any]], box_width: int = 3,
               box_color=(255, 0, 0), text_bg=(255, 255, 255), text_color=(0, 0, 0)) -> Image.Image:
    """Draw boxes from items (supports both Qwen-1000 and YOLO formats)."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    img_w, img_h = img.size

    for idx, item in enumerate(items, start=1):
        fmt = item.get("format", "qwen")
        
        if fmt == "yolo":
            yolo_data = item.get("yolo")
            if not yolo_data or len(yolo_data) != 4:
                continue
            xc, yc, w, h = map(float, yolo_data)
            X1, Y1, X2, Y2 = yolo_norm_to_pixels(xc, yc, w, h, img_w, img_h)
            cls = item.get("class", 0)
        else:  # qwen format
            arr = item.get("bbox_2d")
            if not arr or len(arr) < 4:
                continue
            if len(arr) >= 5:
                x1, y1, x2, y2 = map(float, arr[1:5])
                cls = int(float(arr[0]))
            else:
                x1, y1, x2, y2 = map(float, arr[0:4])
                cls = item.get("class")
            X1, Y1, X2, Y2 = qwen1000_corners_to_pixels(x1, y1, x2, y2, img_w, img_h)

        # Draw box
        for o in range(box_width):
            draw.rectangle([X1 - o, Y1 - o, X2 + o, Y2 + o], outline=box_color)

        # Draw label
        desc = item.get("description") or "bbox"
        label = f"{idx}: {desc}"
        if cls is not None:
            label = f"{idx}: cls={cls} {desc}"
        
        tw, th = draw.textlength(label, font=font), font.size + 6
        lx, ly = X1, max(0, Y1 - th - 2)
        draw.rectangle([lx, ly, lx + tw + 8, ly + th], fill=text_bg)
        draw.text((lx + 4, ly + (th - font.size) // 2 - 1), label, fill=text_color, font=font)

    return img


# ---------------- Export Functions ----------------

def export_yolo(items: List[Dict[str, Any]], out_path: str) -> None:
    """Write YOLO-normalized labels (cls xc yc w h)."""
    lines: List[str] = []
    for it in items:
        fmt = it.get("format", "qwen")
        cls = int(it.get("class", 0))
        
        if fmt == "yolo":
            yolo_data = it.get("yolo")
            if not yolo_data or len(yolo_data) != 4:
                continue
            xc, yc, w, h = map(float, yolo_data)
        else:  # qwen format
            arr = it.get("bbox_2d")
            if not arr or len(arr) < 4:
                continue
            if len(arr) >= 5:
                x1, y1, x2, y2 = map(float, arr[1:5])
            else:
                x1, y1, x2, y2 = map(float, arr[0:4])
            xc, yc, w, h = qwen1000_to_yolo_norm(x1, y1, x2, y2)
        
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def export_qwen(items: List[Dict[str, Any]], out_path: str) -> None:
    """Write Qwen-1000 labels (cls x1 y1 x2 y2)."""
    lines: List[str] = []
    for it in items:
        fmt = it.get("format", "qwen")
        cls = int(it.get("class", 0))
        
        if fmt == "yolo":
            yolo_data = it.get("yolo")
            if not yolo_data or len(yolo_data) != 4:
                continue
            xc, yc, w, h = map(float, yolo_data)
            x1, y1, x2, y2 = yolo_norm_to_qwen1000(xc, yc, w, h)
        else:  # qwen format
            arr = it.get("bbox_2d")
            if not arr or len(arr) < 4:
                continue
            if len(arr) >= 5:
                x1, y1, x2, y2 = map(float, arr[1:5])
            else:
                x1, y1, x2, y2 = map(float, arr[0:4])
        
        lines.append(f"{cls} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser(description="Annotate bounding boxes from Qwen-1000, YOLO, or JSON formats")
    
    # Image source
    group = p.add_mutually_exclusive_group()
    group.add_argument("--url", help="Image URL to download")
    group.add_argument("--file", help="Local image file path")
    
    # Label source (mutually exclusive)
    label_group = p.add_mutually_exclusive_group()
    label_group.add_argument("--labels-qwen", help="Qwen-1000 label file (cls x1 y1 x2 y2; coords in 0..1000)")
    label_group.add_argument("--labels-yolo", help="YOLO label file (cls xc yc w h; normalized 0..1)")
    label_group.add_argument("--data", help="JSON with objects containing 'bbox_2d' in Qwen-1000 format")
    
    # Export options
    p.add_argument("--export-yolo", help="Save YOLO-normalized labels to this file")
    p.add_argument("--export-qwen", help="Save Qwen-1000 labels to this file")
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

    # Get items
    items: List[Dict[str, Any]] = []
    if args.labels_qwen:
        if not os.path.exists(args.labels_qwen):
            sys.stderr.write(f"Label file does not exist: {args.labels_qwen}\n")
            sys.exit(5)
        print(f"Reading Qwen-1000 labels from: {args.labels_qwen}")
        items = read_qwen_labels(args.labels_qwen)
    elif args.labels_yolo:
        if not os.path.exists(args.labels_yolo):
            sys.stderr.write(f"Label file does not exist: {args.labels_yolo}\n")
            sys.exit(5)
        print(f"Reading YOLO labels from: {args.labels_yolo}")
        items = read_yolo_labels(args.labels_yolo)
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
            sys.stderr.write("No bbox items found.\n")
            sys.exit(1)

    print(f"Found {len(items)} bounding boxes")

    # Draw & save
    annotated = draw_boxes(img, items)
    annotated.save(args.out, quality=95)
    print(f"Saved annotated image: {args.out} | size: {annotated.size[0]}x{annotated.size[1]}")

    # Export conversions
    if args.export_yolo:
        export_yolo(items, args.export_yolo)
        print(f"Exported YOLO-normalized labels -> {args.export_yolo}")
    
    if args.export_qwen:
        export_qwen(items, args.export_qwen)
        print(f"Exported Qwen-1000 labels -> {args.export_qwen}")


if __name__ == "__main__":
    main()