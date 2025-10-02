#!/usr/bin/env python3
"""
Annotate bounding boxes on an image loaded from a URL or file.

- Loads image from local disk or downloads from a URL
- Draws each bbox from JSON input or a YOLO label file
- Supports formats: corners [x1,y1,x2,y2], YOLO [xc,yc,w,h] (norm/abs), YOLO 5-tuple [cls,xc,yc,w,h], and Qwen-1000 corners [x1,y1,x2,y2] where coords are in 0..1000 regardless of resize
- Can optionally remap boxes created at a different source resolution

Dependencies: pillow, requests
    pip install pillow requests

Usage examples:
    # 1) Corners format [x1,y1,x2,y2]
    python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" \
      --data '{"cracks":[{"bbox_2d":[100,120,180,160]}]}' --bbox-format corners

    # 2) YOLO normalized [xc,yc,w,h] in [0,1]
    python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" \
      --data '{"cracks":[{"bbox_2d":[0.52,0.41,0.08,0.05]}]}' --bbox-format yolo_norm

    # 3) Qwen-1000 corners [x1,y1,x2,y2] with coordinates in 0..1000 (independent of resize)
    python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" \
      --data '{"cracks":[{"bbox_2d":[250,250,500,500]}]}' --bbox-format qwen1000

    # 4) Read a YOLO label file (one line per box: cls xc yc w h)
    python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" --labels "C:/path/to/img.txt" --bbox-format yolo_norm5

    # 5) If labels were generated on a resized image (e.g., 640x640), map to original size
    python annotate_bboxes_from_url.py --file img.jpg --labels img.txt --bbox-format yolo_norm5 --source-size 640x640

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

# --- Default inputs (edit these if you like) ---
DEFAULT_IMAGE_URL = (
    "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/"
    "Testpulje_small/Folder%202/DJI_0942.JPG"
)

DEFAULT_DATA = {
    "cracks": [
        {"bbox_2d": [352, 449, 392, 471], "description": "horizontal crack in brick wall near roofline"},
        {"bbox_2d": [357, 552, 419, 579], "description": "horizontal crack in brick wall, slightly wider and more pronounced"},
    ]
}


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def load_image_from_file(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _is_normalized(vals: List[float]) -> bool:
    return all(0.0 <= v <= 1.0 for v in vals)


def _yolo_to_corners(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int, normalized: bool) -> Tuple[int, int, int, int]:
    if normalized:
        xc *= img_w
        yc *= img_h
        w *= img_w
        h *= img_h
    x1 = int(round(xc - w / 2))
    y1 = int(round(yc - h / 2))
    x2 = int(round(xc + w / 2))
    y2 = int(round(yc + h / 2))
    # clamp
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2


def _qwen1000_to_corners(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    """Qwen-3 style: coords in [0,1000] regardless of internal resize.
    Convert to pixel corners for the current image size.
    """
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


def _apply_source_size_scale(nums: List[float], src_w: int, src_h: int, fmt: str) -> List[float]:
    """If bboxes were produced on a different source size, scale to 0..1 or pixels accordingly.
    For YOLO normalized, no change needed. For YOLO absolute: map from src->current done later by _yolo_to_corners via img size.
    For corners: scale coordinates from source to current later via separate path.
    Here we just return the same list; scaling is handled in draw_bboxes where we know img size.
    """
    return nums


def _read_yolo_label_file(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            items.append({"bbox_2d": [float(cls), float(xc), float(yc), float(w), float(h)], "description": "label"})
    return items

def draw_bboxes(img: Image.Image, items: List[Dict[str, Any]],
                box_color=(255, 0, 0), text_bg=(255, 255, 255), text_color=(0, 0, 0),
                box_width: int = 3,
                bbox_format: str = "auto",
                source_size: Tuple[int, int] | None = None) -> Image.Image:
    """
    bbox_format options:
      - "auto" (default)
      - "corners": [x1,y1,x2,y2]
      - "yolo_norm": [xc,yc,w,h] normalized (no class)
      - "yolo_abs": [xc,yc,w,h] in pixels (no class)
      - "yolo_norm5": [class,xc,yc,w,h] normalized
      - "yolo_abs5": [class,xc,yc,w,h] in pixels
      - "qwen1000": [x1,y1,x2,y2] with each coord in 0..1000 (independent of resize)
      - "qwen1000_5": [class,x1,y1,x2,y2] with coords in 0..1000
    """
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    img_w, img_h = img.size

    for idx, item in enumerate(items, start=1):
        bbox = item.get("bbox_2d") or item.get("bbox") or item.get("yolo")
        if not bbox:
            continue
        # to floats
        try:
            nums = [float(v) for v in bbox]
        except Exception:
            continue

        x1 = y1 = x2 = y2 = None
        fmt = (bbox_format or "auto").lower()

        # Optional rescaling if corners were authored at a different source size
        if fmt == "corners" and source_size is not None:
            src_w, src_h = source_size
            if len(nums) == 4 and src_w > 0 and src_h > 0:
                sx = img_w / float(src_w)
                sy = img_h / float(src_h)
                nums = [nums[0] * sx, nums[1] * sy, nums[2] * sx, nums[3] * sy]

        if fmt == "corners":
            if len(nums) != 4:
                continue
            x1, y1, x2, y2 = map(int, nums)
        elif fmt in ("yolo_norm", "yolo_abs", "yolo_norm5", "yolo_abs5"):
            normalized = fmt in ("yolo_norm", "yolo_norm5")
            if len(nums) == 5:
                xc, yc, w, h = nums[1], nums[2], nums[3], nums[4]
            elif len(nums) == 4:
                xc, yc, w, h = nums
            else:
                continue
            # If YOLO ABS produced at a different source size, scale to current image
            if not normalized and source_size is not None:
                src_w, src_h = source_size
                sx = img_w / float(src_w)
                sy = img_h / float(src_h)
                xc, yc, w, h = xc * sx, yc * sy, w * sx, h * sy
            x1, y1, x2, y2 = _yolo_to_corners(xc, yc, w, h, img_w, img_h, normalized)
        elif fmt in ("qwen1000", "qwen1000_5"):
            # Qwen 0..1000 corners
            if len(nums) == 5:
                x1, y1, x2, y2 = nums[1], nums[2], nums[3], nums[4]
            elif len(nums) == 4:
                x1, y1, x2, y2 = nums
            else:
                continue
            x1, y1, x2, y2 = _qwen1000_to_corners(x1, y1, x2, y2, img_w, img_h)
        else:  # auto
            if len(nums) == 5:
                last4 = nums[1:5]
                if _is_normalized(last4):  # YOLO normalized with class
                    x1, y1, x2, y2 = _yolo_to_corners(*last4, img_w, img_h, True)
                elif all(0.0 <= v <= 1000.0 for v in last4) and any(v > 1.0 for v in last4):  # Qwen-1000 with class
                    x1, y1, x2, y2 = _qwen1000_to_corners(*last4, img_w, img_h)
                else:
                    x1, y1, x2, y2 = _yolo_to_corners(*last4, img_w, img_h, False)
            elif len(nums) == 4:
                if _is_normalized(nums):  # YOLO normalized
                    x1, y1, x2, y2 = _yolo_to_corners(*nums, img_w, img_h, True)
                elif all(0.0 <= v <= 1000.0 for v in nums) and any(v > 1.0 for v in nums):  # Qwen-1000 corners
                    x1, y1, x2, y2 = _qwen1000_to_corners(*nums, img_w, img_h)
                else:
                    x1, y1, x2, y2 = map(int, nums)
            else:
                continue

        for offset in range(box_width):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=box_color)

        # Label
        desc = item.get("description") or "bbox"
        label_cls = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 5:
            try:
                label_cls = int(float(bbox[0]))
            except Exception:
                label_cls = None
        label = f"{idx}: {'cls='+str(label_cls)+' ' if label_cls is not None else ''}{desc}"
        tw, th = draw.textlength(label, font=font), font.size + 6
        lx = x1
        ly = max(0, y1 - th - 2)
        draw.rectangle([lx, ly, lx + tw + 8, ly + th], fill=text_bg)
        draw.text((lx + 4, ly + (th - font.size) // 2 - 1), label, fill=text_color, font=font)

    return img


def parse_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for v in data.values():
        if isinstance(v, list):
            for el in v:
                if isinstance(el, dict) and ("bbox_2d" in el or "bbox" in el or "yolo" in el):
                    items.append(el)
    return items


def main():
    p = argparse.ArgumentParser(description="Annotate bboxes on an image from URL or file")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--url", help="Image URL to download")
    group.add_argument("--file", help="Local image file path")
    p.add_argument("--labels", help="YOLO label file path (lines: cls xc yc w h)")
    p.add_argument("--data", default=None,
                   help="JSON string with bbox data. If omitted, uses DEFAULT_DATA in the script unless --labels is provided")
    p.add_argument("--out", default="annotated_output.jpg", help="Output image filename")
    p.add_argument("--bbox-format", choices=["auto", "corners", "yolo_norm", "yolo_abs", "yolo_norm5", "yolo_abs5", "qwen1000", "qwen1000_5"], default="auto",
                   help="Format of bbox lists in the JSON or labels. 'auto' tries to infer format.")
    p.add_argument("--source-size", default=None,
                   help="If bboxes were produced on a different resolution, give WxH (e.g., 640x640) so we can map to the current image size.")
    args = p.parse_args()

    # Load image
    try:
        if args.file:
            if not os.path.exists(args.file):
                sys.stderr.write(f"File does not exist: {args.file}\n")
                sys.exit(4)
            else:
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

    # Parse bbox input (labels take precedence over data)
    items: List[Dict[str, Any]] = []
    if args.labels:
        if not os.path.exists(args.labels):
            sys.stderr.write(f"Label file does not exist: {args.labels}\n")
            sys.exit(5)
        print(f"Reading YOLO labels from: {args.labels}")
        items = _read_yolo_label_file(args.labels)
        # If labels were read, default to yolo_norm5 unless user set format explicitly
        if args.bbox_format == "auto":
            args.bbox_format = "yolo_norm5"
    else:
        if args.data:
            try:
                data = json.loads(args.data)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Failed to parse --data JSON: {e}\n")
                sys.exit(2)
        else:
            data = DEFAULT_DATA
        items = parse_items(data)
        if not items:
            sys.stderr.write("No bbox items found in provided data. Expected objects with a bbox field.\n")
            sys.exit(1)

    # Parse source size if provided
    source_size = None
    if args.source_size:
        try:
            w_str, h_str = args.source_size.lower().split("x")
            source_size = (int(w_str), int(h_str))
            print(f"Mapping boxes from source size {source_size} to image size {img.size}")
        except Exception:
            sys.stderr.write("--source-size must be like 640x640\n")
            sys.exit(6)

    annotated = draw_bboxes(img, items, bbox_format=args.bbox_format, source_size=source_size)
    annotated.save(args.out, quality=95)
    print(f"Saved: {args.out} | size: {annotated.size[0]}x{annotated.size[1]}")


if __name__ == "__main__":
    main()
