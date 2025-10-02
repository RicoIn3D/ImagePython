#!/usr/bin/env python3
"""
Annotate bounding boxes on an image loaded from a URL or file.

- Downloads the image from a URL or loads it from local disk
- Draws each bbox from the provided data (x1, y1, x2, y2)
- Writes an annotated copy next to the script

Dependencies: pillow, requests
    pip install pillow requests

Usage examples:
    # (1) Corners format [x1,y1,x2,y2]
    python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" --data '{"cracks":[{"bbox_2d":[100,120,180,160]}]}' --bbox-format corners

    # (2) YOLO normalized [xc,yc,w,h] in [0,1]
    python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" --data '{"cracks"
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


def draw_bboxes(img: Image.Image, items: List[Dict[str, Any]],
                box_color=(255, 0, 0), text_bg=(255, 255, 255), text_color=(0, 0, 0),
                box_width: int = 3,
                bbox_format: str = "auto") -> Image.Image:
    """
    bbox_format options:
      - "auto" (default):
          * len==4 & all in [0,1] -> YOLO normalized [xc,yc,w,h]
          * len==5 & first not used as class if all in [0,1] for last 4 -> YOLO normalized with class
          * len==5 & last 4 >1 -> YOLO abs with class
          * else len==4 -> assume corners [x1,y1,x2,y2]
      - "corners": [x1,y1,x2,y2]
      - "yolo_norm": [xc,yc,w,h] normalized (no class)
      - "yolo_abs": [xc,yc,w,h] in pixels (no class)
      - "yolo_norm5": [class,xc,yc,w,h] normalized
      - "yolo_abs5": [class,xc,yc,w,h] in pixels
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
        # normalize list of numbers
        try:
            nums = [float(v) for v in bbox]
        except Exception:
            continue

        x1 = y1 = x2 = y2 = None
        fmt = bbox_format.lower() if bbox_format else "auto"

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
            x1, y1, x2, y2 = _yolo_to_corners(xc, yc, w, h, img_w, img_h, normalized)
        else:  # auto
            if len(nums) == 5:
                # Likely YOLO with class id
                last4 = nums[1:5]
                normalized = _is_normalized(last4)
                x1, y1, x2, y2 = _yolo_to_corners(*last4, img_w, img_h, normalized)
            elif len(nums) == 4:
                if _is_normalized(nums):
                    x1, y1, x2, y2 = _yolo_to_corners(*nums, img_w, img_h, True)
                else:
                    # Assume corners
                    x1, y1, x2, y2 = map(int, nums)
            else:
                continue
        for offset in range(box_width):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=box_color)

                # Compose label (include class if present)
        desc = item.get("description") or "bbox"
        label_cls = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 5:
            try:
                label_cls = int(float(bbox[0]))
            except Exception:
                label_cls = None
        if label_cls is not None:
            label = f"{idx}: cls={label_cls} {desc}"
        else:
            label = f"{idx}: {desc}"

        tw, th = draw.textlength(label, font=font), font.size + 6
        label_x = x1
        label_y = max(0, y1 - th - 2)

        draw.rectangle([label_x, label_y, label_x + tw + 8, label_y + th], fill=text_bg)
        draw.text((label_x + 4, label_y + (th - font.size) // 2 - 1), label, fill=text_color, font=font)

    return img


def parse_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for v in data.values():
        if isinstance(v, list):
            for el in v:
                if isinstance(el, dict) and "bbox_2d" in el:
                    items.append(el)
    return items


def main():
    p = argparse.ArgumentParser(description="Annotate bboxes on an image from URL or file")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--url", help="Image URL to download")
    group.add_argument("--file", help="Local image file path")
    p.add_argument("--data", default=None,
                   help="JSON string with bbox data. If omitted, uses DEFAULT_DATA in the script")
    p.add_argument("--out", default="annotated_output.jpg", help="Output image filename")
    p.add_argument("--bbox-format", choices=["auto", "corners", "yolo_norm", "yolo_abs", "yolo_norm5", "yolo_abs5"], default="auto",
                   help="Format of bbox lists in the JSON. 'auto' tries to infer format.")
    args = p.parse_args()

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
        sys.stderr.write("No bbox items found in provided data. Expected objects with a 'bbox_2d' field.\n")
        sys.exit(1)

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

    annotated = draw_bboxes(img, items, bbox_format=args.bbox_format)
    annotated.save(args.out, quality=95)
    print(f"Saved: {args.out} | size: {annotated.size[0]}x{annotated.size[1]}")


if __name__ == "__main__":
    main()
