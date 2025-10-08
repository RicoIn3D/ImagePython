#!/usr/bin/env python3
"""
Analyze drone images for structural defects using Ollama vision models.

Usage:
  python analyze_drone_image_yolo.py --url "https://example.com/image.jpg"
  python analyze_drone_image_yolo.py --url "https://example.com/image.jpg" --model "qwen2.5vl:latest"
"""

import requests
import base64
import json
import os
import io
import argparse
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image, ImageOps, ImageDraw, ImageFont
from hashlib import md5

def exif_safe_open_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img

def compute_md5_bytes(image_bytes) -> str:
    return md5(image_bytes).hexdigest()

def yolo_to_px(xc, yc, w, h, W, H):
    Xc, Yc = xc*W, yc*H
    Bw, Bh = w*W, h*H
    x1 = int(round(max(0, Xc - Bw/2)))
    y1 = int(round(max(0, Yc - Bh/2)))
    x2 = int(round(min(W-1, Xc + Bw/2)))
    y2 = int(round(min(H-1, Yc + Bh/2)))
    return x1, y1, x2, y2

def save_debug_grid_overlay(img: Image.Image, cracks: list, out_path: str):
    """Draw a 0..1 grid, mark YOLO centers & boxes from cracks, save to out_path."""
    W, H = img.size
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # grid every 0.1
    for t in [i/10 for i in range(1,10)]:
        x = int(round(t*W)); y = int(round(t*H))
        draw.line([(x,0),(x,H-1)], width=1, fill=(255,255,255))
        draw.line([(0,y),(W-1,y)], width=1, fill=(255,255,255))
        if t in (0.1,0.3,0.5,0.7,0.9):
            draw.text((x+4, 4), f"x={t:.1f}", fill=(0,0,0), font=font)
            draw.text((4, y+4), f"y={t:.1f}", fill=(0,0,0), font=font)

    # boxes from cracks (YOLO [cls,xc,yc,w,h])
    for i,c in enumerate(cracks, start=1):
        bbox = c.get("bbox_2d", [])
        if len(bbox) >= 5:
            _, xc, yc, w, h = map(float, bbox[:5])
            x1,y1,x2,y2 = yolo_to_px(xc,yc,w,h,W,H)
            # rect
            for o in range(3):
                draw.rectangle([x1-o,y1-o,x2+o,y2+o], outline=(255,0,0))
            # center crosshair
            Xc,Yc = int(round(xc*W)), int(round(yc*H))
            draw.line([(Xc-10,Yc),(Xc+10,Yc)], fill=(0,255,0), width=2)
            draw.line([(Xc,Yc-10),(Xc,Yc+10)], fill=(0,255,0), width=2)
            # label
            label = c.get("description","bbox")
            tw = draw.textlength(label, font=font)
            draw.rectangle([x1, max(0,y1-22), x1+int(tw)+8, max(22,y1)], fill=(255,255,255))
            draw.text((x1+4, max(0,y1-20)), label, fill=(0,0,0), font=font)

    im.save(out_path, quality=95)

def get_filename_from_url(url: str) -> str:
    """Extract filename from URL without extension."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    return os.path.splitext(filename)[0]

def create_results_folder() -> str:
    """Create results subfolder if it doesn't exist."""
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

def save_yolo_classes(output_path: str) -> None:
    """Save YOLO classes.txt file with class names."""
    classes = [
        "crack",
        "spalling",
        "mortar_erosion",
        "water_damage",
        "displacement",
        "efflorescence",
        "hole",
        "deformation"
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(classes))
        f.write("\n")
    
    print(f"✓ Saved class definitions to: {output_path}")

def save_yolo_labels_simple(cracks: list, output_path: str) -> None:
    """Save cracks in YOLO format - expects [class_id, xc, yc, w, h]."""
    lines = []
    for crack in cracks:
        bbox = crack.get("bbox_2d", [])
        if len(bbox) >= 5:
            class_id = int(bbox[0])
            x_center = float(bbox[1])
            y_center = float(bbox[2])
            width = float(bbox[3])
            height = float(bbox[4])
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    
    print(f"✓ Saved {len(lines)} crack(s) to: {output_path}")

def convert_qwen1000_to_yolo(bbox_qwen: list, class_id: int = 0) -> list:
    """Convert Qwen-1000 format [x1,y1,x2,y2] to YOLO [class,xc,yc,w,h]."""
    if len(bbox_qwen) != 4:
        return None
    
    x1, y1, x2, y2 = bbox_qwen
    
    # Convert from 0-1000 range to 0-1 range
    x1 = x1 / 1000.0
    y1 = y1 / 1000.0
    x2 = x2 / 1000.0
    y2 = y2 / 1000.0
    
    # Convert corners to center + dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    return [class_id, x_center, y_center, width, height]

def save_yolo_labels_auto(cracks: list, output_path: str, format_type: str) -> None:
    """Save cracks in YOLO format, auto-converting from Qwen-1000 if needed."""
    lines = []
    
    for crack in cracks:
        bbox = crack.get("bbox_2d", [])
        
        if not bbox:
            continue
        
        # Detect format and convert if necessary
        if format_type == "Qwen-1000" and len(bbox) == 4:
            # Convert Qwen-1000 to YOLO
            yolo_bbox = convert_qwen1000_to_yolo(bbox, class_id=0)
            if yolo_bbox:
                class_id, x_center, y_center, width, height = yolo_bbox
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif len(bbox) >= 5:
            # Already YOLO format
            class_id = int(bbox[0])
            x_center = float(bbox[1])
            y_center = float(bbox[2])
            width = float(bbox[3])
            height = float(bbox[4])
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    
    print(f"✓ Saved {len(lines)} crack(s) to: {output_path} (YOLO format)")


def analyze_drone_image(image_url: str, model_name: str = "llava:13b"):
    """Analyze a drone image from the given URL."""
    # Create results folder
    results_folder = create_results_folder()
    base_filename = get_filename_from_url(image_url)
    
    print("Downloading image...")
    # Download and convert to base64
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_data = response.content
        img = exif_safe_open_bytes(image_data)
        W, H = img.size
        digest = compute_md5_bytes(image_data)

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"✓ Image downloaded successfully ({len(image_base64)} bytes)")
        
        # Save image to results folder
        image_path = os.path.join(results_folder, f"{base_filename}.JPG")
        with open(image_path, "wb") as f:
            f.write(image_data)
        print(f"✓ Image saved to: {image_path}")
        
    except Exception as e:
        print(f"✗ Failed to download image: {e}")
        return
    
    # Prepare Ollama API request
    ollama_url = "http://192.168.87.207:11434/api/chat"
    
    # Focused prompt: Find brick wall FIRST, then find cracks
    prompt = (
        "TASK: Find cracks and defects in the brick wall of this building.\n\n"
        
        "STEP 1 - IDENTIFY THE BRICK WALL:\n"
        "Look at the entire image carefully.\n"
        "The brick wall is the reddish-brown masonry surface with visible bricks and mortar.\n"
        "It may be anywhere in the image - left, right, center, upper, or lower areas.\n"
        "Identify the EXACT region where you see the brick wall.\n"
        "IGNORE: sky, clouds, trees, grass, roads, other buildings in background.\n\n"
        
        "STEP 2 - FIND DEFECTS ON THE BRICK WALL:\n"
        "Now look ONLY at the brick wall area you identified.\n"
        "Find these types of defects:\n"
        "- Cracks in bricks or mortar joints\n"
        "- Mortar erosion (gaps in white mortar between bricks)\n"
        "- Spalled or damaged bricks\n"
        "- Color variations indicating damage\n\n"
        
        "STEP 3 - CREATE BOUNDING BOXES:\n"
        "For each defect found, create ONE small bounding box.\n"
        "The box should tightly fit around the defect.\n\n"
        
        "FORMAT: [class_id, x_center, y_center, width, height]\n"
        "- class_id: Always 0\n"
        "- x_center, y_center: Center of box (0.0-1.0, where 0,0 is top-left corner)\n"
        "- width, height: Size of box (0.02-0.10 for small defects)\n\n"
        
        "IMPORTANT VALIDATION:\n"
        "Before adding each box, ask yourself:\n"
        "1. Is this box ON the brick wall surface? (not on sky/background)\n"
        "2. Is there a visible defect at this location?\n"
        "3. Is the box small and tight around the defect?\n\n"
      
        
        "Find 3-12 defects if visible. Return ONLY valid JSON. No extra text."
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a vision AI analyzing building images. Follow the format instructions EXACTLY. Each bbox_2d must have exactly 5 numbers."
            },
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    
    print(f"\nSending request to Ollama ({model_name})...")
    print("This may take 30-60 seconds depending on your hardware...\n")
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        
        # Extract content
        content = result.get("message", {}).get("content", "")
        
        print("=" * 70)
        print("ANALYSIS RESULT")
        print("=" * 70)
        print(content)
        
        # Try to parse and pretty-print JSON
        try:
            findings = json.loads(content)
            print("\n" + "=" * 70)
            print("FORMATTED FINDINGS")
            print("=" * 70)
            print(json.dumps(findings, indent=2))
            
            # Display summary and save YOLO labels
            if isinstance(findings, dict):
                cracks = findings.get("cracks", []) or findings.get("boxes", [])
                if cracks:
                    print(f"\n✓ Found {len(cracks)} crack(s)")
                    findings["_meta"] = {
                        "source_image": os.path.basename(image_path),
                        "image_size": [W, H],
                        "md5": digest,
                        "exif_transposed": True
                    }
                    # Save in YOLO format (simple version)
                    output_file = os.path.join(results_folder, f"{base_filename}.txt")
                    save_yolo_labels_simple(cracks, output_file)
                    
                    # Save YOLO classes file
                    classes_file = os.path.join(results_folder, "classes.txt")
                    save_yolo_classes(classes_file)
                    
                    # Also save JSON for reference
                    json_output_file = os.path.join(results_folder, f"{base_filename}_analysis.json")
                    with open(json_output_file, "w", encoding="utf-8") as f:
                        json.dump(findings, f, indent=2)
                    # Save a debug grid overlay next to the JSON and record its path
                    dbg_path = os.path.join(results_folder, f"{base_filename}_debug_grid.jpg")
                    save_debug_grid_overlay(img, cracks, dbg_path)
                    findings["_meta"]["debug_grid"] = os.path.basename(dbg_path)
                    with open(json_output_file, "w", encoding="utf-8") as f:
                        json.dump(findings, f, indent=2)

                    print(f"✓ Saved debug overlay to: {dbg_path}")
                    print(f"✓ Saved detailed analysis to: {json_output_file}")
                    
                    print(f"\n📁 All files saved to: {results_folder}/")
                    print("\n" + "=" * 70)
                    print("NEXT STEP: Annotate the image with bounding boxes")
                    print("=" * 70)
                    print(f"Run: python annotate_bboxes_from_url.py \\")
                    print(f"       --file \"{image_path}\" \\")
                    print(f"       --json-file \"{json_output_file}\" \\")
                    print(f"       --out \"annotated_{base_filename}.jpg\" \\")
                    print(f"       --export-yolo \"{base_filename}_yolo.txt\"")
                else:
                    print("\n✓ No cracks detected")
                
                if "findings" in findings:
                    print(f"✓ Found {len(findings['findings'])} issue(s)")
                if "overall_assessment" in findings:
                    print(f"✓ Assessment: {findings['overall_assessment']}")
                    
        except json.JSONDecodeError:
            print("\n(Note: Response is not in JSON format)")
            print("Cannot save YOLO labels - invalid JSON response")
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out. The image might be too large or Ollama is slow.")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Make sure it's running on http://192.168.87.207:11434")
        print("  Run: ollama serve")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze drone images for structural defects")
    parser.add_argument(
        "--url", 
        required=True,
        help="Image URL to analyze"
    )
    parser.add_argument(
        "--model",
        default="llava:13b",
        help="Ollama model to use (default: llava:13b, try: qwen2.5vl:latest)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"DRONE IMAGE ANALYSIS - {os.path.basename(urlparse(args.url).path)}")
    print("=" * 70)
    print()
    
    # Check if Ollama is running
    try:
        health_check = requests.get("http://192.168.87.207:11434/api/tags", timeout=5)
        if health_check.status_code == 200:
            models = health_check.json().get("models", [])
            model_names = [m.get("name") for m in models]
            print(f"✓ Ollama is running")
            print(f"✓ Available models: {', '.join(model_names)}")
            
            # Check if requested model is available
            if not any(args.model in name for name in model_names):
                print(f"\n⚠ WARNING: {args.model} not found!")
                print(f"  Install it with: ollama pull {args.model}")
                response = input("\nContinue anyway? (y/n): ")
                if response.lower() != 'y':
                    exit(0)
            print()
    except:
        print("✗ Cannot connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        print("  And the model is installed")
        exit(1)
    
    analyze_drone_image(args.url, args.model)