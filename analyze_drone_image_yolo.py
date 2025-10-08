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
import argparse
from pathlib import Path
from urllib.parse import urlparse

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

def get_prompt_for_model(model_name: str) -> tuple[str, str]:
    """Get appropriate prompt based on model type. Returns (user_prompt, format_type)."""
    
    is_qwen = "qwen" in model_name.lower()
    
    if is_qwen:
        # Qwen-1000 format: [x1, y1, x2, y2] in 0..1000 range
        format_description = (
            "OUTPUT FORMAT - Qwen-1000 corners format:\n"
            "\"bbox_2d\": [x1, y1, x2, y2]\n\n"
            
            "Where ALL values are in the range 0-1000 (NOT 0-1):\n"
            "- x1: left edge X coordinate (0 = left edge, 1000 = right edge)\n"
            "- y1: top edge Y coordinate (0 = TOP edge, 1000 = BOTTOM edge)\n"
            "- x2: right edge X coordinate (0 = left edge, 1000 = right edge)\n"
            "- y2: bottom edge Y coordinate (0 = TOP edge, 1000 = BOTTOM edge)\n\n"
            
            "CRITICAL: Values are 0-1000, independent of actual image size.\n"
            "A box from 20% to 30% horizontally and 35% to 40% vertically would be:\n"
            "[200, 350, 300, 400]\n\n"
        )
        
        example = (
            "REALISTIC EXAMPLE - defects on brick wall in middle of image (Qwen-1000 format):\n"
            "{\n"
            "  \"cracks\": [\n"
            "    {\"bbox_2d\": [320, 350, 400, 380], \"description\": \"horizontal crack in mortar - values are 0-1000\"},\n"
            "    {\"bbox_2d\": [450, 380, 500, 400], \"description\": \"vertical crack in brick - x1,y1 is top-left, x2,y2 is bottom-right\"},\n"
            "    {\"bbox_2d\": [250, 320, 290, 340], \"description\": \"mortar erosion - all coordinates in 0-1000 range\"}\n"
            "  ]\n"
            "}\n\n"
        )
    else:
        # YOLO format: [class_id, x_center, y_center, width, height] in 0..1 range
        format_description = (
            "OUTPUT FORMAT - YOLO normalized format:\n"
            "\"bbox_2d\": [class_id, x_center, y_center, width, height]\n\n"
            
            "Where ALL values are normalized to 0.0-1.0 range:\n"
            "- class_id: Always 0 for cracks\n"
            "- x_center: horizontal center (0.0=left, 1.0=right)\n"
            "- y_center: vertical center measured FROM TOP (0.0=top, 1.0=bottom)\n"
            "- width: box width as fraction (typically 0.02-0.10)\n"
            "- height: box height as fraction (typically 0.02-0.10)\n\n"
            
            "CRITICAL: All coordinates are 0.0-1.0, representing fractions of image dimensions.\n\n"
        )
        
        example = (
            "REALISTIC EXAMPLE - defects on brick wall in middle of image (YOLO format):\n"
            "{\n"
            "  \"cracks\": [\n"
            "    {\"bbox_2d\": [0, 0.35, 0.38, 0.08, 0.03], \"description\": \"horizontal crack - center at 35%,38% with 8%x3% size\"},\n"
            "    {\"bbox_2d\": [0, 0.50, 0.42, 0.05, 0.02], \"description\": \"vertical crack - all values are 0-1 fractions\"},\n"
            "    {\"bbox_2d\": [0, 0.28, 0.35, 0.04, 0.02], \"description\": \"mortar erosion - class_id=0, then center_x, center_y, width, height\"}\n"
            "  ]\n"
            "}\n\n"
        )
    
    common_instructions = (
        "You are inspecting a BRICK WALL for structural defects. Focus ONLY on the brick/masonry surfaces.\n\n"
        
        "IGNORE these areas:\n"
        "- Sky and clouds (usually in upper portion)\n"
        "- Trees, vehicles, ground\n"
        "- Any non-brick surfaces\n\n"
        
        "FOCUS ONLY on the brick wall and scan it systematically for:\n"
        "- Cracks (hairline, vertical, horizontal, diagonal)\n"
        "- Mortar erosion or gaps in joints\n"
        "- Spalled or damaged bricks\n"
        "- Color variations indicating water damage\n"
        "- Any irregularities in the brickwork\n\n"
        
        "CRITICAL INSTRUCTIONS:\n"
        "1. Each defect needs its OWN SMALL bounding box - draw tight boxes around individual cracks\n"
        "2. If you see 5 different cracks, create 5 separate bounding boxes\n"
        "3. Small cracks should have small boxes\n"
        "4. DO NOT create one large box covering multiple defects\n"
        "5. ONLY place boxes on the BRICK SURFACE, never on sky/clouds\n\n"
        
        "VISUAL GUIDANCE:\n"
        "The brick gable/wall is typically reddish-brown colored masonry.\n"
        "Place bounding boxes ONLY on visible defects in this brick surface.\n"
        "If you see white mortar lines between bricks, defects are usually along these lines.\n\n"
        
        "COORDINATE SYSTEM:\n"
        "Origin (0,0) is at TOP-LEFT corner. Y increases downwards.\n"
        "- Y near 0 = top of image (often sky)\n"
        "- Y near middle = center of image (often brick wall)\n"
        "- Y near max = bottom of image (often ground/roof)\n\n"
        
        "⚠️ CRITICAL MISTAKES TO AVOID:\n"
        "1. Do NOT place bounding boxes in the sky/clouds\n"
        "2. Do NOT place boxes on background buildings, trees, or ground\n"
        "3. ONLY place boxes directly on the brick wall surface where you see actual defects\n"
        "4. Measure carefully relative to the ENTIRE image\n\n"
    )
    
    closing = (
        "BEFORE YOU RESPOND:\n"
        "1. Identify where the brick wall is located in the image\n"
        "2. Ignore sky, clouds, and background\n"
        "3. Look for actual defects ONLY on the brick surface\n"
        "4. Measure coordinates carefully\n\n"
        
        "Return ONLY valid JSON - no extra text. If no defects found, return {\"cracks\": []}.\n"
        "⚠️ Be thorough and ACCURATE - focus on BRICK WALL only, typical images have 5-15 defects."
    )
    
    full_prompt = common_instructions + format_description + example + closing
    format_type = "Qwen-1000" if is_qwen else "YOLO"
    
    return full_prompt, format_type

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
    ollama_url = "http://localhost:11434/api/chat"
    
    # Get model-specific prompt
    user_prompt, format_type = get_prompt_for_model(model_name)
    print(f"Using {format_type} coordinate format for {model_name}")
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert structural engineer specializing in drone-based building inspections. "
                    "Your task is to analyze images for structural defects with high precision. "
                    "You must identify cracks, mortar erosion, spalling, water damage, and other defects. "
                    "CRITICAL: Focus ONLY on the brick/masonry surfaces. Ignore sky, clouds, trees, and background. "
                    "Provide accurate bounding box coordinates where origin (0,0) is at TOP-LEFT corner. "
                    "Be thorough, systematic, and precise with spatial measurements."
                )
            },
            {
                "role": "user",
                "content": user_prompt,
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
                    
                    # Save in YOLO format to results folder (auto-convert if needed)
                    output_file = os.path.join(results_folder, f"{base_filename}.txt")
                    save_yolo_labels_auto(cracks, output_file, format_type)
                    
                    # Save YOLO classes file
                    classes_file = os.path.join(results_folder, "classes.txt")
                    save_yolo_classes(classes_file)
                    
                    # Also save JSON for reference
                    json_output_file = os.path.join(results_folder, f"{base_filename}_analysis.json")
                    with open(json_output_file, "w", encoding="utf-8") as f:
                        json.dump(findings, f, indent=2)
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
        print("✗ Cannot connect to Ollama. Make sure it's running on http://localhost:11434")
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
        health_check = requests.get("http://localhost:11434/api/tags", timeout=5)
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