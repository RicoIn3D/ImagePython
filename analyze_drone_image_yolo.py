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

def save_yolo_labels(cracks: list, output_path: str) -> None:
    """Save cracks in YOLO format to a text file."""
    lines = []
    for crack in cracks:
        bbox = crack.get("bbox_2d", [])
        if len(bbox) >= 5:
            # Format: class_id x_center y_center width height
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

def create_results_folder() -> str:
    """Create results subfolder if it doesn't exist."""
    results_folder = "results"
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

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
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert structural engineer specializing in drone-based building inspections. "
                    "Your task is to analyze images for structural defects with high precision. "
                    "You must identify cracks, mortar erosion, spalling, water damage, and other defects. "
                    "CRITICAL: You must provide accurate bounding box coordinates in YOLO format where the origin (0,0) is at the TOP-LEFT corner. "
                    "Always measure y-coordinates from the TOP of the image downwards. "
                    "Be thorough, systematic, and precise with spatial measurements."
                )
            },
            {
                "role": "user",
                "content": (
                    "You are inspecting a brick wall for structural defects. Scan EVERY part of the image systematically and identify ALL visible issues:\n"
                    "- Cracks (hairline, vertical, horizontal, diagonal)\n"
                    "- Mortar erosion or gaps in joints\n"
                    "- Spalled or damaged bricks\n"
                    "- Color variations indicating water damage\n"
                    "- Any irregularities in the brickwork\n\n"
                    
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. Each defect needs its OWN SMALL bounding box - draw tight boxes around individual cracks, not large areas\n"
                    "2. If you see 5 different cracks, create 5 separate bounding boxes\n"
                    "3. Small cracks should have small boxes (width/height around 0.02-0.10)\n"
                    "4. DO NOT create one large box covering multiple defects\n\n"
                    
                    "COORDINATE SYSTEM - MEASURE VERY CAREFULLY:\n"
                    "CRITICAL: The origin (0,0) is at the TOP-LEFT corner. Y coordinates start at 0 at the TOP.\n"
                    "- To find y_center: Count pixels from the TOP of the image, not the bottom\n"
                    "- x_center: 0.0 = left edge, 0.5 = horizontal center, 1.0 = right edge\n"
                    "- y_center: 0.0 = very TOP of image, 0.5 = vertical middle, 1.0 = very BOTTOM\n\n"
                    
                    "POSITION GUIDE - measure distance from TOP edge:\n"
                    "- If defect is near the TOP of image: y_center = 0.05-0.25 (small numbers!)\n"
                    "- If defect is in upper-middle area: y_center = 0.25-0.45\n"
                    "- If defect is in center area: y_center = 0.45-0.55\n"
                    "- If defect is in lower-middle area: y_center = 0.55-0.75\n"
                    "- If defect is near the BOTTOM: y_center = 0.75-0.95\n\n"
                    
                    "⚠️ COMMON MISTAKE TO AVOID:\n"
                    "Do NOT use large y values (like 0.6-0.9) for defects that are in the UPPER part of the image.\n"
                    "Carefully measure from the TOP edge downwards.\n\n"
                    
                    "OUTPUT FORMAT - You MUST return EXACTLY 5 values in bbox_2d array:\n"
                    "\"bbox_2d\": [class_id, x_center, y_center, width, height]\n\n"
                    
                    "Where:\n"
                    "- class_id: Always 0 for cracks\n"
                    "- x_center: horizontal center (0.0=left, 1.0=right)\n"
                    "- y_center: vertical center measured FROM TOP (0.0=top, 1.0=bottom)\n"
                    "- width: box width as fraction (typically 0.02-0.10)\n"
                    "- height: box height as fraction (typically 0.02-0.10)\n\n"
                    
                    "EXAMPLE - for defects in UPPER third of image:\n"
                    "{\n"
                    "  \"cracks\": [\n"
                    "    {\"bbox_2d\": [0, 0.35, 0.20, 0.08, 0.03], \"description\": \"crack near top - y=0.20 means 20% down from top\"},\n"
                    "    {\"bbox_2d\": [0, 0.50, 0.15, 0.05, 0.02], \"description\": \"crack at upper area - y=0.15 means 15% from top\"},\n"
                    "    {\"bbox_2d\": [0, 0.70, 0.30, 0.04, 0.02], \"description\": \"crack in upper-middle - y=0.30 means 30% from top\"}\n"
                    "  ]\n"
                    "}\n\n"
                    
                    "Return ONLY valid JSON - no extra text. If you find nothing, return {\"cracks\": []}.\n"
                    "⚠️ Be thorough and ACCURATE with coordinates - typical drone images have 5-15 detectable defects."
                ),
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
                    
                    # Save in YOLO format to results folder
                    output_file = os.path.join(results_folder, f"{base_filename}.txt")
                    save_yolo_labels(cracks, output_file)
                    
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