#!/usr/bin/env python3
"""
Quick script to analyze your drone image and save cracks in YOLO format
"""

import requests
import base64
import json
import os
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

def analyze_drone_image():
    # Your image URL
    image_url = "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje/uploaded/DJI_0942.JPG"
    
    print("Downloading image...")
    # Download and convert to base64
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        print(f"✓ Image downloaded successfully ({len(image_base64)} bytes)")
    except Exception as e:
        print(f"✗ Failed to download image: {e}")
        return
    
    # Prepare Ollama API request
    ollama_url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": "llava:13b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an experienced architect inspecting structures from drone imagery. "
                    "Analyze images for structural issues, cracks, mortar problems, water damage, "
                    "and any defects. Be specific about locations and severity."                  
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
                    
                    "OUTPUT FORMAT - You MUST return EXACTLY 5 values in bbox_2d array:\n"
                    "\"bbox_2d\": [0, x_center, y_center, width, height]\n\n"
                    
                    "Where:\n"
                    "- First value is ALWAYS 0 (class_id for crack)\n"
                    "- x_center: horizontal center of the crack (0.0=left edge, 1.0=right edge)\n"
                    "- y_center: vertical center of the crack (0.0=top, 1.0=bottom)\n"
                    "- width: crack width as fraction of image (typically 0.02-0.10 for thin cracks)\n"
                    "- height: crack height as fraction of image (typically 0.02-0.10 for small cracks)\n\n"
                    
                    "EXAMPLE of correct format:\n"
                    "{\n"
                    "  \"cracks\": [\n"
                    "    {\"bbox_2d\": [0, 0.356, 0.568, 0.076, 0.026], \"description\": \"horizontal hairline crack in mortar joint\"},\n"
                    "    {\"bbox_2d\": [0, 0.291, 0.473, 0.033, 0.014], \"description\": \"faint horizontal mortar erosion\"},\n"
                    "    {\"bbox_2d\": [0, 0.371, 0.457, 0.035, 0.018], \"description\": \"slight vertical mortar separation\"}\n"
                    "  ]\n"
                    "}\n\n"
                    
                    "Return ONLY valid JSON - no extra text. If you find nothing, return {\"cracks\": []}.\n"
                    "⚠️ Be thorough - typical drone images have 5-15 detectable defects."
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
    
    print("\nSending request to Ollama (llava model)...")
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
                cracks = findings.get("cracks", [])
                if cracks:
                    print(f"\n✓ Found {len(cracks)} crack(s)")
                    
                    # Generate output filename based on image URL
                    base_filename = get_filename_from_url(image_url)
                    output_file = f"{base_filename}.txt"
                    
                    # Save in YOLO format
                    save_yolo_labels(cracks, output_file)
                    
                    # Also save JSON for reference
                    json_output_file = f"{base_filename}_analysis.json"
                    with open(json_output_file, "w", encoding="utf-8") as f:
                        json.dump(findings, f, indent=2)
                    print(f"✓ Saved detailed analysis to: {json_output_file}")
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
    print("=" * 70)
    print("DRONE IMAGE ANALYSIS - DJI_0942.JPG")
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
            
            if not any("llava" in name for name in model_names):
                print("\n⚠ WARNING: llava model not found!")
                print("  Install it with: ollama pull llava")
                response = input("\nContinue anyway? (y/n): ")
                if response.lower() != 'y':
                    exit(0)
            print()
    except:
        print("✗ Cannot connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        print("  And llava model is installed: ollama pull llava")
        exit(1)
    
    analyze_drone_image()