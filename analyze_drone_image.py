#!/usr/bin/env python3
"""
Quick script to analyze your drone image
"""

import requests
import base64
import json

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
                "content": (""
                    " Your task is to scan the "
                    "entire brick wall surface in this image and identify every visible crack, hairline fracture, spalled brick,"
                    " eroded mortar joint, displaced brick, or any deviation from uniform brickwork — no matter how small, faint, or ambiguous. "
                    " \n\nAssume defects exist — your priority is sensitivity, not precision. Do not skip anything that could be a crack or mortar issue.  "
                    "\n\nFor each defect:  \n- Output a tight bounding box in pixel coordinates: [x1, y1, x2, y2] (top-left origin).  "
                    "\n- Write a concise technical description: e.g., 'vertical hairline crack in mortar', 'spalled brick at upper left',"
                    " 'horizontal mortar erosion near roofline'.  \n\nReturn ONLY valid JSON in this exact format  "
                    "coordinates (x_min, y_min, x_max, y_max), where (x_min, y_min) is the top-left corner of the box and (x_max, y_max) is the bottom-right corner"
                    " — no extra text, no explanations: "
                    " \n{ \"cracks\": [ {\"bbox_2d\": [x1,y1,x2,y2], \"description\": \"...\"} ] }  \n\nIf you find absolutely nothing (unlikely), "
                    "return { \"cracks\": [] }."
                    "  \n\n⚠️ Never ignore thin lines, color variations, or irregularities — they may indicate early-stage cracking."
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
            
            # Display summary
            if isinstance(findings, dict):
                if "findings" in findings:
                    print(f"\n✓ Found {len(findings['findings'])} issue(s)")
                if "overall_assessment" in findings:
                    print(f"✓ Assessment: {findings['overall_assessment']}")
                    
        except json.JSONDecodeError:
            print("\n(Note: Response is not in JSON format)")
            
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