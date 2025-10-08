#!/usr/bin/env python3
"""
Batch wrapper that reuses analyze_drone_image_yolo.py and annotate_bboxes_from_url.py

This wrapper:
1. Reads URLs from a text file
2. Calls analyze_drone_image_yolo.py for each URL
3. Calls annotate_bboxes_from_url.py to create annotated images
4. Organizes results in folders: results/{RunID}_{filename}/

Usage:
  python batch_process_images.py --urls image_urls.txt
  python batch_process_images.py --urls image_urls.txt --run-id R001

URLs file format (one URL per line):
  https://example.com/image1.jpg
  https://example.com/image2.jpg
  # Comments start with #
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from urllib.parse import urlparse

def get_filename_from_url(url: str) -> Tuple[str, str]:
    """Extract filename from URL. Returns (name_without_ext, extension)."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    name, ext = os.path.splitext(filename)
    return name, ext

def generate_run_id() -> str:
    """Generate a RunID based on timestamp: RYYYYMMDD_HHMMSS"""
    now = datetime.now()
    return f"R{now.strftime('%Y%m%d_%H%M%S')}"

def read_urls_from_file(filepath: str) -> List[str]:
    """Read URLs from text file, one per line. Skip empty lines and comments."""
    urls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls

def create_output_folder(run_id: str, filename: str) -> str:
    """Create output folder: results/{RunID}_{filename}/"""
    folder_name = f"{run_id}_{filename}"
    output_path = os.path.join("results", folder_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def process_single_image(url: str, run_id: str, index: int, total: int) -> bool:
    """Process a single image through both scripts."""
    
    print(f"\n{'='*70}")
    print(f"Processing image {index}/{total}")
    print(f"URL: {url}")
    print(f"{'='*70}")
    
    # Get filename without extension
    filename, ext = get_filename_from_url(url)
    
    # Step 1: Run analyze_drone_image_yolo.py with URL parameter  models: --model "llava:13b" or --model "qwen2.5vl:latest"
    print("\n[Step 1/3] Running analysis...")
    
    cmd = f'python analyze_drone_image_yolo.py --url "{url}" --model "qwen2.5vl:latest" '
    result = os.system(cmd)
    
    if result != 0:
        print(f"  ✗ Analysis failed with exit code {result}")
        return False
    
    print("  ✓ Analysis complete")
    
    # Check if results were created
    results_base = "results"
    image_file = os.path.join(results_base, f"{filename}{ext}")
    json_file = os.path.join(results_base, f"{filename}_analysis.json")
    labels_file = os.path.join(results_base, f"{filename}.txt")
    
    if not os.path.exists(json_file):
        print(f"  ✗ Analysis JSON not found: {json_file}")
        return False
    
    # Step 2: Create dedicated folder for this run
    print("\n[Step 2/3] Organizing files...")
    output_folder = create_output_folder(run_id, filename)
    print(f"  Output folder: {output_folder}")
    
    # Move files to dedicated folder
    files_to_move = [
        (image_file, f"{filename}{ext}"),
        (json_file, f"{filename}_analysis.json"),
        (labels_file, f"{filename}.txt"),
        (os.path.join(results_base, "classes.txt"), "classes.txt")
    ]
    
    for src, dst_name in files_to_move:
        if os.path.exists(src):
            dst = os.path.join(output_folder, dst_name)
            shutil.move(src, dst)
            print(f"  ✓ Moved: {dst_name}")
    
    # Step 3: Run annotate_bboxes_from_url.py
    print("\n[Step 3/3] Creating annotated image...")
    
    image_path = os.path.join(output_folder, f"{filename}{ext}")
    json_path = os.path.join(output_folder, f"{filename}_analysis.json")
    annotated_output = f"{filename}_annotated{ext}"
    
    cmd = (
        f'python annotate_bboxes_from_url.py '
        f'--file "{image_path}" '
        f'--json-file "{json_path}" '
        f'--out "{annotated_output}" '
        f'--results-folder "{output_folder}"'
    )
    
    result = os.system(cmd)
    
    if result != 0:
        print(f"  ⚠ Annotation failed with exit code {result}")
        print(f"  Files still saved in: {output_folder}")
        return True  # Still consider it a success since analysis worked
    
    print(f"  ✓ Annotated image created")
    print(f"\n  📁 All files saved to: {output_folder}/")
    
    return True

def check_dependencies() -> bool:
    """Check if required scripts exist."""
    required = ["analyze_drone_image_yolo.py", "annotate_bboxes_from_url.py"]
    missing = [f for f in required if not os.path.exists(f)]
    
    if missing:
        print("✗ Missing required scripts:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("✓ All required scripts found")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Batch process drone images using existing analysis and annotation scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_images.py --urls image_urls.txt
  python batch_process_images.py --urls urls.txt --run-id R001
  python batch_process_images.py --urls urls.txt --run-id BUILDING_A_INSPECTION
        """
    )
    
    parser.add_argument("--urls", required=True, help="Text file with image URLs (one per line)")
    parser.add_argument("--run-id", help="Custom Run ID (default: auto-generated timestamp)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("BATCH IMAGE PROCESSOR")
    print("="*70)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if URLs file exists
    if not os.path.exists(args.urls):
        print(f"✗ URLs file not found: {args.urls}")
        sys.exit(1)
    
    # Read URLs
    urls = read_urls_from_file(args.urls)
    if not urls:
        print(f"✗ No URLs found in {args.urls}")
        sys.exit(1)
    
    print(f"✓ Found {len(urls)} URL(s) to process")
    
    # Generate or use provided RunID
    run_id = args.run_id if args.run_id else generate_run_id()
    print(f"✓ Run ID: {run_id}")
    print(f"✓ Results will be saved to: results/{run_id}_<filename>/")
    
    # Confirm before starting
    print(f"\nReady to process {len(urls)} image(s)")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Process each image
    success_count = 0
    failed_count = 0
    
    for idx, url in enumerate(urls, start=1):
        try:
            if process_single_image(url, run_id, idx, len(urls)):
                success_count += 1
            else:
                failed_count += 1
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
            break
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images: {len(urls)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"\nResults location: results/{run_id}_*/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()