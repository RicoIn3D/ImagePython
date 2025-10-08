
pip install pillow requests

python annotate_bboxes_from_url.py

vision pc
192.168.87.207

python annotate_bboxes_from_url.py  --file "results\DJI_0942.JPG"        --json-file "results\DJI_0942_analysis.json"        --out "annotated_DJI_0942.jpg"        --export-yolo "DJI_0942_yolo.txt"

python batch_process_images.py --urls image_urls.txt --run-id PILOT_01



python analyze_drone_image_yolo.py --url "https://example.com/image.jpg"

python batch_process_images.py --urls image_urls.txt --run-id BATCH_001