
pip install pillow requests

python annotate_bboxes_from_url.py


DEFAULT_DATA = {
    "cracks": [
        {"bbox_2d": [352, 449, 392, 471], "description": "horizontal crack in brick wall near roofline"},
        {"bbox_2d": [357, 552, 419, 579], "description": "horizontal crack in brick wall, slightly wider and more pronounced"},
    ]
}


python annotate_bboxes_from_url.py --url "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje_small/Folder%202/DJI_0942.JPG" --export-yolo "yolo.txt" --out url_DJI_0942.jpg


python annotate_bboxes_from_url.py --file "C:/Users/rico/Pictures/DJI_0942.jpg" --export-yolo "yolo.txt" --out an_DJI_0942.jpg


# Læs YOLO labels og vis på billede
python annotate_bboxes_multi_format.py --file "img.jpg" --labels-yolo "labels.txt"

# Konverter YOLO → Qwen-1000
python annotate_bboxes_multi_format.py --file "img.jpg" \
  --labels-yolo "input.yolo.txt" --export-qwen "output.qwen.txt"

# Konverter Qwen-1000 → YOLO
python annotate_bboxes_multi_format.py --file "img.jpg" \
  --labels-qwen "input.qwen.txt" --export-yolo "output.yolo.txt"

# Læs YOLO og eksporter begge formater
python annotate_bboxes_multi_format.py --file "img.jpg" \
  --labels-yolo "input.txt" --export-yolo "out.yolo.txt" --export-qwen "out.qwen.txt"

  python annotate_bboxes_from_url.py --url "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje_small/Folder%202/DJI_0942.JPG" --out url_DJI_0942.jpg --labels-yolo "DJI_0942.txt"

python  annotate_bboxes_from_url.py  --url "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje_small/Folder%202/DJI_0942.JPG" --out url_DJI_0942.jpg  --json-file "DJI_0942_analysis.json"
python  annotate_bboxes_from_url.py  --url "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje_small/Folder%202/DJI_0942.JPG" --out url_DJI_0942.jpg --json-file "DJI_0942_analysis.json"

python annotate_bboxes_from_url.py  --file "results\DJI_0942.JPG"        --json-file "results\DJI_0942_analysis.json"        --out "annotated_DJI_0942.jpg"        --export-yolo "DJI_0942_yolo.txt"

python batch_process_images.py --urls image_urls.txt --run-id PILOT_01



python analyze_drone_image_yolo.py --url "https://example.com/image.jpg"

python batch_process_images.py --urls image_urls.txt --run-id BATCH_001