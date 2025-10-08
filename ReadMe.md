
pip install pillow requests

python annotate_bboxes_from_url.py

vision pc
192.168.87.207

python annotate_bboxes_from_url.py  --file "results\DJI_0942.JPG"        --json-file "results\DJI_0942_analysis.json"        --out "annotated_DJI_0942.jpg"        --export-yolo "DJI_0942_yolo.txt"

python batch_process_images.py --urls image_urls.txt --run-id PILOT_01



python analyze_drone_image_yolo.py --url "https://example.com/image.jpg"

python batch_process_images.py --urls image_urls.txt --run-id BATCH_001

python analyze_drone_image_yolo.py --url "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje/uploaded/DJI_4420.JPG" --model "qwen2.5vl:latest"

  python analyze_drone_image_yolo.py --url "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje/uploaded/DJI_4420.JPG" --model "llava:13b"

  python annotate_bboxes_from_url.py   --file "results/DJI_4419.JPG"   --data '{"cracks":[{"bbox_2d":[0,0.535,0.305,0.035,0.025],"description":"hole in gable"},{"bbox_2d":[0,0.540,0.505,0.040,0.015],"description":"white marking"}]}'   --out "my_manual_annotation.jpg"

  python annotate_bboxes_from_url.py   --file "results/DJI_4419.JPG"  --data '{"cracks":[{"bbox_2d":[0,0.535,0.505,0.035,0.025],"description":"hole"},{"bbox_2d":[0,0.540,0.705,0.040,0.015],"description":"white marking"}]}'   --out "my_manual_annotation_v2.jpg"

  python annotate_bboxes_from_url.py   --file "results/DJI_4419.JPG"   --data '{"cracks":[{"bbox_2d":[0,0.495,0.505,0.035,0.025],"description":"hole"},{"bbox_2d":[0,0.500,0.705,0.040,0.015],"description":"white marking"}]}'   --out "my_manual_annotation_v3.jpg"

  python annotate_bboxes_from_url.py   --file "results/DJI_4419.JPG"   --data '{"cracks":[{"bbox_2d":[0,0.495,0.505,0.035,0.025],"description":"hole"},{"bbox_2d":[0,0.350,0.705,0.040,0.015],"description":"white marking"}]}'   --out "my_manual_annotation_v4.jpg"

  python annotate_bboxes_from_url.py   --file "results/DJI_4419.JPG"   --data '{"cracks":[{"bbox_2d":[0,0.495,0.505,0.035,0.025],"description":"hole"},{"bbox_2d":[0,0.350,0.535,0.040,0.015],"description":"white marking"}]}'   --out "my_manual_annotation_final.jpg"
  
  python annotate_bboxes_from_url.py   --file "results/DJI_4419.JPG"   --data '{"cracks":[{"bbox_2d":[0,0.495,0.505,0.035,0.025],"description":"hole"},{"bbox_2d":[0,0.350,0.735,0.040,0.015],"description":"white marking"}]}'   --out "my_manual_annotation_final.jpg"