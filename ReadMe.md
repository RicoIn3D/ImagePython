
pip install pillow requests

python annotate_bboxes_from_url.py



python annotate_bboxes_from_url.py --file C:\Users\rico\Pictures\DJI_0942.jiff


python annotate_bboxes_from_url.py --file "C:/path/to/img.jpg" \
  --data '{"cracks":[{"bbox_2d":[0.52,0.41,0.08,0.05]}]}' \
  --bbox-format yolo_norm

  python annotate_bboxes_from_url.py --file "C:/Users/rico/Pictures/DJI_0942.jpg" --bbox-format yolo_norm