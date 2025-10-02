
pip install pillow requests

python annotate_bboxes_from_url.py


DEFAULT_DATA = {
    "cracks": [
        {"bbox_2d": [352, 449, 392, 471], "description": "horizontal crack in brick wall near roofline"},
        {"bbox_2d": [357, 552, 419, 579], "description": "horizontal crack in brick wall, slightly wider and more pronounced"},
    ]
}


python annotate_bboxes_from_url.py --file "C:/Users/rico/Pictures/DJI_0942.jpg" --export-yolo "yolo.txt" --out an_DJI_0942.jpg