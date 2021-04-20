from detectron2.engine import DefaultTrainer
from detectron2.data.datsets import register_coco_instances

# Train faster-rcnn in customized dataset using detectron2


def update_cat(data_path, annotation_path, img_path):
    # the dataset should be in coco format, otherwise it needs to be transformed to coco format
    # annotation example:
    # "annotations":[
    #       { "id":
    #         "category_id":
    #         "iscrowd": 0
    #         "segmentation": [[]]
    #         "image_id":
    #         "area":
    #         "bbox": [a,b,c,d]
    # More details in this link
    # "https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
    register_coco_instances(data_path, {}, annotation_path, img_path)
