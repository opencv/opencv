import os
from dataclasses import dataclass


@dataclass
class CommonConfig:
    output_data_root_dir: str = "dnn/dnn_conversion"
    log: str = os.path.join(output_data_root_dir, "{}_log.txt")


@dataclass
class TestClsConfig:
    batch_size: int = 25
    frame_size: int = 224
    img_root_dir: str = "./ILSVRC2012_img_val"
    # location of image-class matching
    img_cls_file: str = "./val.txt"
    bgr_to_rgb: bool = True


@dataclass
class TestSegmConfig:
    batch_size: int = 100
    img_root_dir: str = "./VOC2012"
    img_dir: str = os.path.join(img_root_dir, "JPEGImages/")
    img_segm_gt_dir: str = os.path.join(img_root_dir, "SegmentationClass/")
    # reduced val: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt
    segm_val_file: str = os.path.join(img_root_dir, "ImageSets/Segmentation/seg11valid.txt")
    cls_file: str = os.path.join(img_root_dir, "ImageSets/Segmentation/pascal-classes.txt")

    bgr_to_rgb: bool = True
