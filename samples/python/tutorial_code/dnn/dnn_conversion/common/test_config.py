from dataclasses import dataclass


@dataclass
class TestConfig:
    frame_size: int = 224
    imgs_class_dir: str = "./ILSVRC2012_img_val"
    batch_size: int = 100
    # location of image-class matching
    img_cls_file: str = "./val.txt"
    log: str = "./log.txt"
    bgr_to_rgb: bool = True
