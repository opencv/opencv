from dataclasses import dataclass


@dataclass
class TestConfig:
    frame_size: int = 224
    imgs_dir: str = "./ILSVRC2012_img_val"
    in_blob: str = "input"
    out_blob: str = "output"
    batch_size: int = 100
    img_cls_file: str = "./val.txt"
    log: str = "./log.txt"
    bgr_to_rgb: bool = True
