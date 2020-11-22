import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class CommonConfig:
    output_data_root_dir: str = "dnn_model_runner/dnn_conversion"
    logs_dir: str = os.path.join(output_data_root_dir, "logs")
    log_file_path: str = os.path.join(logs_dir, "{}_log.txt")


@dataclass
class TestClsConfig:
    batch_size: int = 1
    frame_size: int = 224
    img_root_dir: str = "./ILSVRC2012_img_val"
    # location of image-class matching
    img_cls_file: str = "./val.txt"
    bgr_to_rgb: bool = True


@dataclass
class TestClsModuleConfig:
    cls_test_data_dir: str = "../data"
    test_module_name: str = "classification"
    test_module_path: str = "classification.py"
    input_img: str = os.path.join(cls_test_data_dir, "squirrel_cls.jpg")
    model: str = ""

    frame_height: str = str(TestClsConfig.frame_size)
    frame_width: str = str(TestClsConfig.frame_size)
    scale: str = "1.0"
    mean: List[str] = field(default_factory=lambda: ["0.0", "0.0", "0.0"])
    std: List[str] = field(default_factory=list)
    crop: str = "False"
    rgb: str = "True"
    rsz_height: str = ""
    rsz_width: str = ""
    classes: str = os.path.join(cls_test_data_dir, "dnn", "classification_classes_ILSVRC2012.txt")
