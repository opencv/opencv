BASE_IMG_SCALE_FACTOR = 1 / 255.0
PYTORCH_RSZ_HEIGHT = 256
PYTORCH_RSZ_WIDTH = 256

pytorch_resize_input_blob = {
    "mean": ["123.675", "116.28", "103.53"],
    "scale": str(BASE_IMG_SCALE_FACTOR),
    "std": ["0.229", "0.224", "0.225"],
    "crop": "True",
    "rgb": True,
    "rsz_height": str(PYTORCH_RSZ_HEIGHT),
    "rsz_width": str(PYTORCH_RSZ_WIDTH)
}

pytorch_input_blob = {
    "mean": ["123.675", "116.28", "103.53"],
    "scale": str(BASE_IMG_SCALE_FACTOR),
    "std": ["0.229", "0.224", "0.225"],
    "crop": "True",
    "rgb": True
}

tf_input_blob = {
    "scale": str(1 / 127.5),
    "mean": ["127.5", "127.5", "127.5"],
    "std": [],
    "crop": "True",
    "rgb": True
}

tf_model_blob_caffe_mode = {
    "mean": ["103.939", "116.779", "123.68"],
    "scale": "1.0",
    "std": [],
    "crop": "True",
    "rgb": False
}
