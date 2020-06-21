import numpy as np


# https://github.com/opencv/opencv/blob/master/modules/dnn/test/cityscapes_semsegm_test_enet.py#L22
class NormalizePreproc:
    def __init__(self):
        pass

    @staticmethod
    def process(img):
        image_data = np.array(img).transpose(2, 0, 1).astype(np.float32)
        image_data = np.expand_dims(image_data, 0)
        image_data /= 255.0
        return image_data
