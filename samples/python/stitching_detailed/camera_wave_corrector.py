from collections import OrderedDict
import cv2 as cv
import numpy as np


class WaveCorrector:
    """https://docs.opencv.org/master/d7/d74/group__stitching__rotation.html#ga83b24d4c3e93584986a56d9e43b9cf7f"""  # noqa
    choices = OrderedDict()
    choices['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
    choices['vert'] = cv.detail.WAVE_CORRECT_VERT
    choices['auto'] = cv.detail.WAVE_CORRECT_AUTO
    choices['no'] = None

    default = list(choices.keys())[0]

    def __init__(self, wave_correct_kind=default):
        self.wave_correct_kind = WaveCorrector.choices[wave_correct_kind]

    def correct(self, cameras):
        if self.wave_correct_kind:
            rmats = []
            for cam in cameras:
                rmats.append(np.copy(cam.R))
            rmats = cv.detail.waveCorrect(rmats, self.wave_correct_kind)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
        return cameras
