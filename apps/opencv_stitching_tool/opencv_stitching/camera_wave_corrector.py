from collections import OrderedDict
import cv2 as cv
import numpy as np


class WaveCorrector:
    """https://docs.opencv.org/4.x/d7/d74/group__stitching__rotation.html#ga83b24d4c3e93584986a56d9e43b9cf7f"""  # noqa
    WAVE_CORRECT_CHOICES = OrderedDict()
    WAVE_CORRECT_CHOICES['horiz'] = cv.detail.WAVE_CORRECT_HORIZ
    WAVE_CORRECT_CHOICES['vert'] = cv.detail.WAVE_CORRECT_VERT
    WAVE_CORRECT_CHOICES['auto'] = cv.detail.WAVE_CORRECT_AUTO
    WAVE_CORRECT_CHOICES['no'] = None

    DEFAULT_WAVE_CORRECTION = list(WAVE_CORRECT_CHOICES.keys())[0]

    def __init__(self, wave_correct_kind=DEFAULT_WAVE_CORRECTION):
        self.wave_correct_kind = WaveCorrector.WAVE_CORRECT_CHOICES[
            wave_correct_kind
            ]

    def correct(self, cameras):
        if self.wave_correct_kind is not None:
            rmats = [np.copy(cam.R) for cam in cameras]
            rmats = cv.detail.waveCorrect(rmats, self.wave_correct_kind)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
            return cameras
        return cameras
