import cv2 as cv
import numpy as np


class Timelapser:

    TIMELAPSE_CHOICES = ('no', 'as_is', 'crop',)
    DEFAULT_TIMELAPSE = 'no'

    def __init__(self, timelapse=DEFAULT_TIMELAPSE):
        self.do_timelapse = True
        self.timelapse_type = None
        self.timelapser = None

        if timelapse == "as_is":
            self.timelapse_type = cv.detail.Timelapser_AS_IS
        elif timelapse == "crop":
            self.timelapse_type = cv.detail.Timelapser_CROP
        else:
            self.do_timelapse = False

        if self.do_timelapse:
            self.timelapser = cv.detail.Timelapser_createDefault(
                self.timelapse_type
                )

    def initialize(self, *args):
        """https://docs.opencv.org/master/dd/dac/classcv_1_1detail_1_1Timelapser.html#aaf0f7c4128009f02473332a0c41f6345"""  # noqa
        self.timelapser.initialize(*args)

    def process_and_save_frame(self, img_name, img, corner):
        self.save_frame(self.get_fixed_filename(img_name), self.get_frame())

    def process_frame(self, img, corner):
        mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
        self.timelapser.process(img, mask, corner)

    def get_frame(self):
        return self.timelapser.getDst()

    def save_frame(self, filename, frame):
        cv.imwrite(filename, frame)

    def get_fixed_filename(self, img_name):
        pos_s = img_name.rfind("/")
        if pos_s == -1:
            fixed_file_name = "fixed_" + img_name
        else:
            fixed_file_name = (img_name[:pos_s + 1] +
                               "fixed_" +
                               img_name[pos_s + 1:])
        return fixed_file_name
