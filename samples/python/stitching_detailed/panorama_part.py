import cv2 as cv


class PanoramaPart:

    def __init__(self, img_name):
        self.img_name = img_name
        self.full_img_size = (None, None)
        self.work_scale = None
        self.work_img = None
        self.seam_scale = None
        self.seam_img = None
        self.seam_work_aspect = None

    def read_image(self):
        self.full_img = cv.imread(self.img_name)
        self.full_img_size = (self.full_img.shape[1], self.full_img.shape[0])
        if self.full_img is None:
            print("Cannot read image ", self.img_name)
            exit()

    def set_work_image(self, scaler):
        self.work_img = scaler.set_scale_and_downscale(self.full_img)

    def set_seam_image(self, scaler):
        self.seam_img = scaler.set_scale_and_downscale(self.full_img)

    def __update_seam_work_aspect(self):
        if self.work_scale and self.seam_scale:
            self.seam_work_aspect = self.seam_scale / self.work_scale
