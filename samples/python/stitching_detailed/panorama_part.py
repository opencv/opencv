import cv2 as cv


class PanoramaPart:

    def __init__(self, img_name):
        self.img_name = img_name
        self.full_img = None
        self.full_img_size = (None, None)

    def read_image(self):
        self.full_img = cv.imread(self.img_name)
        self.full_img_size = (self.full_img.shape[1], self.full_img.shape[0])
        if self.full_img is None:
            print("Cannot read image ", self.img_name)
            exit()

    def remove_image_from_memory(self):
        self.full_img = None
