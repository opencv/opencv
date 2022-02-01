from collections import namedtuple
import cv2 as cv

from .blender import Blender
from .stitching_error import StitchingError


class Rectangle(namedtuple('Rectangle', 'x y width height')):
    __slots__ = ()

    @property
    def area(self):
        return self.width * self.height

    @property
    def corner(self):
        return (self.x, self.y)

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height

    def times(self, x):
        return Rectangle(*(int(round(i*x)) for i in self))

    def draw_on(self, img, color=(0, 0, 255), size=1):
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        start_point = (self.x, self.y)
        end_point = (self.x2-1, self.y2-1)
        cv.rectangle(img, start_point, end_point, color, size)
        return img


class Cropper:

    DEFAULT_CROP = False

    def __init__(self, crop=DEFAULT_CROP):
        self.do_crop = crop
        self.overlapping_rectangles = []
        self.cropping_rectangles = []

    def prepare(self, imgs, masks, corners, sizes):
        if self.do_crop:
            mask = self.estimate_panorama_mask(imgs, masks, corners, sizes)
            self.compile_numba_functionality()
            lir = self.estimate_largest_interior_rectangle(mask)
            corners = self.get_zero_center_corners(corners)
            rectangles = self.get_rectangles(corners, sizes)
            self.overlapping_rectangles = self.get_overlaps(
                rectangles, lir)
            self.intersection_rectangles = self.get_intersections(
                rectangles, self.overlapping_rectangles)

    def crop_images(self, imgs, aspect=1):
        for idx, img in enumerate(imgs):
            yield self.crop_img(img, idx, aspect)

    def crop_img(self, img, idx, aspect=1):
        if self.do_crop:
            intersection_rect = self.intersection_rectangles[idx]
            scaled_intersection_rect = intersection_rect.times(aspect)
            cropped_img = self.crop_rectangle(img, scaled_intersection_rect)
            return cropped_img
        return img

    def crop_rois(self, corners, sizes, aspect=1):
        if self.do_crop:
            scaled_overlaps = \
                [r.times(aspect) for r in self.overlapping_rectangles]
            cropped_corners = [r.corner for r in scaled_overlaps]
            cropped_corners = self.get_zero_center_corners(cropped_corners)
            cropped_sizes = [r.size for r in scaled_overlaps]
            return cropped_corners, cropped_sizes
        return corners, sizes

    @staticmethod
    def estimate_panorama_mask(imgs, masks, corners, sizes):
        _, mask = Blender.create_panorama(imgs, masks, corners, sizes)
        return mask

    def compile_numba_functionality(self):
        # numba functionality is only imported if cropping
        # is explicitely desired
        try:
            import numba
        except ModuleNotFoundError:
            raise StitchingError("Numba is needed for cropping but not installed")
        from .largest_interior_rectangle import largest_interior_rectangle
        self.largest_interior_rectangle = largest_interior_rectangle

    def estimate_largest_interior_rectangle(self, mask):
        lir = self.largest_interior_rectangle(mask)
        lir = Rectangle(*lir)
        return lir

    @staticmethod
    def get_zero_center_corners(corners):
        min_corner_x = min([corner[0] for corner in corners])
        min_corner_y = min([corner[1] for corner in corners])
        return [(x - min_corner_x, y - min_corner_y) for x, y in corners]

    @staticmethod
    def get_rectangles(corners, sizes):
        rectangles = []
        for corner, size in zip(corners, sizes):
            rectangle = Rectangle(*corner, *size)
            rectangles.append(rectangle)
        return rectangles

    @staticmethod
    def get_overlaps(rectangles, lir):
        return [Cropper.get_overlap(r, lir) for r in rectangles]

    @staticmethod
    def get_overlap(rectangle1, rectangle2):
        x1 = max(rectangle1.x, rectangle2.x)
        y1 = max(rectangle1.y, rectangle2.y)
        x2 = min(rectangle1.x2, rectangle2.x2)
        y2 = min(rectangle1.y2, rectangle2.y2)
        if x2 < x1 or y2 < y1:
            raise StitchingError("Rectangles do not overlap!")
        return Rectangle(x1, y1, x2-x1, y2-y1)

    @staticmethod
    def get_intersections(rectangles, overlapping_rectangles):
        return [Cropper.get_intersection(r, overlap_r) for r, overlap_r
                in zip(rectangles, overlapping_rectangles)]

    @staticmethod
    def get_intersection(rectangle, overlapping_rectangle):
        x = abs(overlapping_rectangle.x - rectangle.x)
        y = abs(overlapping_rectangle.y - rectangle.y)
        width = overlapping_rectangle.width
        height = overlapping_rectangle.height
        return Rectangle(x, y, width, height)

    @staticmethod
    def crop_rectangle(img, rectangle):
        return img[rectangle.y:rectangle.y2, rectangle.x:rectangle.x2]
