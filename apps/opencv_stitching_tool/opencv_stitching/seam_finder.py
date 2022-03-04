from collections import OrderedDict
import cv2 as cv
import numpy as np

from .blender import Blender


class SeamFinder:
    """https://docs.opencv.org/4.x/d7/d09/classcv_1_1detail_1_1SeamFinder.html"""  # noqa
    SEAM_FINDER_CHOICES = OrderedDict()
    SEAM_FINDER_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
    SEAM_FINDER_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
    SEAM_FINDER_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)  # noqa
    SEAM_FINDER_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)  # noqa

    DEFAULT_SEAM_FINDER = list(SEAM_FINDER_CHOICES.keys())[0]

    def __init__(self, finder=DEFAULT_SEAM_FINDER):
        self.finder = SeamFinder.SEAM_FINDER_CHOICES[finder]

    def find(self, imgs, corners, masks):
        """https://docs.opencv.org/4.x/d0/dd5/classcv_1_1detail_1_1DpSeamFinder.html#a7914624907986f7a94dd424209a8a609"""  # noqa
        imgs_float = [img.astype(np.float32) for img in imgs]
        return self.finder.find(imgs_float, corners, masks)

    @staticmethod
    def resize(seam_mask, mask):
        dilated_mask = cv.dilate(seam_mask, None)
        resized_seam_mask = cv.resize(dilated_mask, (mask.shape[1],
                                                     mask.shape[0]),
                                      0, 0, cv.INTER_LINEAR_EXACT)
        return cv.bitwise_and(resized_seam_mask, mask)

    @staticmethod
    def draw_seam_mask(img, seam_mask, color=(0, 0, 0)):
        seam_mask = cv.UMat.get(seam_mask)
        overlayed_img = np.copy(img)
        overlayed_img[seam_mask == 0] = color
        return overlayed_img

    @staticmethod
    def draw_seam_polygons(panorama, blended_seam_masks, alpha=0.5):
        return add_weighted_image(panorama, blended_seam_masks, alpha)

    @staticmethod
    def draw_seam_lines(panorama, blended_seam_masks,
                        linesize=1, color=(0, 0, 255)):
        seam_lines = \
            SeamFinder.exctract_seam_lines(blended_seam_masks, linesize)
        panorama_with_seam_lines = panorama.copy()
        panorama_with_seam_lines[seam_lines == 255] = color
        return panorama_with_seam_lines

    @staticmethod
    def exctract_seam_lines(blended_seam_masks, linesize=1):
        seam_lines = cv.Canny(np.uint8(blended_seam_masks), 100, 200)
        seam_indices = (seam_lines == 255).nonzero()
        seam_lines = remove_invalid_line_pixels(
            seam_indices, seam_lines, blended_seam_masks
            )
        kernelsize = linesize + linesize - 1
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        return cv.dilate(seam_lines, kernel)

    @staticmethod
    def blend_seam_masks(seam_masks, corners, sizes):
        imgs = colored_img_generator(sizes)
        blended_seam_masks, _ = \
            Blender.create_panorama(imgs, seam_masks, corners, sizes)
        return blended_seam_masks


def colored_img_generator(sizes, colors=(
            (255, 000, 000),      # Blue
            (000, 000, 255),      # Red
            (000, 255, 000),      # Green
            (000, 255, 255),      # Yellow
            (255, 000, 255),      # Magenta
            (128, 128, 255),      # Pink
            (128, 128, 128),      # Gray
            (000, 000, 128),      # Brown
            (000, 128, 255))      # Orange
            ):
    for idx, size in enumerate(sizes):
        if idx+1 > len(colors):
            raise ValueError("Not enough default colors! Pass additional "
                             "colors to \"colors\" parameter")
        yield create_img_by_size(size, colors[idx])


def create_img_by_size(size, color=(0, 0, 0)):
    width, height = size
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    return img


def add_weighted_image(img1, img2, alpha):
    return cv.addWeighted(
        img1, alpha, img2, (1.0 - alpha), 0.0
        )


def remove_invalid_line_pixels(indices, lines, mask):
    for x, y in zip(*indices):
        if check_if_pixel_or_neighbor_is_black(mask, x, y):
            lines[x, y] = 0
    return lines


def check_if_pixel_or_neighbor_is_black(img, x, y):
    check = [is_pixel_black(img, x, y),
             is_pixel_black(img, x+1, y), is_pixel_black(img, x-1, y),
             is_pixel_black(img, x, y+1), is_pixel_black(img, x, y-1)]
    return any(check)


def is_pixel_black(img, x, y):
    return np.all(get_pixel_value(img, x, y) == 0)


def get_pixel_value(img, x, y):
    try:
        return img[x, y]
    except IndexError:
        pass
