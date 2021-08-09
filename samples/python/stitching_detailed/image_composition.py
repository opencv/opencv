import numpy as np


from stitching_detailed.warper import Warper
from stitching_detailed.exposure_error_compensator import ExposureErrorCompensator  # noqa
from stitching_detailed.seam_finder import SeamFinder
from stitching_detailed.blender import Blender
from stitching_detailed.timelapser import Timelapser


class ImageComposition:
    def __init__(self,
                 warper=Warper(),
                 seam_finder=SeamFinder(),
                 compensator=ExposureErrorCompensator(),
                 blender=Blender(),
                 timelapser=Timelapser()):

        self.warper = warper
        self.seam_finder = seam_finder
        self.compensator = compensator
        self.blender = blender
        self.timelapser = timelapser

    def compose(self, img_names, images, images_small, cameras):
        imgs, masks, corners = self.warp_images(images_small, cameras)
        seam_masks = self.find_seam_masks(imgs, masks, corners)
        self.estimate_exposure_errors(imgs, masks, corners)
        imgs, masks, corners = self.warp_images(images, cameras)
        imgs = self.compensate_exposure_errors(imgs, masks, corners)
        seam_masks = self.resize_seam_masks_to_original_resolution(seam_masks,
                                                                   masks)
        if self.timelapser.do_timelapse:
            self.create_timelapse(img_names, imgs, corners)

        final_pano = self.blend_images(imgs, corners, sizes, seam_masks)
        return final_pano

    def warp_images(self, imgs, cameras, aspect=1):
        imgs_warped = []
        masks_warped = []
        corners = []
        for img, camera in zip(imgs, cameras):
            corner, img = self.warper.warp_image(img, camera, aspect)
            imgs_warped.append(img)
            corners.append(corner)

            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            _, mask = self.warper.warp_image(mask, camera, aspect, mask=True)
            masks_warped.append(mask)

        return imgs_warped, masks_warped, corners

    def find_seam_masks(self, imgs, masks, corners):
        return self.seam_finder.find(imgs, corners, masks)

    def estimate_exposure_errors(self, imgs, masks, corners):
        self.compensator.feed(corners, imgs, masks)

    def compensate_exposure_errors(self, imgs, masks, corners):
        return [self.compensator.apply(idx, corner, img, mask)
                for idx, (img, mask, corner)
                in enumerate(zip(imgs, masks, corners))]

    def resize_seam_masks_to_original_resolution(self, seam_masks, masks):
        return [self.seam_finder.resize(seam_mask, mask)
                for seam_mask, mask in zip(seam_masks, masks)]

    def create_timelapse(self, img_names, imgs, corners):
        for img_name, img, corner in zip(img_names, imgs, corners):
            self.timelapser.process_and_save_frame(img_name, img, corner)

    def blend_images(self, imgs, corners, sizes, seam_masks):
        self.blender.prepare(corners, sizes)
        for img, corner, seam_mask in zip(imgs, corners, seam_masks):
            self.blender.feed(img, seam_mask, corner)
        return self.blender.blend()
