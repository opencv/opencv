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

    def compose(self, img_data, cameras, panorama_scale):
        imgs, masks, corners = self.warp_images(img_data.get_seam_images(), cameras, panorama_scale, img_data.get_seam_work_aspect())
        seam_masks = self.find_seam_masks(imgs, masks, corners)
        self.estimate_exposure_errors(imgs, masks, corners)
        imgs, masks, corners = self.warp_images(img_data.get_compose_images(), cameras, panorama_scale, img_data.get_compose_work_aspect())
        imgs = self.compensate_exposure_errors(imgs, masks, corners)
        seam_masks = self.resize_seam_masks_to_original_resolution(seam_masks,
                                                                   masks)
        if self.timelapser.do_timelapse:
            self.create_timelapse(img_data.img_names, imgs, corners)

        final_pano = self.blend_images(imgs, corners, seam_masks)
        return final_pano

    def warp_images(self, imgs, cameras, scale=1, aspect=1):
        imgs_warped = []
        masks_warped = []
        corners = []
        self.warper.set_scale(scale*aspect)
        for img, camera in zip(imgs, cameras):
            corner, img_warped = self.warper.warp_image(img, camera, aspect)
            imgs_warped.append(img_warped)
            corners.append(corner)

            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            _, mask_warped = self.warper.warp_image(mask, camera,
                                                    aspect, mask=True)
            masks_warped.append(mask_warped)

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
        return [SeamFinder.resize(seam_mask, mask)
                for seam_mask, mask in zip(seam_masks, masks)]

    def create_timelapse(self, img_names, imgs, corners):
        self.timelapser.initialize(corners, self.__get_sizes(imgs))
        for img_name, img, corner in zip(img_names, imgs, corners):
            self.timelapser.process_and_save_frame(img_name, img, corner)

    def blend_images(self, imgs, corners, seam_masks):
        self.blender.prepare(corners, self.__get_sizes(imgs))
        for img, corner, seam_mask in zip(imgs, corners, seam_masks):
            self.blender.feed(img, seam_mask, corner)
        return self.blender.blend()

    @staticmethod
    def __get_sizes(imgs):
        return [(img.shape[1], img.shape[0]) for img in imgs]
