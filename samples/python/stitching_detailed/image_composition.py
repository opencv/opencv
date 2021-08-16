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
        img_names = img_data.img_names
        img_sizes = img_data.full_img_sizes
        compose_imgs = img_data.compose_imgs
        seam_imgs = img_data.seam_imgs
        compose_scale = img_data.compose_megapix_scaler.scale
        compose_work_aspect = img_data.compose_work_aspect
        seam_work_aspect = img_data.seam_work_aspect

        imgs, masks, corners = self.warp_images(seam_imgs, cameras, panorama_scale, seam_work_aspect)
        seam_masks = self.find_seam_masks(imgs, masks, corners)
        self.estimate_exposure_errors(imgs, masks, corners)
        imgs, masks, corners = self.warp_images(compose_imgs, cameras, panorama_scale, compose_work_aspect)
        corners2, sizes = self.warp_rois(cameras, img_sizes, compose_scale, compose_work_aspect)
        assert [(img.shape[1], img.shape[0]) for img in imgs] == sizes
        assert corners == corners2
        imgs = self.compensate_exposure_errors(imgs, masks, corners)
        seam_masks = self.resize_seam_masks_to_original_resolution(seam_masks,
                                                                   masks)
        if self.timelapser.do_timelapse:
            self.create_timelapse(img_names, imgs, corners, sizes)

        final_pano = self.blend_images(imgs, corners, sizes, seam_masks)
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

    def create_timelapse(self, img_names, imgs, corners, sizes):
        self.timelapser.initialize(corners, sizes)
        for img_name, img, corner in zip(img_names, imgs, corners):
            self.timelapser.process_and_save_frame(img_name, img, corner)

    def blend_images(self, imgs, corners, sizes, seam_masks):
        self.blender.prepare(corners, sizes)
        for img, corner, seam_mask in zip(imgs, corners, seam_masks):
            self.blender.feed(img, seam_mask, corner)
        return self.blender.blend()

    def warp_rois(self, cameras, full_img_sizes, compose_scale, compose_work_aspect):
        corners = []
        sizes = []
        for size, camera in zip(full_img_sizes, cameras):
            sz = (int(round(size[0] * compose_scale)),
                  int(round(size[1] * compose_scale)))
            roi = self.warper.warp_roi(*sz, camera, compose_work_aspect)
            corners.append(roi[0:2])
            sizes.append(roi[2:4])
        return corners, sizes
