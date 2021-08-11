from types import SimpleNamespace

import cv2 as cv
import numpy as np

from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .subsetter import Subsetter
from .camera_estimator import CameraEstimator
from .camera_adjuster import CameraAdjuster
from .camera_wave_corrector import WaveCorrector

from .image_to_megapix_scaler import ImageToMegapixScaler
from .image_registration import ImageRegistration
from .warper import Warper
from .exposure_error_compensator import ExposureErrorCompensator
from .seam_finder import SeamFinder
from .blender import Blender
from .timelapser import Timelapser
from .image_composition import ImageComposition


class Stitcher:

    def __init__(self, img_names, **kwargs):
        self.img_names = img_names
        self.args = SimpleNamespace(**kwargs)

    def stitch(self):
        args = self.args
        img_names = self.img_names

        full_img_sizes = []
        work_imgs = []
        seam_imgs = []
        compose_imgs = []

        work_megapix_scaler = ImageToMegapixScaler(args.work_megapix)
        seam_megapix_scaler = ImageToMegapixScaler(args.seam_megapix)
        compose_megapix_scaler = ImageToMegapixScaler(args.compose_megapix)

        for img in img_names:
            full_img = read_image(img)
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            work_imgs.append(work_megapix_scaler.set_scale_and_downscale(full_img))
            seam_imgs.append(seam_megapix_scaler.set_scale_and_downscale(full_img))
            compose_imgs.append(compose_megapix_scaler.set_scale_and_downscale(full_img))

        seam_work_aspect = (seam_megapix_scaler.scale /
                            work_megapix_scaler.scale)

        compose_work_aspect = (compose_megapix_scaler.scale /
                               work_megapix_scaler.scale)


# =============================================================================
# REGISTRATION PART
# =============================================================================


        image_registration = get_image_registration_object(args)
        indices, cameras, scale = image_registration.register(img_names,
                                                              work_imgs)

        img_names = Subsetter.subset_list(img_names, indices)
        seam_imgs = Subsetter.subset_list(seam_imgs, indices)
        compose_imgs = Subsetter.subset_list(compose_imgs, indices)
        full_img_sizes = Subsetter.subset_list(full_img_sizes, indices)


# =============================================================================
# COMPOSITION PART
# =============================================================================

        image_composition = get_image_composition_object(args)

        warp_type = args.warp
        result_name = args.output

        image_composition.warper.set_scale(warp_type, scale * seam_work_aspect)
        images_warped, masks_warped, corners = image_composition.warp_images(seam_imgs,
                                                                             cameras,
                                                                             seam_work_aspect)

        seam_masks = image_composition.find_seam_masks(images_warped, masks_warped, corners)
        image_composition.estimate_exposure_errors(images_warped, masks_warped, corners)

        corners = []
        sizes = []
        scale *= compose_work_aspect
        image_composition.warper.set_scale(warp_type, scale)
        warper = image_composition.warper.warper
        for i in range(0, len(img_names)):
            sz = (int(round(full_img_sizes[i][0] * compose_megapix_scaler.scale)),
                  int(round(full_img_sizes[i][1] * compose_megapix_scaler.scale)))
            roi = warper.warpRoi(sz, Warper.get_K(cameras[i], compose_work_aspect), cameras[i].R)
            corners.append(roi[0:2])
            sizes.append(roi[2:4])

        images_warped, masks_warped, corners = image_composition.warp_images(compose_imgs,cameras,compose_work_aspect)
        images_warped = image_composition.compensate_exposure_errors(images_warped, masks_warped, corners)
        seam_masks = image_composition.resize_seam_masks_to_original_resolution(seam_masks, masks_warped)
        if image_composition.timelapser.do_timelapse:
            image_composition.create_timelapse(img_names, images_warped, corners)
        if not image_composition.timelapser.do_timelapse:
            self.result = image_composition.blend_images(images_warped, corners, sizes, seam_masks)
            cv.imwrite(result_name, self.result)
            zoom_x = 600.0 / self.result.shape[1]
            dst = cv.normalize(src=self.result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            self.dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)

        print("Done")


def read_image(img_name):
    img = cv.imread(img_name)
    if img is None:
        print("Cannot read image ", img_name)
        exit()
    return img

def get_image_registration_object(args):
    if not args.match_conf:
        args.match_conf = FeatureMatcher.get_default_match_conf(args.features)

    finder = FeatureDetector(args.features)
    matcher = FeatureMatcher(args.matcher, args.rangewidth,
                             try_use_gpu=args.try_cuda,
                             match_conf=args.match_conf)
    subsetter = Subsetter(args.conf_thresh, args.save_graph)
    camera_estimator = CameraEstimator(args.estimator)
    camera_adjuster = CameraAdjuster(args.ba, args.ba_refine_mask)
    wave_corrector = WaveCorrector(args.wave_correct)

    return ImageRegistration(finder, matcher, subsetter, camera_estimator,
                             camera_adjuster, wave_corrector)

def get_image_composition_object(args):
    warper = Warper(args.warp)
    seam_finder = SeamFinder(args.seam)
    compensator = ExposureErrorCompensator(args.expos_comp,
                                           args.expos_comp_nr_feeds,
                                           args.expos_comp_block_size)
    blender = Blender(args.blend, args.blend_strength)
    timelapser = Timelapser(args.timelapse)

    return ImageComposition(warper, seam_finder, compensator, blender,
                            timelapser)
