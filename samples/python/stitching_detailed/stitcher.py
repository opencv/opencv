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

        work_imgs = []
        images = []
        full_img_sizes = []
        compose_imgs = []

        work_megapix_scaler = ImageToMegapixScaler(args.work_megapix)
        seam_megapix_scaler = ImageToMegapixScaler(args.seam_megapix)
        compose_megapix_scaler = ImageToMegapixScaler(args.compose_megapix)

        for img in img_names:
            full_img = read_image(img)
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            work_imgs.append(work_megapix_scaler.set_scale_and_downscale(full_img))
            compose_imgs.append(compose_megapix_scaler.set_scale_and_downscale(full_img))
            images.append(seam_megapix_scaler.set_scale_and_downscale(full_img))

# =============================================================================
# REGISTRATION PART
# =============================================================================


        image_registration = get_image_registration_object(args)
        indices, cameras = image_registration.register(img_names, work_imgs)
        seam_work_aspect = (work_megapix_scaler.scale /
                            seam_megapix_scaler.scale )



        img_names = Subsetter.subset_list(img_names, indices)
        images = Subsetter.subset_list(images, indices)
        full_img_sizes = Subsetter.subset_list(full_img_sizes, indices)
        num_images = len(images)


        import statistics
        focals = [cam.focal for cam in cameras]
        focals.sort()
        warped_image_scale = statistics.median(focals)

# =============================================================================
# COMPOSITION PART
# =============================================================================

        image_composition = get_image_composition_object(args)

        warp_type = args.warp
        result_name = args.output

        image_composition.warper.set_scale(warped_image_scale * seam_work_aspect)
        images_warped, masks_warped, corners = image_composition.warp_images(images,
                                                                             cameras,
                                                                             seam_work_aspect)

        masks_warped = image_composition.find_seam_masks(images_warped, masks_warped, corners)
        image_composition.estimate_exposure_errors(images_warped, masks_warped, corners)

        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        timelapser = Timelapser(args.timelapse)
        # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, name in enumerate(img_names):
            img = compose_imgs[idx]
            compose_work_aspect = compose_megapix_scaler.scale / work_megapix_scaler.scale
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                      int(round(full_img_sizes[i][1] * compose_scale)))
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])

            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            image_warped = image_composition.compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            if blender is None and not timelapser.do_timelapse:
                blender = Blender(args.blend, args.blend_strength)
                blender.prepare(corners, sizes)
            if timelapser.do_timelapse:
                timelapser.process_and_save_frame(img_names[idx],
                                                  image_warped_s,
                                                  corners[idx])
            else:
                blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
        if not timelapser.do_timelapse:
            self.result = blender.blend()
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
