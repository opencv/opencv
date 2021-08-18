import statistics
from types import SimpleNamespace

import cv2 as cv
import numpy as np

from .image_to_megapix_scaler import ImageToMegapixScaler
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .subsetter import Subsetter
from .camera_estimator import CameraEstimator
from .camera_adjuster import CameraAdjuster
from .camera_wave_corrector import WaveCorrector
from .warper import Warper
from .exposure_error_compensator import ExposureErrorCompensator
from .seam_finder import SeamFinder
from .blender import Blender
from .timelapser import Timelapser


class Stitcher:
    DEFAULT_SETTINGS = {
         "try_cuda": False,
         "work_megapix": 0.6,
         "features": FeatureDetector.DEFAULT_DETECTOR,
         "n_features": 500,
         "matcher": FeatureMatcher.DEFAULT_MATCHER,
         "rangewidth": FeatureMatcher.DEFAULT_RANGE_WIDTH,
         "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
         "match_conf": None,
         "conf_thresh": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
         "ba": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
         "ba_refine_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
         "wave_correct": WaveCorrector.DEFAULT_WAVE_CORRECTION,
         "save_graph": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
         "warp": Warper.DEFAULT_WARP_TYPE,
         "seam_megapix": 0.1,
         "seam": SeamFinder.DEFAULT_SEAM_FINDER,
         "compose_megapix": -1,
         "expos_comp": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
         "expos_comp_nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
         "expos_comp_block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
         "blend": Blender.DEFAULT_BLENDER,
         "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
         "timelapse": Timelapser.DEFAULT_TIMELAPSE}

    @staticmethod
    def stitch(img_names, **kwargs):
        args = Stitcher.DEFAULT_SETTINGS.copy()
        args.update(kwargs)
        args = SimpleNamespace(**args)

        full_img_sizes = []
        features = []
        seam_images = []

        work_megapix_scaler = ImageToMegapixScaler(args.work_megapix)
        seam_megapix_scaler = ImageToMegapixScaler(args.seam_megapix)
        finder = FeatureDetector(args.features,
                                 nfeatures=args.n_features)

        for name in img_names:
            full_img = Stitcher.read_image(name)
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            img = work_megapix_scaler.set_scale_and_downscale(full_img)
            img_features = finder.detect_features(img)
            features.append(img_features)
            img = seam_megapix_scaler.set_scale_and_downscale(full_img)
            seam_images.append(img)

        seam_work_aspect = (seam_megapix_scaler.scale /
                            work_megapix_scaler.scale)

        if not args.match_conf:
            args.match_conf = FeatureMatcher.get_default_match_conf(args.features)
        matcher = FeatureMatcher(args.matcher,
                                 args.rangewidth,
                                 try_use_gpu=args.try_cuda,
                                 match_conf=args.match_conf)

        matches = matcher.match_features(features)

        subsetter = Subsetter(args.conf_thresh,
                              args.save_graph)

        subsetter.save_matches_graph_dot_file(img_names, matches)

        indices = subsetter.get_indices_to_keep(features, matches)

        img_names = Subsetter.subset_list(img_names, indices)
        full_img_sizes = Subsetter.subset_list(full_img_sizes, indices)
        seam_images = Subsetter.subset_list(seam_images, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)

        num_images = len(img_names)
        if num_images < 2:
            print("Need more images")
            exit()

        camera_estimator = CameraEstimator(args.estimator)
        cameras = camera_estimator.estimate(features, matches)

        camera_adjuster = CameraAdjuster(args.ba,
                                         args.ba_refine_mask)
        cameras = camera_adjuster.adjust(features, matches, cameras)

        wave_corrector = WaveCorrector(args.wave_correct)
        cameras = wave_corrector.correct(cameras)

        focals = [cam.focal for cam in cameras]
        warped_image_scale = statistics.median(focals)

        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []

        warper = Warper(args.warp)
        warper.set_scale(warped_image_scale*seam_work_aspect)

        for img, camera in zip(seam_images, cameras):
            corner, img_warped = warper.warp_image(img, camera, seam_work_aspect)
            images_warped.append(img_warped)
            corners.append(corner)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            _, mask_warped = warper.warp_image(mask, camera, seam_work_aspect, mask=True)
            masks_warped.append(mask_warped)

        compensator = ExposureErrorCompensator(
            args.expos_comp,
            args.expos_comp_nr_feeds,
            args.expos_comp_block_size
            )
        compensator.feed(corners, images_warped, masks_warped)

        seam_finder = SeamFinder(args.seam)
        seam_masks = seam_finder.find(images_warped, corners, masks_warped)

        corners = []
        sizes = []

        compose_scale = 1
        if args.compose_megapix > 0:
            compose_scale = min(1.0, np.sqrt(args.compose_megapix * 1e6 / (full_img_sizes[0][1] * full_img_sizes[0][0])))
        compose_work_aspect = compose_scale / work_megapix_scaler.scale
        warper.set_scale(warped_image_scale * compose_work_aspect)
        for size, camera in zip(full_img_sizes, cameras):
            sz = (int(round(size[0] * compose_scale)),
                  int(round(size[1] * compose_scale)))
            roi = warper.warp_roi(*sz, camera, compose_work_aspect)
            corners.append(roi[0:2])
            sizes.append(roi[2:4])

        blender = Blender(args.blend, args.blend_strength)
        timelapser = Timelapser(args.timelapse)
        if timelapser.do_timelapse:
            timelapser.initialize(corners, sizes)
        else:
            blender.prepare(corners, sizes)

        compose_megapix_scaler = ImageToMegapixScaler(args.compose_megapix)
        for idx, (name, camera, seam_mask) in enumerate(
                zip(img_names, cameras, seam_masks)):

            full_img = Stitcher.read_image(name)
            img = compose_megapix_scaler.set_scale_and_downscale(full_img)

            corner, image_warped = warper.warp_image(img, camera, compose_work_aspect)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            _, mask_warped = warper.warp_image(mask, camera, compose_work_aspect, mask=True)

            compensator.apply(idx, corner, image_warped, mask_warped)

            resized_seam_mask = SeamFinder.resize(seam_mask, mask_warped)

            if timelapser.do_timelapse:
                timelapser.process_and_save_frame(name, image_warped, corner)
            else:
                blender.feed(image_warped, resized_seam_mask, corner)

        if not timelapser.do_timelapse:
            return blender.blend()

    @staticmethod
    def read_image(img_name):
        img = cv.imread(img_name)
        if img is None:
            print("Cannot read image ", img_name)
            exit()
        return img
