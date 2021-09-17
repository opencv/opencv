import statistics
from types import SimpleNamespace

import cv2 as cv

from .image_handler import ImageHandler
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
         "work_megapix": ImageHandler.DEFAULT_MEDIUM_MEGAPIX,
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
         "seam_megapix": ImageHandler.DEFAULT_LOW_MEGAPIX,
         "seam": SeamFinder.DEFAULT_SEAM_FINDER,
         "compose_megapix": ImageHandler.DEFAULT_FINAL_MEGAPIX,
         "expos_comp": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
         "expos_comp_nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
         "expos_comp_block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
         "blend": Blender.DEFAULT_BLENDER,
         "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
         "timelapse": Timelapser.DEFAULT_TIMELAPSE}

    def __init__(self, **kwargs):
        args = Stitcher.DEFAULT_SETTINGS.copy()
        args.update(kwargs)
        args = SimpleNamespace(**args)

        self.img_handler = ImageHandler(args.work_megapix,
                                        args.seam_megapix,
                                        args.compose_megapix)
        self.finder = \
            FeatureDetector(args.features, nfeatures=args.n_features)
        match_conf = FeatureMatcher.get_match_conf(args.match_conf,
                                                   args.features)
        self.matcher = FeatureMatcher(args.matcher,
                                      args.rangewidth,
                                      try_use_gpu=args.try_cuda,
                                      match_conf=match_conf)
        self.subsetter = \
            Subsetter(args.conf_thresh, args.save_graph)
        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = CameraAdjuster(args.ba, args.ba_refine_mask)
        self.wave_corrector = WaveCorrector(args.wave_correct)
        self.warper = Warper(args.warp)
        self.compensator = \
            ExposureErrorCompensator(args.expos_comp, args.expos_comp_nr_feeds,
                                     args.expos_comp_block_size)
        self.seam_finder = SeamFinder(args.seam)
        self.blender = Blender(args.blend, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse)

    def stitch(self, img_names):
        self.img_handler.set_img_names(img_names)

        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs)
        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        panorama_scale, panorama_corners, panorama_sizes = \
            self.estimate_final_panorama_dimensions(cameras)

        self.initialize_composition(panorama_corners, panorama_sizes)

        imgs = self.resize_low_resolution(imgs)
        imgs = self.warp_low_resolution_images(imgs, cameras)
        self.estimate_exposure_errors(imgs)
        seam_masks = self.find_seam_masks(imgs)

        imgs = self.resize_final_resolution()
        imgs = self.warp_final_resolution_images(imgs, cameras)
        imgs = self.compensate_exposure_errors(imgs)
        seam_masks = self.resize_seam_masks(seam_masks)
        self.blend_images(imgs, seam_masks)

        return self.create_final_panorama()

    def resize_medium_resolution(self):
        return list(self.img_handler.resize_to_medium_resolution())

    def find_features(self, imgs):
        return [self.finder.detect_features(img) for img in imgs]

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, imgs, features, matches):
        self.subsetter.save_matches_graph_dot_file(self.img_handler.img_names,
                                                   matches)

        indices = self.subsetter.get_indices_to_keep(features, matches)

        self.img_handler.subset(indices)
        imgs = Subsetter.subset_list(imgs, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)

        return imgs, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_final_panorama_dimensions(self, cameras):
        self.img_handler.final_scaler.set_scale_by_img_size(self.img_handler.img_sizes[0])
        compose_work_aspect = self.img_handler.get_compose_work_aspect()

        focals = [cam.focal for cam in cameras]
        panorama_scale_determined_on_work_img = statistics.median(focals)

        panorama_scale = (panorama_scale_determined_on_work_img *
                          compose_work_aspect)
        panorama_corners = []
        panorama_sizes = []

        self.warper.set_scale(panorama_scale)
        for size, camera in zip(self.img_handler.img_sizes, cameras):
            sz = self.img_handler.final_scaler.get_scaled_img_size(size)
            roi = self.warper.warp_roi(*sz, camera, compose_work_aspect)
            panorama_corners.append(roi[0:2])
            panorama_sizes.append(roi[2:4])

        return panorama_scale, panorama_corners, panorama_sizes

    def estimate_final_panorama_scale(self, cameras):
        focals = [cam.focal for cam in cameras]
        panorama_scale = statistics.median(focals)
        return panorama_scale

    def initialize_composition(self, corners, sizes):
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes)
        else:
            self.blender.prepare(corners, sizes)

    def resize_low_resolution(self, imgs=None):
        return list(self.img_handler.resize_to_low_resolution(imgs))

    def warp_low_resolution_images(self, imgs, cameras):
        return list(self.warp_images(self.img_handler.low_scaler, imgs, cameras))

    def warp_final_resolution_images(self, imgs, cameras):
        return self.warp_images(self.img_handler.final_scaler, imgs, cameras)

    def warp_images(self, scaler, imgs, cameras):
        self._masks = []
        self._corners = []
        panorama_scale = self.estimate_final_panorama_scale(cameras)
        self.warper.set_scale(panorama_scale *
                              scaler.get_aspect_to(self.img_handler.medium_scaler))
        for img, camera in zip(imgs, cameras):
            img, mask, corner = \
                self.warper.warp_image_and_image_mask(
                    img, camera, scaler.get_aspect_to(self.img_handler.medium_scaler)
                    )
            self._masks.append(mask)
            self._corners.append(corner)
            yield img

    def estimate_exposure_errors(self, imgs):
        self.compensator.feed(self._corners, imgs, self._masks)

    def find_seam_masks(self, imgs):
        return self.seam_finder.find(imgs, self._corners, self._masks)

    def resize_final_resolution(self):
        return self.img_handler.resize_to_final_resolution()

    def compensate_exposure_errors(self, imgs):
        for idx, img in enumerate(imgs):
            yield self.compensator.apply(idx, self._corners[idx],
                                         img, self._masks[idx])

    def resize_seam_masks(self, seam_masks):
        for idx, seam_mask in enumerate(seam_masks):
            yield SeamFinder.resize(seam_mask, self._masks[idx])

    def blend_images(self, imgs, seam_masks):
        for idx, (img, seam_mask) in enumerate(zip(imgs, seam_masks)):
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(self._img_names[idx],
                                                       img,
                                                       self._corners[idx])
            else:
                self.blender.feed(img, seam_mask, self._corners[idx])

    def create_final_panorama(self):
        if not self.timelapser.do_timelapse:
            return self.blender.blend()

    def collect_garbage(self):
        del self.img_handler.img_names, self.img_handler.img_sizes, self._corners, self._masks
