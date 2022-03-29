from types import SimpleNamespace

from .image_handler import ImageHandler
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .subsetter import Subsetter
from .camera_estimator import CameraEstimator
from .camera_adjuster import CameraAdjuster
from .camera_wave_corrector import WaveCorrector
from .warper import Warper
from .cropper import Cropper
from .exposure_error_compensator import ExposureErrorCompensator
from .seam_finder import SeamFinder
from .blender import Blender
from .timelapser import Timelapser
from .stitching_error import StitchingError


class Stitcher:
    DEFAULT_SETTINGS = {
         "medium_megapix": ImageHandler.DEFAULT_MEDIUM_MEGAPIX,
         "detector": FeatureDetector.DEFAULT_DETECTOR,
         "nfeatures": 500,
         "matcher_type": FeatureMatcher.DEFAULT_MATCHER,
         "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
         "try_use_gpu": False,
         "match_conf": None,
         "confidence_threshold": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
         "matches_graph_dot_file": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
         "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
         "adjuster": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
         "refinement_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
         "wave_correct_kind": WaveCorrector.DEFAULT_WAVE_CORRECTION,
         "warper_type": Warper.DEFAULT_WARP_TYPE,
         "low_megapix": ImageHandler.DEFAULT_LOW_MEGAPIX,
         "crop": Cropper.DEFAULT_CROP,
         "compensator": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
         "nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
         "block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
         "finder": SeamFinder.DEFAULT_SEAM_FINDER,
         "final_megapix": ImageHandler.DEFAULT_FINAL_MEGAPIX,
         "blender_type": Blender.DEFAULT_BLENDER,
         "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
         "timelapse": Timelapser.DEFAULT_TIMELAPSE}

    def __init__(self, **kwargs):
        self.initialize_stitcher(**kwargs)

    def initialize_stitcher(self, **kwargs):
        self.settings = Stitcher.DEFAULT_SETTINGS.copy()
        self.validate_kwargs(kwargs)
        self.settings.update(kwargs)

        args = SimpleNamespace(**self.settings)
        self.img_handler = ImageHandler(args.medium_megapix,
                                        args.low_megapix,
                                        args.final_megapix)
        self.detector = \
            FeatureDetector(args.detector, nfeatures=args.nfeatures)
        match_conf = \
            FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher(args.matcher_type, args.range_width,
                                      try_use_gpu=args.try_use_gpu,
                                      match_conf=match_conf)
        self.subsetter = \
            Subsetter(args.confidence_threshold, args.matches_graph_dot_file)
        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = \
            CameraAdjuster(args.adjuster, args.refinement_mask)
        self.wave_corrector = WaveCorrector(args.wave_correct_kind)
        self.warper = Warper(args.warper_type)
        self.cropper = Cropper(args.crop)
        self.compensator = \
            ExposureErrorCompensator(args.compensator, args.nr_feeds,
                                     args.block_size)
        self.seam_finder = SeamFinder(args.finder)
        self.blender = Blender(args.blender_type, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse)

    def stitch(self, img_names):
        self.initialize_registration(img_names)
        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs)
        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        self.estimate_scale(cameras)

        imgs = self.resize_low_resolution(imgs)
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras)
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = \
            self.crop_low_resolution(imgs, masks, corners, sizes)
        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        imgs, masks, corners, sizes = \
            self.crop_final_resolution(imgs, masks, corners, sizes)
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        return self.create_final_panorama()

    def initialize_registration(self, img_names):
        self.img_handler.set_img_names(img_names)

    def resize_medium_resolution(self):
        return list(self.img_handler.resize_to_medium_resolution())

    def find_features(self, imgs):
        return [self.detector.detect_features(img) for img in imgs]

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, imgs, features, matches):
        names, sizes, imgs, features, matches = \
            self.subsetter.subset(self.img_handler.img_names,
                                  self.img_handler.img_sizes,
                                  imgs, features, matches)
        self.img_handler.img_names, self.img_handler.img_sizes = names, sizes
        return imgs, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        self.warper.set_scale(cameras)

    def resize_low_resolution(self, imgs=None):
        return list(self.img_handler.resize_to_low_resolution(imgs))

    def warp_low_resolution(self, imgs, cameras):
        sizes = self.img_handler.get_low_img_sizes()
        camera_aspect = self.img_handler.get_medium_to_low_ratio()
        imgs, masks, corners, sizes = \
            self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        sizes = self.img_handler.get_final_img_sizes()
        camera_aspect = self.img_handler.get_medium_to_final_ratio()
        return self.warp(imgs, cameras, sizes, camera_aspect)

    def warp(self, imgs, cameras, sizes, aspect=1):
        imgs = self.warper.warp_images(imgs, cameras, aspect)
        masks = self.warper.create_and_warp_masks(sizes, cameras, aspect)
        corners, sizes = self.warper.warp_rois(sizes, cameras, aspect)
        return imgs, masks, corners, sizes

    def prepare_cropper(self, imgs, masks, corners, sizes):
        self.cropper.prepare(imgs, masks, corners, sizes)

    def crop_low_resolution(self, imgs, masks, corners, sizes):
        imgs, masks, corners, sizes = self.crop(imgs, masks, corners, sizes)
        return list(imgs), list(masks), corners, sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        lir_aspect = self.img_handler.get_low_to_final_ratio()
        return self.crop(imgs, masks, corners, sizes, lir_aspect)

    def crop(self, imgs, masks, corners, sizes, aspect=1):
        masks = self.cropper.crop_images(masks, aspect)
        imgs = self.cropper.crop_images(imgs, aspect)
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def estimate_exposure_errors(self, corners, imgs, masks):
        self.compensator.feed(corners, imgs, masks)

    def find_seam_masks(self, imgs, corners, masks):
        return self.seam_finder.find(imgs, corners, masks)

    def resize_final_resolution(self):
        return self.img_handler.resize_to_final_resolution()

    def compensate_exposure_errors(self, corners, imgs):
        for idx, (corner, img) in enumerate(zip(corners, imgs)):
            yield self.compensator.apply(idx, corner, img, self.get_mask(idx))

    def resize_seam_masks(self, seam_masks):
        for idx, seam_mask in enumerate(seam_masks):
            yield SeamFinder.resize(seam_mask, self.get_mask(idx))

    def set_masks(self, mask_generator):
        self.masks = mask_generator
        self.mask_index = -1

    def get_mask(self, idx):
        if idx == self.mask_index + 1:
            self.mask_index += 1
            self.mask = next(self.masks)
            return self.mask
        elif idx == self.mask_index:
            return self.mask
        else:
            raise StitchingError("Invalid Mask Index!")

    def initialize_composition(self, corners, sizes):
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes)
        else:
            self.blender.prepare(corners, sizes)

    def blend_images(self, imgs, masks, corners):
        for idx, (img, mask, corner) in enumerate(zip(imgs, masks, corners)):
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(
                    self.img_handler.img_names[idx], img, corner
                    )
            else:
                self.blender.feed(img, mask, corner)

    def create_final_panorama(self):
        if not self.timelapser.do_timelapse:
            panorama, _ = self.blender.blend()
            return panorama

    @staticmethod
    def validate_kwargs(kwargs):
        for arg in kwargs:
            if arg not in Stitcher.DEFAULT_SETTINGS:
                raise StitchingError("Invalid Argument: " + arg)
