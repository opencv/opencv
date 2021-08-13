from types import SimpleNamespace

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

from .image_data import ImageData
from .image_registration import ImageRegistration
from .image_composition import ImageComposition


class Stitcher:

    def __init__(self, img_names, **kwargs):
        self.img_names = img_names
        settings = Stitcher.DEFAULT_SETTINGS.copy()
        settings.update(kwargs)
        self.settings = SimpleNamespace(**settings)

    def stitch(self):
        image_data = self.get_image_data_object()
        image_registration = self.get_image_registration_object()
        indices, cameras, panorama_scale = image_registration.register(image_data.img_names,
                                                                       image_data.work_imgs)
        image_data.subset(indices)
        image_composition = self.get_image_composition_object()
        return image_composition.compose(image_data, cameras, panorama_scale)

    def get_image_data_object(self):
        return ImageData(self.img_names,
                         self.settings.work_megapix,
                         self.settings.seam_megapix,
                         self.settings.compose_megapix)

    def get_image_registration_object(self):
        if not self.settings.match_conf:
            self.settings.match_conf = FeatureMatcher.get_default_match_conf(
                self.settings.features
                )

        finder = FeatureDetector(self.settings.features)
        matcher = FeatureMatcher(self.settings.matcher,
                                 self.settings.rangewidth,
                                 try_use_gpu=self.settings.try_cuda,
                                 match_conf=self.settings.match_conf)
        subsetter = Subsetter(self.settings.conf_thresh,
                              self.settings.save_graph)
        camera_estimator = CameraEstimator(self.settings.estimator)
        camera_adjuster = CameraAdjuster(self.settings.ba,
                                         self.settings.ba_refine_mask)
        wave_corrector = WaveCorrector(self.settings.wave_correct)

        return ImageRegistration(finder,
                                 matcher,
                                 subsetter,
                                 camera_estimator,
                                 camera_adjuster,
                                 wave_corrector)

    def get_image_composition_object(self):
        warper = Warper(self.settings.warp)
        seam_finder = SeamFinder(self.settings.seam)
        compensator = ExposureErrorCompensator(
            self.settings.expos_comp,
            self.settings.expos_comp_nr_feeds,
            self.settings.expos_comp_block_size
            )
        blender = Blender(self.settings.blend, self.settings.blend_strength)
        timelapser = Timelapser(self.settings.timelapse)

        return ImageComposition(warper,
                                seam_finder,
                                compensator,
                                blender,
                                timelapser)

    DEFAULT_SETTINGS = {
         "try_cuda": False,
         "work_megapix": ImageData.DEFAULT_WORK_MEGAPIX,
         "features": FeatureDetector.DEFAULT_DETECTOR,
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
         "seam_megapix": ImageData.DEFAULT_SEAM_MEGAPIX,
         "seam": SeamFinder.DEFAULT_SEAM_FINDER,
         "compose_megapix": ImageData.DEFAULT_COMPOSE_MEGAPIX,
         "expos_comp": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
         "expos_comp_nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
         "expos_comp_block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
         "blend": Blender.DEFAULT_BLENDER,
         "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
         "timelapse": Timelapser.DEFAULT_TIMELAPSE}
