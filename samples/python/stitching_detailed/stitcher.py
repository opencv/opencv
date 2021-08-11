from types import SimpleNamespace

from .image_data import ImageData

from .image_registration import ImageRegistration
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .subsetter import Subsetter
from .camera_estimator import CameraEstimator
from .camera_adjuster import CameraAdjuster
from .camera_wave_corrector import WaveCorrector

from .image_composition import ImageComposition
from .warper import Warper
from .exposure_error_compensator import ExposureErrorCompensator
from .seam_finder import SeamFinder
from .blender import Blender
from .timelapser import Timelapser


class Stitcher:

    def __init__(self, img_names, **kwargs):
        self.img_names = img_names
        self.args = SimpleNamespace(**kwargs)

    def stitch(self):
        image_data = Stitcher.get_image_data_object(self.img_names, self.args)
        image_registration = Stitcher.get_image_registration_object(self.args)
        indices, cameras, panorama_scale = image_registration.register(image_data.img_names,
                                                                       image_data.work_imgs)
        image_data.subset(indices)
        image_composition = Stitcher.get_image_composition_object(self.args)
        return image_composition.compose(image_data, cameras, panorama_scale)

    def get_image_data_object(img_names, args):
        return ImageData(img_names,
                         args.work_megapix,
                         args.seam_megapix,
                         args.compose_megapix)

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
