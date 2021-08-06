"""
Stitching sample (advanced)
===========================

Show how to use Stitcher API from python.
"""

# Python 2/3 compatibility
from __future__ import print_function

import argparse

import cv2 as cv
import numpy as np

from stitching_detailed.stitcher import Stitcher

from stitching_detailed.feature_detector import FeatureDetector
from stitching_detailed.feature_matcher import FeatureMatcher
from stitching_detailed.subsetter import Subsetter
from stitching_detailed.camera_estimator import CameraEstimator
from stitching_detailed.camera_adjuster import CameraAdjuster
from stitching_detailed.camera_wave_corrector import WaveCorrector
from stitching_detailed.warper import Warper
from stitching_detailed.exposure_error_compensator import ExposureErrorCompensator  # noqa
from stitching_detailed.seam_finder import SeamFinder
from stitching_detailed.blender import Blender
from stitching_detailed.timelapser import Timelapser

parser = argparse.ArgumentParser(
    prog="stitching_detailed.py", description="Rotation model images stitcher"
)
parser.add_argument(
    'img_names', nargs='+',
    help="Files to stitch", type=str
)
parser.add_argument(
    '--try_cuda',
    action='store',
    default=False,
    help="Try to use CUDA. The default value is no. "
         "All default values are for CPU mode.",
    type=bool, dest='try_cuda'
)
parser.add_argument(
    '--work_megapix', action='store', default=0.6,
    help="Resolution for image registration step. The default is 0.6 Mpx",
    type=float, dest='work_megapix'
)
parser.add_argument(
    '--features', action='store',
    default=FeatureDetector.DEFAULT_DETECTOR,
    help="Type of features used for images matching. "
         "The default is '%s'." % FeatureDetector.DEFAULT_DETECTOR,
    choices=FeatureDetector.DETECTOR_CHOICES.keys(),
    type=str, dest='features'
)
parser.add_argument(
    '--matcher', action='store', default=FeatureMatcher.DEFAULT_MATCHER,
    help="Matcher used for pairwise image matching. "
         "The default is '%s'." % FeatureMatcher.DEFAULT_MATCHER,
    choices=FeatureMatcher.MATCHER_CHOICES,
    type=str, dest='matcher'
)
parser.add_argument(
    '--rangewidth', action='store', default=FeatureMatcher.DEFAULT_RANGE_WIDTH,
    help="uses range_width to limit number of images to match with.",
    type=int, dest='rangewidth'
)
parser.add_argument(
    '--estimator', action='store',
    default=CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
    help="Type of estimator used for transformation estimation. "
         "The default is '%s'." % CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
    choices=CameraEstimator.CAMERA_ESTIMATOR_CHOICES.keys(),
    type=str, dest='estimator'
)
parser.add_argument(
    '--match_conf', action='store',
    help="Confidence for feature matching step. "
         "The default is 0.3 for ORB and 0.65 for other feature types.",
    type=float, dest='match_conf'
)
parser.add_argument(
    '--conf_thresh', action='store',
    default=Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
    help="Threshold for two images are from the same panorama confidence. "
         "The default is '%s'." % Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
    type=float, dest='conf_thresh'
)
parser.add_argument(
    '--ba', action='store', default=CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
    help="Bundle adjustment cost function. "
         "The default is '%s'." % CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
    choices=CameraAdjuster.CAMERA_ADJUSTER_CHOICES.keys(),
    type=str, dest='ba'
)
parser.add_argument(
    '--ba_refine_mask', action='store',
    default=CameraAdjuster.DEFAULT_REFINEMENT_MASK,
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
         "where 'x' means refine respective parameter and '_' means don't "
         "refine, and has the following format:<fx><skew><ppx><aspect><ppy>. "
         "The default mask is '%s'. "
         "If bundle adjustment doesn't support estimation of selected "
         "parameter then the respective flag is ignored."
         "" % CameraAdjuster.DEFAULT_REFINEMENT_MASK,
    type=str, dest='ba_refine_mask'
)
parser.add_argument(
    '--wave_correct', action='store',
    default=WaveCorrector.DEFAULT_WAVE_CORRECTION,
    help="Perform wave effect correction. "
         "The default is '%s'" % WaveCorrector.DEFAULT_WAVE_CORRECTION,
    choices=WaveCorrector.WAVE_CORRECT_CHOICES.keys(),
    type=str, dest='wave_correct'
)
parser.add_argument(
    '--save_graph', action='store',
    default=Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
    help="Save matches graph represented in DOT language to <file_name> file.",
    type=str, dest='save_graph'
)
parser.add_argument(
    '--warp', action='store', default=Warper.DEFAULT_WARP_TYPE,
    help="Warp surface type. The default is '%s'." % Warper.DEFAULT_WARP_TYPE,
    choices=Warper.WARP_TYPE_CHOICES,
    type=str, dest='warp'
)
parser.add_argument(
    '--seam_megapix', action='store', default=0.1,
    help="Resolution for seam estimation step. The default is 0.1 Mpx.",
    type=float, dest='seam_megapix'
)
parser.add_argument(
    '--seam', action='store', default=SeamFinder.DEFAULT_SEAM_FINDER,
    help="Seam estimation method. "
         "The default is '%s'." % SeamFinder.DEFAULT_SEAM_FINDER,
    choices=SeamFinder.SEAM_FINDER_CHOICES.keys(),
    type=str, dest='seam'
)
parser.add_argument(
    '--compose_megapix', action='store', default=-1,
    help="Resolution for compositing step. Use -1 for original resolution. "
         "The default is -1",
    type=float, dest='compose_megapix'
)
parser.add_argument(
    '--expos_comp', action='store',
    default=ExposureErrorCompensator.DEFAULT_COMPENSATOR,
    help="Exposure compensation method. "
         "The default is '%s'." % ExposureErrorCompensator.DEFAULT_COMPENSATOR,
    choices=ExposureErrorCompensator.COMPENSATOR_CHOICES.keys(),
    type=str, dest='expos_comp'
)
parser.add_argument(
    '--expos_comp_nr_feeds', action='store',
    default=ExposureErrorCompensator.DEFAULT_NR_FEEDS,
    help="Number of exposure compensation feed.",
    type=np.int32, dest='expos_comp_nr_feeds'
)
parser.add_argument(
    '--expos_comp_block_size', action='store',
    default=ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
    help="BLock size in pixels used by the exposure compensator. "
         "The default is '%s'." % ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
    type=np.int32, dest='expos_comp_block_size'
)
parser.add_argument(
    '--blend', action='store', default=Blender.DEFAULT_BLENDER,
    help="Blending method. The default is '%s'." % Blender.DEFAULT_BLENDER,
    choices=Blender.BLENDER_CHOICES,
    type=str, dest='blend'
)
parser.add_argument(
    '--blend_strength', action='store', default=Blender.DEFAULT_BLEND_STRENGTH,
    help="Blending strength from [0,100] range. "
         "The default is '%s'." % Blender.DEFAULT_BLEND_STRENGTH,
    type=np.int32, dest='blend_strength'
)
parser.add_argument(
    '--output', action='store', default='result.jpg',
    help="The default is 'result.jpg'",
    type=str, dest='output'
)
parser.add_argument(
    '--timelapse', action='store', default=Timelapser.DEFAULT_TIMELAPSE,
    help="Output warped images separately as frames of a time lapse movie, "
         "with 'fixed_' prepended to input file names. "
         "The default is '%s'." % Timelapser.DEFAULT_TIMELAPSE,
    choices=Timelapser.TIMELAPSE_CHOICES,
    type=str, dest='timelapse'
)

__doc__ += '\n' + parser.format_help()

if __name__ == '__main__':
    print(__doc__)
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)
    img_names = args_dict.pop("img_names")
    img_names = [cv.samples.findFile(img_name) for img_name in img_names]

    stitcher = Stitcher(img_names, **args_dict)
    stitcher.stitch()

    cv.imshow(args.output, stitcher.dst)
    cv.waitKey()
    cv.destroyAllWindows()
