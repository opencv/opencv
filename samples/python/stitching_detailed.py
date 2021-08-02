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
from stitching_detailed.stitcher_choices import (ESTIMATOR_CHOICES,
                                                 BA_COST_CHOICES,
                                                 WAVE_CORRECT_CHOICES,
                                                 WARP_CHOICES,
                                                 SEAM_FIND_CHOICES,
                                                 EXPOS_COMP_CHOICES,
                                                 BLEND_CHOICES)

from stitching_detailed.feature_detector import FeatureDetector

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
    default=FeatureDetector.default,
    help="Type of features used for images matching. "
         "The default is '%s'." % FeatureDetector.default,
    choices=FeatureDetector.choices.keys(),
    type=str, dest='features'
)
parser.add_argument(
    '--matcher', action='store', default='homography',
    help="Matcher used for pairwise image matching. "
         "The default is 'homography'.",
    choices=('homography', 'affine'),
    type=str, dest='matcher'
)
parser.add_argument(
    '--estimator', action='store', default=list(ESTIMATOR_CHOICES.keys())[0],
    help="Type of estimator used for transformation estimation. "
         "The default is '%s'." % list(ESTIMATOR_CHOICES.keys())[0],
    choices=ESTIMATOR_CHOICES.keys(),
    type=str, dest='estimator'
)
parser.add_argument(
    '--match_conf', action='store',
    help="Confidence for feature matching step. "
         "The default is 0.3 for ORB and 0.65 for other feature types.",
    type=float, dest='match_conf'
)
parser.add_argument(
    '--conf_thresh', action='store', default=1.0,
    help="Threshold for two images are from the same panorama confidence. "
         "The default is 1.0.",
    type=float, dest='conf_thresh'
)
parser.add_argument(
    '--ba', action='store', default=list(BA_COST_CHOICES.keys())[0],
    help="Bundle adjustment cost function. "
         "The default is '%s'." % list(BA_COST_CHOICES.keys())[0],
    choices=BA_COST_CHOICES.keys(),
    type=str, dest='ba'
)
parser.add_argument(
    '--ba_refine_mask', action='store', default='xxxxx',
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
         "where 'x' means refine respective parameter and '_' means don't "
         "refine, and has the following format:<fx><skew><ppx><aspect><ppy>. "
         "The default mask is 'xxxxx'. "
         "If bundle adjustment doesn't support estimation of selected "
         "parameter then the respective flag is ignored.",
    type=str, dest='ba_refine_mask'
)
parser.add_argument(
    '--wave_correct', action='store',
    default=list(WAVE_CORRECT_CHOICES.keys())[0],
    help="Perform wave effect correction. "
         "The default is '%s'" % list(WAVE_CORRECT_CHOICES.keys())[0],
    choices=WAVE_CORRECT_CHOICES.keys(),
    type=str, dest='wave_correct'
)
parser.add_argument(
    '--save_graph', action='store', default=None,
    help="Save matches graph represented in DOT language to <file_name> file.",
    type=str, dest='save_graph'
)
parser.add_argument(
    '--warp', action='store', default=WARP_CHOICES[0],
    help="Warp surface type. The default is '%s'." % WARP_CHOICES[0],
    choices=WARP_CHOICES,
    type=str, dest='warp'
)
parser.add_argument(
    '--seam_megapix', action='store', default=0.1,
    help="Resolution for seam estimation step. The default is 0.1 Mpx.",
    type=float, dest='seam_megapix'
)
parser.add_argument(
    '--seam', action='store', default=list(SEAM_FIND_CHOICES.keys())[0],
    help="Seam estimation method. "
         "The default is '%s'." % list(SEAM_FIND_CHOICES.keys())[0],
    choices=SEAM_FIND_CHOICES.keys(),
    type=str, dest='seam'
)
parser.add_argument(
    '--compose_megapix', action='store', default=-1,
    help="Resolution for compositing step. Use -1 for original resolution. "
         "The default is -1",
    type=float, dest='compose_megapix'
)
parser.add_argument(
    '--expos_comp', action='store', default=list(EXPOS_COMP_CHOICES.keys())[0],
    help="Exposure compensation method. "
         "The default is '%s'." % list(EXPOS_COMP_CHOICES.keys())[0],
    choices=EXPOS_COMP_CHOICES.keys(),
    type=str, dest='expos_comp'
)
parser.add_argument(
    '--expos_comp_nr_feeds', action='store', default=1,
    help="Number of exposure compensation feed.",
    type=np.int32, dest='expos_comp_nr_feeds'
)
parser.add_argument(
    '--expos_comp_nr_filtering', action='store', default=2,
    help="Number of filtering iterations of the exposure compensation gains.",
    type=float, dest='expos_comp_nr_filtering'
)
parser.add_argument(
    '--expos_comp_block_size', action='store', default=32,
    help="BLock size in pixels used by the exposure compensator. "
         "The default is 32.",
    type=np.int32, dest='expos_comp_block_size'
)
parser.add_argument(
    '--blend', action='store', default=BLEND_CHOICES[0],
    help="Blending method. The default is '%s'." % BLEND_CHOICES[0],
    choices=BLEND_CHOICES,
    type=str, dest='blend'
)
parser.add_argument(
    '--blend_strength', action='store', default=5,
    help="Blending strength from [0,100] range. The default is 5",
    type=np.int32, dest='blend_strength'
)
parser.add_argument(
    '--output', action='store', default='result.jpg',
    help="The default is 'result.jpg'",
    type=str, dest='output'
)
parser.add_argument(
    '--timelapse', action='store', default=None,
    help="Output warped images separately as frames of a time lapse movie, "
         "with 'fixed_' prepended to input file names.",
    type=str, dest='timelapse'
)
parser.add_argument(
    '--rangewidth', action='store', default=-1,
    help="uses range_width to limit number of images to match with.",
    type=int, dest='rangewidth'
)

__doc__ += '\n' + parser.format_help()

if __name__ == '__main__':
    print(__doc__)
    args, unknown = parser.parse_known_args()
    args_dict = vars(args)
    stitcher = Stitcher(args_dict.pop("img_names"), **args_dict)
    stitcher.stitch()

    cv.imshow(args.output, stitcher.dst)
    cv.waitKey()
    cv.destroyAllWindows()
