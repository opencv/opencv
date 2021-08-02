import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', '..')))

from stitching_detailed.stitcher import Stitcher


# Eine Klasse erstellen, die von unittest.TestCase erbt
class TestStitcher(unittest.TestCase):
    def tearDown(self):
        try:
            os.remove("result.jpg")
        except OSError:
            pass

    def test_stitcher(self):
        settings = {
            "try_cuda": False,
            "work_megapix": 0.6,
            "features": "orb",
            "matcher": "homography",
            "estimator": "homography",
            "match_conf": None,
            "conf_thresh": 1.0,
            "ba": "ray",
            "ba_refine_mask": "xxxxx",
            "wave_correct": "horiz",
            "save_graph": None,
            "warp": "spherical",
            "seam_megapix": 0.1,
            "seam": "dp_color",
            "compose_megapix": -1,
            "expos_comp": "gain_blocks",
            "expos_comp_nr_feeds": 1,
            "expos_comp_nr_filtering": 2,
            "expos_comp_block_size": 32,
            "blend": "multiband",
            "blend_strength": 5,
            "output": "result.jpg",
            "timelapse": None,
            "rangewidth": -1
        }
        stitcher = Stitcher(["s1.jpg", "s2.jpg"], **settings)
        stitcher.stitch()

        max_image_shape_derivation_per_pixel = 3
        np.testing.assert_allclose(stitcher.result.shape,
                                   (700, 1811, 3),
                                   max_image_shape_derivation_per_pixel)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
