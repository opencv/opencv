# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

set(the_description "SLAM (Simultaneous Localization and Mapping) — visual odometry and mapping")

ocv_define_module(slam
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_features
    opencv_3d
    OPTIONAL opencv_dnn
    WRAP python)
