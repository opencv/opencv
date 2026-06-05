// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_SLAM_ODOMETRY_PARAMS_HPP
#define OPENCV_SLAM_ODOMETRY_PARAMS_HPP

#include <opencv2/core.hpp>

namespace cv { namespace slam {

/** @brief Tunable parameters for the VisualOdometry pipeline.

    All fields have sensible defaults. Override any subset before
    passing to VisualOdometry::create().

    @ingroup slam_odometry
*/
struct CV_EXPORTS_W_SIMPLE OdometryParams
{
    CV_WRAP OdometryParams() = default;

    // --- Bootstrap (two-view init) -------------------------------------------

    CV_PROP_RW int    min_init_inliers            = 80;    //!< Min essential-matrix RANSAC inliers.
    CV_PROP_RW double min_init_parallax_deg       = 3.0;   //!< Min median parallax to trigger bootstrap.
    CV_PROP_RW int    min_init_points             = 50;    //!< Min triangulated points to accept init.
    CV_PROP_RW double hf_ratio_thresh             = 0.45;  //!< H/(H+F) ratio; > thresh uses Homography.
    CV_PROP_RW double min_growth_parallax_deg     = 1.0;   //!< Min parallax for a new map point to survive.
    CV_PROP_RW double essential_ransac_thresh     = 1.0;   //!< RANSAC reprojection threshold (px).
    CV_PROP_RW double essential_ransac_confidence = 0.999; //!< RANSAC confidence for findEssentialMat.
};

}} // namespace cv::slam

#endif // OPENCV_SLAM_ODOMETRY_PARAMS_HPP
