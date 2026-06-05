// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_SLAM_TYPES_HPP
#define OPENCV_SLAM_TYPES_HPP

#include <opencv2/core.hpp>
#include <map>
#include <vector>

namespace cv { namespace slam {

/** @brief Current state of the VisualOdometry pipeline.
    @ingroup slam_odometry
*/
enum OdometryState
{
    NOT_INITIALIZED = 0, //!< No frames processed yet.
    INITIALIZING    = 1, //!< Reference frame set; waiting for bootstrap.
    TRACKING        = 2  //!< Map initialised; localising via PnP.
};

/** @brief Feature data for one image frame.
    @ingroup slam_odometry
*/
struct CV_EXPORTS_W_SIMPLE FrameFeatures
{
    std::vector<KeyPoint> keypoints;
    Mat                   descriptors;
    Size                  imageSize;
};

struct MapPoint;
struct KeyFrame;

/** @brief A triangulated 3-D landmark owned by Map.
    @ingroup slam_odometry
*/
struct CV_EXPORTS_W_SIMPLE MapPoint
{
    int     id  = -1;
    Point3d pos;
    bool    bad = false;

    std::map<KeyFrame*, int> observations; //!< Observing keyframe → keypoint index.
};

/** @brief A frame whose pose has been committed to the map.
    @ingroup slam_odometry
*/
struct CV_EXPORTS_W_SIMPLE KeyFrame
{
    int     id = -1;
    Matx44d pose_cw; //!< World-to-camera transform (row-major 4x4).

    std::vector<KeyPoint> keypoints;
    Mat                   descriptors;
    std::vector<Point2f>  undist_kpts;
    Size                  imageSize;

    std::vector<int>       kpt_to_mp;  //!< Map-point id per keypoint, -1 if unmatched.
    std::vector<MapPoint*> mappoints;  //!< Direct pointer per keypoint, nullptr if unmatched.

    std::vector<std::pair<KeyFrame*, int>> ordered_covisibility; //!< Neighbours by shared point count.
};

}} // namespace cv::slam

#endif // OPENCV_SLAM_TYPES_HPP
