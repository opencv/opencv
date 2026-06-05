// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_SLAM_VISUAL_ODOMETRY_HPP
#define OPENCV_SLAM_VISUAL_ODOMETRY_HPP

#include "types.hpp"
#include "map.hpp"
#include "odometry_params.hpp"
#include <opencv2/core.hpp>
#include <opencv2/features.hpp>

namespace cv { namespace slam {

/** @brief Monocular visual odometry pipeline.

    Use in batch mode (process a whole folder) or frame-by-frame:

    @code
    auto vo = cv::slam::VisualOdometry::create(
        detector, matcher, imagesFolder, outputFolder, K, dist, params);
    vo->run();                  // batch
    vo->processFrame(img);      // or incremental
    @endcode

    Output files written by run() to outputFolder:
    - trajectory.bin  — binary T_cw pose stream
    - trajectory.txt  — camera centres in world coords
    - images.txt      — COLMAP-compatible pose file
    - map_points.txt  — 3-D point cloud
    - keypoints.txt   — per-keyframe observations
    - vo.log          — per-frame processing trace

    All poses are world-to-camera (T_cw). Camera centre = -R^T * t.

    @ingroup slam_odometry
*/
class CV_EXPORTS_W VisualOdometry : public Algorithm
{
public:
    VisualOdometry();
    virtual ~VisualOdometry();

    /** @brief Create a VisualOdometry instance.
        @param detector      Feature detector + descriptor extractor.
        @param matcher       Descriptor matcher.
        @param imagesFolder  Input image folder used by run(). May be empty.
        @param outputFolder  Artifact output folder. May be empty.
        @param cameraMatrix  3x3 intrinsic matrix K.
        @param distCoeffs    Distortion coefficients. Pass empty Mat for rectified input.
        @param params        Odometry parameters.
    */
    CV_WRAP static Ptr<VisualOdometry> create(
        const Ptr<Feature2D>&         detector,
        const Ptr<DescriptorMatcher>& matcher,
        const String&                 imagesFolder,
        const String&                 outputFolder,
        InputArray                    cameraMatrix,
        InputArray                    distCoeffs,
        const OdometryParams&         params = OdometryParams());

    /** @brief Process all images in imagesFolder and write output artifacts.
        @return true if at least one pose was emitted.
    */
    CV_WRAP virtual bool run() = 0;

    /** @brief Process a single frame.
        @return true if a pose was emitted.
    */
    CV_WRAP virtual bool processFrame(InputArray image) = 0;

    /** @brief Reset to NOT_INITIALIZED state and clear the map. */
    CV_WRAP virtual void reset() = 0;

    CV_WRAP virtual OdometryState                  getState()      const = 0;
    CV_WRAP virtual Matx44d                        getLastPose()   const = 0;
    CV_WRAP virtual Map&                           getMap()              = 0;
    CV_WRAP virtual const std::vector<Matx44d>&   getTrajectory() const = 0;

    CV_WRAP virtual OdometryParams getParams() const = 0;
    CV_WRAP virtual void           setParams(const OdometryParams& params) = 0;

    /** @brief Enable/disable pose-only BA after each PnP step. No-op without g2o. Default: true. */
    CV_WRAP virtual void setPoseOptimization(bool enable) = 0;
    CV_WRAP virtual bool getPoseOptimization() const = 0;

    /** @brief Enable/disable local BA at each keyframe. No-op without g2o. Default: true. */
    CV_WRAP virtual void setLocalBA(bool enable) = 0;
    CV_WRAP virtual bool getLocalBA() const = 0;

    CV_WRAP virtual String getImagesFolder() const = 0;
    CV_WRAP virtual void   setImagesFolder(const String& path) = 0;

    CV_WRAP virtual String getOutputFolder() const = 0;
    CV_WRAP virtual void   setOutputFolder(const String& path) = 0;
};

}} // namespace cv::slam

#endif // OPENCV_SLAM_VISUAL_ODOMETRY_HPP
