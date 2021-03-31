// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_TRACKER_MIL_MODEL_HPP__
#define __OPENCV_TRACKER_MIL_MODEL_HPP__

#include "opencv2/video/detail/tracking.detail.hpp"
#include "tracker_mil_state.hpp"

namespace cv {
inline namespace tracking {
namespace impl {

using namespace cv::detail::tracking;

/**
 * \brief Implementation of TrackerModel for MIL algorithm
 */
class TrackerMILModel : public detail::TrackerModel
{
public:
    enum
    {
        MODE_POSITIVE = 1,  // mode for positive features
        MODE_NEGATIVE = 2,  // mode for negative features
        MODE_ESTIMATON = 3  // mode for estimation step
    };

    /**
   * \brief Constructor
   * \param boundingBox The first boundingBox
   */
    TrackerMILModel(const Rect& boundingBox);

    /**
   * \brief Destructor
   */
    ~TrackerMILModel() {};

    /**
   * \brief Set the mode
   */
    void setMode(int trainingMode, const std::vector<Mat>& samples);

    /**
   * \brief Create the ConfidenceMap from a list of responses
   * \param responses The list of the responses
   * \param confidenceMap The output
   */
    void responseToConfidenceMap(const std::vector<Mat>& responses, ConfidenceMap& confidenceMap);

protected:
    void modelEstimationImpl(const std::vector<Mat>& responses) CV_OVERRIDE;
    void modelUpdateImpl() CV_OVERRIDE;

private:
    int mode;
    std::vector<Mat> currentSample;

    int width;  //initial width of the boundingBox
    int height;  //initial height of the boundingBox
};

}}}  // namespace cv::tracking::impl

#endif
