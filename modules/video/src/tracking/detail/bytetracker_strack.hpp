// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VIDEO_DETAIL_TRACKING_BYTETRACKER_STRACK_HPP
#define OPENCV_VIDEO_DETAIL_TRACKING_BYTETRACKER_STRACK_HPP

#include "opencv2/video/tracking.hpp"

namespace cv {
namespace detail {
namespace tracking {

//! @addtogroup tracking_detail
//! @{

enum TrackState { NEW = 0, TRACKED, LOST};


class CV_EXPORTS_W Strack : public Track {
public:
    Strack();
    Strack(const Rect2f& tlwh, int classId, float score);
    int getId() const;
    cv::Rect2f getTlwh() const;
    void setTlwh(const cv::Rect2f& tlwh);
    TrackState getState() const;
    void setState(TrackState);
    int getClass();
    cv::Rect2f predict();
    void update(Strack& track);
    void activate(int frame, int id);
    void reactivate(Strack& track, int frame);
    int getTrackletLen() const;
    void setTrackletLen(int val);
    void incrementTrackletLen();
    float getScore() const;
    ~Strack();

private:
    //cv::Rect tlwh_; //rect
    //int trackId_; //trackingId
    //int classId_; //classLabel
    TrackState state_;
    int trackletLen_;
    //float score_; //classScore
    int startFrame_;
    cv::KalmanFilter kalmanFilter_;
};

//! @}

}}}  // namespace cv::detail::tracking

#endif
