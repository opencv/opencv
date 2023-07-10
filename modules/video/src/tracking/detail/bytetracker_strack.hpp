#ifndef BYTETRACKER_STRACK_HPP
#define BYTETRACKER_STRACK_HPP

#include "opencv2/video/detail/tracking.detail.hpp"

namespace cv {
namespace detail {
namespace tracking { 

enum TrackState { New = 0, Tracked, Lost};

struct Detection
{
    int classId;
    float confidence;
    cv::Rect box;
};

class Strack {
public:
    Strack();
    Strack(Rect tlwh, float score);
    int getId() const;
    cv::Rect getTlwh() const;
    void setTlwh(cv::Mat tlwh);
    TrackState getState() const;
    void setState(TrackState);
    cv::Mat predict();
    void update(const Strack& track);
    void activate(int frame, int id);
    void reactivate(const Strack& track, int frame);
    int getTrackletLen();
    void incrementTrackletLen();
    float getScore() const;
    ~Strack();

private:
    cv::Rect tlwh_;
    int trackId_;
    TrackState state_;
    int trackletLen_;
    float score_;
    int startFrame_;
    cv::KalmanFilter kalmanFilter_;

};
}
}
}
#endif // STRACK_HPP
