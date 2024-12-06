// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//#include "opencv2/video/tracking.hpp"
#include "bytetracker_strack.hpp"

namespace cv {
namespace detail {
namespace tracking {


Strack::Strack()
{
    trackingId = 0;
    classLabel = 0;
    state_ = TrackState::NEW;
    trackletLen_ = 0;
    startFrame_ = 0;
    kalmanFilter_ = cv::KalmanFilter(8,4);
}

Strack::~Strack()
{
    //nothing
}

Strack::Strack(const cv::Rect2f& tlwh, int classId, float score)
{
    rect = tlwh;
    classLabel = classId;
    classScore = score;
    trackingId = 0;
    state_ = TrackState::NEW;
    trackletLen_ = 0;
    startFrame_ = 0;
    kalmanFilter_ = cv::KalmanFilter(8,4);
}


int Strack::getId() const
{
  return trackingId;
}

cv::Rect2f Strack::getTlwh() const
{
  return rect;
}

void Strack::setTlwh(const cv::Rect2f& tlwh)
{
    rect = tlwh;
}

TrackState Strack::getState() const
{
    return state_;
}

void Strack::setState(TrackState state)
{
    state_ = state;
}

int Strack::getClass()
{
    return classLabel;
}

void Strack::activate(int frame, int id)
{
    startFrame_ = frame;
    trackletLen_ = 0;
    state_ = TrackState::TRACKED;
    trackingId = id;

    kalmanFilter_.measurementMatrix = cv::Mat::eye(4, 8, CV_32F); //H mat

    cv::Mat_<float> transitionMatrix(8,8); //make it a createTransitionMatrix() or maybe a big wrapper for everything.
    for (int row = 0; row < 8; ++row)
    {
        for (int col = 0; col < 8; ++col)
        {
            if (row == col || col == (row + 4))
                transitionMatrix(row,col) = 1.0;
            else
                transitionMatrix(row,col) = 0.0;
        }
    }
    kalmanFilter_.transitionMatrix = transitionMatrix; // F mat

    //Q matrix
    /*
    setIdentity(kalmanFilter_.processNoiseCov, cv::Scalar::all(1e-2));
    for (int i=0; i < 4; i++){
        kalmanFilter_.processNoiseCov.at<float>(i,i) *= 10000;
    }
    kalmanFilter_.processNoiseCov.at<float>(7,7) = static_cast<float>(1e-4);
    //vx  vy = 1
    kalmanFilter_.processNoiseCov.at<float>(4,4) = static_cast<float>(1e-4);
    kalmanFilter_.processNoiseCov.at<float>(5,5) = static_cast<float>(1e-4);
    */

    cv::Mat noiseCov = cv::Mat::zeros(8, 8, CV_32F);
    float stdWeightPosition = 1.0f/20;
    float stdWeightVelocity = 1.0f/160;

    noiseCov.at<float>(0, 0) = 2 * stdWeightPosition * rect.height;
    noiseCov.at<float>(1, 1) = 2 * stdWeightPosition * rect.height;
    noiseCov.at<float>(2, 2) = 1e-2f;
    noiseCov.at<float>(3, 3) = 2 * stdWeightPosition * rect.height;
    noiseCov.at<float>(4, 4) = 10 * stdWeightVelocity * rect.height;
    noiseCov.at<float>(5, 5) = 10 * stdWeightVelocity * rect.height;
    noiseCov.at<float>(6, 6) = 1e-5f;
    noiseCov.at<float>(7, 7) = 10 * stdWeightVelocity * rect.height;

    kalmanFilter_.processNoiseCov = noiseCov;

    float cx = rect.x + rect.width;
    float cy = rect.y + rect.height;
    float w = rect.width;
    float h = rect.height;
    kalmanFilter_.statePre = (cv::Mat_<float>(8,1,CV_32F) << cx, cy, w, h, 0, 0, 0, 0);
    kalmanFilter_.statePost = (cv::Mat_<float>(8,1,CV_32F) << cx, cy, w, h, 0, 0, 0, 0);

}

void Strack::update(Strack& track)
{
    trackletLen_++;

    float cx = track.rect.x + track.rect.width;
    float cy = track.rect.y + track.rect.height;
    float w = track.rect.width;
    float h = track.rect.height;

    cv::Mat measurement = (cv::Mat_<float>(4,1) << cx, cy, w, h);

    kalmanFilter_.correct(measurement);
    cv::Mat noiseCov = cv::Mat::zeros(8, 8, CV_32F);
    float stdWeightPosition = 1.0f/10;
    float stdWeightVelocity = 1.0f/160;

    noiseCov.at<float>(0, 0) = stdWeightPosition * h;
    noiseCov.at<float>(1, 1) = stdWeightPosition * h;
    noiseCov.at<float>(2, 2) = 1e-2f;
    noiseCov.at<float>(3, 3) = stdWeightPosition * h;
    noiseCov.at<float>(4, 4) = stdWeightVelocity * h;
    noiseCov.at<float>(5, 5) = stdWeightVelocity * h;
    noiseCov.at<float>(6, 6) = 1e-5f;
    noiseCov.at<float>(7, 7) = stdWeightVelocity * h;

    kalmanFilter_.processNoiseCov += noiseCov;

    classScore = track.classScore;
    rect = track.rect;

}

void Strack::reactivate(Strack& track, int frame)
{
    update(track);
    startFrame_ = frame;
    trackletLen_ = 0;
    state_ = TrackState::TRACKED;
}

void Strack::incrementTrackletLen()
{
    trackletLen_++;
}

int Strack::getTrackletLen() const
{
    return trackletLen_;
}
void Strack::setTrackletLen(int val)
{
    trackletLen_= val;
}

cv::Rect2f Strack::predict()
{
    cv::Mat predictionMat = kalmanFilter_.predict();
    cv::Rect2f prediction(
        predictionMat.at<float>(0),
        predictionMat.at<float>(1),
        predictionMat.at<float>(2),
        predictionMat.at<float>(3)
    );
    return prediction;
}

float Strack::getScore() const
{
    return classScore;
}

}
}
}
