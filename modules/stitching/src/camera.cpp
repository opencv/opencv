#include "precomp.hpp"

using namespace std;
using namespace cv;

cv::CameraParams::CameraParams() : focal(1), R(Mat::eye(3, 3, CV_64F)), t(Mat::zeros(3, 1, CV_64F)) {}

cv::CameraParams::CameraParams(const CameraParams &other) { *this = other; }

const cv::CameraParams& CameraParams::operator =(const CameraParams &other)
{
    focal = other.focal;
    R = other.R.clone();
    t = other.t.clone();
    return *this;
}
