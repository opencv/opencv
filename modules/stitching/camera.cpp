#include <fstream>
#include "camera.hpp"

using namespace std;
using namespace cv;


CameraParams::CameraParams() : focal(1), R(Mat::eye(3, 3, CV_64F)), t(Mat::zeros(3, 1, CV_64F)) {}

CameraParams::CameraParams(const CameraParams &other) { *this = other; }

const CameraParams& CameraParams::operator =(const CameraParams &other)
{
    focal = other.focal;
    R = other.R.clone();
    t = other.t.clone();
    return *this;
}
