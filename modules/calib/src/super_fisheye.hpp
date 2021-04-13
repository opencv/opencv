#ifndef SUPER_FISHEYE_INTERNAL_H
#define SUPER_FISHEYE_INTERNAL_H
#include "precomp.hpp"
#include "fisheye.hpp"

namespace cv { namespace internal2 {

void projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
                   cv::InputArray _rvec,cv::InputArray _tvec,
                   const cv::internal::IntrinsicParams& param, cv::OutputArray jacobian);


CV_EXPORTS Mat NormalizePixels(const Mat& imagePoints, const cv::internal::IntrinsicParams& param);

}}

#endif
