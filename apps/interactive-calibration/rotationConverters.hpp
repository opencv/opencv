#ifndef RAOTATION_CONVERTERS_HPP
#define RAOTATION_CONVERTERS_HPP

#include <opencv2/core.hpp>

namespace calib
{
#define CALIB_RADIANS 0
#define CALIB_DEGREES 1

    void Euler(const cv::Mat& src, cv::Mat& dst, int argType = CALIB_RADIANS);
    void RodriguesToEuler(const cv::Mat& src, cv::Mat& dst, int argType = CALIB_RADIANS);
    void EulerToRodrigues(const cv::Mat& src, cv::Mat& dst, int argType = CALIB_RADIANS);

}
#endif
