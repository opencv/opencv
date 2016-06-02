#ifndef LINALG_HPP
#define LINALG_HPP

#include <opencv2/core.hpp>

namespace cvfork {

double invert( cv::InputArray _src, cv::OutputArray _dst, int method );
bool solve(cv::InputArray _src, cv::InputArray _src2arg, cv::OutputArray _dst, int method );

}

#endif
