// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
#define OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP

//! @cond IGNORED

namespace cv {
namespace detail {

template<template<typename> class Functor, typename... Args>
static inline void depthDispatch(const int depth, Args&&... args)
{
    switch (depth)
    {
        case CV_8U:
            Functor<uint8_t>{}(std::forward<Args>(args)...);
            break;
        case CV_8S:
            Functor<int8_t>{}(std::forward<Args>(args)...);
            break;
        case CV_16U:
            Functor<uint16_t>{}(std::forward<Args>(args)...);
            break;
        case CV_16S:
            Functor<int16_t>{}(std::forward<Args>(args)...);
            break;
        case CV_32S:
            Functor<int32_t>{}(std::forward<Args>(args)...);
            break;
        case CV_32F:
            Functor<float>{}(std::forward<Args>(args)...);
            break;
        case CV_64F:
            Functor<double>{}(std::forward<Args>(args)...);
            break;
        case CV_16F:
        default:
            CV_Error(cv::Error::BadDepth, "Unsupported matrix type.");
    };
}

}}

//! @endcond

#endif //OPENCV_CORE_DETAIL_DISPATCH_HELPER_IMPL_HPP
