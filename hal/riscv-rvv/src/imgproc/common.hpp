// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2025, SpaceMIT Inc., all rights reserved.
// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_HAL_RVV_IMGPROC_COMMON_HPP_INCLUDED
#define OPENCV_HAL_RVV_IMGPROC_COMMON_HPP_INCLUDED

#include "opencv2/imgproc/hal/interface.h"

namespace cv { namespace rvv_hal { namespace imgproc { namespace common {

inline int borderInterpolate( int p, int len, int borderType )
{
    if ((unsigned)p < (unsigned)len)
        ;
    else if (borderType == BORDER_REPLICATE)
        p = p < 0 ? 0 : len - 1;
    else if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        int delta = borderType == BORDER_REFLECT_101;
        if (len == 1)
            return 0;
        do
        {
            if (p < 0)
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if (borderType == BORDER_WRAP)
    {
        if (p < 0)
            p -= ((p-len+1)/len)*len;
        if (p >= len)
            p %= len;
    }
    else if (borderType == BORDER_CONSTANT)
        p = -1;
    return p;
}

class FilterInvoker : public ParallelLoopBody
{
public:
    template<typename... Args>
    FilterInvoker(std::function<int(int, int, Args...)> _func, Args&&... args)
    {
        func = std::bind(_func, std::placeholders::_1, std::placeholders::_2, std::forward<Args>(args)...);
    }

    virtual void operator()(const Range& range) const override
    {
        func(range.start, range.end);
    }

private:
    std::function<int(int, int)> func;
};

template<typename... Args>
inline int invoke(int height, std::function<int(int, int, Args...)> func, Args&&... args)
{
    cv::parallel_for_(Range(1, height), FilterInvoker(func, std::forward<Args>(args)...), cv::getNumThreads());
    return func(0, 1, std::forward<Args>(args)...);
}

}}}} // cv::rvv_hal::imgproc::common

#endif // OPENCV_HAL_RVV_IMGPROC_COMMON_HPP_INCLUDED
