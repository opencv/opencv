// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TEST_REF_REDUCE_ARG_HPP
#define OPENCV_TEST_REF_REDUCE_ARG_HPP

#include "opencv2/core/detail/dispatch_helper.impl.hpp"

#include <algorithm>
#include <numeric>

namespace cvtest {

template <class Cmp, typename T>
struct reduceMinMaxImpl
{
    void operator()(const cv::Mat& src, cv::Mat& dst, const int axis) const
    {
        Cmp cmp;
        std::vector<int> sizes(src.dims);
        std::copy(src.size.p, src.size.p + src.dims, sizes.begin());

        std::vector<cv::Range> idx(sizes.size(), cv::Range(0, 1));
        idx[axis] = cv::Range::all();
        const int n = std::accumulate(begin(sizes), end(sizes), 1, std::multiplies<int>());
        const std::vector<int> newShape{1, src.size[axis]};
        for (int i = 0; i < n ; ++i)
        {
            cv::Mat sub = src(idx);

            auto begin = sub.begin<T>();
            auto it = std::min_element(begin, sub.end<T>(), cmp);
            *dst(idx).ptr<int32_t>() = static_cast<int32_t>(std::distance(begin, it));

            for (int j = static_cast<int>(idx.size()) - 1; j >= 0; --j)
            {
                if (j == axis)
                {
                    continue;
                }
                const int old_s = idx[j].start;
                const int new_s = (old_s + 1) % sizes[j];
                if (new_s > old_s)
                {
                    idx[j] = cv::Range(new_s, new_s + 1);
                    break;
                }
                idx[j] = cv::Range(0, 1);
            }
        }
    }
};

template<template<class> class Cmp>
struct MinMaxReducer{
    template <typename T>
    using Impl = reduceMinMaxImpl<Cmp<T>, T>;

    static void reduce(const Mat& src, Mat& dst, int axis)
    {
        axis = (axis + src.dims) % src.dims;
        CV_Assert(src.channels() == 1 && axis >= 0 && axis < src.dims);

        std::vector<int> sizes(src.dims);
        std::copy(src.size.p, src.size.p + src.dims, sizes.begin());
        sizes[axis] = 1;

        dst.create(sizes, CV_32SC1); // indices
        dst.setTo(cv::Scalar::all(0));

        cv::detail::depthDispatch<Impl>(src.depth(), src, dst, axis);
    }
};

}

#endif //OPENCV_TEST_REF_REDUCE_ARG_HPP
