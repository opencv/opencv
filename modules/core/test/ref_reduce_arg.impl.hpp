// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TEST_REF_REDUCE_ARG_HPP
#define OPENCV_TEST_REF_REDUCE_ARG_HPP

#include "opencv2/core/detail/reduce_arg_helper.impl.hpp"

#include <algorithm>
#include <numeric>

namespace cvtest {

template <typename T>
struct reduceMinMaxImpl
{
    void operator()(const cv::Mat& src, cv::Mat& dst, cv::detail::ReduceMode mode, const int axis) const
    {
        switch(mode)
        {
            case cv::detail::ReduceMode::FIRST_MIN:
                reduceMinMaxApply<true, true>(src, dst, axis);
                break;
            case cv::detail::ReduceMode::LAST_MIN:
                reduceMinMaxApply<true, false>(src, dst, axis);
                break;
            case cv::detail::ReduceMode::FIRST_MAX:
                reduceMinMaxApply<false, true>(src, dst, axis);
                break;
            case cv::detail::ReduceMode::LAST_MAX:
                reduceMinMaxApply<false, false>(src, dst, axis);
                break;
        }
    }

    template <bool isMin, bool isFirst>
    static void reduceMinMaxApply(const cv::Mat& src, cv::Mat& dst, const int axis)
    {
        std::vector<int> sizes(src.dims);
        std::copy(src.size.p, src.size.p + src.dims, sizes.begin());

        std::vector<cv::Range> idx(sizes.size(), cv::Range(0, 1));
        idx[axis] = cv::Range::all();
        const int n = std::accumulate(begin(sizes), end(sizes), 1, std::multiplies<int>());
        const std::vector<int> newShape{1, src.size[axis]};
        for (int i = 0; i < n ; ++i)
        {
            cv::Mat sub = src(idx).clone().reshape(1, newShape);

            double minVal, maxVal;
            cv::minMaxLoc(sub, &minVal, &maxVal);

            double val = isMin ? minVal : maxVal;

            // not sure what minMaxLoc guarantees (first/last/any occurrence of min/max)
            int32_t res = 0;
            if (isFirst)
            {
                auto begin = sub.begin<T>();
                auto end = sub.end<T>();
                res = static_cast<int32_t>(std::distance(begin, std::find(begin, end, val)));
            }
            else
            {
                auto rbegin = sub.rbegin<T>();
                auto rend = sub.rend<T>();
                res = static_cast<int32_t>(std::distance(std::find(rbegin, rend, val), std::prev(rend)));
            }
            *dst(idx).ptr<int32_t>() = res;

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

static void reduceMinMax(const Mat& src, Mat& dst, cv::detail::ReduceMode mode, int axis)
{
    axis = (axis + src.dims) % src.dims;
    CV_Assert(src.channels() == 1 && axis >= 0 && axis < src.dims);

    std::vector<int> sizes(src.dims);
    std::copy(src.size.p, src.size.p + src.dims, sizes.begin());
    sizes[axis] = 1;

    dst.create(sizes, CV_32SC1); // indices
    dst.setTo(cv::Scalar::all(0));

    cv::detail::depthDispatch<reduceMinMaxImpl>(src.depth(), src, dst, mode, axis);
}

}

#endif //OPENCV_TEST_REF_REDUCE_ARG_HPP
