// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ONE_HOT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ONE_HOT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/one_hot.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T, class TIdx>
    class OneHotOp final : public CUDABackendNode {
    public:
        OneHotOp(csl::Stream stream_, int axis_, const Mat& values)
            : stream(std::move(stream_)), axis(axis_),
              offValue(static_cast<T>(0.f)), onValue(static_cast<T>(1.f))
        {
            if (values.empty())
                return;
            CV_Assert(values.total() == 2);
            // convert straight to T's depth so int64 values stay exact; half has no host type, so read it as float
            using HostT = typename std::conditional<std::is_same<T, half>::value, float, T>::type;
            Mat typed;
            values.convertTo(typed, DataType<HostT>::depth);
            offValue = static_cast<T>(typed.ptr<HostT>()[0]);
            onValue = static_cast<T>(typed.ptr<HostT>()[1]);
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 3 && outputs.size() == 1);

            MatShape outShape = cv::dnn::shape(outputs[0]);
            const int rank = outShape.dims;
            const int a = axis < 0 ? axis + rank : axis;
            CV_Assert(0 <= a && a < rank);
            if (outputs[0].total() == 0)
                return;

            std::size_t inner = 1;
            for (int i = a + 1; i < rank; i++)
                inner *= static_cast<std::size_t>(outShape[i]);

            auto indices = csl::viewOf<TIdx>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);
            kernels::one_hot<T, TIdx>(stream, output, indices, static_cast<std::size_t>(outShape[a]), inner,
                                      offValue, onValue);
        }

    private:
        csl::Stream stream;
        int axis;
        T offValue, onValue;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ONE_HOT_HPP */
