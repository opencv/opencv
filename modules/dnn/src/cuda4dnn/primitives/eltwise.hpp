// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ELTWISE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ELTWISE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/eltwise_ops.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class EltwiseOpType {
        MAX,
        SUM,
        PRODUCT,
        DIV,
        MIN,
    };

    class EltwiseOpBase : public CUDABackendNode {
    public:
        EltwiseOpBase(csl::Stream stream_, EltwiseOpType op_, std::vector<float> coeffs_)
            : stream(std::move(stream_)), op(op_), coeffs(std::move(coeffs_))
        {
        }

    protected:
        csl::Stream stream;

    public:
        EltwiseOpType op;
        std::vector<float> coeffs;
    };

    template <class T>
    class EltwiseOp final : public EltwiseOpBase {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        EltwiseOp(csl::Stream stream_, EltwiseOpType op_, std::vector<float> coeffs_)
            : EltwiseOpBase(std::move(stream_), op_, std::move(coeffs_))
        {
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() >= 2);
            CV_Assert(outputs.size() == 1);

            CV_Assert(coeffs.size() == 0 || op == EltwiseOpType::SUM);
            CV_Assert(coeffs.size() == 0 || inputs.size() == coeffs.size());

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            if (inputs.size() == 2)
            {
                auto input_wrapper_x = inputs[0].dynamicCast<wrapper_type>();
                auto input_x = input_wrapper_x->getView();

                auto input_wrapper_y = inputs[1].dynamicCast<wrapper_type>();
                auto input_y = input_wrapper_y->getView();

                switch (op)
                {
                case EltwiseOpType::MAX: kernels::eltwise_max_2<T>(stream, output, input_x, input_y); break;
                case EltwiseOpType::MIN: kernels::eltwise_min_2<T>(stream, output, input_x, input_y); break;
                case EltwiseOpType::PRODUCT: kernels::eltwise_prod_2<T>(stream, output, input_x, input_y); break;
                case EltwiseOpType::DIV: kernels::eltwise_div_2<T>(stream, output, input_x, input_y); break;
                case EltwiseOpType::SUM:
                    if (coeffs.empty() || (coeffs[0] == 1 && coeffs[1] == 1))
                        kernels::eltwise_sum_2<T>(stream, output, input_x, input_y);
                    else
                        kernels::eltwise_sum_coeff_2<T>(stream, output, coeffs[0], input_x, coeffs[1], input_y);
                    break;
                }
            }
            else
            {
                auto input_wrapper_0 = inputs[0].dynamicCast<wrapper_type>();
                auto input_0 = input_wrapper_0->getView();

                /* we first make a copy and then apply EltwiseOp cumulatively */
                csl::tensor_ops::copy(stream, output, input_0);

                for (int i = 1; i < inputs.size(); i++)
                {
                    auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                    auto input = input_wrapper->getView();

                    switch (op)
                    {
                    case EltwiseOpType::MAX: kernels::eltwise_max_2<T>(stream, output, output, input); break;
                    case EltwiseOpType::MIN: kernels::eltwise_min_2<T>(stream, output, output, input); break;
                    case EltwiseOpType::PRODUCT: kernels::eltwise_prod_2<T>(stream, output, output, input); break;
                    case EltwiseOpType::DIV: kernels::eltwise_div_2<T>(stream, output, output, input); break;
                    case EltwiseOpType::SUM:
                        if (coeffs.empty() || coeffs[i] == 1)
                            kernels::eltwise_sum_2<T>(stream, output, output, input);
                        else
                        {
                            /* if this is the first op, we must scale output too */
                            T coeff_x = (i == 1) ? coeffs[0] : 1.0;
                            kernels::eltwise_sum_coeff_2<T>(stream, output, coeff_x, output, coeffs[i], input);
                        }
                        break;
                    }
                }
            }
        }
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ELTWISE_HPP */
