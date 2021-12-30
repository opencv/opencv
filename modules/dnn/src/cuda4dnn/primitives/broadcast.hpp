// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BROADCAST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BROADCAST_HPP

#include "../../op_cuda.hpp"
#include "../../broadcast_common.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    struct BroadcastOp : public CUDABackendNode
    {
    public:
        BroadcastOp(csl::Stream stream_, std::vector<InputCache> m_cache_, std::vector<size_t> m_prods_, std::vector<int> m_outShape_)
            : stream(std::move(stream_)), m_cache(std::move(m_cache_)), m_prods(std::move(m_prods_)), m_outShape(std::move(m_outShape_)) { }

    protected:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            auto copier = [this](typename csl::TensorView<T>::const_pointer src,
                                 size_t src_offset,
                                 typename csl::TensorSpan<T>::pointer dst,
                                 size_t dst_offset,
                                 const size_t size, const size_t times) {
                src += src_offset;
                dst += dst_offset;
                for (size_t i = 0; i < times; ++i)
                {
                    memcpy(dst, src, size, stream);
                    dst += size;
                }
            };

            for (int i = 0; i < inputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                const auto input_ptr = input.get();
                auto output_ptr = output.get();

                broadcast(input_ptr, output_ptr, m_cache[i], m_prods, m_outShape, copier);
            }
        }
    private:
        csl::Stream stream;
        std::vector<InputCache> m_cache;
        std::vector<size_t> m_prods;
        std::vector<int> m_outShape;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BROADCAST_HPP */
