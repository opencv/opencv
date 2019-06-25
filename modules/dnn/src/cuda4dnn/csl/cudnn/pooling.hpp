// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_POOLING_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_POOLING_HPP

#include "cudnn.h"
#include "../pointer.hpp"

#include <opencv2/core.hpp>

#include <cudnn.h>

#include <cstddef>
#include <array>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <iterator>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    class PoolingDescriptor {
    public:
        enum class pooling_type {
            MAX,
            MAX_DETERMINISTIC,
            AVERAGE_EXCLUDE_PADDING,
            AVERAGE_INCLUDE_PADDING
        };

        PoolingDescriptor() noexcept : descriptor{ nullptr } { }
        PoolingDescriptor(const PoolingDescriptor&) = delete;
        PoolingDescriptor(PoolingDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        PoolingDescriptor(
            const SequenceContainer& window_size,
            const SequenceContainer& padding,
            const SequenceContainer& stride,
            pooling_type type)
        {
            constructor(window_size, padding, stride, type);
        }

        ~PoolingDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyPoolingDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(descriptor));
            }
        }

        PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;
        PoolingDescriptor& operator=(PoolingDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnPoolingDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class SequenceContainer>
        void constructor(
            const SequenceContainer& window_size,
            const SequenceContainer& padding,
            const SequenceContainer& stride,
            pooling_type type)
        {
            CV_Assert(window_size.size() == padding.size());
            CV_Assert(window_size.size() == stride.size());

            auto get_pooling_type = [] (pooling_type type) {
                switch (type) {
                case pooling_type::MAX:
                    return CUDNN_POOLING_MAX;
                case pooling_type::MAX_DETERMINISTIC:
                    return CUDNN_POOLING_MAX_DETERMINISTIC;
                case pooling_type::AVERAGE_EXCLUDE_PADDING:
                    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                case pooling_type::AVERAGE_INCLUDE_PADDING:
                    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                }
                CV_Error(Error::StsBadArg, "unknown pooling type");
            };

            CUDA4DNN_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&descriptor));
            try {
                const auto rank = window_size.size();
                if (rank == 2) {
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetPooling2dDescriptor(
                            descriptor,
                            get_pooling_type(type), CUDNN_PROPAGATE_NAN,
                            window_size[0], window_size[1],
                            padding[0], padding[1],
                            stride[0], stride[1]
                        )
                    );
                }
                else {
                    std::vector<int> iwindow_size(std::begin(window_size), std::end(window_size));
                    std::vector<int> ipadding(std::begin(padding), std::end(padding));
                    std::vector<int> istride(std::begin(stride), std::end(stride));
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetPoolingNdDescriptor(
                            descriptor,
                            get_pooling_type(type), CUDNN_PROPAGATE_NAN,
                            rank, iwindow_size.data(), ipadding.data(), istride.data()
                        )
                    );
                }
            } catch (...) {
                /* cudnnDestroyPoolingDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(descriptor));
                throw;
            }
        }

        cudnnPoolingDescriptor_t descriptor;
    };

    template <class T> inline
    void getPoolingForwardOutputDim(
        const PoolingDescriptor& poolingDesc,
        const TensorDescriptor<T>& inputDesc,
        std::vector<int>& output_dim)
    {
        output_dim.clear();
        output_dim.resize(CUDNN_DIM_MAX); /* we use `output_dim` to hold temporaries */

        std::vector<int> temp(CUDNN_DIM_MAX);
        cudnnDataType_t tempDataType;
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetTensorNdDescriptor(
                inputDesc.get(),
                CUDNN_DIM_MAX + 1, /* according to docs, this is what we do to get the rank */
                &tempDataType,
                output_dim.data(),
                temp.data(),
                temp.data()
            )
        );

        const auto rank = output_dim[0];
        output_dim.resize(rank);
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetPoolingNdForwardOutputDim(poolingDesc.get(), inputDesc.get(), rank, output_dim.data())
        );
    }

    template <class T> inline
    void pool(
        const Handle& handle,
        const PoolingDescriptor& poolingDesc,
        const TensorDescriptor<T>& inputDesc,
        const DevicePtr<const T> inputPtr,
        T alpha, T beta,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnPoolingForward(
                HandleAccessor::get(handle),
                poolingDesc.get(),
                &alpha, inputDesc.get(), inputPtr.get(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_POOLING_HPP */
