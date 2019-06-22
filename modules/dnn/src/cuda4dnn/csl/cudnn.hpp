// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP

#include <opencv2/dnn/csl/cudnn.hpp>

#include "pointer.hpp"

#include <cudnn.h>

#define CUDA4DNN_CHECK_CUDNN(call) \
    ::cv::dnn::cuda4dnn::csl::cudnn::detail::check((call), CV_Func, __FILE__, __LINE__)

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    namespace detail {
        inline void check(cudnnStatus_t status, const char* func, const char* file, int line) {
            if (status != CUDNN_STATUS_SUCCESS)
                throw cuDNNException(Error::GpuApiCallError, cudnnGetErrorString(status), func, file, line);
        }

        /** get_data_type<T> returns the equivalent cudnn enumeration constant for type T */
        template <class> auto get_data_type()->decltype(CUDNN_DATA_FLOAT);
        template <> inline auto get_data_type<float>()->decltype(CUDNN_DATA_FLOAT) { return CUDNN_DATA_FLOAT; }
        template <> inline auto get_data_type<double>()->decltype(CUDNN_DATA_FLOAT) { return CUDNN_DATA_DOUBLE; }
    }


    /** used to access the raw cuDNN handle held by Handle */
    class HandleAccessor {
    public:
        static cudnnHandle_t get(const Handle& handle);
    };

    template <class T>
    class TensorDescriptor {
    public:
        TensorDescriptor() noexcept : descriptor{ nullptr } { }
        TensorDescriptor(const TensorDescriptor&) = delete;
        TensorDescriptor(TensorDescriptor&& other)
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        TensorDescriptor(std::size_t N, std::size_t chans, std::size_t height, std::size_t width) {
            CUDA4DNN_CHECK_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
            try {
                CUDA4DNN_CHECK_CUDNN(cudnnSetTensor4dDescriptor(descriptor,
                    CUDNN_TENSOR_NCHW, detail::get_data_type<T>(),
                    static_cast<int>(N), static_cast<int>(chans),
                    static_cast<int>(height), static_cast<int>(width)));
            }
            catch (...) {
                /* cudnnDestroyTensorDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
                throw;
            }
        }

        ~TensorDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyTensorDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
            }
        }

        TensorDescriptor& operator=(const TensorDescriptor&) = delete;
        TensorDescriptor& operator=(TensorDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnTensorDescriptor_t get() const noexcept { return descriptor; }

    private:
        cudnnTensorDescriptor_t descriptor;
    };

    /** @brief element-wise addition with broadcasting
     *
     * \f$ C = \alpha A + \beta C \f$
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
     *
     * @param           handle      valid cuDNN handle
     * @param           alpha       scale factor for A
     * @param           aDesc       tensor descriptor for A
     * @param[in]       A           pointer to tensor in device memory
     * @param           beta        scale factor for C
     * @param           cDesc       tensor descriptor for C
     * @param[in]       C           pointer to tensor in device memory
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type add(const Handle& handle,
        T alpha, const TensorDescriptor<T>& aDesc, DevicePtr<const T> A,
        T beta, const TensorDescriptor<T>& cDesc, DevicePtr<T> C)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnAddTensor(HandleAccessor::get(handle),
                &alpha, aDesc.get(), A.get(),
                &beta, cDesc.get(), C.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP */
