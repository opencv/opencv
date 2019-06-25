// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP

#include <opencv2/dnn/csl/cudnn.hpp>

#include "pointer.hpp"

#include <cudnn.h>

#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>
#include <iterator>

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

    /** creates a cuDNN tensor descriptor for a given shape */
    template <class T>
    class TensorDescriptor {
    public:
        TensorDescriptor() noexcept : descriptor{ nullptr } { }
        TensorDescriptor(const TensorDescriptor&) = delete;
        TensorDescriptor(TensorDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a tensor descriptor from the axis lengths provided in \p shape */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        TensorDescriptor(const SequenceContainer& shape) {
            constructor(shape.begin(), shape.end());
        }

        /** constructs a tensor descriptor from the axis lengths provided in [begin, end) */
        template <class ForwardItr, typename = typename std::enable_if<!std::is_integral<ForwardItr>::value, void>::type> // TODO is_iterator
        TensorDescriptor(ForwardItr begin, ForwardItr end) {
            constructor(begin, end);
        }

        /** constructs a tensor descriptor from the axis lengths provided as arguments */
        template <class ...Sizes>
        TensorDescriptor(Sizes ...sizes) {
            static_assert(sizeof...(Sizes) <= CUDNN_DIM_MAX, "required rank exceeds maximum supported rank");
            std::array<int, sizeof...(Sizes)> dims = { static_cast<int>(sizes)... };
            constructor(std::begin(dims), std::end(dims));
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
        template <class ForwardItr>
        void constructor(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) <= CUDNN_DIM_MAX);

            CUDA4DNN_CHECK_CUDNN(cudnnCreateTensorDescriptor(&descriptor));
            try {
                const auto rank = std::distance(start, end);
                if (rank <= 4) {
                    std::array<int, 4> dims;
                    std::fill(std::begin(dims), std::end(dims), 1);

                    /* suppose we have a 3d tensor, the first axis is the batch axis and
                     * the second axis is the channel axis (generally)
                     *
                     * cuDNN frequently assumes that the first axis is the batch axis and the
                     * second axis is the channel axis; hence, we copy the shape of a lower rank
                     * tensor to the begining of `dims`
                     */
                    std::copy(start, end, std::begin(dims));

                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetTensor4dDescriptor(descriptor,
                            CUDNN_TENSOR_NCHW, detail::get_data_type<T>(),
                            dims[0], dims[1], dims[2], dims[3]
                        )
                    );
                } else {
                    std::vector<int> stride(rank);
                    stride.back() = 1;
                    /* WHAT WE HAVE NOW:
                     * stride[-1] = 1
                     * stride[-2] = garbage
                     * stride[-3] = garbage
                     * stride[-4] = garbage
                     * ...
                     */

                    std::copy(start + 1, end, stride.begin());
                    /* WHAT WE HAVE NOW:
                     * stride[-1] = 1
                     * stride[-2] = dim[-1]
                     * stride[-3] = dim[-2]
                     * stride[-4] = dim[-3]
                     * ...
                     */

                    std::partial_sum(std::rbegin(stride), std::rend(stride), std::rbegin(stride), std::multiplies<int>());
                    /* WHAT WE HAVE NOW:
                     * stride[-1] = 1
                     * stride[-2] = stride[-1] * dim[-1]
                     * stride[-3] = stride[-2] * dim[-2]
                     * stride[-4] = stride[-3] * dim[-3]
                     * ...
                     */

                    std::vector<int> dims(start, end);
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetTensorNdDescriptor(descriptor,
                            detail::get_data_type<T>(), rank,
                            dims.data(), stride.data()
                        )
                    );
                }
            } catch (...) {
                /* cudnnDestroyTensorDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyTensorDescriptor(descriptor));
                throw;
            }
        }

        cudnnTensorDescriptor_t descriptor;
    };

    /** creates a cuDNN filter descriptor for the given filter shape
     *
     * Dimension Ordering:
     * 0: number of output feature maps
     * 1: number of input feature maps
     * 2..n: kernel dimensions
     */
    template <class T>
    class FilterDescriptor {
    public:
        FilterDescriptor() noexcept : descriptor{ nullptr } { }
        FilterDescriptor(const FilterDescriptor&) = delete;
        FilterDescriptor(FilterDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        /** constructs a filter descriptor from the filter dimensions provided in \p shape */
        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        FilterDescriptor(const SequenceContainer& shape) {
            constructor(shape.begin(), shape.end());
        }

        /** constructs a filter descriptor from the filter dimensions provided in [begin, end) */
        template <class ForwardItr, typename = typename std::enable_if<!std::is_integral<ForwardItr>::value, void>::type> // TODO is_iterator
        FilterDescriptor(ForwardItr begin, ForwardItr end) {
            constructor(begin, end);
        }

        /** constructs a filter descriptor from the filter dimensions provided as arguments */
        template <class ...Sizes>
        FilterDescriptor(Sizes ...sizes) {
            static_assert(sizeof...(Sizes) >= 3, "filter descriptors must have at least three dimensions");
            static_assert(sizeof...(Sizes) <= CUDNN_DIM_MAX, "required rank exceeds maximum supported rank");
            std::array<int, sizeof...(Sizes)> dims = { static_cast<int>(sizes)... };
            constructor(std::begin(dims), std::end(dims));
        }

        ~FilterDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyFilterDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyFilterDescriptor(descriptor));
            }
        }

        FilterDescriptor& operator=(const FilterDescriptor&) = delete;
        FilterDescriptor& operator=(FilterDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnFilterDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class ForwardItr>
        void constructor(ForwardItr start, ForwardItr end) {
            CV_Assert(start != end);
            CV_Assert(std::distance(start, end) >= 3);
            CV_Assert(std::distance(start, end) <= CUDNN_DIM_MAX);

            CUDA4DNN_CHECK_CUDNN(cudnnCreateFilterDescriptor(&descriptor));
            try {
                const auto rank = std::distance(start, end);
                if (rank == 4) {
                    std::array<int, 4> dims;
                    std::copy(start, end, std::begin(dims));

                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetFilter4dDescriptor(descriptor,
                            detail::get_data_type<T>(), CUDNN_TENSOR_NCHW,
                            dims[0], dims[1], dims[2], dims[3]
                        )
                    );
                } else {
                    std::vector<int> dims(start, end);
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetFilterNdDescriptor(descriptor,
                            detail::get_data_type<T>(), CUDNN_TENSOR_NCHW,
                            dims.size(), dims.data()
                        )
                    );
                }
            } catch (...) {
                /* cudnnDestroyFilterDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyFilterDescriptor(descriptor));
                throw;
            }
        }

        cudnnFilterDescriptor_t descriptor;
    };

    /** creates a cuDNN convolution descriptor */
    template <class T>
    class ConvolutionDescriptor {
    public:
        ConvolutionDescriptor() noexcept : descriptor{ nullptr } { }
        ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
        ConvolutionDescriptor(ConvolutionDescriptor&& other) noexcept
            : descriptor{ other.descriptor } {
            other.descriptor = nullptr;
        }

        template <class SequenceContainer, typename = decltype(std::begin(std::declval<SequenceContainer>()))>
        ConvolutionDescriptor(
            const SequenceContainer& zero_padding,
            const SequenceContainer& stride,
            const SequenceContainer& dialation,
            std::size_t group_count)
        {
            constructor(zero_padding, stride, dialation, group_count);
        }

        ~ConvolutionDescriptor() noexcept {
            if (descriptor != nullptr) {
                /* cudnnDestroyConvolutionDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(descriptor));
            }
        }

        ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;
        ConvolutionDescriptor& operator=(ConvolutionDescriptor&& other) noexcept {
            descriptor = other.descriptor;
            other.descriptor = nullptr;
            return *this;
        };

        cudnnConvolutionDescriptor_t get() const noexcept { return descriptor; }

    private:
        template <class SequenceContainer>
        void constructor(
            const SequenceContainer& zero_padding,
            const SequenceContainer& stride,
            const SequenceContainer& dialation,
            std::size_t group_count)
        {
            CV_Assert(std::size(zero_padding) == std::size(stride));
            CV_Assert(std::size(zero_padding) == std::size(dialation));

            CUDA4DNN_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&descriptor));
            try {
                const auto rank = std::size(zero_padding);
                if (rank == 2) {
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetConvolution2dDescriptor(descriptor,
                            zero_padding[0], zero_padding[1],
                            stride[0], stride[1],
                            dialation[0], dialation[1],
                            CUDNN_CROSS_CORRELATION,
                            detail::get_data_type<T>()
                        )
                    );
                } else {
                    std::vector<int> ipadding(std::begin(zero_padding), std::end(zero_padding));
                    std::vector<int> istride(std::begin(stride), std::end(stride));
                    std::vector<int> idialation(std::begin(dialation), std::end(dialation));
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetConvolutionNdDescriptor(descriptor, rank,
                            ipadding.data(),
                            istride.data(),
                            idialation.data(),
                            CUDNN_CROSS_CORRELATION,
                            detail::get_data_type<T>()
                        )
                    );
                }
                CUDA4DNN_CHECK_CUDNN(cudnnSetConvolutionGroupCount(descriptor, group_count));
            } catch (...) {
                /* cudnnDestroyConvolutionDescriptor will not fail */
                CUDA4DNN_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(descriptor));
                throw;
            }
        }

        cudnnConvolutionDescriptor_t descriptor;
    };

    template <class T>
    class ConvolutionAlgorithm {
    public:
        ConvolutionAlgorithm() noexcept : workspace_size{ 0 } { }
        ConvolutionAlgorithm(ConvolutionAlgorithm&) = default;
        ConvolutionAlgorithm(ConvolutionAlgorithm&&) = default;

        ConvolutionAlgorithm(
            Handle& handle,
            ConvolutionDescriptor<T>& conv,
            FilterDescriptor<T>& filter,
            TensorDescriptor<T>& input,
            TensorDescriptor<T>& output)
        {
            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionForwardAlgorithm(HandleAccessor::get(handle),
                    input.get(), filter.get(), conv.get(), output.get(),
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    0, &algo
                )
            );

            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionForwardWorkspaceSize(HandleAccessor::get(handle),
                    input.get(), filter.get(), conv.get(), output.get(),
                    algo, &workspace_size
                )
            );
        }

        ConvolutionAlgorithm& operator=(const ConvolutionAlgorithm&) = default;
        ConvolutionAlgorithm& operator=(ConvolutionAlgorithm&& other) = default;

        auto get() const noexcept { return algo; }
        auto get_workspace_size() const noexcept { return workspace_size; }

    private:
        cudnnConvolutionFwdAlgo_t algo;
        std::size_t workspace_size;
    };

    template <class T> inline
    void getConvolutionForwardOutputDim(
        const ConvolutionDescriptor<T>& conv,
        const FilterDescriptor<T>& filter,
        const TensorDescriptor<T>& input,
        std::vector<int>& output)
    {
        output.clear();
        output.resize(CUDNN_DIM_MAX); /* we use `output` to hold temporaries */

        std::vector<int> temp(CUDNN_DIM_MAX);
        cudnnDataType_t tempDataType;
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetTensorNdDescriptor(
                input.get(),
                CUDNN_DIM_MAX + 1, /* according to docs, this is what we do to get the rank */
                &tempDataType,
                output.data(),
                temp.data(),
                temp.data()
            )
        );
        const auto rank = output[0];
        output.resize(rank);

        CUDA4DNN_CHECK_CUDNN(
            cudnnGetConvolutionNdForwardOutputDim(conv.get(), input.get(), filter.get(),
                rank,
                output.data()
            )
        );
    }

    template <class T> inline
    void convolve(const Handle& handle,
        const FilterDescriptor<T>& filter_desc,
        DevicePtr<const T> filter_data,
        const ConvolutionDescriptor<T>& conv_desc,
        const ConvolutionAlgorithm<T>& algo,
        DevicePtr<unsigned char> workspace,
        const TensorDescriptor<T>& input_desc,
        DevicePtr<const T> input_data,
        T alpha,
        T beta,
        const TensorDescriptor<T>& output_desc,
        DevicePtr<T> output_data)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnConvolutionForward(
                HandleAccessor::get(handle),
                &alpha, input_desc.get(), input_data.get(),
                filter_desc.get(), filter_data.get(), conv_desc.get(), algo.get(), workspace.get(),
                algo.get_workspace_size(), &beta, output_desc.get(), output_data.get()
            )
        );
    }

    class PoolingDescriptor {
    public:
        enum class pooling_type {
            max,
            average_exclude_padding,
            average_include_padding
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
            CV_Assert(std::size(window_size) == std::size(padding));
            CV_Assert(std::size(window_size) == std::size(stride));

            auto get_pooling_type = [] (pooling_type type) {
                switch (type) {
                case pooling_type::max:
                    return CUDNN_POOLING_MAX;
                case pooling_type::average_exclude_padding:
                    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
                case pooling_type::average_include_padding:
                    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
                }
                return CUDNN_POOLING_MAX;
            };

            CUDA4DNN_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&descriptor));
            try {
                const auto rank = std::size(window_size);
                if (rank == 2) {
                    CUDA4DNN_CHECK_CUDNN(
                        cudnnSetPooling2dDescriptor(descriptor,
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
                        cudnnSetPoolingNdDescriptor(descriptor,
                            get_pooling_type(type), CUDNN_PROPAGATE_NAN,
                            rank,
                            iwindow_size.data(),
                            ipadding.data(),
                            istride.data()
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
    void getPoolingForwardOutputDim(const PoolingDescriptor& pooling_desc,
            const TensorDescriptor<T>& input,
            std::vector<int>& output) {
        output.clear();
        output.resize(CUDNN_DIM_MAX); /* we use `output` to hold temporaries */

        std::vector<int> temp(CUDNN_DIM_MAX);
        cudnnDataType_t tempDataType;
        CUDA4DNN_CHECK_CUDNN(
            cudnnGetTensorNdDescriptor(
                input.get(),
                CUDNN_DIM_MAX + 1, /* according to docs, this is what we do to get the rank */
                &tempDataType,
                output.data(),
                temp.data(),
                temp.data()
            )
        );
        const auto rank = output[0];
        output.resize(rank);

        CUDA4DNN_CHECK_CUDNN(
            cudnnGetPoolingNdForwardOutputDim(pooling_desc.get(), input.get(), rank, output.data())
        );
    }

    template <class T> inline
    void pool(Handle& handle,
            PoolingDescriptor& pooling_desc,
            TensorDescriptor<T>& input_desc,
            DevicePtr<const T> input_data,
            T alpha, T beta,
            TensorDescriptor<T>& output_desc,
            DevicePtr<T> output_data)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnPoolingForward(HandleAccessor::get(handle),
                pooling_desc.get(),
                &alpha, input_desc.get(), input_data.get(),
                &beta, output_desc.get(), output_data.get()
            )
        );
    }

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

    /** @brief computes softmax (or log softmax)
     *
     * @tparam          T           matrix element type (must be `float` or `double`)
     *
     * @param           handle      valid cuDNN handle
     * @param           outputDesc  tensor descriptor for A
     * @param[out]      output      pointer to tensor in device memory
     * @param           inputDesc   tensor descriptor for C
     * @param[in]       input       pointer to tensor in device memory
     * @param           log         apply log on probabilities
     *
     * Exception Guarantee: Basic
     */
    template <class T>
    typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>
    ::type softmax(const cudnn::Handle& handle,
        const TensorDescriptor<T>& outputDesc, DevicePtr<T> output,
        const TensorDescriptor<T>& inputDesc, DevicePtr<const T> input,
        bool log)
    {
        T alpha = 1.0, beta = 0.0;
        cudnnSoftmaxAlgorithm_t algo = log ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
        CUDA4DNN_CHECK_CUDNN(
            cudnnSoftmaxForward(
                HandleAccessor::get(handle),
                algo, CUDNN_SOFTMAX_MODE_CHANNEL,
                &alpha, inputDesc.get(), input.get(),
                &beta, outputDesc.get(), output.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_HPP */
