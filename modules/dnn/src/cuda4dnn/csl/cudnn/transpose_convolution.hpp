// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSPOSE_CONVOLUTION_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSPOSE_CONVOLUTION_HPP

#include "cudnn.hpp"
#include "convolution.hpp"

#include "../pointer.hpp"
#include "../workspace.hpp"

#include <cudnn.h>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

    /** wrapper around a transpose convolution algorithm
     *
     * @tparam  T   type of elements being transpose-convolved
     */
    template <class T>
    class TransposeConvolutionAlgorithm {
    public:
        TransposeConvolutionAlgorithm() noexcept : workspace_size{ 0 } { }
        TransposeConvolutionAlgorithm(TransposeConvolutionAlgorithm&) = default;
        TransposeConvolutionAlgorithm(TransposeConvolutionAlgorithm&&) = default;

        TransposeConvolutionAlgorithm(
            const Handle& handle,
            const ConvolutionDescriptor<T>& convDesc,
            const FilterDescriptor<T>& filterDesc,
            const TensorDescriptor<T>& inputDesc,
            const TensorDescriptor<T>& outputDesc)
        {
#if CUDNN_MAJOR >= 8
            int requestedAlgoCount = 0, returnedAlgoCount = 0;
            CUDA4DNN_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.get(), &requestedAlgoCount));
            std::vector<cudnnConvolutionBwdDataAlgoPerf_t> results(requestedAlgoCount);
            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionBackwardDataAlgorithm_v7(
                    handle.get(),
                    filterDesc.get(), inputDesc.get(), convDesc.get(), outputDesc.get(),
                    requestedAlgoCount,
                    &returnedAlgoCount,
                    &results[0]
                )
            );

            size_t free_memory, total_memory;
            CUDA4DNN_CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));

            bool found_conv_algorithm = false;
            for (int i = 0; i < returnedAlgoCount; i++)
            {
                if (results[i].status == CUDNN_STATUS_SUCCESS &&
                    results[i].algo != CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED &&
                    results[i].memory < free_memory)
                {
                    found_conv_algorithm = true;
                    dalgo = results[i].algo;
                    workspace_size = results[i].memory;
                    break;
                }
            }

            if (!found_conv_algorithm)
                CV_Error (cv::Error::GpuApiCallError, "cuDNN did not return a suitable algorithm for transpose convolution.");
#else
            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionBackwardDataAlgorithm(
                    handle.get(),
                    filterDesc.get(), inputDesc.get(), convDesc.get(), outputDesc.get(),
                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                    0, /* no memory limit */
                    &dalgo
                )
            );

            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionBackwardDataWorkspaceSize(
                    handle.get(),
                    filterDesc.get(), inputDesc.get(), convDesc.get(), outputDesc.get(),
                    dalgo, &workspace_size
                )
            );
#endif
        }

        TransposeConvolutionAlgorithm& operator=(const TransposeConvolutionAlgorithm&) = default;
        TransposeConvolutionAlgorithm& operator=(TransposeConvolutionAlgorithm&& other) = default;

        cudnnConvolutionBwdDataAlgo_t get() const noexcept { return dalgo; }

        std::size_t get_workspace_size() const noexcept { return workspace_size; }

    private:
        cudnnConvolutionBwdDataAlgo_t dalgo;
        std::size_t workspace_size;
    };

    /** @brief performs transpose convolution
      *
      * dstValue = alpha * result + beta * priorDstValue
      *
      * @tparam          T              transpose convolution element type (must be `half` or `float`)
      *
      * @param           handle         valid cuDNN Handle
      * @param           convDesc       convolution description
      * @param           transConvAlgo  algorithm to use for convolution
      * @param           workspace      workspace memory which meets the requirements of \p convAlgo
      * @param           filterDesc     filter descriptor
      * @param[in]       filterPtr      pointer to device memory containing the filters
      * @param           inputDesc      tensor descriptor describing the input
      * @param[in]       inputPtr       pointer to input tensor in device memory
      * @param           alpha          result scale factor
      * @param           beta           previous value scale factor
      * @param           outputDesc     tensor descriptor describing the output
      * @param[out]      outputPtr      pointer to output tensor in device memory
      *
      * Exception Guarantee: Basic
      */
    template <class T>
    void transpose_convolve(
        const Handle& handle,
        const ConvolutionDescriptor<T>& convDesc,
        const TransposeConvolutionAlgorithm<T>& transConvAlgo,
        WorkspaceInstance workspace,
        const FilterDescriptor<T>& filterDesc,
        DevicePtr<const T> filterPtr,
        const TensorDescriptor<T>& inputDesc,
        DevicePtr<const T> inputPtr,
        T alpha, T beta,
        const TensorDescriptor<T>& outputDesc,
        DevicePtr<T> outputPtr)
    {
        CUDA4DNN_CHECK_CUDNN(
            cudnnConvolutionBackwardData(
                handle.get(),
                &alpha,
                filterDesc.get(), filterPtr.get(),
                inputDesc.get(), inputPtr.get(),
                convDesc.get(), transConvAlgo.get(),
                static_cast<void*>(workspace.get()), workspace.size_in_bytes(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

    template <> inline
    void transpose_convolve(
        const Handle& handle,
        const ConvolutionDescriptor<half>& convDesc,
        const TransposeConvolutionAlgorithm<half>& convAlgo,
        WorkspaceInstance workspace,
        const FilterDescriptor<half>& filterDesc,
        DevicePtr<const half> filterPtr,
        const TensorDescriptor<half>& inputDesc,
        DevicePtr<const half> inputPtr,
        half alpha, half beta,
        const TensorDescriptor<half>& outputDesc,
        DevicePtr<half> outputPtr)
    {
        /* we specalize for fp16 as the scaling factors must be provided as `float` */
        float alpha_ = alpha, beta_ = beta;
        CUDA4DNN_CHECK_CUDNN(
            cudnnConvolutionBackwardData(
                handle.get(),
                &alpha_,
                filterDesc.get(), filterPtr.get(),
                inputDesc.get(), inputPtr.get(),
                convDesc.get(), convAlgo.get(),
                static_cast<void*>(workspace.get()), workspace.size_in_bytes(),
                &beta_, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSPOSE_CONVOLUTION_HPP */
