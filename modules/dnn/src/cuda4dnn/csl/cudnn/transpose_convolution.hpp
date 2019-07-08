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

    template <class T>
    class TransposeConvolutionAlgorithm {
    public:
        TransposeConvolutionAlgorithm() noexcept : workspace_size{ 0 } { }
        TransposeConvolutionAlgorithm(TransposeConvolutionAlgorithm&) = default;
        TransposeConvolutionAlgorithm(TransposeConvolutionAlgorithm&&) = default;

        TransposeConvolutionAlgorithm(
            const Handle& handle,
            const ConvolutionDescriptor<T>& conv,
            const FilterDescriptor<T>& filter,
            const TensorDescriptor<T>& input,
            const TensorDescriptor<T>& output)
        {
            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionBackwardDataAlgorithm(
                    HandleAccessor::get(handle),
                    filter.get(), input.get(), conv.get(), output.get(),
                    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                    0, /* no memory limit */
                    &dalgo
                )
            );

            CUDA4DNN_CHECK_CUDNN(
                cudnnGetConvolutionBackwardDataWorkspaceSize(
                    HandleAccessor::get(handle),
                    filter.get(), input.get(), conv.get(), output.get(),
                    dalgo, &workspace_size
                )
            );
        }

        TransposeConvolutionAlgorithm& operator=(const TransposeConvolutionAlgorithm&) = default;
        TransposeConvolutionAlgorithm& operator=(TransposeConvolutionAlgorithm&& other) = default;

        cudnnConvolutionBwdDataAlgo_t get() const noexcept { return dalgo; }

        std::size_t get_workspace_size() const noexcept { return workspace_size; }

    private:
        cudnnConvolutionBwdDataAlgo_t dalgo;
        std::size_t workspace_size;
    };

    template <class T> inline
    void transpose_convolve(
        const Handle& handle,
        const ConvolutionDescriptor<T>& convDesc,
        const TransposeConvolutionAlgorithm<T>& convAlgo,
        const Workspace& workspace,
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
                HandleAccessor::get(handle),
                &alpha,
                filterDesc.get(), filterPtr.get(),
                inputDesc.get(), inputPtr.get(),
                convDesc.get(), convAlgo.get(),
                WorkspaceAccessor::get(workspace).get(), workspace.size(),
                &beta, outputDesc.get(), outputPtr.get()
            )
        );
    }

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif /* OPENCV_DNN_CUDA4DNN_CSL_CUDNN_TRANSPOSE_CONVOLUTION_HPP */
