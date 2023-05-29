// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_CSL_CUDNN_RECURRENT_HPP
#define OPENCV_DNN_CUDA4DNN_CSL_CUDNN_RECURRENT_HPP

#include "cudnn.hpp"
#include <cudnn.h>


namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace cudnn {

/**
 */
class DropoutDescriptor
{
public:
    DropoutDescriptor() noexcept = default;
    DropoutDescriptor(const DropoutDescriptor &) = delete;
    DropoutDescriptor(DropoutDescriptor &&other) noexcept : descriptor{other.descriptor}
    {
        states = std::move(other.states);
        other.descriptor = nullptr;
    }

    /**
     */
    DropoutDescriptor(const Handle &handle, float dropout)
    {
        CUDA4DNN_CHECK_CUDNN(cudnnCreateDropoutDescriptor(&descriptor));

        // we need additional memory for dropout descriptor
        size_t stateSize;
        CUDA4DNN_CHECK_CUDNN(cudnnDropoutGetStatesSize(handle.get(), &stateSize));
        states.reset(stateSize);

        try
        {
            auto seed = 1234ull; // Pick a seed.
            CUDA4DNN_CHECK_CUDNN(cudnnSetDropoutDescriptor(descriptor, handle.get(), dropout,
                                                           states.get().get(), stateSize, seed));
        }
        catch (...)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyDropoutDescriptor(descriptor));
            throw;
        }
    }

    ~DropoutDescriptor() noexcept
    {
        if (descriptor)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyDropoutDescriptor(descriptor));
        }
    }

    DropoutDescriptor &operator=(const DropoutDescriptor &) = delete;
    DropoutDescriptor &operator=(DropoutDescriptor &&other) noexcept
    {
        descriptor = other.descriptor;
        states = std::move(other.states);
        other.descriptor = nullptr;
        return *this;
    };

    cudnnDropoutDescriptor_t get() const noexcept { return descriptor; }

private:
    cudnnDropoutDescriptor_t descriptor{nullptr};

    using value_type = typename ManagedPtr<char>::element_type;
    ManagedPtr<value_type> states;
};

/**
 */
template<class T>
class RNNDescriptor
{
public:
    enum class RNNMode
    {
        RNN_RELU,
        RNN_TANH,
        LSTM,
        GRU
    };

    RNNDescriptor() noexcept = default;
    RNNDescriptor(const RNNDescriptor &) = delete;
    RNNDescriptor(RNNDescriptor &&other) noexcept : descriptor{other.descriptor}
    {
        other.descriptor = nullptr;
    }

    /**
    */
    RNNDescriptor(const Handle &handle, RNNMode mode, int hidden_size, int num_layers,
                  bool bidirectional, const DropoutDescriptor &dropoutDesc)
    {
        CUDA4DNN_CHECK_CUDNN(cudnnCreateRNNDescriptor(&descriptor));
        const auto rnn_mode = [mode] {
            switch (mode)
            {
            case RNNMode::RNN_RELU:
                return CUDNN_RNN_RELU;
            case RNNMode::RNN_TANH:
                return CUDNN_RNN_TANH;
            case RNNMode::LSTM:
                return CUDNN_LSTM;
            case RNNMode::GRU:
                return CUDNN_GRU;
            default:
                return CUDNN_LSTM;
            }
        }();

        try
        {
            CUDA4DNN_CHECK_CUDNN(cudnnSetRNNDescriptor_v6(
                handle.get(), descriptor, hidden_size, num_layers, dropoutDesc.get(),
                CUDNN_LINEAR_INPUT, bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                rnn_mode,
                algo, //CUDNN_RNN_ALGO_STANDARD,
                detail::get_data_type<T>()));
        }
        catch (...)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyRNNDescriptor(descriptor));
            throw;
        }
    }

    ~RNNDescriptor() noexcept
    {
        if (descriptor)
        {
            CUDA4DNN_CHECK_CUDNN(cudnnDestroyRNNDescriptor(descriptor));
        }
    }

    RNNDescriptor &operator=(const RNNDescriptor &) = delete;
    RNNDescriptor &operator=(RNNDescriptor &&other) noexcept
    {
        descriptor = other.descriptor;
        other.descriptor = nullptr;
        return *this;
    };

    cudnnRNNDescriptor_t get() const noexcept { return descriptor; }

private:
    cudnnRNNDescriptor_t descriptor{nullptr};
    cudnnRNNMode_t mode{CUDNN_LSTM};
    // support only one algo for a while
    cudnnRNNAlgo_t algo{CUDNN_RNN_ALGO_STANDARD};
};

template<class T>
size_t getRNNWorkspaceSize(const Handle &handle, const RNNDescriptor<T> &rnnDesc,
                           const int seqLength, const TensorDescriptorsArray<T> &inputDesc)
{
    size_t workSize;
    CUDA4DNN_CHECK_CUDNN(cudnnGetRNNWorkspaceSize(handle.get(), rnnDesc.get(), seqLength,
                                                  inputDesc.get().data(), &workSize));
    return workSize;
}

template<class T>
void LSTMForward(const Handle &handle, const RNNDescriptor<T> &rnnDesc,
                 const FilterDescriptor<T> &filterDesc, DevicePtr<const T> filterPtr,
                 const TensorDescriptorsArray<T> &inputDesc, DevicePtr<const T> inputPtr,
                 const TensorDescriptor<T> &initialHDesc, DevicePtr<const T> initialH,
                 const TensorDescriptor<T> &initialCDesc, DevicePtr<const T> initialC,
                 const int seqLength, const TensorDescriptorsArray<T> &outputDesc,
                 DevicePtr<T> yOutputPtr, DevicePtr<T> ycOutputPtr, WorkspaceInstance workspace)
{
    CV_Assert(handle);

    CUDA4DNN_CHECK_CUDNN(cudnnRNNForwardInference(handle.get(), rnnDesc.get(), seqLength,
                                                  inputDesc.get().data(), inputPtr.get(), // input sequence
                                                  initialHDesc.get(), initialH.get(),
                                                  initialCDesc.get(), initialC.get(), // hidden
                                                  filterDesc.get(), filterPtr.get(), // weights
                                                  outputDesc.get().data(), yOutputPtr.get(), // output
                                                  nullptr, nullptr,
                                                  initialCDesc.get(), ycOutputPtr.get(),
                                                  static_cast<void*>(workspace.get()), workspace.size_in_bytes()));
}

}}}}} /* namespace cv::dnn::cuda4dnn::csl::cudnn */

#endif //OPENCV_DNN_CUDA4DNN_CSL_CUDNN_RECURRENT_HPP