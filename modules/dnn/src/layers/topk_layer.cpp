// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

namespace {

template<typename T>
class ComparatorGreater {
public:
    ComparatorGreater(const T* data, size_t step)
        : data_(data), step_(step) {}

    void addOffset(size_t offset) {
        data_ += offset;
    }

    void minusOffset(size_t offset) {
        data_ -= offset;
    }

    bool operator()(const size_t lhs_idx, const size_t rhs_idx) {
        T lhs = *(data_ + lhs_idx * step_),
          rhs = *(data_ + rhs_idx * step_);
        return (lhs > rhs || (lhs == rhs && lhs_idx < rhs_idx));
    }

private:
    const T* data_;
    size_t step_;
};

template<typename T>
class ComparatorLess {
public:
    ComparatorLess(const T* data, size_t step)
        : data_(data), step_(step) {}

    void addOffset(size_t offset) {
        data_ += offset;
    }

    void minusOffset(size_t offset) {
        data_ -= offset;
    }

    bool operator()(const size_t lhs_idx, const size_t rhs_idx) {
        T lhs = *(data_ + lhs_idx * step_),
          rhs = *(data_ + rhs_idx * step_);
        return (lhs < rhs || (lhs == rhs && lhs_idx < rhs_idx));
    }

private:
    const T* data_;
    size_t step_;
};
}

class TopKLayerImpl CV_FINAL : public TopKLayer
{
public:
    TopKLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", -1);
        largest = params.get<int>("largest", 1) == 1;
        sorted = params.get<int>("sorted", 1) == 1;
        CV_CheckTrue(sorted, "TopK: sorted == false is not supported"); // TODO: support sorted

        CV_CheckTrue(params.has("k"), "TopK: parameter k is required but missing");
        K = params.get<int>("k");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        const auto &input_shape = inputs.front();
        int input_dims = input_shape.size();

        // Check if axis is valid
        CV_CheckGE(axis, -input_dims, "TopK: axis is out of range");
        CV_CheckLT(axis, input_dims, "TopK: axis is out of range");
        // Normalize axis
        int axis_normalized = normalize_axis(axis, input_shape.size());

        // Check if K is in range (0, input_shape[axis])
        CV_CheckGT(K, 0, "TopK: K needs to be a positive integer");
        CV_CheckLT(K, input_shape[axis_normalized], "TopK: K is out of range");

        // Assign output shape
        auto output_shape = input_shape;
        output_shape[axis_normalized] = K;
        outputs.assign(2, output_shape);

        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE {
        // [TODO] Check depth of inputs[1](K) once K becomes one of the inputs
        outputs.resize(2);
        outputs[0] = inputs.front();
        outputs[1] = CV_64S; // [TODO] Replace with inputs.back() once K becomes one of the inputs
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        // Normalize axis
        auto input_shape = shape(inputs.front());
        axis = normalize_axis(axis, input_shape.size());
    }

    template<class Comparator, typename T>
    void FindTopK(const Mat &input, Mat &output_value, Mat &output_index) {
        const auto input_shape = shape(input);
        size_t loops = std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1, std::multiplies<int>());
        size_t step = std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), 1, std::multiplies<int>());
        int dim_axis = input_shape[axis];
        if (loops == 1) {
            auto worker = [&](const Range &r) {
                const auto *input_ptr = input.ptr<const T>();
                auto *output_value_ptr = output_value.ptr<T>();
                auto *output_index_ptr = output_index.ptr<int64_t>();

                Comparator cmp(input_ptr, step);

                AutoBuffer<int> buffer_index(dim_axis);
                auto *buffer_index_ptr = buffer_index.data();
                for (int offset = r.start; offset < r.end; offset++) {
                    const auto *input_offset_ptr = input_ptr + offset;
                    cmp.addOffset(offset);

                    std::iota(buffer_index_ptr, buffer_index_ptr + dim_axis, 0);
                    std::stable_sort(buffer_index_ptr, buffer_index_ptr + dim_axis, cmp);

                    auto *output_value_offset_ptr = output_value_ptr + offset;
                    auto *output_index_offset_ptr = output_index_ptr + offset;
                    for (int i = 0; i < K; i++) {
                        int source_index = buffer_index_ptr[i];
                        output_value_offset_ptr[i * step] = *(input_offset_ptr + source_index * step);
                        output_index_offset_ptr[i * step] = source_index;
                    }
                    cmp.minusOffset(offset);
                }
            };
            parallel_for_(Range(0, step), worker);
        } else {
            auto worker = [&](const Range &r) {
                const auto *input_ptr = input.ptr<const T>();
                auto *output_value_ptr = output_value.ptr<T>();
                auto *output_index_ptr = output_index.ptr<int64_t>();

                Comparator cmp(input_ptr, step);

                AutoBuffer<int> buffer_index(dim_axis);
                auto *buffer_index_ptr = buffer_index.data();
                for (int batch_index = r.start; batch_index < r.end; batch_index++) {
                    for (size_t offset = 0; offset < step; offset++) {
                        const auto *input_offset_ptr = input_ptr + batch_index * dim_axis * step + offset;
                        cmp.addOffset(batch_index * dim_axis * step + offset);

                        std::iota(buffer_index_ptr, buffer_index_ptr + dim_axis, 0);
                        std::stable_sort(buffer_index_ptr, buffer_index_ptr + dim_axis, cmp);

                        auto *output_value_offset_ptr = output_value_ptr + batch_index * K * step + offset;
                        auto *output_index_offset_ptr = output_index_ptr + batch_index * K * step + offset;
                        for (int i = 0; i < K; i++) {
                            int source_index = buffer_index_ptr[i];
                            output_value_offset_ptr[i * step] = *(input_offset_ptr + source_index * step);
                            output_index_offset_ptr[i * step] = source_index;
                        }
                        cmp.minusOffset(batch_index * dim_axis * step + offset);
                    }
                }
            };
            parallel_for_(Range(0, loops), worker);
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto &input = inputs.front();
        auto &output_value = outputs.front();
        auto &output_index = outputs.back();

        if (largest) {
            switch (input.depth()) {
                case CV_8U: FindTopK<ComparatorGreater<uint8_t>, uint8_t>(input, output_value, output_index); break;
                case CV_8S: FindTopK<ComparatorGreater<int8_t>, int8_t>(input, output_value, output_index); break;
                case CV_16U: FindTopK<ComparatorGreater<uint16_t>, uint16_t>(input, output_value, output_index); break;
                case CV_16S: FindTopK<ComparatorGreater<int16_t>, int16_t>(input, output_value, output_index); break;
                case CV_16F: FindTopK<ComparatorGreater<hfloat>, hfloat>(input, output_value, output_index); break;
                case CV_32U: FindTopK<ComparatorGreater<unsigned>, unsigned>(input, output_value, output_index); break;
                case CV_32S: FindTopK<ComparatorGreater<int>, int>(input, output_value, output_index); break;
                case CV_32F: FindTopK<ComparatorGreater<float>, float>(input, output_value, output_index); break;
                case CV_64U: FindTopK<ComparatorGreater<uint64_t>, uint64_t>(input, output_value, output_index); break;
                case CV_64S: FindTopK<ComparatorGreater<int64_t>, int64_t>(input, output_value, output_index); break;
                case CV_64F: FindTopK<ComparatorGreater<double>, double>(input, output_value, output_index); break;
                default: CV_Error(Error::BadDepth, "Unsupported input data type");
            }
        } else {
            switch (input.depth()) {
                case CV_8U: FindTopK<ComparatorLess<uint8_t>, uint8_t>(input, output_value, output_index); break;
                case CV_8S: FindTopK<ComparatorLess<int8_t>, int8_t>(input, output_value, output_index); break;
                case CV_16U: FindTopK<ComparatorLess<uint16_t>, uint16_t>(input, output_value, output_index); break;
                case CV_16S: FindTopK<ComparatorLess<int16_t>, int16_t>(input, output_value, output_index); break;
                case CV_16F: FindTopK<ComparatorLess<hfloat>, hfloat>(input, output_value, output_index); break;
                case CV_32U: FindTopK<ComparatorLess<unsigned>, unsigned>(input, output_value, output_index); break;
                case CV_32S: FindTopK<ComparatorLess<int>, int>(input, output_value, output_index); break;
                case CV_32F: FindTopK<ComparatorLess<float>, float>(input, output_value, output_index); break;
                case CV_64U: FindTopK<ComparatorLess<uint64_t>, uint64_t>(input, output_value, output_index); break;
                case CV_64S: FindTopK<ComparatorLess<int64_t>, int64_t>(input, output_value, output_index); break;
                case CV_64F: FindTopK<ComparatorLess<double>, double>(input, output_value, output_index); break;
                default: CV_Error(Error::BadDepth, "Unsupported input data type");
            }
        }
    }

private:
    int axis;
    bool largest;
    bool sorted;

    int K; // FIXIT: make it layer input once dynamic shape is supported
};

Ptr<TopKLayer> TopKLayer::create(const LayerParams& params)
{
    return makePtr<TopKLayerImpl>(params);
}

}} // namespace cv::dnn
