// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

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
        outputs.assign(1, output_shape);
        outputs.assign(2, output_shape); // TODO: support indices of type CV_32S on 5.x

        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        // Normalize axis
        auto input_shape = shape(inputs.front());
        axis = normalize_axis(axis, input_shape.size());
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

        const auto input_shape = shape(input);
        size_t loops = std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1, std::multiplies<int>());
        size_t step = std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), 1, std::multiplies<int>());
        int dim_axis = input_shape[axis];
        if (loops == 1) {
            auto worker = [&](const Range &r) {
                const auto *input_ptr = input.ptr<const float>();   // TODO: support other input type
                auto *output_value_ptr = output_value.ptr<float>();
                auto *output_index_ptr = output_index.ptr<float>(); // TODO: use CV_32S on 5.x

                AutoBuffer<int> buffer_index(dim_axis);
                auto *buffer_index_ptr = buffer_index.data();
                for (int offset = r.start; offset < r.end; offset++) {
                    const auto *input_offset_ptr = input_ptr + offset;

                    std::iota(buffer_index_ptr, buffer_index_ptr + dim_axis, 0);
                    std::function<bool(int, int)> cmp;
                    if (largest) { // TODO: replace func ptr with template
                        cmp = [&](int i1, int i2) {
                            return *(input_offset_ptr + i1 * step) > *(input_offset_ptr + i2 * step);
                        };
                    } else {
                        cmp = [&](int i1, int i2) {
                            return *(input_offset_ptr + i1 * step) < *(input_offset_ptr + i2 * step);
                        };
                    }
                    std::stable_sort(buffer_index_ptr, buffer_index_ptr + dim_axis, cmp);

                    for (int i = 0; i < K; i++) {
                        int source_index = buffer_index_ptr[i];
                        output_value_ptr[i * step + offset] = *(input_offset_ptr + source_index * step);
                        output_index_ptr[i * step + offset] = source_index;
                    }
                }
            };
            parallel_for_(Range(0, step), worker, double(step / 1024.0));
        } else {
            auto worker = [&](const Range &r) {
                const auto *input_ptr = input.ptr<const float>();
                auto *output_value_ptr = output_value.ptr<float>();
                auto *output_index_ptr = output_index.ptr<float>();

                AutoBuffer<int> buffer_index(dim_axis);
                auto *buffer_index_ptr = buffer_index.data();
                for (int batch_index = r.start; batch_index < r.end; batch_index++) {
                    for (size_t offset = 0; offset < step; offset++) {
                        const auto *input_offset_ptr = input_ptr + batch_index * dim_axis * step + offset;

                        std::iota(buffer_index_ptr, buffer_index_ptr + dim_axis, 0);
                        std::function<bool(int, int)> cmp;
                        if (largest) {
                            cmp = [&](int i1, int i2) {
                                return *(input_offset_ptr + i1 * step) > *(input_offset_ptr + i2 * step);
                            };
                        } else {
                            cmp = [&](int i1, int i2) {
                                return *(input_offset_ptr + i1 * step) < *(input_offset_ptr + i2 * step);
                            };
                        }
                        std::stable_sort(buffer_index_ptr, buffer_index_ptr + dim_axis, cmp);

                        for (int i = 0; i < K; i++) {
                            int source_index = buffer_index_ptr[i];
                            output_value_ptr[batch_index * K * step + i * step + offset] = *(input_offset_ptr + source_index * step);
                            output_index_ptr[batch_index * K * step + i * step + offset] = source_index;
                        }
                    }
                }
            };
            parallel_for_(Range(0, loops), worker, double(loops / 1024.0));
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
