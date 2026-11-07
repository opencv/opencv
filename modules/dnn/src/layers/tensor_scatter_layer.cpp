// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ONNX TensorScatter operator
// Spec: https://onnx.ai/onnx/operators/onnx__TensorScatter.html
// Supported opsets: 24

#include "../precomp.hpp"
#include "tensor_scatter.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

class TensorScatterLayerImpl CV_FINAL : public TensorScatterLayer
{
public:
    TensorScatterLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis_ = params.get<int>("axis", -2);
        mode_ = toLowerCase(params.get<String>("mode", "linear"));
        CV_Check(mode_, mode_ == "linear" || mode_ == "circular", "TensorScatter: unsupported mode");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        outputs.assign(1, inputs[0]);
        return false;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                          const int requiredOutputs,
                          const int requiredInternals,
                          std::vector<MatType>& outputs,
                          std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_CheckTypeEQ(inputs[0], inputs[1], "TensorScatter: past_cache and update must have the same type");
        outputs.assign(1, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() >= 2);
        const Mat& past_cache = inputs[0];
        const Mat& update = inputs[1];

        CV_Assert(past_cache.isContinuous());
        CV_Assert(update.isContinuous());
        CV_CheckEQ(past_cache.dims, update.dims, "TensorScatter: past_cache and update must have the same rank");
        CV_CheckTypeEQ(past_cache.type(), update.type(), "TensorScatter: past_cache and update must have the same type");

        int ndims = past_cache.dims;
        int axis = axis_;
        if (axis < 0)
            axis += ndims;
        CV_Check(axis, axis > 0 && axis < ndims, "TensorScatter: axis must be in range [1, ndims-1]");

        for (int d = 0; d < ndims; ++d)
        {
            if (d != axis)
            {
                CV_CheckEQ(past_cache.size[d], update.size[d], "TensorScatter: shape mismatch on non-axis dimension");
            }
        }

        int max_seq_len = past_cache.size[axis];
        int seq_len = update.size[axis];
        CV_CheckLE(seq_len, max_seq_len, "TensorScatter: sequence_length cannot exceed max_sequence_length");

        past_cache.copyTo(outputs[0]);
        CV_Assert(outputs[0].isContinuous());

        int batch_size = past_cache.size[0];
        if (batch_size == 0 || seq_len == 0 || past_cache.total() == 0 || update.total() == 0)
            return;

        bool is_circular = (mode_ == "circular");
        std::vector<size_t> write_offsets(batch_size, 0);

        const Mat* write_indices_mat = inputs.size() > 2 ? &inputs[2] : nullptr;
        if (write_indices_mat && !write_indices_mat->empty())
        {
            CV_Assert(write_indices_mat->isContinuous());
            CV_CheckEQ((int)write_indices_mat->total(), batch_size,
                       "TensorScatter: write_indices must have length equal to batch_size");
            int depth = write_indices_mat->depth();
            CV_Check(depth, depth == CV_32S || depth == CV_64S,
                     "TensorScatter: write_indices must be int32 or int64");

            const int32_t* ptr32 = depth == CV_32S ? write_indices_mat->ptr<int32_t>() : nullptr;
            const int64_t* ptr64 = depth == CV_64S ? write_indices_mat->ptr<int64_t>() : nullptr;

            for (int b = 0; b < batch_size; ++b)
            {
                int64_t w = ptr32 ? static_cast<int64_t>(ptr32[b]) : ptr64[b];
                if (is_circular)
                {
                    int64_t mod = (w % max_seq_len + max_seq_len) % max_seq_len;
                    write_offsets[b] = static_cast<size_t>(mod);
                }
                else
                {
                    CV_Assert(w >= 0 && "TensorScatter: write_index must be >= 0 in linear mode");
                    CV_Assert(w <= static_cast<int64_t>(max_seq_len - seq_len) &&
                              "TensorScatter: write_index + sequence_length exceeds max_sequence_length in linear mode");
                    write_offsets[b] = static_cast<size_t>(w);
                }
            }
        }
        else if (!is_circular)
        {
            CV_Assert(seq_len <= max_seq_len &&
                      "TensorScatter: sequence_length exceeds max_sequence_length in linear mode");
        }

        size_t outer_size = 1;
        for (int d = 0; d < axis; ++d)
        {
            size_t dim = static_cast<size_t>(past_cache.size[d]);
            CV_Assert(dim == 0 || outer_size <= std::numeric_limits<size_t>::max() / dim);
            outer_size *= dim;
        }

        size_t inner_size = 1;
        for (int d = axis + 1; d < ndims; ++d)
        {
            size_t dim = static_cast<size_t>(past_cache.size[d]);
            CV_Assert(dim == 0 || inner_size <= std::numeric_limits<size_t>::max() / dim);
            inner_size *= dim;
        }

        CV_Assert(past_cache.elemSize() == 0 || inner_size <= std::numeric_limits<size_t>::max() / past_cache.elemSize());
        size_t inner_bytes = inner_size * past_cache.elemSize();
        size_t stride_batch = outer_size / static_cast<size_t>(batch_size);

        const uchar* src_ptr = update.ptr<uchar>();
        uchar* dst_ptr = outputs[0].ptr<uchar>();

        if (outer_size == 0 || seq_len == 0 || inner_bytes == 0)
            return;

        const size_t max_outer_chunk = static_cast<size_t>(std::numeric_limits<int>::max() / 2);
        std::vector<TensorScatterChunk> chunks = planTensorScatterChunks(outer_size, max_outer_chunk);

        for (const auto& chunk : chunks)
        {
            double chunk_work_bytes = static_cast<double>(chunk.len) * static_cast<double>(seq_len) * static_cast<double>(inner_bytes);
            double nstripes = chunk_work_bytes / (64.0 * 1024.0);

            parallel_for_(Range(0, static_cast<int>(chunk.len)), [&](const Range& r) {
                for (int ob = r.start; ob < r.end; ++ob)
                {
                    size_t outer_idx = chunk.start + static_cast<size_t>(ob);
                    int b = static_cast<int>(outer_idx / stride_batch);
                    size_t write_offset = write_offsets[b];

                    for (int s = 0; s < seq_len; ++s)
                    {
                        size_t target_seq_idx = write_offset + static_cast<size_t>(s);
                        if (target_seq_idx >= static_cast<size_t>(max_seq_len))
                            target_seq_idx -= static_cast<size_t>(max_seq_len);

                        size_t src_offset = (outer_idx * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * inner_bytes;
                        size_t dst_offset = (outer_idx * static_cast<size_t>(max_seq_len) + target_seq_idx) * inner_bytes;

                        std::memcpy(dst_ptr + dst_offset, src_ptr + src_offset, inner_bytes);
                    }
                }
            }, nstripes);
        }
    }

private:
    int axis_;
    String mode_;
};

Ptr<TensorScatterLayer> TensorScatterLayer::create(const LayerParams& params)
{
    return makePtr<TensorScatterLayerImpl>(params);
}

} // namespace dnn
} // namespace cv
