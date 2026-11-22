// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// ONNX TensorScatter operator
// Spec: https://onnx.ai/onnx/operators/onnx__TensorScatter.html
// Supported opsets: 24

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <limits>
#include <algorithm>

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

    virtual bool alwaysSupportInplace() const CV_OVERRIDE { return true; }

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

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

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

        int batch_size = past_cache.size[0];
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
                    if (max_seq_len > 0)
                    {
                        int64_t mod = (w % max_seq_len + max_seq_len) % max_seq_len;
                        write_offsets[b] = static_cast<size_t>(mod);
                    }
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

        auto kind = outputs_arr.kind();
        std::vector<Mat> tmpOuts;
        std::vector<Mat>* pOuts = nullptr;
        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            pOuts = &outputs_arr.getMatVecRef();
            CV_Assert(!pOuts->empty());
            (*pOuts)[0].fit(past_cache.shape(), past_cache.type());
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(!uouts.empty());
            tmpOuts.resize(uouts.size());
            tmpOuts[0].fit(past_cache.shape(), past_cache.type());
            pOuts = &tmpOuts;
        }
        else
        {
            CV_Error(Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }

        Mat& dst = (*pOuts)[0];
        CV_Assert(dst.isContinuous());

        if (batch_size == 0 || seq_len == 0 || past_cache.total() == 0 || update.total() == 0)
        {
            if (past_cache.total() > 0 && past_cache.ptr<uchar>() != dst.ptr<uchar>())
                past_cache.copyTo(dst);
            if (kind == _InputArray::STD_VECTOR_UMAT)
                dst.copyTo(outputs_arr.getUMatVecRef()[0]);
            return;
        }

        const size_t outer_size = past_cache.total(0, axis);
        const size_t inner_bytes = past_cache.total(axis + 1) * past_cache.elemSize();
        const size_t stride_batch = outer_size / static_cast<size_t>(batch_size);

        const uchar* cache_ptr = past_cache.ptr<uchar>();
        const uchar* src_ptr = update.ptr<uchar>();
        uchar* dst_ptr = dst.ptr<uchar>();
        const bool copy_untouched = (cache_ptr != dst_ptr);

        const size_t seq_len_u = static_cast<size_t>(seq_len);
        const size_t max_seq_len_u = static_cast<size_t>(max_seq_len);
        const size_t src_row_bytes = seq_len_u * inner_bytes;
        const size_t dst_row_bytes = max_seq_len_u * inner_bytes;

        const size_t work_row_bytes = copy_untouched ? dst_row_bytes : src_row_bytes;
        const double total_work_bytes = static_cast<double>(outer_size) * static_cast<double>(work_row_bytes);
        const double max_int_d = static_cast<double>(std::numeric_limits<int>::max());
        int nstripes = static_cast<int>(std::max(1.0, std::min({static_cast<double>(outer_size), total_work_bytes / (64.0 * 1024.0), max_int_d})));
        if (nstripes <= 0)
            nstripes = 1;

        const size_t q = outer_size / static_cast<size_t>(nstripes);
        const size_t rem = outer_size % static_cast<size_t>(nstripes);

        parallel_for_(Range(0, nstripes), [&](const Range& r) {
            size_t st = static_cast<size_t>(r.start) * q + std::min(static_cast<size_t>(r.start), rem);
            size_t en = static_cast<size_t>(r.end) * q + std::min(static_cast<size_t>(r.end), rem);

            for (size_t outer_idx = st; outer_idx < en; ++outer_idx)
            {
                size_t b = outer_idx / stride_batch;
                size_t write_offset = write_offsets[b];

                const uchar* cache_row = cache_ptr + outer_idx * dst_row_bytes;
                const uchar* src_row = src_ptr + outer_idx * src_row_bytes;
                uchar* dst_row = dst_ptr + outer_idx * dst_row_bytes;

                size_t first_len = std::min(seq_len_u, max_seq_len_u - write_offset);
                size_t first_bytes = first_len * inner_bytes;
                std::memcpy(dst_row + write_offset * inner_bytes, src_row, first_bytes);

                size_t second_len = seq_len_u - first_len;
                if (second_len > 0)
                {
                    std::memcpy(dst_row, src_row + first_bytes, second_len * inner_bytes);
                }

                if (copy_untouched)
                {
                    if (second_len > 0)
                    {
                        if (write_offset > second_len)
                        {
                            size_t mid_off = second_len * inner_bytes;
                            std::memcpy(dst_row + mid_off, cache_row + mid_off, (write_offset - second_len) * inner_bytes);
                        }
                    }
                    else
                    {
                        if (write_offset > 0)
                        {
                            std::memcpy(dst_row, cache_row, write_offset * inner_bytes);
                        }
                        size_t suffix_start = write_offset + seq_len_u;
                        if (suffix_start < max_seq_len_u)
                        {
                            size_t suffix_off = suffix_start * inner_bytes;
                            std::memcpy(dst_row + suffix_off, cache_row + suffix_off, (max_seq_len_u - suffix_start) * inner_bytes);
                        }
                    }
                }
            }
        }, nstripes);

        if (kind == _InputArray::STD_VECTOR_UMAT)
            dst.copyTo(outputs_arr.getUMatVecRef()[0]);
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
