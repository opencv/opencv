// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <limits>
#include <algorithm>

namespace cv {
namespace dnn {

struct Col2ImChunk
{
    size_t start;
    size_t len;
};

static std::vector<Col2ImChunk> planCol2ImChunks(size_t total_work, size_t max_chunk_size)
{
    CV_CheckGT(max_chunk_size, (size_t)0, "Col2Im: max_chunk_size must be positive");
    CV_CheckLE(max_chunk_size, static_cast<size_t>(std::numeric_limits<int>::max()),
               "Col2Im: max_chunk_size must not exceed int range");

    std::vector<Col2ImChunk> chunks;
    if (total_work == 0)
        return chunks;

    chunks.reserve((total_work + max_chunk_size - 1) / max_chunk_size);
    for (size_t start = 0; start < total_work; start += max_chunk_size)
    {
        size_t len = std::min(max_chunk_size, total_work - start);
        chunks.push_back({start, len});
    }
    return chunks;
}

// ONNX Col2Im operator
// Spec: https://onnx.ai/onnx/operators/onnx__Col2Im.html
// Supported opsets: 18-24

class Col2ImLayerImpl CV_FINAL : public Col2ImLayer
{
public:
    Col2ImLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        if (params.has("dilations"))
        {
            DictValue val = params.get("dilations");
            dilations_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                dilations_[i] = val.get<int>(i);
        }
        else if (params.has("dilation"))
        {
            DictValue val = params.get("dilation");
            dilations_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                dilations_[i] = val.get<int>(i);
        }

        if (params.has("pads"))
        {
            DictValue val = params.get("pads");
            pads_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                pads_[i] = val.get<int>(i);
        }
        else if (params.has("pad"))
        {
            DictValue val = params.get("pad");
            pads_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                pads_[i] = val.get<int>(i);
        }

        if (params.has("strides"))
        {
            DictValue val = params.get("strides");
            strides_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                strides_[i] = val.get<int>(i);
        }
        else if (params.has("stride"))
        {
            DictValue val = params.get("stride");
            strides_.resize(val.size());
            for (int i = 0; i < val.size(); ++i)
                strides_[i] = val.get<int>(i);
        }
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_ || this->inputs.size() != 3)
            return true;
        return !(netimpl_->isConstArg(this->inputs[1]) && netimpl_->isConstArg(this->inputs[2]));
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_ || this->inputs.size() != 3)
        {
            outputs.assign(1, shape(0));
            internals.clear();
            return false;
        }

        const Mat image_shape_mat = netimpl_->argTensor(this->inputs[1]);
        const Mat block_shape_mat = netimpl_->argTensor(this->inputs[2]);
        CV_Assert(!dynamicOutputShapes() || (!image_shape_mat.empty() && !block_shape_mat.empty()));

        Col2ImSpec spec;
        parseCol2ImSpec(inputs[0], image_shape_mat, block_shape_mat, spec);
        outputs.assign(1, spec.outShape);
        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 3);
        CV_CheckTypeEQ(inputs[0], CV_32F, "Col2Im currently supports only CV_32F");
        CV_CheckType(inputs[1], inputs[1] == CV_32S || inputs[1] == CV_64S,
                     "Col2Im: image_shape must be int32 or int64");
        CV_CheckType(inputs[2], inputs[2] == CV_32S || inputs[2] == CV_64S,
                     "Col2Im: block_shape must be int32 or int64");
        CV_Assert(requiredInternals == 0);
        outputs.assign(std::max(1, requiredOutputs), inputs[0]);
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());
        CV_UNUSED(internals_arr);

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 3);

        const Mat& data = inputs[0];
        const Mat& image_shape_mat = inputs[1];
        const Mat& block_shape_mat = inputs[2];

        CV_CheckTypeEQ(data.type(), CV_32F, "Col2Im currently supports only CV_32F");
        CV_Assert(data.isContinuous());

        Col2ImSpec spec;
        parseCol2ImSpec(data.shape(), image_shape_mat, block_shape_mat, spec);

        auto kind = outputs_arr.kind();
        std::vector<Mat> tmpOuts;
        std::vector<Mat>* pOuts = nullptr;

        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            pOuts = &outputs_arr.getMatVecRef();
            CV_Assert(!pOuts->empty());
            (*pOuts)[0].fit(spec.outShape, data.type());
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(!uouts.empty());
            tmpOuts.resize(uouts.size());
            tmpOuts[0].fit(spec.outShape, data.type());
            pOuts = &tmpOuts;
        }
        else
        {
            CV_Error(Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }

        Mat& out = (*pOuts)[0];
        if (spec.N == 0 || spec.C == 0)
        {
            if (kind == _InputArray::STD_VECTOR_UMAT)
            {
                std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
                uouts[0].fit(spec.outShape, data.type());
                tmpOuts[0].copyTo(uouts[0]);
            }
            return;
        }
        out.setTo(0);

        const int D = spec.D;
        AutoBuffer<size_t, 4> block_steps(D);
        size_t b_step = 1;
        for (int d = D - 1; d >= 0; --d)
        {
            block_steps[d] = b_step;
            CV_CheckLE(static_cast<size_t>(spec.block_shape[d]), SIZE_MAX / b_step, "Col2Im: b_step overflow");
            b_step *= static_cast<size_t>(spec.block_shape[d]);
        }

        AutoBuffer<size_t, 4> out_steps(D);
        size_t o_step = 1;
        for (int d = D - 1; d >= 0; --d)
        {
            out_steps[d] = o_step;
            CV_CheckLE(static_cast<size_t>(spec.out_spatial_dims[d]), SIZE_MAX / o_step, "Col2Im: o_step overflow");
            o_step *= static_cast<size_t>(spec.out_spatial_dims[d]);
        }

        AutoBuffer<size_t, 4> img_steps(D);
        size_t im_step = 1;
        for (int d = D - 1; d >= 0; --d)
        {
            img_steps[d] = im_step;
            CV_CheckLE(static_cast<size_t>(spec.image_shape[d]), SIZE_MAX / im_step, "Col2Im: im_step overflow");
            im_step *= static_cast<size_t>(spec.image_shape[d]);
        }

        CV_CheckLE(static_cast<size_t>(spec.N), SIZE_MAX / static_cast<size_t>(spec.C), "Col2Im: N * C overflow");
        CV_CheckLE(static_cast<size_t>(spec.L), SIZE_MAX / spec.block_prod, "Col2Im: L * block_prod overflow");
        const size_t totalNC = static_cast<size_t>(spec.N) * static_cast<size_t>(spec.C);
        const size_t input_channel_stride = spec.block_prod * static_cast<size_t>(spec.L);
        const size_t max_chunk = static_cast<size_t>(std::numeric_limits<int>::max() / 2);
        const size_t last_stride = static_cast<size_t>(spec.strides[D - 1]);
        const bool unit_last_stride = (last_stride == 1);

        std::vector<Col2ImChunk> chunks = planCol2ImChunks(totalNC, max_chunk);

        for (const auto& chunk : chunks)
        {
            parallel_for_(Range(0, static_cast<int>(chunk.len)), [&](const Range& r) {
                AutoBuffer<int, 4> l_starts(D);
                AutoBuffer<int, 4> l_ends(D);
                AutoBuffer<int64_t, 4> bases(D);
                AutoBuffer<int, 4> outer_l(D - 1);

                for (int i = r.start; i < r.end; ++i)
                {
                    size_t nc = chunk.start + static_cast<size_t>(i);
                    const float* src_nc = data.ptr<float>() + nc * input_channel_stride;
                    float* dst_nc = out.ptr<float>() + nc * spec.spatial_size;

                    for (size_t k = 0; k < spec.block_prod; ++k)
                    {
                        size_t rem_k = k;
                        bool tap_valid = true;
                        for (int d = 0; d < D; ++d)
                        {
                            int k_d = static_cast<int>(rem_k / block_steps[d]);
                            rem_k %= block_steps[d];

                            // Valid block coordinates l_d satisfy 0 <= l_d * stride_d + base_d < image_shape[d].
                            int64_t base_d = static_cast<int64_t>(k_d) * static_cast<int64_t>(spec.dilations[d])
                                             - static_cast<int64_t>(spec.pads[d]);
                            int64_t l_start_d = 0;
                            if (base_d < 0)
                            {
                                int64_t need = -base_d;
                                l_start_d = (need + static_cast<int64_t>(spec.strides[d]) - 1)
                                            / static_cast<int64_t>(spec.strides[d]);
                            }

                            int64_t limit_d = static_cast<int64_t>(spec.image_shape[d]) - 1 - base_d;
                            int64_t l_end_d = 0;
                            if (limit_d >= 0)
                            {
                                l_end_d = std::min(static_cast<int64_t>(spec.out_spatial_dims[d]),
                                                   limit_d / static_cast<int64_t>(spec.strides[d]) + 1);
                            }

                            if (l_start_d >= l_end_d)
                            {
                                tap_valid = false;
                                break;
                            }

                            l_starts[d] = static_cast<int>(l_start_d);
                            l_ends[d] = static_cast<int>(l_end_d);
                            bases[d] = base_d;
                        }

                        if (!tap_valid)
                            continue;

                        const float* src_k = src_nc + k * static_cast<size_t>(spec.L);
                        const int row_start = l_starts[D - 1];
                        const int row_len = l_ends[D - 1] - row_start;
                        const size_t last_img_pos = static_cast<size_t>(
                            static_cast<int64_t>(row_start) * static_cast<int64_t>(spec.strides[D - 1])
                            + bases[D - 1]);

                        size_t src_outer_off = static_cast<size_t>(row_start);
                        size_t dst_outer_off = last_img_pos;
                        for (int d = 0; d < D - 1; ++d)
                        {
                            outer_l[d] = l_starts[d];
                            size_t pos_d = static_cast<size_t>(
                                static_cast<int64_t>(outer_l[d]) * static_cast<int64_t>(spec.strides[d])
                                + bases[d]);
                            src_outer_off += static_cast<size_t>(outer_l[d]) * out_steps[d];
                            dst_outer_off += pos_d * img_steps[d];
                        }

                        for (;;)
                        {
                            const float* src_row = src_k + src_outer_off;
                            float* dst_row = dst_nc + dst_outer_off;
                            if (unit_last_stride)
                            {
                                for (int idx = 0; idx < row_len; ++idx)
                                    dst_row[idx] += src_row[idx];
                            }
                            else
                            {
                                for (int idx = 0; idx < row_len; ++idx)
                                    dst_row[static_cast<size_t>(idx) * last_stride] += src_row[idx];
                            }

                            int axis = D - 2;
                            while (axis >= 0)
                            {
                                if (outer_l[axis] + 1 < l_ends[axis])
                                {
                                    ++outer_l[axis];
                                    src_outer_off += out_steps[axis];
                                    dst_outer_off += static_cast<size_t>(spec.strides[axis]) * img_steps[axis];
                                    break;
                                }
                                size_t reset_span = static_cast<size_t>(outer_l[axis] - l_starts[axis]);
                                src_outer_off -= reset_span * out_steps[axis];
                                dst_outer_off -= (reset_span * static_cast<size_t>(spec.strides[axis])) * img_steps[axis];
                                outer_l[axis] = l_starts[axis];
                                --axis;
                            }
                            if (axis < 0)
                                break;
                        }
                    }
                }
            });
        }

        if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            uouts[0].fit(spec.outShape, data.type());
            tmpOuts[0].copyTo(uouts[0]);
        }
    }

private:
    struct Col2ImSpec
    {
        int N = 0;
        int C = 0;
        int L = 0;
        int D = 0;
        size_t block_prod = 1;
        size_t spatial_size = 1;
        AutoBuffer<int, 4> image_shape;
        AutoBuffer<int, 4> block_shape;
        AutoBuffer<int, 4> dilations;
        AutoBuffer<int, 4> strides;
        AutoBuffer<int, 8> pads;
        AutoBuffer<int, 4> out_spatial_dims;
        MatShape outShape;
    };

    void parseCol2ImSpec(const MatShape& dataShape,
                         const Mat& image_shape_mat,
                         const Mat& block_shape_mat,
                         Col2ImSpec& spec) const
    {
        CV_Assert(dataShape.dims == 3);
        CV_Assert(image_shape_mat.isContinuous());
        CV_Assert(block_shape_mat.isContinuous());

        const int N = dataShape[0];
        const int C_x_block = dataShape[1];
        const int L = dataShape[2];
        CV_CheckGE(N, 0, "Col2Im: batch dimension N must be non-negative");
        CV_CheckGE(C_x_block, 0, "Col2Im: channel*block dimension must be non-negative");
        CV_CheckGE(L, 0, "Col2Im: spatial length L must be non-negative");

        const size_t total_img_dims = image_shape_mat.total();
        CV_CheckGE(total_img_dims, (size_t)2, "Col2Im: image_shape must have at least 2 spatial dimensions");
        CV_CheckLE(total_img_dims, (size_t)(MatShape::MAX_DIMS - 2),
                   "Col2Im: spatial dimensions exceed maximum supported MatShape rank");
        const int D = (int)total_img_dims;
        CV_CheckEQ(block_shape_mat.total(), total_img_dims, "Col2Im: block_shape length must match image_shape");

        spec.image_shape.allocate(D);
        spec.block_shape.allocate(D);
        size_t block_prod = 1;

        if (image_shape_mat.type() == CV_64S)
        {
            const int64_t* ptr = image_shape_mat.ptr<int64_t>();
            for (int d = 0; d < D; ++d)
            {
                if (ptr[d] <= 0 || ptr[d] > (int64_t)std::numeric_limits<int>::max())
                    CV_Error(Error::StsBadArg, "Col2Im: image_shape elements must be positive and not exceed INT_MAX");
                spec.image_shape[d] = (int)ptr[d];
            }
        }
        else if (image_shape_mat.type() == CV_32S)
        {
            const int32_t* ptr = image_shape_mat.ptr<int32_t>();
            for (int d = 0; d < D; ++d)
            {
                CV_CheckGT(ptr[d], 0, "Col2Im: image_shape elements must be positive");
                spec.image_shape[d] = (int)ptr[d];
            }
        }
        else
        {
            CV_Error(Error::StsBadArg, "Col2Im: image_shape must be int32 or int64");
        }

        if (block_shape_mat.type() == CV_64S)
        {
            const int64_t* ptr = block_shape_mat.ptr<int64_t>();
            for (int d = 0; d < D; ++d)
            {
                if (ptr[d] <= 0 || ptr[d] > (int64_t)std::numeric_limits<int>::max())
                    CV_Error(Error::StsBadArg, "Col2Im: block_shape elements must be positive and not exceed INT_MAX");
                CV_CheckLE((size_t)ptr[d], SIZE_MAX / block_prod, "Col2Im: block_shape product overflow");
                spec.block_shape[d] = (int)ptr[d];
                block_prod *= (size_t)ptr[d];
            }
        }
        else if (block_shape_mat.type() == CV_32S)
        {
            const int32_t* ptr = block_shape_mat.ptr<int32_t>();
            for (int d = 0; d < D; ++d)
            {
                CV_CheckGT(ptr[d], 0, "Col2Im: block_shape elements must be positive");
                CV_CheckLE((size_t)ptr[d], SIZE_MAX / block_prod, "Col2Im: block_shape product overflow");
                spec.block_shape[d] = (int)ptr[d];
                block_prod *= (size_t)ptr[d];
            }
        }
        else
        {
            CV_Error(Error::StsBadArg, "Col2Im: block_shape must be int32 or int64");
        }

        CV_Assert(block_prod > 0);
        CV_CheckEQ((size_t)C_x_block % block_prod, (size_t)0,
                   "Col2Im: channel dimension must be divisible by block_shape product");
        const int C = (int)((size_t)C_x_block / block_prod);

        if (!dilations_.empty())
            CV_CheckEQ(dilations_.size(), (size_t)D, "Col2Im: dilations attribute length must match spatial dimensions");
        if (!strides_.empty())
            CV_CheckEQ(strides_.size(), (size_t)D, "Col2Im: strides attribute length must match spatial dimensions");
        if (!pads_.empty())
            CV_CheckEQ(pads_.size(), (size_t)(2 * D), "Col2Im: pads attribute length must be 2 * spatial dimensions");

        spec.dilations.allocate(D);
        for (int d = 0; d < D; ++d)
        {
            spec.dilations[d] = (d < (int)dilations_.size()) ? dilations_[d] : 1;
            CV_CheckGT(spec.dilations[d], 0, "Col2Im: dilation must be positive");
        }

        spec.strides.allocate(D);
        for (int d = 0; d < D; ++d)
        {
            spec.strides[d] = (d < (int)strides_.size()) ? strides_[d] : 1;
            CV_CheckGT(spec.strides[d], 0, "Col2Im: stride must be positive");
        }

        spec.pads.allocate(2 * D);
        for (int d = 0; d < 2 * D; ++d)
        {
            spec.pads[d] = (d < (int)pads_.size()) ? pads_[d] : 0;
            CV_CheckGE(spec.pads[d], 0, "Col2Im: pad must be non-negative");
        }

        spec.out_spatial_dims.allocate(D);
        size_t expected_L = 1;
        for (int d = 0; d < D; ++d)
        {
            int64_t dim_padded = (int64_t)spec.image_shape[d] + (int64_t)spec.pads[d] + (int64_t)spec.pads[d + D];
            int64_t eff_kernel = (int64_t)(spec.block_shape[d] - 1) * (int64_t)spec.dilations[d] + 1;
            if (dim_padded < eff_kernel)
                CV_Error(Error::StsBadArg, "Col2Im: padded image dimension must be >= effective kernel size");
            int64_t l_d = (dim_padded - eff_kernel) / spec.strides[d] + 1;
            if (l_d <= 0 || l_d > (int64_t)std::numeric_limits<int>::max())
                CV_Error(Error::StsBadArg, "Col2Im: output spatial dimension must be in range [1, INT_MAX]");
            spec.out_spatial_dims[d] = (int)l_d;
            CV_CheckLE((size_t)l_d, SIZE_MAX / expected_L, "Col2Im: expected L overflow");
            expected_L *= (size_t)l_d;
        }
        CV_CheckEQ((size_t)L, expected_L, "Col2Im: input spatial length L does not match calculated blocks");

        MatShape outShape;
        outShape.reserve(2 + D);
        outShape.push_back(N);
        outShape.push_back(C);
        size_t spatial_size = 1;
        for (int d = 0; d < D; ++d)
        {
            outShape.push_back(spec.image_shape[d]);
            CV_CheckLE((size_t)spec.image_shape[d], SIZE_MAX / spatial_size, "Col2Im: image spatial size overflow");
            spatial_size *= (size_t)spec.image_shape[d];
        }

        spec.N = N;
        spec.C = C;
        spec.L = L;
        spec.D = D;
        spec.block_prod = block_prod;
        spec.spatial_size = spatial_size;
        spec.outShape = outShape;
    }

    std::vector<int> dilations_;
    std::vector<int> pads_;
    std::vector<int> strides_;
};

Ptr<Col2ImLayer> Col2ImLayer::create(const LayerParams& params)
{
    return makePtr<Col2ImLayerImpl>(params);
}

} // namespace dnn
} // namespace cv
