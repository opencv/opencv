// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

// com.microsoft GatherBlockQuantized (contrib op, since com.microsoft opset 1):
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgatherblockquantized
// Gathers rows of a block-quantized table and dequantises them in one step:
// out = (value - zero_point) * scale, where value is a {2,4,8}-bit field of the uint8 data.

namespace cv { namespace dnn {

class GatherBlockQuantizedLayerImpl CV_FINAL : public GatherBlockQuantizedLayer
{
public:
    GatherBlockQuantizedLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        gather_axis = params.get<int>("gather_axis", 0);
        quantize_axis = params.get<int>("quantize_axis", 1);
        block_size = params.get<int>("block_size", 128);
        bits = params.get<int>("bits", 4);

        CV_Check(bits, bits == 2 || bits == 4 || bits == 8,
                 "DNN/GatherBlockQuantized: bits must be 2, 4 or 8");
        CV_Check(block_size, block_size >= 16 && (block_size & (block_size - 1)) == 0,
                 "DNN/GatherBlockQuantized: block_size must be a power of two and at least 16");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    // Resolved axes and the block counts derived from them, in unpacked elements.
    struct BlockInfo
    {
        int gather_axis, quantize_axis, components;
        int64_t gather_M, gather_N, gather_axis_dim, gather_block;
        int64_t quantize_axis_dim, quantize_N, n_blk;
    };

    BlockInfo resolve(const MatShape& data, const MatShape& indices) const
    {
        BlockInfo info;
        info.components = 8 / bits;
        info.gather_axis = normalize_axis(gather_axis, data.dims);
        info.quantize_axis = normalize_axis(quantize_axis, data.dims);
        CV_Check(info.gather_axis, info.gather_axis == 0,
                 "DNN/GatherBlockQuantized: uint8 data requires gather_axis 0");
        CV_Check(info.quantize_axis, info.quantize_axis == data.dims - 1,
                 "DNN/GatherBlockQuantized: uint8 data requires quantize_axis to be the last one");

        info.gather_axis_dim = data[info.gather_axis];
        info.gather_M = 1;
        for (int i = 0; i < info.gather_axis; i++)
            info.gather_M *= data[i];
        info.gather_block = info.components;
        for (int i = info.gather_axis + 1; i < data.dims; i++)
            info.gather_block *= data[i];
        info.gather_N = (int64_t)indices.total();
        info.quantize_axis_dim = (int64_t)data[info.quantize_axis] * info.components;
        info.quantize_N = 1;
        for (int i = info.quantize_axis + 1; i < data.dims; i++)
            info.quantize_N *= data[i];
        info.n_blk = (info.quantize_axis_dim + block_size - 1) / block_size;
        return info;
    }

    // The gathered axis becomes the indices' shape, and the packed last axis unpacks.
    MatShape getOutShape(const MatShape& data, const MatShape& indices) const
    {
        const int axis_ = normalize_axis(gather_axis, data.dims);
        MatShape out(data.dims - 1 + indices.dims);
        int k = 0;
        for (int i = 0; i < axis_; i++)
            out[k++] = data[i];
        for (int i = 0; i < indices.dims; i++)
            out[k++] = indices[i];
        for (int i = axis_ + 1; i < data.dims; i++)
            out[k++] = data[i];
        CV_Assert(k == out.dims && out.dims > 0);
        out[out.dims - 1] *= 8 / bits;
        return out;
    }

    void checkQuantParams(const MatShape& data, const MatShape& scales,
                          const MatShape& zp, const BlockInfo& info) const
    {
        CV_CheckEQ(scales.dims, data.dims, "DNN/GatherBlockQuantized: data and scales rank differ");
        for (int i = 0; i < data.dims; i++) {
            const int want = i == info.quantize_axis ? (int)info.n_blk : data[i];
            CV_CheckEQ(scales[i], want, "DNN/GatherBlockQuantized: scales shape mismatch");
        }
        // Optional, and its shape is not always known here; forward reads the tensor itself.
        if (zp.dims <= 0)
            return;
        CV_CheckEQ(zp.dims, scales.dims,
                   "DNN/GatherBlockQuantized: scales and zero_points rank differ");
        for (int i = 0; i < scales.dims; i++) {
            const int64_t packed = (info.n_blk + info.components - 1) / info.components;
            const int want = i == info.quantize_axis ? (int)packed : scales[i];
            CV_CheckEQ(zp[i], want, "DNN/GatherBlockQuantized: zero_points shape mismatch");
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Check((int)inputs.size(), inputs.size() == 3 || inputs.size() == 4,
                 "DNN/GatherBlockQuantized: expected (data, indices, scales[, zero_points])");
        CV_CheckGE(inputs[0].dims, 2, "DNN/GatherBlockQuantized: data must have rank 2 or more");

        const BlockInfo info = resolve(inputs[0], inputs[1]);
        checkQuantParams(inputs[0], inputs[2], inputs.size() == 4 ? inputs[3] : MatShape(), info);
        outputs.assign(1, getOutShape(inputs[0], inputs[1]));
        internals.clear();
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Check((int)inputs.size(), inputs.size() == 3 || inputs.size() == 4,
                 "DNN/GatherBlockQuantized: expected (data, indices, scales[, zero_points])");
        CV_CheckType(inputs[0], inputs[0] == CV_8U,
                     "DNN/GatherBlockQuantized: data must be CV_8U");
        CV_CheckType(inputs[1], inputs[1] == CV_32S || inputs[1] == CV_64S,
                     "DNN/GatherBlockQuantized: indices must be CV_32S or CV_64S");
        CV_CheckType(inputs[2], inputs[2] == CV_32F,
                     "DNN/GatherBlockQuantized: scales must be CV_32F");
        if (inputs.size() == 4 && inputs[3] != CV_8U && inputs[3] != -1)
            CV_Error(Error::StsNotImplemented,
                     "DNN/GatherBlockQuantized: zero_points must be CV_8U");
        outputs.assign(requiredOutputs, inputs[2]);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& data = inputs[0];
        const Mat& indices = inputs[1];
        const Mat& scales = inputs[2];
        const Mat zeroPoints = inputs.size() == 4 ? inputs[3] : Mat();
        Mat& out = outputs[0];
        CV_Assert(data.isContinuous() && indices.isContinuous() && scales.isContinuous());
        CV_Assert(out.isContinuous() && (zeroPoints.empty() || zeroPoints.isContinuous()));

        const BlockInfo info = resolve(data.shape(), indices.shape());

        // Resolved once here, so the parallel loop reads a plain row; -1 means zero-fill.
        std::vector<int64_t> rows((size_t)info.gather_N);
        const int indType = indices.type();
        for (int64_t j = 0; j < info.gather_N; j++) {
            int64_t v = indType == CV_32S ? (int64_t)indices.ptr<int32_t>()[j]
                                          : indices.ptr<int64_t>()[j];
            v += v < 0 ? info.gather_axis_dim : 0;
            // Unlike ONNX Gather, an out-of-range index zeroes its slice instead of failing.
            rows[(size_t)j] = (v < 0 || v >= info.gather_axis_dim) ? -1 : v;
        }

        const uchar* dataPtr = data.ptr<uchar>();
        const float* scalePtr = scales.ptr<float>();
        const uchar* zpPtr = zeroPoints.empty() ? nullptr : zeroPoints.ptr<uchar>();
        float* outPtr = out.ptr<float>();

        const int64_t total = info.gather_M * info.gather_N;
        parallel_for_(Range(0, (int)total), [&](const Range& r) {
            for (int i = r.start; i < r.end; i++)
                gatherOne(dataPtr, scalePtr, zpPtr, outPtr, rows, info, i);
        }, std::min((double)total, (double)total * info.gather_block / 1e5));
    }

    // One gathered row: blocks outer so the scale and zero point are read once each.
    void gatherOne(const uchar* data, const float* scales, const uchar* zp, float* out,
                   const std::vector<int64_t>& rows, const BlockInfo& info, int idx) const
    {
        const int64_t m = idx / info.gather_N, n = idx % info.gather_N;
        const int64_t outBase = (int64_t)idx * info.gather_block;
        if (rows[(size_t)n] < 0) {
            std::memset(out + outBase, 0, (size_t)info.gather_block * sizeof(float));
            return;
        }
        const int64_t dataBase = m * info.gather_axis_dim * info.gather_block
                               + rows[(size_t)n] * info.gather_block;
        const int64_t zpStride = (info.n_blk + info.components - 1) / info.components;
        const int mask = (1 << bits) - 1;
        const int defaultZp = 1 << (bits - 1);

        for (int64_t sub = 0; sub < info.gather_block; sub += info.quantize_axis_dim) {
            const int64_t x = (dataBase + sub) / info.quantize_axis_dim;
            for (int64_t b = 0; b < info.n_blk; b++) {
                const float scale = scales[x * info.n_blk + b];
                int zpVal = defaultZp;
                if (zp) {
                    const uchar packed = zp[x * zpStride + b / info.components];
                    zpVal = (packed >> ((b % info.components) * bits)) & mask;
                }
                const int64_t lo = b * block_size;
                const int64_t hi = std::min(lo + block_size, info.quantize_axis_dim);
                for (int64_t y = lo; y < hi; y++) {
                    const int64_t di = dataBase + sub + y;
                    const uchar byte = data[di / info.components];
                    const int v = (byte >> ((di % info.components) * bits)) & mask;
                    out[outBase + sub + y] = (float)(v - zpVal) * scale;
                }
            }
        }
    }
};

Ptr<GatherBlockQuantizedLayer> GatherBlockQuantizedLayer::create(const LayerParams& params)
{
    return makePtr<GatherBlockQuantizedLayerImpl>(params);
}

}}  // namespace cv::dnn
