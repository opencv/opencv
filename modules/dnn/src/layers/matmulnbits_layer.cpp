// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/matmulnbits_kernels.hpp"

// com.microsoft MatMulNBits (contrib op, since com.microsoft opset 1):
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits
// Y = A @ dequant(B).T. Weights are block-quantized to {4,8} bits (symmetric, default
// zero-point 2^(bits-1)) and dequantized per block inside the GEMM (no fp32 copy).

namespace cv { namespace dnn {

class MatMulNBitsLayerImpl CV_FINAL : public MatMulNBitsLayer {
 public:
    MatMulNBitsLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        K = params.get<int>("K");
        N = params.get<int>("N");
        bits = params.get<int>("bits", 4);
        block_size = params.get<int>("block_size");

        CV_Check(bits, bits == 4 || bits == 8,
                 "DNN/MatMulNBits: only 4/8-bit weights are supported");
        CV_Check(block_size, block_size > 0 && (block_size * bits) % 8 == 0,
                 "DNN/MatMulNBits: block_size*bits must be a positive multiple of 8");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1),
                   "DNN/MatMulNBits: expected a single activation input; weights and scales must be constants");
        CV_Check((int)blobs.size(), blobs.size() == 2 || blobs.size() == 3,
                   "DNN/MatMulNBits: expected packed weights and scales as constant blobs");

        const int n_blk = (K + block_size - 1) / block_size;   // ceil: K need not divide block_size; last block may be partial
        CV_CheckEQ(static_cast<int>(blobs[0].total()), N * n_blk * (block_size * bits / 8),
                   "DNN/MatMulNBits: packed weight size mismatch");
        CV_CheckEQ(static_cast<int>(blobs[1].total()), N * n_blk,
                   "DNN/MatMulNBits: scales size mismatch");
        if (blobs.size() == 3) {
            const int epb = 8 / bits;
            CV_CheckEQ(static_cast<int>(blobs[2].total()), N * ((n_blk + epb - 1) / epb),
                       "DNN/MatMulNBits: zero_points size mismatch");
            CV_CheckType(blobs[2].type(), blobs[2].type() == CV_8U,
                         "DNN/MatMulNBits: zero_points must be CV_8U");
        }
        CV_CheckType(blobs[0].type(), blobs[0].type() == CV_8U,
                     "DNN/MatMulNBits: packed weights must be CV_8U; int32-packed weights are not supported");
        CV_CheckType(blobs[1].type(), blobs[1].type() == CV_32F,
                     "DNN/MatMulNBits: scales must be CV_32F");

        MatShape shape_A = inputs[0];
        CV_CheckGE(shape_A.size(), static_cast<size_t>(1), "DNN/MatMulNBits: invalid shape of input A");
        CV_CheckEQ(shape_A.back(), K, "DNN/MatMulNBits: last dimension of A must equal K");

        MatShape out = shape_A;
        out.back() = N;
        outputs.assign(1, out);
        // A as even and odd halves for the fused M=1 path; K floats hold both.
        internals.assign(bits == 4 ? 1 : 0, MatShape({K}));
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "DNN/MatMulNBits: expected a single activation input");
        CV_CheckType(inputs[0], inputs[0] == CV_32F, "DNN/MatMulNBits: activation must be CV_32F");
        outputs.assign(requiredOutputs, CV_32F);
        internals.assign(requiredInternals, CV_32F);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &) const CV_OVERRIDE {
        CV_Assert(!inputs.empty());
        const int64 M = total(inputs[0]) / K;
        return CV_BIG_INT(2) * M * N * K;
    }

    void runImpl(const Mat& A, const Mat& B, const Mat& scales, const Mat& zeroPoints,
                 Mat& Y, Mat& split) const {
        const int M = static_cast<int>(A.total() / K);
        const int n_blk = (K + block_size - 1) / block_size;

        const float* a = A.ptr<const float>();
        const uchar* bq = B.ptr<const uchar>();
        const float* sc = scales.ptr<const float>();
        const uchar* zp = zeroPoints.empty() ? nullptr : zeroPoints.ptr<const uchar>();
        float* y = Y.ptr<float>();

        // Staging costs 4 bytes written and read back per 0.5 byte of weight and amortizes
        // only over the M rows, so at M=1 it is pure overhead. Everything else stays staged.
        const bool fused = M == 1 && bits == 4 && K % 2 == 0 && !split.empty() &&
                           block_size / 2 >= matMulNBitsGemvMinBlobBytes();

        const float* a_even = nullptr;
        const float* a_odd = nullptr;
        if (fused) {
            float* buf = split.ptr<float>();
            matMulNBitsSplitA(a, buf, buf + K / 2, K / 2);
            a_even = buf;
            a_odd = buf + K / 2;
        }

        // Each range owns a disjoint set of output columns n, so the column writes are race-free.
        // A stripe spans >= 16 columns: that is one 64-byte output line, and the per-stripe
        // dequant buffer makes finer splitting cost more than it saves.
        parallel_for_(Range(0, N), [&](const Range& r) {
            if (fused) {
                matMulNBitsGemv(a_even, a_odd, bq, sc, zp, y, K, block_size, r.start, r.end);
                return;
            }
            // whole blocks (>= K): the last block is dequantized in full
            AutoBuffer<float, 1024> wcol(n_blk * block_size);
            matMulNBitsStripe(a, bq, sc, zp, y, M, N, K, block_size, bits,
                              r.start, r.end, wcol.data());
        }, std::min(N / 16.0, 8.0 * std::max(1, getNumThreads())));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        const Mat& A = inputs[0];
        const Mat& B = blobs[0];        // packed weights, uint8 [N, n_blk, block_size*bits/8]
        const Mat& scales = blobs[1];   // fp32 [N, n_blk]
        const Mat zeroPoints = blobs.size() == 3 ? blobs[2] : Mat();   // uint8, packed per column
        Mat& Y = outputs[0];

        CV_Assert(A.isContinuous() && Y.isContinuous());

        Mat split = internals.empty() ? Mat() : internals[0];
        runImpl(A, B, scales, zeroPoints, Y, split);
    }
};

Ptr<MatMulNBitsLayer> MatMulNBitsLayer::create(const LayerParams& params)
{
    return makePtr<MatMulNBitsLayerImpl>(params);
}

}} // cv::dnn
