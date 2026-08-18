// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/hal/intrin.hpp>

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
        CV_CheckEQ(blobs.size(), static_cast<size_t>(2),
                   "DNN/MatMulNBits: expected packed weights and scales as constant blobs");

        const int n_blk = (K + block_size - 1) / block_size;   // ceil: K need not divide block_size; last block may be partial
        CV_CheckEQ(static_cast<int>(blobs[0].total()), N * n_blk * (block_size * bits / 8),
                   "DNN/MatMulNBits: packed weight size mismatch");
        CV_CheckEQ(static_cast<int>(blobs[1].total()), N * n_blk,
                   "DNN/MatMulNBits: scales size mismatch");
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
        internals.clear();
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
        internals.assign(requiredInternals, MatType(-1));
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &) const CV_OVERRIDE {
        CV_Assert(!inputs.empty());
        const int64 M = total(inputs[0]) / K;
        return CV_BIG_INT(2) * M * N * K;
    }

    // Dequantize one block into w[0..block_size-1]. BITS is compile-time so the per-byte
    // unpack unrolls and the shift folds to a constant (the hot half of the kernel).
    template<int BITS>
    static inline void dequantBlock(const uchar* packed, int blob_bytes, float scale, float* w) {
        const int EPB = 8 / BITS;            // elements per byte: 4-bit:2  8-bit:1
        const int QMASK = (1 << BITS) - 1;
        const int ZP = 1 << (BITS - 1);      // symmetric default zero-point
        for (int j = 0; j < blob_bytes; j++) {
            const uchar byte = packed[j];
            for (int e = 0; e < EPB; e++)     // LSB-first within the byte
                w[j * EPB + e] = (((byte >> (e * BITS)) & QMASK) - ZP) * scale;
        }
    }

    template<int BITS>
    void runImpl(const Mat& A, const Mat& B, const Mat& scales, Mat& Y) const {
        const int M = static_cast<int>(A.total() / K);
        const int n_blk = (K + block_size - 1) / block_size;
        const int blob_bytes = block_size * BITS / 8;   // packed weights per block

        const float* a = A.ptr<const float>();
        const uchar* bq = B.ptr<const uchar>();
        const float* sc = scales.ptr<const float>();
        float* y = Y.ptr<float>();

        // Each range owns a disjoint set of output columns n, so the column writes are race-free.
        parallel_for_(Range(0, N), [&](const Range& r) {
            // Dequant the whole column once so it amortizes across the M rows.
            AutoBuffer<float, 1024> wcol(n_blk * block_size);   // whole blocks (>= K): the last block is dequantized in full
            for (int n = r.start; n < r.end; n++) {
                const uchar* bq_n = bq + static_cast<size_t>(n) * n_blk * blob_bytes;
                const float* sc_n = sc + static_cast<size_t>(n) * n_blk;
                for (int blk = 0; blk < n_blk; blk++)
                    dequantBlock<BITS>(bq_n + static_cast<size_t>(blk) * blob_bytes, blob_bytes,
                                       sc_n[blk], wcol.data() + blk * block_size);

                const float* w = wcol.data();
                for (int m = 0; m < M; m++) {
                    const float* arow = a + static_cast<size_t>(m) * K;
                    float acc = 0.f;
                    int t = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    const int vlanes = VTraits<v_float32>::vlanes();
                    v_float32 vacc0 = vx_setzero_f32(), vacc1 = vx_setzero_f32();
                    v_float32 vacc2 = vx_setzero_f32(), vacc3 = vx_setzero_f32();
                    for (; t <= K - 4 * vlanes; t += 4 * vlanes) {
                        vacc0 = v_fma(vx_load(arow + t),              vx_load(w + t),              vacc0);
                        vacc1 = v_fma(vx_load(arow + t + vlanes),     vx_load(w + t + vlanes),     vacc1);
                        vacc2 = v_fma(vx_load(arow + t + 2 * vlanes), vx_load(w + t + 2 * vlanes), vacc2);
                        vacc3 = v_fma(vx_load(arow + t + 3 * vlanes), vx_load(w + t + 3 * vlanes), vacc3);
                    }
                    v_float32 vacc = v_add(v_add(vacc0, vacc1), v_add(vacc2, vacc3));
                    for (; t <= K - vlanes; t += vlanes)
                        vacc = v_fma(vx_load(arow + t), vx_load(w + t), vacc);
                    acc = v_reduce_sum(vacc);
#endif
                    for (; t < K; t++)
                        acc += arow[t] * w[t];
                    y[static_cast<size_t>(m) * N + n] = acc;
                }
            }
        }, static_cast<double>(N) * M * K * (1 / 1024.0));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& A = inputs[0];
        const Mat& B = blobs[0];        // packed weights, uint8 [N, n_blk, block_size*bits/8]
        const Mat& scales = blobs[1];   // fp32 [N, n_blk]
        Mat& Y = outputs[0];

        CV_Assert(A.isContinuous() && Y.isContinuous());

        switch (bits) {                 // bits ∈ {4,8} enforced in the constructor
            case 4:  runImpl<4>(A, B, scales, Y); break;
            default: runImpl<8>(A, B, scales, Y); break;
        }
    }
};

Ptr<MatMulNBitsLayer> MatMulNBitsLayer::create(const LayerParams& params)
{
    return makePtr<MatMulNBitsLayerImpl>(params);
}

}} // cv::dnn
