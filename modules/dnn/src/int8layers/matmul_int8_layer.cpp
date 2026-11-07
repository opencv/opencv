// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "../layers/cpu_kernels/fast_gemm.hpp"
#include "matmul_int8_kernels.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class MatMulInt8LayerImpl CV_FINAL : public MatMulInt8Layer
{
public:
    enum { VEC_ALIGN = 32 };

    MatMulInt8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        int32_output = params.get<bool>("int32_output", false);
        CV_CheckTrue(int32_output,
                     "DNN/MatMulInt8: only the ONNX MatMulInteger form is created by type name");
        trans_a = trans_b = false;
        alpha = beta = 1.f;
        input_zp = output_zp = 0;
        input_sc = output_sc = 1.f;
        output_type = CV_32S;
        per_channel = false;
        inp_dims = 2;
        num_output = 0;
    }

    MatMulInt8LayerImpl(const MatMulInt8Params& p)
    {
        name = p.name;
        type = "MatMulInt8";
        trans_a = trans_b = false;
        alpha = beta = 1.f;
        input_sc = p.input_sc;
        input_zp = p.input_zp;
        output_sc = p.output_sc;
        output_zp = p.output_zp;
        output_type = p.output_type;
        per_channel = p.per_channel;
        inp_dims = p.inp_dims;
        num_output = p.num_output;

        if (!p.weights.empty()) {
            int numOutput = p.num_output;
            int innerSize = (int)p.weights.total() / numOutput;

            CV_Assert(p.weights.dims >= 2 && (size_t)(innerSize * numOutput) == p.weights.total());
            CV_Assert((size_t)numOutput == p.bias.total());

            blobs = { p.weights.clone(), p.bias.clone(), p.outputMultiplier.clone() };
            weightsMat = blobs[0] = blobs[0].reshape(1, numOutput);
            int vecsize = weightsMat.cols;

            if (vecsize % VEC_ALIGN != 0)
            {
                int vecsize_aligned = (int)alignSize(vecsize, VEC_ALIGN);
                Mat weightsBuf(weightsMat.rows, vecsize_aligned, weightsMat.type());
                Mat wpadding = weightsBuf.colRange(vecsize, vecsize_aligned);
                wpadding.setTo(Scalar::all(0));
                weightsMat = weightsBuf.colRange(0, vecsize);
                blobs[0].copyTo(weightsMat);
            }
            biasMat = blobs[1] = blobs[1].reshape(1, 1);
            outputMultiplier = blobs[2];
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if (int32_output) {
            CV_CheckGE(inputs.size(), (size_t)2, "DNN/MatMulInteger: A and B are required");
            MatShape ap, bp, full, out;
            matmulShapes(inputs[0], inputs[1], false, false, ap, bp, full, out);
            outputs.assign(1, out);
            int rows = 1;
            for (size_t i = 0; i + 1 < ap.size(); i++)
                rows *= ap[i];
            // per-row correction term, used when B is uint8 or has a nonzero zero point
            internals.assign(1, MatShape({std::max(rows, 1)}));
            return false;
        }
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(blobs[0].dims, 2, "");
        int numOut = blobs[0].size[0];
        CV_Assert((size_t)numOut == blobs[1].total());

        // Preserve all batch dimensions, replace last dim with num_output.
        // Input: [B1, B2, ..., M, K] → Output: [B1, B2, ..., M, num_output]
        int ndims = inputs[0].dims;
        CV_Assert(ndims >= 1);
        MatShape outShape = inputs[0];
        outShape[ndims - 1] = numOut;

        outputs.resize(1, outShape);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        if (int32_output) {
            CV_CheckType(inputs[0], inputs[0] == CV_8U || inputs[0] == CV_8S,
                         "DNN/MatMulInteger: A must be uint8 or int8");
            CV_CheckType(inputs[1], inputs[1] == CV_8U || inputs[1] == CV_8S,
                         "DNN/MatMulInteger: B must be uint8 or int8");
            outputs.assign(requiredOutputs, CV_32S);
            internals.assign(requiredInternals, CV_32S);
            return;
        }
        outputs.assign(requiredOutputs, output_type);
        internals.clear();
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (int32_output) {
            forwardInt32(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> input, output;
        inputs_arr.getMatVector(input);
        outputs_arr.getMatVector(output);

        // Flatten all dimensions except the last (K) into a single batch dim.
        // Input: [B1, B2, ..., M, K] → [B1*B2*...*M, K]
        int ndims = input[0].dims;
        int K = input[0].size[ndims - 1];
        int outerSize = (int)(input[0].total() / K);

        Mat srcMat0 = input[0].reshape(1, outerSize);
        Mat srcMat;
        if (srcMat0.type() == CV_8U) {
            srcMat0.convertTo(srcMat, CV_8S, 1, -128);
        } else {
            srcMat = srcMat0;
        }

        Mat dstMat = output[0].reshape(1, outerSize);
        Mat dstMatInt32 = Mat(shape(dstMat), CV_32S);

        const size_t nouts = (size_t)outerSize * weightsMat.rows;
        const int nstripes = (int)std::max<size_t>(1, std::min<size_t>(getNumThreads(), nouts / 16));
        runInt8Gemm(srcMat, weightsMat, biasMat, outputMultiplier, dstMatInt32, nstripes, output_zp);
        if (output_type == CV_8U) {
            dstMatInt32.convertTo(dstMat, output_type, 1, 128);
        } else {
            dstMatInt32.convertTo(dstMat, output_type);
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        if (int32_output)
            return (int64)total(outputs[0]) * inputs[0][inputs[0].dims - 1];
        long flops = 0;
        int innerSize = blobs[0].size[1];
        for (size_t i = 0; i < outputs.size(); i++) {
            flops += total(outputs[i]) * innerSize;
        }
        return flops;
    }

private:
    // B - 128 (uint8 B) or B (int8 B), each [K, N] matrix of B in the matMulInt8Int32 panel
    // layout one after another, with their column sums
    void packWeightsInteger(const Mat& B)
    {
        CV_CheckType(B.type(), B.type() == CV_8UC1 || B.type() == CV_8SC1,
                     "DNN/MatMulInteger: B must be uint8 or int8");
        CV_CheckGE(B.dims, 2, "DNN/MatMulInteger: B must have at least two dimensions");
        CV_Assert(B.isContinuous());
        constexpr int NR = MATMUL_INT8_NR, KR = MATMUL_INT8_KR;
        const int K = B.size[B.dims - 2], N = B.size[B.dims - 1];
        const int nb = (int)(B.total() / ((size_t)K * N));
        const int Npad = alignSize(N, NR);
        const size_t panelSize = (size_t)alignSize(K, KR) * NR, matSize = Npad / NR * panelSize;
        const int delta = B.type() == CV_8UC1 ? 128 : 0;
        packedB.create(1, (int)(nb * matSize), CV_8S);
        packedB.setTo(Scalar::all(0));
        colsum = Mat::zeros(1, nb * Npad, CV_32S);
        corr = Mat::zeros(1, nb * Npad, CV_32S);
        const uchar* b = B.ptr<uchar>();
        int8_t* dst = packedB.ptr<int8_t>();
        int* cs = colsum.ptr<int>();
        parallel_for_(Range(0, nb * N), [&](const Range& range) {
            for (int j = range.start; j < range.end; j++) {
                const int m = j / N, n = j % N;
                const uchar* bm = b + (size_t)m * K * N;
                int8_t* d = dst + m * matSize + n / NR * panelSize + n % NR * KR;
                int s = 0;
                for (int k = 0; k < K; k++) {
                    const uchar v = bm[(size_t)k * N + n];
                    int w = (delta ? (int)v : (int)(schar)v) - delta;
                    d[k / KR * NR * KR + k % KR] = (int8_t)w;
                    s += w;
                }
                cs[m * Npad + n] = s;
            }
        }, (double)nb * N * K / (1 << 16));
    }

    // ONNX MatMulInteger: https://onnx.ai/onnx/operators/onnx__MatMulInteger.html, opset 10 (its
    // only version); a_zero_point may be per row of A, b_zero_point per column of B
    void forwardInt32(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                      OutputArrayOfArrays internals_arr)
    {
        std::vector<Mat> input, output, internals;
        inputs_arr.getMatVector(input);
        outputs_arr.getMatVector(output);
        internals_arr.getMatVector(internals);
        const int ninputs = (int)input.size();
        const Mat& A = input[0];
        Mat& Y = output[0];
        CV_CheckType(A.type(), A.type() == CV_8UC1 || A.type() == CV_8SC1,
                     "DNN/MatMulInteger: A must be uint8 or int8");
        CV_Assert(A.isContinuous() && Y.isContinuous());
        MatShape ap, bp, full, out;
        matmulShapes(A.shape(), input[1].shape(), false, false, ap, bp, full, out);
        const Mat B = input[1].reshape(1, bp);  // a 1-D B is one column, [K, 1]
        const bool isU8 = A.type() == CV_8UC1;

        const Mat zbm = ninputs > 3 ? input[3] : Mat();
        const int K = B.size[B.dims - 2], N = B.size[B.dims - 1];
        const int R = (int)A.total(0, A.dims - 1);
        int za = 0;
        std::vector<int> zrow;  // per-row a_zero_point, empty when it is a scalar
        if (ninputs > 2 && !input[2].empty()) {
            const Mat& zam = input[2];
            CV_CheckTypeEQ(zam.type(), A.type(),
                           "DNN/MatMulInteger: a_zero_point must have the type of A");
            const uchar* zp = zam.ptr<uchar>();
            if (zam.total() == 1) {
                za = isU8 ? (int)zp[0] : (int)(schar)zp[0];
            } else {
                // [M] for a 2-D A, otherwise A's shape with K -> 1, where batch dims may be 1
                const MatShape as = A.shape();
                MatShape zs = zam.shape();
                if (zs.size() == 1)
                    zs.push_back(1);
                while (zs.size() < as.size())
                    zs.insert(zs.begin(), 1);
                bool ok = zs.size() == as.size() && zs.back() == 1;
                for (size_t d = 0; ok && d + 1 < as.size(); d++)
                    ok = zs[d] == as[d] || zs[d] == 1;
                CV_Check(zam.total(), ok,
                         "DNN/MatMulInteger: a_zero_point must be a scalar or one value per row");
                zrow.resize(R);
                for (int r = 0; r < R; r++) {
                    size_t zi = 0, zstep = 1;
                    for (int d = (int)as.size() - 2, rem = r; d >= 0; d--) {
                        if (zs[d] != 1)
                            zi += (size_t)(rem % as[d]) * zstep;
                        rem /= as[d];
                        zstep *= zs[d];
                    }
                    zrow[r] = isU8 ? (int)zp[zi] : (int)(schar)zp[zi];
                }
            }
        }
        const int* zr = zrow.empty() ? nullptr : zrow.data();
        if (Y.total() == 0)
            return;
        if (K == 0) {
            Y.setTo(Scalar::all(0));
            return;
        }
        const int nb = (int)(B.total() / ((size_t)K * N));
        if (!zbm.empty()) {
            CV_CheckTypeEQ(zbm.type(), B.type(),
                           "DNN/MatMulInteger: b_zero_point must have the type of B");
            CV_Check(zbm.total(), zbm.total() == 1 || zbm.total() == (size_t)N ||
                     zbm.total() == (size_t)nb * N,
                     "DNN/MatMulInteger: b_zero_point must be a scalar or one value per column");
        }

        const bool dynamicWeights = !getNetImpl(this)->isConstArg(inputs[1]);
        if (dynamicWeights || packedB.empty())
            packWeightsInteger(B);

        // (B - zb) = (B' - zb') with B' = B - delta, so zb' leaves -zb'[n] * sum_k (A - za)
        const int delta = B.type() == CV_8UC1 ? 128 : 0;
        const int Npad = alignSize(N, MATMUL_INT8_NR);
        int* cr = corr.ptr<int>();
        bool hasCorr = false;
        const size_t z = zbm.total();
        for (int m = 0; m < nb; m++) {
            for (int n = 0; n < N; n++) {
                const size_t i = z == 1 ? 0 : z == (size_t)N ? n : (size_t)m * N + n;
                int zb = z == 0 ? 0 : delta ? (int)zbm.ptr<uchar>()[i] : (int)zbm.ptr<schar>()[i];
                cr[m * Npad + n] = delta - zb;
                hasCorr = hasCorr || cr[m * Npad + n] != 0;
            }
        }

        const uchar* a = A.ptr<uchar>();
        int* rowterm = internals[0].ptr<int>();
        if (hasCorr) {
            parallel_for_(Range(0, R), [&](const Range& range) {
                for (int r = range.start; r < range.end; r++) {
                    const uchar* ar = a + (size_t)r * K;
                    int s = 0;
                    if (isU8)
                        for (int k = 0; k < K; k++)
                            s += ar[k];
                    else
                        for (int k = 0; k < K; k++)
                            s += (schar)ar[k];
                    rowterm[r] = s - K * (zr ? zr[r] : za);
                }
            }, (double)R * K / (1 << 16));
        }

        // a per-row zero point runs the kernel with a_zp = 0 and subtracts za[r] * colsum[n] here
        auto subtractRowZeroPoints = [&](const int* zrows, const int* csm, int* ym, int rows) {
            parallel_for_(Range(0, rows), [&](const Range& range) {
                for (int r = range.start; r < range.end; r++)
                    for (int n = 0; n < N; n++)
                        ym[(size_t)r * N + n] -= zrows[r] * csm[n];
            }, (double)rows * N / (1 << 16));
        };

        const int8_t* pb = packedB.ptr<int8_t>();
        const int* cs = colsum.ptr<int>();
        int* y = Y.ptr<int>();
        if (nb == 1) {
            matMulInt8Int32(a, K, pb, cs, hasCorr ? cr : nullptr, rowterm, y, N, R, N, K,
                            zr ? 0 : za, isU8);
            if (zr)
                subtractRowZeroPoints(zr, cs, y, R);
        } else {
            // one GEMM per output matrix, at the offsets MatMul uses to broadcast batch dims
            const size_t matSize = packedB.total() / nb;
            MatMulHelper helper;
            helper.compute(false, false, ap, bp, full);
            for (size_t i = 0; i < helper.batch; i++) {
                const size_t m = helper.B_rows[i];
                const size_t ofs = helper.A_offsets[i];
                matMulInt8Int32(a + ofs, K, pb + m * matSize, cs + m * Npad,
                                hasCorr ? cr + m * Npad : nullptr, rowterm + ofs / K,
                                y + helper.C_offsets[i], N, helper.M, N, K, zr ? 0 : za, isU8);
                if (zr)
                    subtractRowZeroPoints(zr + ofs / K, cs + m * Npad, y + helper.C_offsets[i],
                                          helper.M);
            }
        }

        if (dynamicWeights)
            packedB.release();
    }

    // Int8 GEMM: C_int32 = A_int8 @ W_int8^T + bias_int32
    // Then: output = clamp(round(C_int32 * multiplier) + output_zp)
    static void runInt8Gemm(const Mat& srcMat, const Mat& weights, const Mat& biasMat,
                            const Mat& outputMultiplier, Mat& dstMat, int nstripes, int outZp)
    {
        CV_Assert(srcMat.dims == 2 && srcMat.cols == weights.cols &&
                  dstMat.rows == srcMat.rows && dstMat.cols == weights.rows &&
                  srcMat.type() == weights.type() && srcMat.type() == CV_8S &&
                  dstMat.type() == CV_32S && biasMat.type() == CV_32S &&
                  biasMat.isContinuous() && (int)biasMat.total() == dstMat.cols);

        Int8GemmBody p;
        p.srcMat = &srcMat;
        p.weights = &weights;
        p.biasMat = &biasMat;
        p.outputMultiplier = &outputMultiplier;
        p.dstMat = &dstMat;
        p.nstripes = nstripes;
        p.outZp = outZp;
        p.useAVX2 = checkHardwareSupport(CPU_AVX2);
        p.useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX;
        p.useLASX = checkHardwareSupport(CPU_LASX);
        p.useRVV = checkHardwareSupport(CPU_RVV);
        p.useSVE = checkHardwareSupport(CPU_SVE);

        parallel_for_(Range(0, nstripes), p, nstripes);
    }

    class Int8GemmBody : public ParallelLoopBody
    {
    public:
        Int8GemmBody() : srcMat(0), weights(0), biasMat(0), outputMultiplier(0),
                         dstMat(0), nstripes(0), outZp(0),
                         useAVX2(false), useAVX512(false), useLASX(false), useRVV(false), useSVE(false) {}

        void operator()(const Range& r) const CV_OVERRIDE
        {
            int valign = MatMulInt8LayerImpl::VEC_ALIGN;
            int nsamples = srcMat->rows;
            int nw0 = weights->rows;
            int k, vecsize = srcMat->cols;
            int vecsize_aligned = (int)alignSize(vecsize, VEC_ALIGN);
            size_t total = (size_t)nsamples * nw0;
            size_t stripeSize = (total + nstripes - 1) / nstripes;
            size_t stripeStart = r.start * stripeSize;
            size_t stripeEnd = r.end == nstripes ? total : std::min(r.end * stripeSize, total);
            size_t wstep = weights->step1();
            AutoBuffer<int8_t> srcbuf(vecsize_aligned + valign);
            int8_t* sptr = alignPtr(srcbuf.data(), (int)(valign * sizeof(int8_t)));

            for (k = vecsize; k < vecsize_aligned; k++)
                sptr[k] = 0;

            for (size_t ofs = stripeStart; ofs < stripeEnd; )
            {
                int sampleIdx = (int)(ofs / nw0);
                int delta = (int)(ofs - (size_t)sampleIdx * nw0);
                const int8_t* sptr_ = srcMat->ptr<int8_t>(sampleIdx);
                const int8_t* wptr = weights->ptr<int8_t>(delta);
                int* dptr = dstMat->ptr<int>(sampleIdx) + delta;
                const int* biasptr = biasMat->ptr<int>() + delta;
                const float* multptr = outputMultiplier->ptr<float>() + delta;
                int nw = std::min(nw0 - delta, (int)(stripeEnd - ofs));

                memcpy(sptr, sptr_, vecsize * sizeof(sptr[0]));
            #if CV_TRY_AVX512_SKX
                if (useAVX512)
                    opt_AVX512_SKX::fastGEMM1T(sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp);
                else
            #endif
            #if CV_TRY_AVX2
                if (useAVX2)
                    opt_AVX2::fastGEMM1T(sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp);
                else
            #endif
            #if CV_TRY_LASX
                if (useLASX)
                    opt_LASX::fastGEMM1T(sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp);
                else
            #endif
            #if CV_TRY_RVV && CV_RVV
                if (useRVV)
                    opt_RVV::fastGEMM1T(sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp);
                else
            #endif
            #if CV_RVP052
                if (1)
                    opt_RVP052::fastGEMM1T(sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp);
                else
            #endif
            #if CV_TRY_SVE
                if(useSVE){
                    opt_SVE::fastGEMM1T( sptr, wptr, wstep, biasptr, multptr, dptr, nw, vecsize, outZp );
                }
                else
            #endif
                {
                    int i = 0;
            #if CV_SIMD128
                    for (; i <= nw - 4; i += 4, wptr += 4 * wstep)
                    {
                        v_int32x4 vs0 = v_setzero_s32(), vs1 = v_setzero_s32(),
                                  vs2 = v_setzero_s32(), vs3 = v_setzero_s32();
                        v_int32x4 outzp = v_setall_s32(outZp), outmin = v_setall_s32(-128), outmax = v_setall_s32(127);
                        v_int32x4 s = v_load(biasptr + i);
                        v_float32x4 mult = v_load(multptr + i);

                        for (k = 0; k < vecsize; k += 16)
                        {
                            v_int8x16 v = v_load_aligned(sptr + k);
                            vs0 = v_dotprod_expand_fast(v, v_load_aligned(wptr + k), vs0);
                            vs1 = v_dotprod_expand_fast(v, v_load_aligned(wptr + wstep + k), vs1);
                            vs2 = v_dotprod_expand_fast(v, v_load_aligned(wptr + wstep * 2 + k), vs2);
                            vs3 = v_dotprod_expand_fast(v, v_load_aligned(wptr + wstep * 3 + k), vs3);
                        }

                        s = v_add(s, v_int32x4(v_reduce_sum(vs0), v_reduce_sum(vs1), v_reduce_sum(vs2), v_reduce_sum(vs3)));
                        v_int32x4 out = v_add(outzp, v_round(v_mul(v_cvt_f32(s), mult)));
                        v_store(dptr + i, v_min(v_max(out, outmin), outmax));
                    }
            #endif

                    for (; i < nw; i++, wptr += wstep)
                    {
                        int s0 = biasptr[i];
                        float mult0 = multptr[i];

                        for (k = 0; k < vecsize; k++)
                        {
                            int8_t v = sptr[k];
                            s0 += (int)v * wptr[k];
                        }
                        int out0 = outZp + (int)std::round(s0 * mult0);
                        dptr[i] = std::min(std::max(out0, -128), 127);
                    }
                }

                ofs += nw;
            }
        }

        const Mat *srcMat, *weights, *biasMat, *outputMultiplier;
        Mat* dstMat;
        int nstripes, outZp;
        bool useAVX2;
        bool useAVX512;
        bool useLASX;
        bool useRVV;
        bool useSVE;
    };

    int inp_dims;
    int num_output;
    Mat weightsMat, biasMat, outputMultiplier;
    Mat packedB, colsum, corr; // int32 mode
};

Ptr<MatMulInt8Layer> MatMulInt8Layer::create(const LayerParams& params)
{
    return Ptr<MatMulInt8Layer>(new MatMulInt8LayerImpl(params));
}

Ptr<MatMulInt8Layer> MatMulInt8Layer::create(const MatMulInt8Params& params)
{
    return Ptr<MatMulInt8Layer>(new MatMulInt8LayerImpl(params));
}

}
}
