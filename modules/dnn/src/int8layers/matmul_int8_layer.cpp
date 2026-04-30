// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class MatMulInt8LayerImpl CV_FINAL : public MatMulInt8Layer
{
public:
    enum { VEC_ALIGN = 32 };

    MatMulInt8LayerImpl(const MatMulInt8Params& p)
    {
        name = p.name;
        type = "MatMulInt8";
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
                         std::vector<MatShape> &) const CV_OVERRIDE
    {
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

        const int nstripes = outerSize <= 4 ? 1 : getNumThreads();
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
        CV_UNUSED(inputs);
        long flops = 0;
        int innerSize = blobs[0].size[1];
        for (size_t i = 0; i < outputs.size(); i++) {
            flops += total(outputs[i]) * innerSize;
        }
        return flops;
    }

private:
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

        parallel_for_(Range(0, nstripes), p, nstripes);
    }

    class Int8GemmBody : public ParallelLoopBody
    {
    public:
        Int8GemmBody() : srcMat(0), weights(0), biasMat(0), outputMultiplier(0),
                         dstMat(0), nstripes(0), outZp(0),
                         useAVX2(false), useAVX512(false), useLASX(false), useRVV(false) {}

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
    };

    int inp_dims;
    int num_output;
    Mat weightsMat, biasMat, outputMultiplier;
};

Ptr<MatMulInt8Layer> MatMulInt8Layer::create(const MatMulInt8Params& params)
{
    return Ptr<MatMulInt8Layer>(new MatMulInt8LayerImpl(params));
}

}
}
