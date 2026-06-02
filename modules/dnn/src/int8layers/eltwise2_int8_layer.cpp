// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "opencv2/core/hal/intrin.hpp"

#if CV_AVX2
#include <immintrin.h>
#endif
#if CV_NEON
#include <arm_neon.h>
#endif

namespace cv {
namespace dnn {

static Mat broadcastExpand(const Mat& src, const MatShape& dstShape)
{
    MatShape srcShape = src.shape();
    const int ndst = (int)dstShape.size();
    while ((int)srcShape.size() < ndst)
        srcShape.insert(srcShape.begin(), 1);
    if (srcShape == dstShape)
        return src;

    MatShape outShape = dstShape;
    Mat out(outShape, src.type());
    const int total = (int)out.total();

    std::vector<int> dstStrides(ndst);
    { int s = 1; for (int d = ndst-1; d >= 0; d--) { dstStrides[d] = s; s *= dstShape[d]; } }

    std::vector<int> srcStrides(ndst, 0);
    { int s = 1; for (int d = ndst-1; d >= 0; d--) {
        srcStrides[d] = (srcShape[d] > 1) ? s : 0; s *= srcShape[d]; } }

    const uint8_t* sp = src.ptr<uint8_t>();
    uint8_t* dp = out.ptr<uint8_t>();
    for (int i = 0; i < total; i++) {
        int rem = i, srcIdx = 0;
        for (int d = 0; d < ndst; d++) {
            int coord = rem / dstStrides[d]; rem %= dstStrides[d];
            srcIdx += coord * srcStrides[d];
        }
        dp[i] = sp[srcIdx];
    }
    return out;
}

class Eltwise2Int8LayerImpl CV_FINAL : public Eltwise2Int8Layer
{
public:
    enum class Op { ADD, MUL };
    Op op;

    std::vector<float> coeffs;
    float offset;
    float mulCoeff;
    bool withRelu;

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

    Eltwise2Int8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        output_zp = params.get<int>("zeropoints", 0);
        output_sc = params.get<float>("scales", 1.0f);
        withRelu = params.get<bool>("with_relu", false);

        std::string opStr = params.get<String>("operation", "add");
        op = (opStr == "mul" || opStr == "prod") ? Op::MUL : Op::ADD;

        if (params.has("input_scales"))
        {
            DictValue sc = params.get("input_scales");
            int n = sc.size();
            scales.resize(n);
            for (int i = 0; i < n; i++)
                scales[i] = sc.get<float>(i);
        }

        if (params.has("input_zeropoints"))
        {
            DictValue zp = params.get("input_zeropoints");
            int n = zp.size();
            zeropoints.resize(n);
            for (int i = 0; i < n; i++)
                zeropoints[i] = zp.get<int>(i);
        }

        offset = 0.f;
        mulCoeff = 0.f;
    }

    Eltwise2Int8LayerImpl(const Eltwise2Int8Params& p)
    {
        name = p.name;
        type = "Eltwise2Int8";
        scales = p.input_scales;
        zeropoints = p.input_zeropoints;
        output_sc = p.output_sc;
        output_zp = p.output_zp;
        withRelu = p.with_relu;
        op = (p.operation == "mul" || p.operation == "prod") ? Op::MUL : Op::ADD;
        offset = 0.f;
        mulCoeff = 0.f;
    }

    void ensureCoeffs()
    {
        if (!coeffs.empty() || scales.empty())
            return;

        CV_CheckEQ(scales.size(), zeropoints.size(),
                   "Eltwise2Int8: scales and zeropoints sizes must match");
        CV_Assert(output_sc > 0.f);

        coeffs.resize(scales.size());
        if (op == Op::MUL) {
            CV_Assert(scales.size() == 2);
            mulCoeff = (scales[0] * scales[1]) / output_sc;
            coeffs[0] = (float)zeropoints[0];
            coeffs[1] = (float)zeropoints[1];
            offset = (float)output_zp;
        } else {
            offset = (float)output_zp;
            for (size_t i = 0; i < scales.size(); i++)
            {
                coeffs[i] = scales[i] / output_sc;
                offset -= coeffs[i] * zeropoints[i];
            }
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        int ndims = 0;
        DataLayout outLayout = DATA_LAYOUT_UNKNOWN;
        int outC = 0;
        for (const auto& s : inputs) {
            ndims = std::max(ndims, (int)s.size());
            if (s.layout != DATA_LAYOUT_UNKNOWN && outLayout == DATA_LAYOUT_UNKNOWN)
                outLayout = s.layout;
            if (s.C > outC) outC = s.C;
        }
        MatShape outShape(ndims, 1);
        for (const auto& s : inputs) {
            int off = ndims - (int)s.size();
            for (int d = 0; d < (int)s.size(); d++)
                outShape[d + off] = std::max(outShape[d + off], s[d]);
        }
        outShape.layout = outLayout;
        outShape.C = outC;
        outputs.assign(1, outShape);
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        DataLayout defaultLayout = netimpl_->originalLayout;
        size_t ninputs = actualInputs.size(), nblockInputs = 0;
        CV_Assert(ninputs >= 1u);
        for (size_t i = 0; i < ninputs; i++)
            nblockInputs += actualInputs[i] == DATA_LAYOUT_BLOCK;

        desiredInputs = actualInputs;
        if (nblockInputs == ninputs) {
            outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        } else {
            if (nblockInputs < ninputs) {
                for (size_t i = 0; i < ninputs; i++) {
                    DataLayout layout = actualInputs[i];
                    desiredInputs[i] = layout == DATA_LAYOUT_BLOCK ? defaultLayout : layout;
                }
            }
            outputs.assign(requiredOutputs, DATA_LAYOUT_UNKNOWN);
        }
        return outputs[0] == DATA_LAYOUT_BLOCK ? netimpl_->defaultC0 : 0;
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        ensureCoeffs();

        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 2 && coeffs.size() == (size_t)ninputs);

        std::vector<Mat> inputs(ninputs);
        DataLayout outLayout = DATA_LAYOUT_UNKNOWN;
        int outC = 0;
        {
            int ndims = 0;
            for (int k = 0; k < ninputs; k++) {
                inputs[k] = input_arrs.getMat(k);
                const MatShape& sh = inputs[k].shape();
                ndims = std::max(ndims, (int)sh.size());
                if (sh.layout != DATA_LAYOUT_UNKNOWN && outLayout == DATA_LAYOUT_UNKNOWN)
                    outLayout = sh.layout;
                if (sh.C > outC) outC = sh.C;
            }
            MatShape broadShape(ndims, 1);
            for (int k = 0; k < ninputs; k++) {
                const MatShape& sh = inputs[k].shape();
                int off = ndims - (int)sh.size();
                for (int d = 0; d < (int)sh.size(); d++)
                    broadShape[d + off] = std::max(broadShape[d + off], sh[d]);
            }
            broadShape.layout = outLayout;
            broadShape.C = outC;
            for (int k = 0; k < ninputs; k++)
                if (inputs[k].shape() != broadShape)
                    inputs[k] = broadcastExpand(inputs[k], broadShape);
        }

        const Mat& inp0 = inputs[0];
        MatShape outshape = inp0.shape();
        if (outshape.layout == DATA_LAYOUT_UNKNOWN && outLayout != DATA_LAYOUT_UNKNOWN) {
            outshape.layout = outLayout;
            outshape.C = outC;
        }
        int outtype = inp0.type();
        CV_Assert(outtype == CV_8SC1 || outtype == CV_8UC1);
        const bool isU8 = (outtype == CV_8UC1);

        int outkind = output_arrs.kind();
        std::vector<Mat>* outs = nullptr;
        std::vector<UMat>* uouts = nullptr;
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            outs = &output_arrs.getMatVecRef();
            outs->resize(1);
            outs->at(0).fit(outshape, outtype);
            out = outs->at(0);
        } else {
            uouts = &output_arrs.getUMatVecRef();
            uouts->resize(1);
            uouts->at(0).fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        const size_t total_elems = inp0.total();
        const float* cptr = coeffs.data();
        const float off = offset;
        const int8_t* lutptr = !activationLUT.empty() ? activationLUT.ptr<int8_t>() : nullptr;

        const float c0 = cptr[0];
        const float c1 = cptr[1];
        const bool isMul = (op == Op::MUL);
        const float mc = mulCoeff;
        const float zp0f = isMul ? (float)zeropoints[0] : 0.f;
        const float zp1f = isMul ? (float)zeropoints[1] : 0.f;
        const float ozpf = (float)output_zp;

        const int out_min = isU8 ? (withRelu ? output_zp : 0) : (withRelu ? output_zp : -128);
        const int out_max = isU8 ? 255 : 127;

        const double nstripes = (double)std::max(1, getNumThreads() * 4);

        if (isU8) {
            std::vector<const uint8_t*> inptrs(ninputs);
            for (int k = 0; k < ninputs; k++)
                inptrs[k] = inputs[k].ptr<uint8_t>();
            const uint8_t* p0 = inptrs[0];
            const uint8_t* p1 = inptrs[1];
            uint8_t* outptr = out.ptr<uint8_t>();

            parallel_for_(Range(0, (int)total_elems), [&](const Range& r) {
                int i = r.start;
            #if CV_AVX2
                if (isMul && !lutptr && ninputs == 2) {
                    __m256 vmc = _mm256_set1_ps(mc);
                    __m256 vzp0 = _mm256_set1_ps(zp0f);
                    __m256 vzp1 = _mm256_set1_ps(zp1f);
                    __m256 vozp = _mm256_set1_ps(ozpf);
                    __m256i vmin = _mm256_set1_epi32(out_min);
                    __m256i vmax = _mm256_set1_epi32(out_max);
                    for (; i <= r.end - 8; i += 8) {
                        __m128i a8 = _mm_loadl_epi64((const __m128i*)(p0 + i));
                        __m128i b8 = _mm_loadl_epi64((const __m128i*)(p1 + i));
                        __m256i a32 = _mm256_cvtepu8_epi32(a8);
                        __m256i b32 = _mm256_cvtepu8_epi32(b8);
                        __m256 af = _mm256_sub_ps(_mm256_cvtepi32_ps(a32), vzp0);
                        __m256 bf = _mm256_sub_ps(_mm256_cvtepi32_ps(b32), vzp1);
                        __m256 val = _mm256_fmadd_ps(_mm256_mul_ps(af, bf), vmc, vozp);
                        __m256i ival = _mm256_cvtps_epi32(val);
                        ival = _mm256_max_epi32(ival, vmin);
                        ival = _mm256_min_epi32(ival, vmax);
                        __m128i lo = _mm256_castsi256_si128(ival);
                        __m128i hi = _mm256_extracti128_si256(ival, 1);
                        __m128i packed16 = _mm_packs_epi32(lo, hi);
                        __m128i packed8 = _mm_packus_epi16(packed16, packed16);
                        _mm_storel_epi64((__m128i*)(outptr + i), packed8);
                    }
                } else
                if (!isMul && !lutptr && ninputs == 2) {
                    __m256 vc0 = _mm256_set1_ps(c0);
                    __m256 vc1 = _mm256_set1_ps(c1);
                    __m256 voff = _mm256_set1_ps(off);
                    __m256i vmin = _mm256_set1_epi32(out_min);
                    __m256i vmax = _mm256_set1_epi32(out_max);

                    for (; i <= r.end - 8; i += 8) {
                        __m128i a8 = _mm_loadl_epi64((const __m128i*)(p0 + i));
                        __m128i b8 = _mm_loadl_epi64((const __m128i*)(p1 + i));
                        __m256i a32 = _mm256_cvtepu8_epi32(a8);
                        __m256i b32 = _mm256_cvtepu8_epi32(b8);

                        __m256 af = _mm256_cvtepi32_ps(a32);
                        __m256 bf = _mm256_cvtepi32_ps(b32);

                        __m256 val = _mm256_fmadd_ps(vc0, af, _mm256_fmadd_ps(vc1, bf, voff));

                        __m256i ival = _mm256_cvtps_epi32(val);
                        ival = _mm256_max_epi32(ival, vmin);
                        ival = _mm256_min_epi32(ival, vmax);

                        __m128i lo = _mm256_castsi256_si128(ival);
                        __m128i hi = _mm256_extracti128_si256(ival, 1);
                        __m128i packed16 = _mm_packs_epi32(lo, hi);
                        __m128i packed8 = _mm_packus_epi16(packed16, packed16);

                        _mm_storel_epi64((__m128i*)(outptr + i), packed8);
                    }
                }
            #elif CV_NEON
                if (ninputs == 2) {
                    float32x4_t vc0 = vdupq_n_f32(c0);
                    float32x4_t vc1 = vdupq_n_f32(c1);
                    float32x4_t voff = vdupq_n_f32(off);
                    float32x4_t vmc = vdupq_n_f32(mc);
                    float32x4_t vzp0 = vdupq_n_f32(zp0f);
                    float32x4_t vzp1 = vdupq_n_f32(zp1f);
                    float32x4_t vozp = vdupq_n_f32(ozpf);
                    int32x4_t vmin = vdupq_n_s32(out_min);
                    int32x4_t vmax = vdupq_n_s32(out_max);

                    for (; i <= r.end - 8; i += 8) {
                        uint8x8_t a8 = vld1_u8(p0 + i);
                        uint8x8_t b8 = vld1_u8(p1 + i);
                        uint16x8_t a16 = vmovl_u8(a8);
                        uint16x8_t b16 = vmovl_u8(b8);

                        float32x4_t af_lo = vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a16))));
                        float32x4_t bf_lo = vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b16))));
                        float32x4_t af_hi = vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(a16))));
                        float32x4_t bf_hi = vcvtq_f32_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(b16))));

                        float32x4_t val_lo, val_hi;
                        if (isMul) {
                            val_lo = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(af_lo, vzp0), vsubq_f32(bf_lo, vzp1)), vmc), vozp);
                            val_hi = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(af_hi, vzp0), vsubq_f32(bf_hi, vzp1)), vmc), vozp);
                        } else {
                            val_lo = vaddq_f32(vaddq_f32(voff, vmulq_f32(vc0, af_lo)), vmulq_f32(vc1, bf_lo));
                            val_hi = vaddq_f32(vaddq_f32(voff, vmulq_f32(vc0, af_hi)), vmulq_f32(vc1, bf_hi));
                        }

#if CV_NEON_AARCH64
                        int32x4_t ival_lo = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_lo), vmin), vmax);
                        int32x4_t ival_hi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_hi), vmin), vmax);
#else
                        float32x4_t half = vdupq_n_f32(0.5f);
                        int32x4_t ival_lo = vminq_s32(vmaxq_s32(vcvtq_s32_f32(vaddq_f32(val_lo,
                            vbslq_f32(vcgeq_f32(val_lo, vdupq_n_f32(0.f)), half, vnegq_f32(half)))), vmin), vmax);
                        int32x4_t ival_hi = vminq_s32(vmaxq_s32(vcvtq_s32_f32(vaddq_f32(val_hi,
                            vbslq_f32(vcgeq_f32(val_hi, vdupq_n_f32(0.f)), half, vnegq_f32(half)))), vmin), vmax);
#endif

                        int16x8_t p16 = vcombine_s16(vqmovn_s32(ival_lo), vqmovn_s32(ival_hi));
                        uint8x8_t p8u = vqmovun_s16(p16);
                        if (lutptr) {
                            uint8_t tmp[8];
                            vst1_u8(tmp, p8u);
                            for (int k = 0; k < 8; k++)
                                outptr[i + k] = (uint8_t)lutptr[tmp[k]];
                        } else {
                            vst1_u8(outptr + i, p8u);
                        }
                    }
                }
            #endif
                for (; i < r.end; i++)
                {
                    float val;
                    if (isMul) {
                        val = mc * ((float)p0[i] - zp0f) * ((float)p1[i] - zp1f) + ozpf;
                    } else {
                        val = c0 * (float)p0[i] + c1 * (float)p1[i] + off;
                        for (int k = 2; k < ninputs; k++)
                            val += cptr[k] * (float)inptrs[k][i];
                    }

                    int ival = cvRound(val);
                    ival = std::max(out_min, std::min(out_max, ival));

                    if (lutptr)
                        ival = (int)(uint8_t)lutptr[ival];

                    outptr[i] = (uint8_t)ival;
                }
            }, nstripes);
        } else {
            std::vector<const int8_t*> inptrs(ninputs);
            for (int k = 0; k < ninputs; k++)
                inptrs[k] = inputs[k].ptr<int8_t>();
            const int8_t* p0 = inptrs[0];
            const int8_t* p1 = inptrs[1];
            int8_t* outptr = out.ptr<int8_t>();

            parallel_for_(Range(0, (int)total_elems), [&](const Range& r) {
                int i = r.start;
            #if CV_AVX2
                if (isMul && !lutptr && ninputs == 2) {
                    __m256 vmc = _mm256_set1_ps(mc);
                    __m256 vzp0 = _mm256_set1_ps(zp0f);
                    __m256 vzp1 = _mm256_set1_ps(zp1f);
                    __m256 vozp = _mm256_set1_ps(ozpf);
                    __m256i vmin = _mm256_set1_epi32(out_min);
                    __m256i vmax = _mm256_set1_epi32(out_max);
                    for (; i <= r.end - 8; i += 8) {
                        __m128i a8 = _mm_loadl_epi64((const __m128i*)(p0 + i));
                        __m128i b8 = _mm_loadl_epi64((const __m128i*)(p1 + i));
                        __m256i a32 = _mm256_cvtepi8_epi32(a8);
                        __m256i b32 = _mm256_cvtepi8_epi32(b8);
                        __m256 af = _mm256_sub_ps(_mm256_cvtepi32_ps(a32), vzp0);
                        __m256 bf = _mm256_sub_ps(_mm256_cvtepi32_ps(b32), vzp1);
                        __m256 val = _mm256_fmadd_ps(_mm256_mul_ps(af, bf), vmc, vozp);
                        __m256i ival = _mm256_cvtps_epi32(val);
                        ival = _mm256_max_epi32(ival, vmin);
                        ival = _mm256_min_epi32(ival, vmax);
                        __m128i lo = _mm256_castsi256_si128(ival);
                        __m128i hi = _mm256_extracti128_si256(ival, 1);
                        __m128i packed16 = _mm_packs_epi32(lo, hi);
                        __m128i packed8 = _mm_packs_epi16(packed16, packed16);
                        _mm_storel_epi64((__m128i*)(outptr + i), packed8);
                    }
                } else
                if (!isMul && !lutptr && ninputs == 2) {
                    __m256 vc0 = _mm256_set1_ps(c0);
                    __m256 vc1 = _mm256_set1_ps(c1);
                    __m256 voff = _mm256_set1_ps(off);
                    __m256i vmin = _mm256_set1_epi32(out_min);
                    __m256i vmax = _mm256_set1_epi32(out_max);

                    for (; i <= r.end - 8; i += 8) {
                        __m128i a8 = _mm_loadl_epi64((const __m128i*)(p0 + i));
                        __m128i b8 = _mm_loadl_epi64((const __m128i*)(p1 + i));
                        __m256i a32 = _mm256_cvtepi8_epi32(a8);
                        __m256i b32 = _mm256_cvtepi8_epi32(b8);

                        __m256 af = _mm256_cvtepi32_ps(a32);
                        __m256 bf = _mm256_cvtepi32_ps(b32);

                        __m256 val = _mm256_fmadd_ps(vc0, af, _mm256_fmadd_ps(vc1, bf, voff));

                        __m256i ival = _mm256_cvtps_epi32(val);
                        ival = _mm256_max_epi32(ival, vmin);
                        ival = _mm256_min_epi32(ival, vmax);

                        __m128i lo = _mm256_castsi256_si128(ival);
                        __m128i hi = _mm256_extracti128_si256(ival, 1);
                        __m128i packed16 = _mm_packs_epi32(lo, hi);
                        __m128i packed8 = _mm_packs_epi16(packed16, packed16);

                        _mm_storel_epi64((__m128i*)(outptr + i), packed8);
                    }
                }
            #elif CV_NEON
                if (ninputs == 2) {
                    float32x4_t vc0 = vdupq_n_f32(c0);
                    float32x4_t vc1 = vdupq_n_f32(c1);
                    float32x4_t voff = vdupq_n_f32(off);
                    float32x4_t vmc = vdupq_n_f32(mc);
                    float32x4_t vzp0 = vdupq_n_f32(zp0f);
                    float32x4_t vzp1 = vdupq_n_f32(zp1f);
                    float32x4_t vozp = vdupq_n_f32(ozpf);
                    int32x4_t vmin = vdupq_n_s32(out_min);
                    int32x4_t vmax = vdupq_n_s32(out_max);

                    for (; i <= r.end - 8; i += 8) {
                        int8x8_t a8 = vld1_s8(p0 + i);
                        int8x8_t b8 = vld1_s8(p1 + i);
                        int16x8_t a16 = vmovl_s8(a8);
                        int16x8_t b16 = vmovl_s8(b8);

                        float32x4_t af_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a16)));
                        float32x4_t bf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                        float32x4_t af_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a16)));
                        float32x4_t bf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));

                        float32x4_t val_lo, val_hi;
                        if (isMul) {
                            val_lo = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(af_lo, vzp0), vsubq_f32(bf_lo, vzp1)), vmc), vozp);
                            val_hi = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(af_hi, vzp0), vsubq_f32(bf_hi, vzp1)), vmc), vozp);
                        } else {
                            val_lo = vaddq_f32(vaddq_f32(voff, vmulq_f32(vc0, af_lo)), vmulq_f32(vc1, bf_lo));
                            val_hi = vaddq_f32(vaddq_f32(voff, vmulq_f32(vc0, af_hi)), vmulq_f32(vc1, bf_hi));
                        }

#if CV_NEON_AARCH64
                        int32x4_t ival_lo = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_lo), vmin), vmax);
                        int32x4_t ival_hi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_hi), vmin), vmax);
#else
                        float32x4_t half = vdupq_n_f32(0.5f);
                        int32x4_t ival_lo = vminq_s32(vmaxq_s32(vcvtq_s32_f32(vaddq_f32(val_lo,
                            vbslq_f32(vcgeq_f32(val_lo, vdupq_n_f32(0.f)), half, vnegq_f32(half)))), vmin), vmax);
                        int32x4_t ival_hi = vminq_s32(vmaxq_s32(vcvtq_s32_f32(vaddq_f32(val_hi,
                            vbslq_f32(vcgeq_f32(val_hi, vdupq_n_f32(0.f)), half, vnegq_f32(half)))), vmin), vmax);
#endif

                        int16x8_t p16 = vcombine_s16(vqmovn_s32(ival_lo), vqmovn_s32(ival_hi));
                        int8x8_t p8 = vqmovn_s16(p16);
                        if (lutptr) {
                            int8_t tmp[8];
                            vst1_s8(tmp, p8);
                            for (int k = 0; k < 8; k++)
                                outptr[i + k] = (int8_t)lutptr[(int)tmp[k] + 128];
                        } else {
                            vst1_s8(outptr + i, p8);
                        }
                    }
                }
            #endif
                for (; i < r.end; i++)
                {
                    float val;
                    if (isMul) {
                        val = mc * ((float)p0[i] - zp0f) * ((float)p1[i] - zp1f) + ozpf;
                    } else {
                        val = c0 * (float)p0[i] + c1 * (float)p1[i] + off;
                        for (int k = 2; k < ninputs; k++)
                            val += cptr[k] * (float)inptrs[k][i];
                    }

                    int ival = cvRound(val);
                    ival = std::max(out_min, std::min(out_max, ival));

                    if (lutptr)
                        ival = (int)lutptr[ival + 128];

                    outptr[i] = (int8_t)ival;
                }
            }, nstripes);
        }

        if (uouts) {
            out.copyTo(uouts->at(0));
        }
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty())
        {
            activ = activ_int8;
            if (!activ_int8->blobs.empty())
                activationLUT = activ_int8->blobs[0];
            return true;
        }
        return false;
    }
};

Ptr<Eltwise2Int8Layer> Eltwise2Int8Layer::create(const LayerParams& params)
{
    return Ptr<Eltwise2Int8Layer>(new Eltwise2Int8LayerImpl(params));
}

Ptr<Eltwise2Int8Layer> Eltwise2Int8Layer::create(const Eltwise2Int8Params& params)
{
    return Ptr<Eltwise2Int8Layer>(new Eltwise2Int8LayerImpl(params));
}

} // namespace dnn
} // namespace cv
