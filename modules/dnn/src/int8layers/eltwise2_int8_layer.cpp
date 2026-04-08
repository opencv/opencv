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

class Eltwise2Int8LayerImpl CV_FINAL : public Eltwise2Int8Layer
{
public:
    std::vector<float> coeffs;
    float offset;
    bool withRelu;

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

    Eltwise2Int8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        output_zp = params.get<int>("zeropoints", 0);
        output_sc = params.get<float>("scales", 1.0f);
        withRelu = params.get<bool>("with_relu", false);

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
    }

    void ensureCoeffs()
    {
        if (!coeffs.empty() || scales.empty())
            return;

        CV_CheckEQ(scales.size(), zeropoints.size(),
                   "Eltwise2Int8: scales and zeropoints sizes must match");
        CV_Assert(output_sc > 0.f);

        coeffs.resize(scales.size());
        offset = (float)output_zp;
        for (size_t i = 0; i < scales.size(); i++)
        {
            coeffs[i] = scales[i] / output_sc;
            offset -= coeffs[i] * zeropoints[i];
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        outputs.assign(1, inputs[0]);
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

        const Mat& inp0 = input_arrs.getMat(0);
        MatShape outshape = inp0.shape();
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

        const int out_min = isU8 ? (withRelu ? output_zp : 0) : (withRelu ? output_zp : -128);
        const int out_max = isU8 ? 255 : 127;

        const double nstripes = (double)std::max(1, getNumThreads() * 4);

        if (isU8) {
            std::vector<const uint8_t*> inptrs(ninputs);
            for (int k = 0; k < ninputs; k++)
                inptrs[k] = input_arrs.getMat(k).ptr<uint8_t>();
            const uint8_t* p0 = inptrs[0];
            const uint8_t* p1 = inptrs[1];
            uint8_t* outptr = out.ptr<uint8_t>();

            parallel_for_(Range(0, (int)total_elems), [&](const Range& r) {
                int i = r.start;
            #if CV_AVX2
                if (!lutptr && ninputs == 2) {
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
                    int32x4_t vmin = vdupq_n_s32(out_min);
                    int32x4_t vmax = vdupq_n_s32(out_max);

                    for (; i <= r.end - 8; i += 8) {
                        uint8x8_t a8 = vld1_u8(p0 + i);
                        uint8x8_t b8 = vld1_u8(p1 + i);
                        uint16x8_t a16 = vmovl_u8(a8);
                        uint16x8_t b16 = vmovl_u8(b8);

                        int32x4_t a32_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(a16)));
                        int32x4_t b32_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(b16)));
                        int32x4_t a32_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(a16)));
                        int32x4_t b32_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(b16)));

                        float32x4_t val_lo = vfmaq_f32(vfmaq_f32(voff, vc0, vcvtq_f32_s32(a32_lo)),
                                                        vc1, vcvtq_f32_s32(b32_lo));
                        float32x4_t val_hi = vfmaq_f32(vfmaq_f32(voff, vc0, vcvtq_f32_s32(a32_hi)),
                                                        vc1, vcvtq_f32_s32(b32_hi));

                        int32x4_t ival_lo = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_lo), vmin), vmax);
                        int32x4_t ival_hi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_hi), vmin), vmax);

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
                    float val = c0 * (float)p0[i] + c1 * (float)p1[i] + off;

                    for (int k = 2; k < ninputs; k++)
                        val += cptr[k] * (float)inptrs[k][i];

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
                inptrs[k] = input_arrs.getMat(k).ptr<int8_t>();
            const int8_t* p0 = inptrs[0];
            const int8_t* p1 = inptrs[1];
            int8_t* outptr = out.ptr<int8_t>();

            parallel_for_(Range(0, (int)total_elems), [&](const Range& r) {
                int i = r.start;
            #if CV_AVX2
                if (!lutptr && ninputs == 2) {
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
                    int32x4_t vmin = vdupq_n_s32(out_min);
                    int32x4_t vmax = vdupq_n_s32(out_max);

                    for (; i <= r.end - 8; i += 8) {
                        int8x8_t a8 = vld1_s8(p0 + i);
                        int8x8_t b8 = vld1_s8(p1 + i);
                        int16x8_t a16 = vmovl_s8(a8);
                        int16x8_t b16 = vmovl_s8(b8);

                        int32x4_t a32_lo = vmovl_s16(vget_low_s16(a16));
                        int32x4_t b32_lo = vmovl_s16(vget_low_s16(b16));
                        int32x4_t a32_hi = vmovl_s16(vget_high_s16(a16));
                        int32x4_t b32_hi = vmovl_s16(vget_high_s16(b16));

                        float32x4_t val_lo = vfmaq_f32(vfmaq_f32(voff, vc0, vcvtq_f32_s32(a32_lo)),
                                                        vc1, vcvtq_f32_s32(b32_lo));
                        float32x4_t val_hi = vfmaq_f32(vfmaq_f32(voff, vc0, vcvtq_f32_s32(a32_hi)),
                                                        vc1, vcvtq_f32_s32(b32_hi));

                        int32x4_t ival_lo = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_lo), vmin), vmax);
                        int32x4_t ival_hi = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(val_hi), vmin), vmax);

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
                    float val = c0 * (float)p0[i] + c1 * (float)p1[i] + off;

                    for (int k = 2; k < ninputs; k++)
                        val += cptr[k] * (float)inptrs[k][i];

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

} // namespace dnn
} // namespace cv
