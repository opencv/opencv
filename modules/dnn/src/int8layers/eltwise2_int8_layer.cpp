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

        std::vector<const int8_t*> inptrs(ninputs);
        for (int k = 0; k < ninputs; k++)
            inptrs[k] = input_arrs.getMat(k).ptr<int8_t>();

        int8_t* outptr = out.ptr<int8_t>();

        const float c0 = cptr[0];
        const float c1 = cptr[1];
        const int8_t* p0 = inptrs[0];
        const int8_t* p1 = inptrs[1];

        const int relu_min = withRelu ? output_zp : -128;

        const int grain = std::max(1, (int)(total_elems / (getNumThreads() * 4)));

        parallel_for_(Range(0, (int)total_elems), [&](const Range& r) {
            int i = r.start;
        #if CV_AVX2
            if (!lutptr && ninputs == 2) {
                __m256 vc0 = _mm256_set1_ps(c0);
                __m256 vc1 = _mm256_set1_ps(c1);
                __m256 voff = _mm256_set1_ps(off);
                __m256i vmin = _mm256_set1_epi32(relu_min);
                __m256i vmax = _mm256_set1_epi32(127);

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
                    __m128i packed16 = _mm_packs_epi32(lo, hi); // 8x int16
                    __m128i packed8 = _mm_packs_epi16(packed16, packed16); // 8x int8 in low 64 bits

                    _mm_storel_epi64((__m128i*)(outptr + i), packed8);
                }
            }
        #endif
            for (; i < r.end; i++)
            {
                float val = c0 * (float)p0[i] + c1 * (float)p1[i] + off;

                for (int k = 2; k < ninputs; k++)
                    val += cptr[k] * (float)inptrs[k][i];

                int ival = cvRound(val);
                ival = std::max(relu_min, std::min(127, ival));

                if (lutptr)
                    ival = (int)lutptr[ival + 128];

                outptr[i] = (int8_t)ival;
            }
        }, grain);

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
