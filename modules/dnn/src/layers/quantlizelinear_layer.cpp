
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cv
{
namespace dnn
{

// Fast path for per-tensor quantization: float → uint8 with AVX2
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void quantizeLinearChunk_f32_u8_avx2(const float* src, uint8_t* dst,
                                             float inv_scale, float zp_f,
                                             int64_t len)
{
    __m256 vscale = _mm256_set1_ps(inv_scale);
    __m256 vzp = _mm256_set1_ps(zp_f);
    __m256 vmin = _mm256_setzero_ps();
    __m256 vmax = _mm256_set1_ps(255.f);

    int64_t j = 0;
    for (; j <= len - 8; j += 8) {
        __m256 v = _mm256_loadu_ps(src + j);
        v = _mm256_add_ps(_mm256_mul_ps(v, vscale), vzp);
        v = _mm256_min_ps(_mm256_max_ps(v, vmin), vmax);
        __m256i vi = _mm256_cvtps_epi32(v);
        __m128i lo = _mm256_castsi256_si128(vi);
        __m128i hi = _mm256_extracti128_si256(vi, 1);
        __m128i packed16 = _mm_packs_epi32(lo, hi);
        __m128i packed8 = _mm_packus_epi16(packed16, packed16);
        _mm_storel_epi64((__m128i*)(dst + j), packed8);
    }
    for (; j < len; j++)
        dst[j] = saturate_cast<uint8_t>(src[j] * inv_scale + zp_f);
}

static void quantizeLinearFast_f32_u8_avx2(const float* inp, uint8_t* out,
                                            float inv_scale, float zp_f,
                                            int64_t total)
{
    const int64_t block = 1024;
    int64_t nblocks = (total + block - 1) / block;

    parallel_for_(Range(0, (int)nblocks), [&](const Range& r) {
        for (int i = r.start; i < r.end; i++) {
            int64_t ofs = i * block;
            int64_t len = std::min(block, total - ofs);
            quantizeLinearChunk_f32_u8_avx2(inp + ofs, out + ofs, inv_scale, zp_f, len);
        }
    });
}
#endif


/*
    QuantizeLinear layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html

    Opset's 10 to 23 are covered.
*/

template <typename _InpTp, typename _ScaleTp, typename _OutTp>
static void quantizeLinear(const _InpTp* inp_, const _ScaleTp* scale_,
                           const _OutTp* zp_, _OutTp* out_,
                           int64_t nslices, int sz_a_,
                           int64_t slice_size_, int block_size_)
{
    int bsz_ = std::max(block_size_, 1);
    int nblocks_per_axis = (sz_a_ + bsz_ - 1) / bsz_;
    int64_t nmacro_blocks = nslices * nblocks_per_axis;
    CV_Assert(nmacro_blocks <= (int64_t)INT_MAX);

    parallel_for_(Range(0, (int)nmacro_blocks), [&](const Range& r) {
        int sz_a = sz_a_;
        int64_t slice_size = slice_size_;
        int block_size = block_size_;
        int delta = 0;
        int64_t scale_step = block_size > 0 ? slice_size : 1;
        int64_t zp_step = zp_ ? scale_step : 0;

        for (int i = r.start; i < r.end; i += delta) {
            int slice_idx = i / nblocks_per_axis;
            int block_idx = i - slice_idx * nblocks_per_axis;
            int64_t block_ofs, scale_ofs;
            if (block_size > 0) {
                delta = std::min(nblocks_per_axis - block_idx, r.end - i);
                block_ofs = (slice_idx*sz_a + block_idx*block_size)*slice_size;
                scale_ofs = (slice_idx*nblocks_per_axis + block_idx)*slice_size;
            } else {
                delta = std::min(sz_a - block_idx, r.end - i);
                block_ofs = (slice_idx*sz_a + block_idx)*slice_size;
                scale_ofs = block_idx;
            }
            const _InpTp* inp = inp_ + block_ofs;
            const _OutTp* zp = zp_ ? zp_ + scale_ofs : nullptr;
            const _ScaleTp* sc = scale_ + scale_ofs;
            _OutTp* out = out_ + block_ofs;

            // [TODO] vectorize using intrinsics
            if (slice_size > 1) {
                for (int k = 0; k < delta; k++, inp += slice_size, out += slice_size,
                                                sc += scale_step, zp += zp_step) {
                    float scval = 1.f/(float)(*sc);
                    _OutTp zpval = zp ? *zp : (_InpTp)0;

                    for (int64_t j = 0; j < slice_size; j++)
                        out[j] = saturate_cast<_OutTp>(inp[j]*scval + zpval);
                }
            } else if (block_size > 0 ) {
                int bsz = block_size;
                for (int k = 0; k < delta; k++, inp += bsz, out += bsz) {
                    bsz = std::min(bsz, sz_a - (block_idx + k)*block_size);
                    float scval = 1.f/(float)sc[k];
                    _OutTp zpval = zp ? zp[k] : (_InpTp)0;

                    for (int j = 0; j < bsz; j++)
                        out[j] = saturate_cast<_OutTp>(inp[j]*scval + zpval);
                }
                sc += delta;
                zp += zp ? delta : 0;
            } else {
                // here we assume that scale's have been inversed in advance in the parent function
                if (zp) {
                    for (int j = 0; j < delta; j++) {
                        float scval = (float)sc[j];
                        _OutTp zpval = zp[j];
                        out[j] = saturate_cast<_OutTp>(inp[j]*scval + zpval);
                    }
                } else {
                    for (int j = 0; j < delta; j++) {
                        float scval = (float)sc[j];
                        out[j] = saturate_cast<_OutTp>(inp[j]*scval);
                    }
                }
                inp += delta;
                out += delta;
            }
        }
    });
}

// Dequantize INT8/UINT8 to FP32/FP16; out must be preallocated
static void quantizeLinear(const Mat& inp, const Mat& scale_, const Mat& zp,
                           int axis, int block_size, Mat& out)
{
    Mat scale = scale_;
    CV_Assert(inp.isContinuous());
    CV_Assert(scale.isContinuous());
    CV_Assert(out.isContinuous());

    int inptype = inp.type();
    int outtype = out.type();
    int sctype = scale.type();
    int zptype = zp.type();
    MatShape inpshape = inp.shape();
    MatShape scshape = scale.shape();
    MatShape zpshape = zp.shape();
    int i, ndims = inpshape.dims;
    int64_t nslices = 1, slice_size = 1;

    CV_Assert(inptype == CV_32F || inptype == CV_16F);
    CV_Assert(sctype == CV_32F || sctype == CV_16F);
    CV_Assert(outtype == CV_8U || outtype == CV_8S);

    if (!zp.empty()) {
        CV_Assert(zp.isContinuous());
        CV_Assert(zptype == outtype);
        CV_Assert(zpshape == scshape);
    }

    axis = normalize_axis(axis, ndims);
    for (i = 0; i < axis; i++)
        nslices *= inpshape[i];
    for (i = axis+1; i < ndims; i++)
        slice_size *= inpshape[i];
    int sz_a = inpshape[axis];

    if (block_size == 0) {
        size_t sc_total = scshape.total();
        CV_Assert(scale.dims <= 1);
        CV_Assert(sc_total == 1 || sc_total == (size_t)sz_a);

        // unroll the innermost loop if the scale's/zp's are the same
        if (sc_total == 1) {
            slice_size *= sz_a;
            sz_a = 1;
        }

        // avoid repeated inversion and FP16 => FP32 conversion inside the innermost loop
        if (slice_size == 1) {
            Mat temp(scale.size(), CV_32F);
            const float* scdata_32f = reinterpret_cast<const float*>(scale.data);
            const hfloat* scdata_16f = reinterpret_cast<const hfloat*>(scale.data);
            float* tempdata = temp.ptr<float>();

            for (size_t i = 0; i < sc_total; i++)
                tempdata[i] = 1.f/(sctype == CV_32F ? scdata_32f[i] : (float)scdata_16f[i]);
            scale = temp;
            sctype = CV_32F;
        }
    } else {
        CV_Assert(block_size > 0);
        CV_Assert(scale.dims == ndims);
        for (int i = 0; i < ndims; i++) {
            int inp_i = inpshape[i];
            int sc_i = scshape[i];
            if (i == axis) {
                CV_Assert((inp_i + block_size - 1)/block_size == sc_i);
            } else {
                CV_Assert(sc_i == inp_i);
            }
        }
    }

    // Fast path: per-tensor quantization float→uint8 with AVX2 + proper parallelism
#if defined(__x86_64__) || defined(_M_X64)
    if (block_size == 0 && sz_a == 1 && inptype == CV_32F && outtype == CV_8U && sctype == CV_32F
        && checkHardwareSupport(CV_CPU_AVX2)) {
        float inv_scale = 1.f / reinterpret_cast<const float*>(scale.data)[0];
        float zp_f = zp.empty() ? 0.f : (float)reinterpret_cast<const uint8_t*>(zp.data)[0];
        int64_t total = nslices * slice_size;
        quantizeLinearFast_f32_u8_avx2(reinterpret_cast<const float*>(inp.data),
                                        reinterpret_cast<uint8_t*>(out.data),
                                        inv_scale, zp_f, total);
        return;
    }
#endif

    if (outtype == CV_8U && sctype == CV_32F && inptype == CV_32F)
        quantizeLinear(reinterpret_cast<const float*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<uint8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8U && sctype == CV_16F && inptype == CV_32F)
        quantizeLinear(reinterpret_cast<const float*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<uint8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8U && sctype == CV_32F && inptype == CV_16F)
        quantizeLinear(reinterpret_cast<const hfloat*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<uint8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8U && sctype == CV_16F && inptype == CV_16F)
        quantizeLinear(reinterpret_cast<const hfloat*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const uint8_t*>(zp.data),
                         reinterpret_cast<uint8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8S && sctype == CV_32F && inptype == CV_32F)
        quantizeLinear(reinterpret_cast<const float*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<int8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8S && sctype == CV_16F && inptype == CV_32F)
        quantizeLinear(reinterpret_cast<const float*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<int8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8S && sctype == CV_32F && inptype == CV_16F)
        quantizeLinear(reinterpret_cast<const hfloat*>(inp.data),
                         reinterpret_cast<const float*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<int8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else if (outtype == CV_8S && sctype == CV_16F && inptype == CV_16F)
        quantizeLinear(reinterpret_cast<const hfloat*>(inp.data),
                         reinterpret_cast<const hfloat*>(scale.data),
                         reinterpret_cast<const int8_t*>(zp.data),
                         reinterpret_cast<int8_t*>(out.data),
                         nslices, sz_a, slice_size, block_size);
    else {
        CV_Error_(Error::StsNotImplemented,
                  ("the following combination of types is not supported in "
                   "QuantizeLinear: inp=%s, scale=%s, out=%s",
                   typeToString(inptype).c_str(),
                   typeToString(sctype).c_str(),
                   typeToString(outtype).c_str()));
    }
}

class QuantizeLinearLayerImpl CV_FINAL : public QuantizeLinearLayer
{
public:
    QuantizeLinearLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", 1);
        block_size = params.get<int>("block_size", 0);
        saturate = params.get<bool>("saturate", true);
        output_dtype = params.get<int>("output_dtype", -1);
        CV_Assert(block_size >= 0);
        CV_Assert(saturate);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(2 <= ninputs && ninputs <= 3);
        CV_Assert(requiredOutputs == 1);
        outputs.assign(1, inputs[0]);
        return true;
    }

    int getOutType(int zptype) const
    {
        return output_dtype >= 0 ? output_dtype : zptype;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(2 <= ninputs && ninputs <= 3);
        int zptype = CV_8U;
        if (ninputs == 3) {
            zptype = inputs[2];
        }
        outputs.assign(1, getOutType(zptype));
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        int ninputs = inputs_arr.size(-1).area();
        CV_Assert(2 <= ninputs && ninputs <= 3);

        Mat inp = inputs_arr.getMat(0);
        Mat scale = inputs_arr.getMat(1);
        Mat zeropoint;
        int zptype = CV_8U, outtype;
        MatShape inpshape = inp.shape();

        if (ninputs >= 3) {
            zeropoint = inputs_arr.getMat(2);
            zptype = zeropoint.type();
        }

        outtype = getOutType(zptype);
        auto kind = outputs_arr.kind();

        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(inpshape, outtype);
            quantizeLinear(inp, scale, zeropoint, axis, block_size, outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(inpshape, outtype);
            Mat temp(inpshape, outtype);
            quantizeLinear(inp, scale, zeropoint, axis, block_size, temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }
};

Ptr<QuantizeLinearLayer> QuantizeLinearLayer::create(const LayerParams& params)
{
    return Ptr<QuantizeLinearLayer>(new QuantizeLinearLayerImpl(params));
}

}}
