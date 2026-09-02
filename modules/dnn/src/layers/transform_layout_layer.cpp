// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace cv
{
namespace dnn
{

#if __cplusplus < 201703L
template<typename T>
static T clamp(T d, T min, T max)
{
    return std::min(std::max(d, min), max);
}
#else
#define clamp std::clamp
#endif

static MatShape inferTransformLayoutShape(const MatShape& inpshape_,
                                          DataLayout outlayout,
                                          DataLayout defaultLayout,
                                          int C0)
{
    MatShape inpshape = inpshape_;
    if (inpshape.layout == DATA_LAYOUT_UNKNOWN) {
        inpshape.layout = defaultLayout;
    }

    return inpshape.toLayout(outlayout, C0);
}

template<typename _Tp>
static inline void transpose8x8(const _Tp* inp_, size_t istep,
                                _Tp* out_, size_t ostep)
{
#if defined(__AVX2__)
    if (sizeof(_Tp) == 4u) {
        // 8x8 32-bit transpose via 256-bit AVX2: 8 unpack + 8 shuffle + 8 perm.
        // Roughly 2x faster than the four v_transpose4x4 path on AVX2 hosts.
        const float* inp = (const float*)inp_;
        float* out = (float*)out_;
        __m256 r0 = _mm256_loadu_ps(inp + istep*0);
        __m256 r1 = _mm256_loadu_ps(inp + istep*1);
        __m256 r2 = _mm256_loadu_ps(inp + istep*2);
        __m256 r3 = _mm256_loadu_ps(inp + istep*3);
        __m256 r4 = _mm256_loadu_ps(inp + istep*4);
        __m256 r5 = _mm256_loadu_ps(inp + istep*5);
        __m256 r6 = _mm256_loadu_ps(inp + istep*6);
        __m256 r7 = _mm256_loadu_ps(inp + istep*7);

        __m256 t0 = _mm256_unpacklo_ps(r0, r1);
        __m256 t1 = _mm256_unpackhi_ps(r0, r1);
        __m256 t2 = _mm256_unpacklo_ps(r2, r3);
        __m256 t3 = _mm256_unpackhi_ps(r2, r3);
        __m256 t4 = _mm256_unpacklo_ps(r4, r5);
        __m256 t5 = _mm256_unpackhi_ps(r4, r5);
        __m256 t6 = _mm256_unpacklo_ps(r6, r7);
        __m256 t7 = _mm256_unpackhi_ps(r6, r7);

        __m256 v0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
        __m256 v1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
        __m256 v2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
        __m256 v3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
        __m256 v4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
        __m256 v5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
        __m256 v6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
        __m256 v7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

        // 0x20 -> {lo of A, lo of B}; 0x31 -> {hi of A, hi of B}
        _mm256_storeu_ps(out + ostep*0, _mm256_permute2f128_ps(v0, v4, 0x20));
        _mm256_storeu_ps(out + ostep*1, _mm256_permute2f128_ps(v1, v5, 0x20));
        _mm256_storeu_ps(out + ostep*2, _mm256_permute2f128_ps(v2, v6, 0x20));
        _mm256_storeu_ps(out + ostep*3, _mm256_permute2f128_ps(v3, v7, 0x20));
        _mm256_storeu_ps(out + ostep*4, _mm256_permute2f128_ps(v0, v4, 0x31));
        _mm256_storeu_ps(out + ostep*5, _mm256_permute2f128_ps(v1, v5, 0x31));
        _mm256_storeu_ps(out + ostep*6, _mm256_permute2f128_ps(v2, v6, 0x31));
        _mm256_storeu_ps(out + ostep*7, _mm256_permute2f128_ps(v3, v7, 0x31));
        return;
    }
#endif
#if CV_SIMD128
    if (sizeof(_Tp) == 4u) {
        const uint32_t* inp = (const uint32_t*)inp_;
        uint32_t* out = (uint32_t*)out_;
        v_uint32x4 a0, a1, a2, a3, b0, b1, b2, b3;

        a0 = v_load(inp + istep*0);
        a1 = v_load(inp + istep*1);
        a2 = v_load(inp + istep*2);
        a3 = v_load(inp + istep*3);
        v_transpose4x4(a0, a1, a2, a3, b0, b1, b2, b3);
        v_store(out + ostep*0, b0);
        v_store(out + ostep*1, b1);
        v_store(out + ostep*2, b2);
        v_store(out + ostep*3, b3);

        a0 = v_load(inp + istep*0 + 4);
        a1 = v_load(inp + istep*1 + 4);
        a2 = v_load(inp + istep*2 + 4);
        a3 = v_load(inp + istep*3 + 4);
        v_transpose4x4(a0, a1, a2, a3, b0, b1, b2, b3);
        v_store(out + ostep*4, b0);
        v_store(out + ostep*5, b1);
        v_store(out + ostep*6, b2);
        v_store(out + ostep*7, b3);

        a0 = v_load(inp + istep*4);
        a1 = v_load(inp + istep*5);
        a2 = v_load(inp + istep*6);
        a3 = v_load(inp + istep*7);
        v_transpose4x4(a0, a1, a2, a3, b0, b1, b2, b3);
        v_store(out + ostep*0 + 4, b0);
        v_store(out + ostep*1 + 4, b1);
        v_store(out + ostep*2 + 4, b2);
        v_store(out + ostep*3 + 4, b3);

        a0 = v_load(inp + istep*4 + 4);
        a1 = v_load(inp + istep*5 + 4);
        a2 = v_load(inp + istep*6 + 4);
        a3 = v_load(inp + istep*7 + 4);
        v_transpose4x4(a0, a1, a2, a3, b0, b1, b2, b3);
        v_store(out + ostep*4 + 4, b0);
        v_store(out + ostep*5 + 4, b1);
        v_store(out + ostep*6 + 4, b2);
        v_store(out + ostep*7 + 4, b3);
    } else
#endif
    {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            out_[i * ostep + j] = inp_[j * istep + i];
    }
}

template <typename _Tp>
void transformLayoutInterleave_(const _Tp* inp_base, _Tp* out_base, int C, size_t len,
                                int nc, int nzc, size_t dlen)
{
    size_t i = 0;
    for (; i + 7u < dlen; i += 8u)
    {
        int c = 0;

        for (; c + 7u < nzc; c += 8u) {
            transpose8x8<_Tp>(inp_base + c * len + i, len, out_base + i * nc + c, nc);
        }

        for (; c < nzc; ++c) {
            _Tp* outptr = out_base + i * nc + c;
            const _Tp* inptr = inp_base + c * len + i;
            outptr[0 * nc] = inptr[0];
            outptr[1 * nc] = inptr[1];
            outptr[2 * nc] = inptr[2];
            outptr[3 * nc] = inptr[3];
            outptr[4 * nc] = inptr[4];
            outptr[5 * nc] = inptr[5];
            outptr[6 * nc] = inptr[6];
            outptr[7 * nc] = inptr[7];
        }

        for (; c < nc; ++c) {
            _Tp* outptr = out_base + i * nc + c;
            outptr[0 * nc] = (_Tp)0; outptr[1 * nc] = (_Tp)0; outptr[2 * nc] = (_Tp)0; outptr[3 * nc] = (_Tp)0;
            outptr[4 * nc] = (_Tp)0; outptr[5 * nc] = (_Tp)0; outptr[6 * nc] = (_Tp)0; outptr[7 * nc] = (_Tp)0;
        }
    }
    for (; i < dlen; ++i) {
        _Tp* outptr = out_base + i * nc;
        for (int c = 0; c < nc; ++c) {
            outptr[c] = c < nzc ? inp_base[c*len + i] : (_Tp)0;
        }
    }
}

template <typename _Tp>
void transformLayoutDeinterleave_(const _Tp* inp_base, _Tp* out_base, int C, size_t len,
                                  int nc, int nzc, size_t dlen)
{
    size_t i = 0;
    for (; i + 7u < dlen; i += 8u)
    {
        int c = 0;

        for (; c + 7u < nzc; c += 8u)
        {
            transpose8x8<_Tp>(inp_base + i * nc + c, nc, out_base + c * len + i, len);
        }

        for (; c < nzc; ++c)
        {
            const _Tp* inptr = inp_base + i * nc + c;
            _Tp* outptr = out_base + c * len + i;
            outptr[0] = inptr[0 * nc];
            outptr[1] = inptr[1 * nc];
            outptr[2] = inptr[2 * nc];
            outptr[3] = inptr[3 * nc];
            outptr[4] = inptr[4 * nc];
            outptr[5] = inptr[5 * nc];
            outptr[6] = inptr[6 * nc];
            outptr[7] = inptr[7 * nc];
        }
    }
    for (; i < dlen; ++i)
    {
        const _Tp* inptr = inp_base + i * nc;
        for (int c = 0; c < nzc; ++c) {
            out_base[c*len + i] = inptr[c];
        }
    }
}

typedef void (*TransformLayoutFunc)(const void* inp, void* out, int C, size_t planesize,
                                    int nc, int nzc, size_t dlen);

#undef DECL_TRANSFORM_LAYOUT
#define DECL_TRANSFORM_LAYOUT(suffix, _Tp) \
static void transformLayoutInterleave_##suffix(const void* inp, void* out, int C, size_t planesize, \
                                               int nc, int nzc, size_t dlen) \
{ \
    transformLayoutInterleave_((const _Tp*)inp, (_Tp*)out, C, planesize, nc, nzc, dlen); \
} \
static void transformLayoutDeinterleave_##suffix(const void* inp, void* out, int C, size_t planesize, \
                                                 int nc, int nzc, size_t dlen) \
{ \
    transformLayoutDeinterleave_((const _Tp*)inp, (_Tp*)out, C, planesize, nc, nzc, dlen); \
}

DECL_TRANSFORM_LAYOUT(8u, uint8_t)
DECL_TRANSFORM_LAYOUT(16u, uint16_t)
DECL_TRANSFORM_LAYOUT(32u, uint32_t)
DECL_TRANSFORM_LAYOUT(64u, uint64_t)

TransformLayoutFunc getTransformLayoutFunc(DataLayout inplayout, DataLayout outlayout, size_t esz)
{
    if (inplayout == DATA_LAYOUT_NCHW &&
        (outlayout == DATA_LAYOUT_BLOCK || outlayout == DATA_LAYOUT_NHWC)) {
        return esz == 1u ? transformLayoutInterleave_8u :
               esz == 2u ? transformLayoutInterleave_16u :
               esz == 4u ? transformLayoutInterleave_32u :
               esz == 8u ? transformLayoutInterleave_64u : nullptr;
    }
    if ((inplayout == DATA_LAYOUT_BLOCK || inplayout == DATA_LAYOUT_NHWC) &&
        outlayout == DATA_LAYOUT_NCHW) {
        return esz == 1u ? transformLayoutDeinterleave_8u :
               esz == 2u ? transformLayoutDeinterleave_16u :
               esz == 4u ? transformLayoutDeinterleave_32u :
               esz == 8u ? transformLayoutDeinterleave_64u : nullptr;
    }
    return nullptr;
}

void transformLayout(const Mat& inp, Mat& out,
                     DataLayout outlayout,
                     DataLayout defaultLayout,
                     int C0)
{
    CV_Assert(defaultLayout == DATA_LAYOUT_NCHW || defaultLayout == DATA_LAYOUT_NHWC);
    CV_Assert(outlayout == DATA_LAYOUT_BLOCK || outlayout == DATA_LAYOUT_NCHW || outlayout == DATA_LAYOUT_NHWC);

    MatShape inpshape = inp.size;
    /*if (inpshape.layout == DATA_LAYOUT_NCHW &&
        inpshape.dims == 4 && inpshape[1] == 272 && inpshape[2] == 14 && inpshape[3] == 14) {
        putchar('.');
    }*/

    if (inpshape.layout == DATA_LAYOUT_UNKNOWN) {
        inpshape.layout = defaultLayout;
    }
    DataLayout inplayout = inpshape.layout;
    MatShape outshape = inferTransformLayoutShape(inpshape, outlayout, defaultLayout, C0);
    out.fit(outshape, inp.type());

    if (inp.empty())
        return;

    if (inplayout == outlayout) {
        inp.copyTo(out);
        return;
    }

    CV_Assert_N(inp.isContinuous(), out.isContinuous());

    size_t esz = inp.elemSize();
    TransformLayoutFunc kernel = getTransformLayoutFunc(inplayout, outlayout, esz);
    CV_Assert(kernel != nullptr);

    int N = inpshape[0];
    int C = inpshape.channels();
    C0 = inplayout == DATA_LAYOUT_BLOCK ? inpshape.back() : C0;
    int C1 = (C + C0 - 1) / C0;

    size_t planesize = 1;
    int inp_sp0 = inplayout == DATA_LAYOUT_NHWC ? 1 : 2;
    int inp_sp1 = inplayout == DATA_LAYOUT_NCHW ? inpshape.dims : inpshape.dims-1;
    for (int i = inp_sp0; i < inp_sp1; i++) {
        planesize *= (size_t)inpshape[i];
    }

    size_t total = N*C1*planesize*C0;
    // Tuned to keep small encoder/decoder transforms (e.g. 256ch * 14*14 ~ 50K elems)
    // from running single-threaded. 16K elems ~ 64 KB ~ L1-resident chunk.
    constexpr size_t min_elems_per_chunk = 1 << 14;
    int nblocks = int((total + min_elems_per_chunk/2) / min_elems_per_chunk);
    int nthreads = std::max(1, getNumThreads());
    nblocks = clamp(nblocks, 1, std::max(N*C1, 1) * 16);
    nblocks = (nblocks + N*C1 - 1)/(N*C1);
    nblocks = std::min(nblocks, std::max(1, nthreads));

    int total_chunks = N * C1 * nblocks;
    double nstripes = std::min((double)total_chunks, (double)nthreads);
    parallel_for_(Range(0, total_chunks), [&](const Range& range)
    {
        int dchunk = 1u;
        bool interleave = inplayout == DATA_LAYOUT_NCHW;
        const uint8_t* inptr = (const uint8_t*)inp.data;
        uint8_t* outptr = (uint8_t*)out.data;

        for (int chunk = range.start; chunk < range.end; chunk += dchunk)
        {
            int n = chunk/(C1*nblocks);
            int c1 = (chunk % (C1*nblocks))/nblocks;
            int block = chunk % nblocks;
            int nc = C0;
            int nzc = std::min(nc, C - c1*C0);
            dchunk = std::min(nblocks - block, range.end - chunk);
            size_t block_start = block * planesize / nblocks;
            size_t block_end = (block + dchunk) * planesize / nblocks;
            size_t dlen = block_end - block_start;
            size_t inpofs = ((n * C1 + c1) * planesize + block_start) * nc * esz;
            size_t outofs = ((n * C + c1 * C0) * planesize + block_start) * esz;
            if (interleave) {
                std::swap(inpofs, outofs);
            }

            kernel(inptr + inpofs, outptr + outofs, C, planesize, nc, nzc, dlen);
        }
    }, nstripes);
}

// Deinterleave (BLOCK/NHWC -> NCHW) fused with an elementwise add against a
// same-shaped NCHW residual tensor -- e.g. a ConvTranspose2 block-layout
// output meeting a plain-NCHW skip connection. Folds what would otherwise be
// two full passes over the tensor (TransformLayout, then NaryEltwise Add)
// into the one pass TransformLayout already has to make.
//
// Float32 only, and only for the exact-shape case (no broadcasting): the
// fusion pass in graph_fusion_transform_add.cpp is the sole caller and only
// fires when both operands match shape and dtype exactly, and when the
// channel count divides evenly by C0 -- so every channel block is fully
// populated (nzc == nc) and the partial-block tail below never has to run.
static void transformLayoutDeinterleaveAddF32(const uint32_t* inp_base, const float* res_base,
                                              float* out_base, size_t len,
                                              int nc, int nzc, size_t dlen)
{
    CV_Assert(nzc == nc);  // guaranteed by the fusion pass (C % C0 == 0)
    size_t i = 0;
    for (; i + 7u < dlen; i += 8u)
    {
        int c = 0;
        for (; c + 7u < nc; c += 8u)
        {
            float* dst = out_base + (size_t)c * len + i;
            transpose8x8<uint32_t>(inp_base + i * nc + c, nc, (uint32_t*)dst, len);
            for (int r = 0; r < 8; r++)
            {
                float* o = out_base + (size_t)(c + r) * len + i;
                const float* rr = res_base + (size_t)(c + r) * len + i;
                for (int j = 0; j < 8; j++)
                    o[j] += rr[j];
            }
        }
        // Unreached while nc is a multiple of 8 (true for the C0 values the
        // block layout uses), kept only so this degrades safely instead of
        // silently dropping channels if that ever changes.
        for (; c < nc; ++c)
        {
            const uint32_t* inptr = inp_base + i * nc + c;
            const float* rr = res_base + (size_t)c * len + i;
            float* outptr = out_base + (size_t)c * len + i;
            for (int j = 0; j < 8; j++)
                outptr[j] = *(const float*)&inptr[j * nc] + rr[j];
        }
    }
    for (; i < dlen; ++i)
    {
        const uint32_t* inptr = inp_base + i * nc;
        for (int c = 0; c < nc; ++c)
            out_base[(size_t)c * len + i] = *(const float*)&inptr[c] + res_base[(size_t)c * len + i];
    }
}

// Mirrors transformLayout()'s chunking (same tuning: ~16K elems/chunk, capped
// by thread count) but is specialized to the one case the fusion pass
// produces: BLOCK or NHWC input, NCHW output, float32, exact shape match.
static void transformLayoutAddF32(const Mat& inp, const Mat& residual, Mat& out,
                                  DataLayout defaultLayout, int C0)
{
    CV_Assert(inp.type() == CV_32F && residual.type() == CV_32F);
    MatShape inpshape = inp.size;
    if (inpshape.layout == DATA_LAYOUT_UNKNOWN)
        inpshape.layout = defaultLayout;
    DataLayout inplayout = inpshape.layout;
    CV_Assert(inplayout == DATA_LAYOUT_BLOCK || inplayout == DATA_LAYOUT_NHWC);

    MatShape outshape = inferTransformLayoutShape(inpshape, DATA_LAYOUT_NCHW, defaultLayout, C0);
    out.fit(outshape, CV_32F);
    CV_Assert(residual.size == out.size);

    if (inp.empty())
        return;
    CV_Assert_N(inp.isContinuous(), residual.isContinuous(), out.isContinuous());

    int N = inpshape[0];
    int C = inpshape.channels();
    C0 = inplayout == DATA_LAYOUT_BLOCK ? inpshape.back() : C0;
    int C1 = (C + C0 - 1) / C0;
    CV_Assert(C % C0 == 0);  // guaranteed by the fusion pass; see kernel comment above

    size_t planesize = 1;
    int inp_sp0 = inplayout == DATA_LAYOUT_NHWC ? 1 : 2;
    int inp_sp1 = inpshape.dims - 1;
    for (int i = inp_sp0; i < inp_sp1; i++)
        planesize *= (size_t)inpshape[i];

    size_t total = (size_t)N * C1 * planesize * C0;
    constexpr size_t min_elems_per_chunk = 1 << 14;
    int nblocks = int((total + min_elems_per_chunk/2) / min_elems_per_chunk);
    int nthreads = std::max(1, getNumThreads());
    nblocks = clamp(nblocks, 1, std::max(N*C1, 1) * 16);
    nblocks = (nblocks + N*C1 - 1)/(N*C1);
    nblocks = std::min(nblocks, std::max(1, nthreads));

    int total_chunks = N * C1 * nblocks;
    double nstripes = std::min((double)total_chunks, (double)nthreads);
    const uint32_t* inpdata = (const uint32_t*)inp.data;
    const float* resdata = (const float*)residual.data;
    float* outdata = (float*)out.data;
    parallel_for_(Range(0, total_chunks), [&](const Range& range)
    {
        int dchunk = 1;
        for (int chunk = range.start; chunk < range.end; chunk += dchunk)
        {
            int n = chunk/(C1*nblocks);
            int c1 = (chunk % (C1*nblocks))/nblocks;
            int block = chunk % nblocks;
            int nc = C0;
            dchunk = std::min(nblocks - block, range.end - chunk);
            size_t block_start = block * planesize / nblocks;
            size_t block_end = (block + dchunk) * planesize / nblocks;
            size_t dlen = block_end - block_start;
            size_t inpofs = ((size_t)(n * C1 + c1) * planesize + block_start) * nc;
            size_t outofs = ((size_t)(n * C + c1 * C0) * planesize + block_start);

            transformLayoutDeinterleaveAddF32(inpdata + inpofs, resdata + outofs, outdata + outofs,
                                              planesize, nc, nc, dlen);
        }
    }, nstripes);
}

class TransformLayoutLayerImpl : public TransformLayoutLayer
{
public:
    TransformLayoutLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        layout = (DataLayout)params.get<int>("layout");
        C0 = params.get<int>("C0", 1);
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "target_layout: \"" << layoutToString(layout) << "\",\n";

        if (layout == DATA_LAYOUT_BLOCK) {
            prindent(strm, indent);
            strm << "C0: " << C0 << ",\n";
        }
        return strm;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape> &inputs,
                             const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        return (int64_t)std::max(inputs[0].total(), outputs[0].total());
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs == (fusedAdd ? 2 : 1));

        outtypes.assign(1, inptypes[0]);
        temptypes.clear();
    }

    MatShape inferShape(const MatShape& inpshape_) const
    {
        return inferTransformLayoutShape(inpshape_, layout,
                                         getNetImpl(this)->originalLayout, C0);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape> &outshapes,
                                 std::vector<MatShape> &tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs == (fusedAdd ? 2u : 1u));

        // The residual (inpshapes[1], when fused) must already match the
        // converted output shape exactly -- graph_fusion_transform_add.cpp
        // only fuses when it has verified that statically; it isn't
        // re-derived here.
        outshapes.assign(1, inferShape(inpshapes[0]));
        tempshapes.clear();
        return true;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        size_t ninputs = inputs_arr.total();
        CV_Assert(ninputs == (fusedAdd ? 2u : 1u));

        int inptype = inputs_arr.type(0);
        MatShape inpshape = inputs_arr.shape(0);
        MatShape outshape = inferShape(inpshape);
        int outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            if (fusedAdd)
                runOpAdd(inp, inputs_arr.getMat(1), outs[0]);
            else
                runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
            if (fusedAdd)
                runOpAdd(inp, inputs_arr.getMat(1), temp);
            else
                runOp(inp, temp);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, Mat& out)
    {
        DataLayout origLayout = getNetImpl(this)->originalLayout;
        transformLayout(inp, out, layout, origLayout, C0);
#if 0
        Mat temp;
        transformLayout(out, temp, layout == DATA_LAYOUT_BLOCK ? origLayout : DATA_LAYOUT_BLOCK, origLayout, C0);
        double err = norm(temp, inp, NORM_INF);
        size_t i, N = inp.total();
        const float* inpdata = inp.ptr<float>();
        const float* tempdata = temp.ptr<float>();
        for (i = 0; i < N; i++) {
            CV_Assert_N(!cvIsNaN(inpdata[i]), !cvIsNaN(tempdata[i]));
        }
        CV_Assert(err == 0.);
#endif
    }

    // graph_fusion_transform_add.cpp guarantees layout == DATA_LAYOUT_NCHW,
    // float32, and an exact shape match between the residual and the
    // converted output before it sets fusedAdd -- so none of that is
    // re-checked against the graph here, only against the actual data.
    void runOpAdd(const Mat& inp, const Mat& residual, Mat& out)
    {
        CV_Assert(layout == DATA_LAYOUT_NCHW);
        // Buffers must be distinct: the kernel reads inp and residual while
        // writing out at different strides, so any aliasing would corrupt
        // the result rather than merely being redundant. alwaysSupportInplace()
        // returning false is what's supposed to keep the buffer pool from
        // handing us an aliased output; check it rather than trust it.
        CV_Assert(out.data != inp.data && out.data != residual.data);
        DataLayout origLayout = getNetImpl(this)->originalLayout;
        transformLayoutAddF32(inp, residual, out, origLayout, C0);
    }
};

Ptr<TransformLayoutLayer> TransformLayoutLayer::create(const LayerParams& params)
{
    return Ptr<TransformLayoutLayer>(new TransformLayoutLayerImpl(params));
}

}}
