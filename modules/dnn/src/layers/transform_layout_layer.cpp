// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

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
    constexpr size_t min_elems_per_chunk = 1 << 17;
    int nblocks = int((total + min_elems_per_chunk/2) / min_elems_per_chunk);
    nblocks = clamp(nblocks, 1, 128);
    nblocks = (nblocks + N*C1 - 1)/(N*C1);

    parallel_for_(Range(0, N*C1*nblocks), [&](const Range& range)
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
    });
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
        CV_Assert(ninputs == 1);

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
        CV_Assert(ninputs == 1);

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
        CV_Assert(ninputs == 1);

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
            runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
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
};

Ptr<TransformLayoutLayer> TransformLayoutLayer::create(const LayerParams& params)
{
    return Ptr<TransformLayoutLayer>(new TransformLayoutLayerImpl(params));
}

}}
