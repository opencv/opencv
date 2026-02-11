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

static MatShape inferTransformLayoutShape(const MatShape& inpshape_,
                                          DataLayout outlayout,
                                          DataLayout defaultLayout,
                                          int C0, int ninpgroups, int noutgroups)
{
    MatShape inpshape = inpshape_;
    if (inpshape.layout == DATA_LAYOUT_UNKNOWN) {
        inpshape.layout = defaultLayout;
    }

    if (ninpgroups > 1) {
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        inpshape = inpshape.toLayout(DATA_LAYOUT_NCHW);
    }

    return inpshape.toLayout(outlayout, C0, noutgroups);
}

// NCHW <=> NHWC or
// NCHW/NHWC <=> BLOCK
template <typename _Tp>
void transformLayout_(const _Tp* inp_, size_t istep, size_t istep0, size_t istep1,
                      _Tp* out_, size_t ostep, size_t ostep0, size_t ostep1,
                      size_t npix, int C0, int C1g, int Cg, int ngroups)
{
    int C = ngroups*Cg;
    CV_Assert(C0 % 8 == 0 || C0 == 4 || C == C0);
    CV_Assert(istep0 == 1u || ostep0 == 1u);
    const int dC0 = std::min(C0, 8);
    for (int g = 0; g < ngroups; g++) {
        for (int c1 = 0; c1 < C1g; c1++) {
            for (int c0 = 0; c0 < C0; c0 += dC0) {
                int dc = std::min(Cg - g*C1g - c0, dC0);
                const _Tp* inp = inp_ + istep0*c0 + istep1*c1 + istepg*g;
                _Tp* out = out_ + ostep0*c0 + ostep1*c1 + ostepg*g;
                
                if (dc == 8) {
                    if (istep0 == 1) {
                        for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                            _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                            _Tp x4 = inp[4], x5 = inp[5], x6 = inp[6], x7 = inp[7];
                            out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                            out[ostep0*4] = x4; out[ostep0*5] = x5; out[ostep0*6] = x6; out[ostep0*7] = x7;
                        }
                    } else {
                        for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                            _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                            _Tp x4 = inp[istep0*4], x5 = inp[istep0*5], x6 = inp[istep0*6], x7 = inp[istep0*7];
                            out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                            out[4] = x4; out[5] = x5; out[6] = x6; out[7] = x7;
                        }
                    }
                } else if (dc == 4) {
                    if (istep0 == 1) {
                        for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                            _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                            out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                        }
                    } else {
                        for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                            _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                            out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                        }
                    }
                } else if (dc == 3 && ostep0 == 1 && ostep == C0) {
                    memset(out, 0, npix*C0*sizeof(out[0]));
                    for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2];
                        out[0] = x0; out[1] = x1; out[2] = x2;
                    }
                } else if (dc == 1 && ostep0 == 1 && ostep == C0) {
                    memset(out, 0, npix*C0*sizeof(out[0]));
                    for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        out[0] = inp[0];
                    }
                } else {
                    for (size_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        int c = 0;
                        for (; c < dc; c++)
                            out[ostep0*c] = inp[istep0*c];
                        if (ostep == C0) {
                            for (; c < dC0; c++)
                                out[ostep0*c] = 0;
                        }
                    }
                }
            }
        }
    }
}

#undef CV_TRANSFORM_LAYOUT_IMPL
#define CV_TRANSFORM_LAYOUT_IMPL(typ, suffix) \
static void transformLayout##suffix(const void* inp_, size_t istep, size_t istep0, \
                                    size_t istep1, size_t istepg, \
                                    void* out_, size_t ostep, size_t ostep0, \
                                    size_t ostep1, size_t ostepg, \
                                    size_t npix, int C0, int C1g, int Cg, int ngroups) \
{ \
    transformLayout_((const typ*)inp_, istep, istep0, istep1, istepg, \
                     (typ*)out_, ostep, ostep0, ostep1, ostepg, \
                     npix, C0, C1g, Cg, ngroups); \
}

CV_TRANSFORM_LAYOUT_IMPL(uint8_t, 8u)
CV_TRANSFORM_LAYOUT_IMPL(uint16_t, 16u)
CV_TRANSFORM_LAYOUT_IMPL(uint32_t, 32u)
CV_TRANSFORM_LAYOUT_IMPL(uint64_t, 64u)

typedef void (*TransformLayoutFunc)(const void*, size_t, size_t, size_t, size_t,
                                    void*, size_t, size_t, size_t, size_t,
                                    size_t, int, int, int, int);

void transformLayout(const Mat& inp, Mat& out,
                     DataLayout outlayout,
                     DataLayout defaultLayout,
                     int C0, int ninpgroups, int noutgroups)
{
    CV_Assert(defaultLayout == DATA_LAYOUT_NCHW || defaultLayout == DATA_LAYOUT_NHWC);
    CV_Assert(outlayout == DATA_LAYOUT_BLOCK || outlayout == DATA_LAYOUT_NCHW || outlayout == DATA_LAYOUT_NHWC);

    MatShape inpshape = inp.size;

    if (inpshape.layout == DATA_LAYOUT_UNKNOWN) {
        inpshape.layout = defaultLayout;
    }
    DataLayout inplayout = inpshape.layout;
    MatShape outshape = inferTransformLayoutShape(inpshape, outlayout, defaultLayout,
                                                  C0, ninpgroups, noutgroups);
    out.fit(outshape, inp.type());

    if (inp.empty())
        return;

    int inp_ndims = inpshape.dims;
    int out_ndims = outshape.dims;
    int C = inplayout == DATA_LAYOUT_BLOCK ? inpshape.C :
        inpshape[inplayout == DATA_LAYOUT_NCHW ? 1 : inp_ndims-1];
    int inpC0 = inplayout == DATA_LAYOUT_BLOCK ? inpshape.back() : inplayout == DATA_LAYOUT_NCHW ? 1 : C;
    int outC0 = outlayout == DATA_LAYOUT_BLOCK ? C0 : outlayout == DATA_LAYOUT_NCHW ? 1 : C;

    if (inplayout == outlayout && inpC0 == outC0 && ninpgroups == noutgroups) {
        inp.copyTo(out);
        return;
    }

    CV_Assert(ninpgroups == 1 || inplayout == DATA_LAYOUT_BLOCK);
    CV_Assert(noutgroups == 1 || outlayout == DATA_LAYOUT_BLOCK);
    CV_Assert_N(ninpgroups > 0, C % ninpgroups == 0);
    CV_Assert_N(noutgroups > 0, C % noutgroups == 0);

    int N = inpshape[0];
    size_t inptotal = inp.total();
    size_t outtotal = out.total();
    size_t inplanesize_C = inptotal / N;
    size_t outplanesize_C = outtotal / N;
    size_t planesize = 1;
    int inp_sp0 = inplayout == DATA_LAYOUT_NHWC ? 1 : 2;
    int inp_sp1 = inplayout == DATA_LAYOUT_NCHW ? inp_ndims : inp_ndims-1;
    for (int i = inp_sp0; i < inp_sp1; i++) {
        planesize *= (size_t)inpshape[i];
    }

    size_t allplanes = planesize*N;

    constexpr size_t BLOCK_SIZE = 1u << 17;
    size_t nblocks = (outtotal + BLOCK_SIZE - 1)/BLOCK_SIZE;
    nblocks = std::min(nblocks, allplanes);

    size_t esz = inp.elemSize();
    size_t istep0, istep1, istepg, istep;
    size_t ostep0, ostep1, ostepg, ostep;

    if (inplayout == DATA_LAYOUT_NCHW) {
        istep = 1;
        istep0 = planesize;
    } else if (inplayout == DATA_LAYOUT_NHWC) {
        istep = C;
        istep0 = 1;
    } else {
        istep = inpC0;
        istep0 = 1;
    }

    if (outlayout == DATA_LAYOUT_NCHW) {
        ostep = 1;
        ostep0 = planesize;
    } else if (outlayout == DATA_LAYOUT_NHWC) {
        ostep = C;
        ostep0 = 1;
    } else {
        ostep = outC0;
        ostep0 = 1;
    }

    const char* inptr0 = (const char*)inp.data;
    char* outptr0 = (char*)out.data;

    TransformLayoutFunc transformLayoutFunc =
        esz == 1 ? transformLayout8u :
        esz == 2 ? transformLayout16u :
        esz == 4 ? transformLayout32u :
        esz == 8 ? transformLayout64u : nullptr;

    CV_Assert(transformLayoutFunc != nullptr);

    parallel_for_(Range(0, int(nblocks)), [&](const Range& range) {
        size_t start = range.start*allplanes/nblocks;
        size_t end = range.end*allplanes/nblocks;
        size_t npix = 0;

        for (size_t ofs = start; ofs < end; ofs += npix) {
            size_t sample_idx = ofs/planesize;
            size_t rawofs = ofs - sample_idx*planesize;
            npix = std::min(planesize - rawofs, end - ofs);
            const char* inptr = inptr0 + (inplanesize_C*sample_idx + istep*rawofs)*esz;
            char* outptr = outptr0 + (outplanesize_C*sample_idx + ostep*rawofs)*esz;
            transformLayoutFunc(inptr, istep, istep0, istep1, istepg,
                                outptr, ostep, ostep0, ostep1, ostepg,
                                npix, C0, C1g, Cg, ngroups);
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
                                         getNetImpl(this)->originalLayout, C0, 1, 1);
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
        CV_Assert(err == 0.);
#endif
    }
};

Ptr<TransformLayoutLayer> TransformLayoutLayer::create(const LayerParams& params)
{
    return Ptr<TransformLayoutLayer>(new TransformLayoutLayerImpl(params));
}

}}
