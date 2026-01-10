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
                                          int C0)
{
    MatShape inpshape = inpshape_;
    int ndims = inpshape.dims;
    DataLayout inplayout = inpshape_.layout == DATA_LAYOUT_UNKNOWN ? defaultLayout : inpshape.layout;
    inpshape.layout = inplayout;

    if (inplayout == outlayout) {
        return inpshape;
    }
    
    // non-block => block
    if (outlayout == DATA_LAYOUT_BLOCK) {
        CV_Assert(inplayout != DATA_LAYOUT_BLOCK);
        return inpshape.toBlock(C0);
    }

    // block => non-block
    if (inplayout == DATA_LAYOUT_BLOCK) {
        CV_Assert(outlayout != DATA_LAYOUT_BLOCK);
        return inpshape.fromBlock(outlayout);
    }

    MatShape outshape = inpshape;
    outshape.layout = outlayout;

    // NHWC => NCHW
    if (outlayout == DATA_LAYOUT_NCHW) {
        CV_Assert(inplayout == DATA_LAYOUT_NHWC);
        int C = inpshape[ndims-1];
        for (int i = 2; i < ndims; i++)
            outshape[i] = inpshape[i-1];
        outshape[1] = C;
    } else {
        // NCHW => NHWC
        CV_Assert(outlayout == DATA_LAYOUT_NHWC && inplayout == DATA_LAYOUT_NCHW);
        int C = inpshape[1];
        for (int i = 2; i < ndims; i++)
            outshape[i-1] = inpshape[i];
        outshape[ndims-1] = C;
    }
    return outshape;
}

template <typename _Tp>
void transform_layout_(const _Tp* inp_, int istep, int istep0, int istep1,
                      _Tp* out_, int ostep, int ostep0, int ostep1,
                      int npix, int C0, int C1, int C)
{
    CV_Assert(C0 % 8 == 0 || C0 == 4 || C1 == 1);
    CV_Assert(istep0 == 1 || ostep0 == 1);
    const int dC0 = std::min(C0, (int)8);
    for (int c1 = 0; c1 < C1; c1++) {
        for (int c0 = 0; c0 < C0; c0 += dC0) {
            const _Tp* inp = inp_ + istep0*c0 + istep1*c1;
            _Tp* out = out_ + ostep0*c0 + ostep1*c1;
            int dc = std::min(C - (c1*C0 + c0), dC0);
            if (dc == 8) {
                if (istep0 == 1) {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                        _Tp x4 = inp[4], x5 = inp[5], x6 = inp[6], x7 = inp[7];
                        out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                        out[ostep0*4] = x4; out[ostep0*5] = x5; out[ostep0*6] = x6; out[ostep0*7] = x7;
                    }
                } else {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                        _Tp x4 = inp[istep0*4], x5 = inp[istep0*5], x6 = inp[istep0*6], x7 = inp[istep0*7];
                        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                        out[4] = x4; out[5] = x5; out[6] = x6; out[7] = x7;
                    }
                }
            } else if (dc == 4) {
                if (istep0 == 1) {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                        out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                    }
                } else {
                    for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                    }
                }
            } else if (dc == 3 && ostep0 == 1 && ostep == C0) {
                memset(out, 0, npix*C0*sizeof(out[0]));
                for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                    _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2];
                    out[0] = x0; out[1] = x1; out[2] = x2;
                }
            } else if (dc == 1 && ostep0 == 1 && ostep == C0) {
                memset(out, 0, npix*C0*sizeof(out[0]));
                for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
                    out[0] = inp[0];
                }
            } else {
                for (int i = 0; i < npix; i++, inp += istep, out += ostep) {
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

#undef CV_TRANSFORM_LAYOUT_IMPL
#define CV_TRANSFORM_LAYOUT_IMPL(typ, suffix) \
static void transform_layout_##suffix(const void* inp_, int istep, int istep0, int istep1, \
                                      void* out_, int ostep, int ostep0, int ostep1, \
                                      int npix, int C0, int C1, int C) \
{ \
    transform_layout_((const typ*)inp_, istep, istep0, istep1, \
                     (typ*)out_, ostep, ostep0, ostep1, npix, C0, C1, C); \
}

CV_TRANSFORM_LAYOUT_IMPL(uint8_t, 8u)
CV_TRANSFORM_LAYOUT_IMPL(uint16_t, 16u)
CV_TRANSFORM_LAYOUT_IMPL(uint32_t, 32u)
CV_TRANSFORM_LAYOUT_IMPL(uint, 64u)

typedef void (*transform_layout_func_t)(const void* inp, int istep, int istep0, int istep1,
                                        void* out, int ostep, int ostep0, int ostep1,
                                        int npix, int C0, int C1, int C);

void transformLayout(const Mat& inp, Mat& out,
                     DataLayout outlayout,
                     DataLayout defaultLayout,
                     int C0)
{
    MatShape inpshape = inp.size;
    MatShape outshape = inferTransformLayoutShape(inpshape, outlayout, defaultLayout, C0);
    DataLayout inplayout = inpshape.layout;
    if (inplayout == DATA_LAYOUT_UNKNOWN && outlayout == DATA_LAYOUT_BLOCK)
        inplayout = defaultLayout;
    
    out.fit(outshape, inp.type());

    if (inp.empty())
        return;

    if (inplayout == outlayout) {
        inp.copyTo(out);
        return;
    }

    int inp_ndims = inpshape.dims;
    int out_ndims = outshape.dims;
    int N = inpshape[0];
    int C = inplayout == DATA_LAYOUT_BLOCK ? inpshape.C :
        inpshape[inplayout == DATA_LAYOUT_NCHW ? 1 : inp_ndims-1];
    int inptotal = (int)inp.total();
    int outtotal = (int)out.total();
    int inplanesize_C = inptotal / N;
    int outplanesize_C = outtotal / N;
    int planesize = (inplayout != DATA_LAYOUT_BLOCK ? inplanesize_C : outplanesize_C)/C;
    int allplanes = planesize*N;

    constexpr int BLOCK_SIZE = 1 << 17;
    int nblocks = (outtotal + BLOCK_SIZE - 1)/BLOCK_SIZE;
    nblocks = std::min(nblocks, allplanes);

    size_t esz = inp.elemSize();
    int istep0, istep1, istep;
    int ostep0, ostep1, ostep;
    int C0_ = C, C1_ = 1;

    if (inplayout == DATA_LAYOUT_BLOCK || outlayout == DATA_LAYOUT_BLOCK) {
        C0_ = inplayout == DATA_LAYOUT_BLOCK ? inpshape[inp_ndims-1] : outshape[out_ndims-1];
        C1_ = (C + C0_ - 1)/C0_;
    }

    if (inplayout == DATA_LAYOUT_NCHW) {
        istep = 1;
        istep0 = planesize;
        istep1 = planesize*C0_;
    } else if (inplayout == DATA_LAYOUT_NHWC) {
        istep = C;
        istep0 = 1;
        istep1 = C0_;
    } else {
        istep = C0_;
        istep0 = 1;
        istep1 = planesize*C0_;
    }

    if (outlayout == DATA_LAYOUT_NCHW) {
        ostep = 1;
        ostep0 = planesize;
        ostep1 = planesize*C0_;
    } else if (outlayout == DATA_LAYOUT_NHWC) {
        ostep = C;
        ostep0 = 1;
        ostep1 = C0_;
    } else {
        ostep = C0_;
        ostep0 = 1;
        ostep1 = planesize*C0_;
    }

    const char* inptr0 = (const char*)inp.data;
    char* outptr0 = (char*)out.data;

    transform_layout_func_t transform_layout_func =
        esz == 1 ? transform_layout_8u :
        esz == 2 ? transform_layout_16u :
        esz == 4 ? transform_layout_32u :
        esz == 8 ? transform_layout_64u : nullptr;

    CV_Assert(transform_layout_func != nullptr);

    parallel_for_(Range(0, nblocks), [&](const Range& r) {
        int start = r.start*allplanes/nblocks;
        int end = r.end*allplanes/nblocks;
        int npix = 0;

        for (int ofs = start; ofs < end; ofs += npix) {
            int sample_idx = ofs/planesize;
            int rawofs = ofs - sample_idx*planesize;
            npix = std::min(planesize - rawofs, end - ofs);
            const char* inptr = inptr0 + (inplanesize_C*sample_idx + istep*rawofs)*esz;
            char* outptr = outptr0 + (outplanesize_C*sample_idx + ostep*rawofs)*esz;
            transform_layout_func(inptr, istep, istep0, istep1,
                                  outptr, ostep, ostep0, ostep1,
                                  npix, C0_, C1_, C);
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
        CV_Assert(err == 0.);
#endif
    }
};

Ptr<TransformLayoutLayer> TransformLayoutLayer::create(const LayerParams& params)
{
    return Ptr<TransformLayoutLayer>(new TransformLayoutLayerImpl(params));
}

}}
