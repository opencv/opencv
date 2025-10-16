// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include "conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace dnn
{

static void avgPool32f(const void* inp_, void* out_,
                       const ConvState& cs, bool count_include_pad_)
{
    constexpr int MAX_POOL_DIMS = ConvState::MAX_CONV_DIMS;
    int C0_ = cs.inpshape.back();
    int NC = cs.inpshape[0]*cs.inpshape[1];
    int nlanes_ = VTraits<v_float32>::vlanes();

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.nspatialdims <= MAX_POOL_DIMS && MAX_POOL_DIMS == 3);
    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.dims == cs.outshape.dims);

    parallel_for_(Range(0, NC), [&](const Range& r) {
        bool count_include_pad = count_include_pad_;
        int sdims = cs.nspatialdims;
        int nc0 = r.start, nc1 = r.end;
        int nlanes = nlanes_, C0 = cs.inpshape.back();
        int Di = sdims > 2 ? cs.inpshape[sdims - 1] : 1;
        int Hi = sdims > 1 ? cs.inpshape[sdims] : 1;
        int Wi = cs.inpshape[sdims + 1];
        int D = sdims > 2 ? cs.outshape[sdims - 1] : 1;
        int H = sdims > 1 ? cs.outshape[sdims] : 1;
        int W = cs.outshape[sdims + 1];
        int iplanesize = Di*Hi*Wi*C0;
        int planesize = D*H*W*C0;
        int SZ = cs.strides[0], SY = cs.strides[1], SX = cs.strides[2];
        int padZ0 = cs.pads[0], padY0 = cs.pads[1], padX0 = cs.pads[2];
        int inner_z0 = cs.inner[0], inner_z1 = cs.inner[MAX_POOL_DIMS];
        int inner_y0 = cs.inner[1], inner_y1 = cs.inner[MAX_POOL_DIMS + 1];
        int inner_x0 = cs.inner[2], inner_x1 = cs.inner[MAX_POOL_DIMS + 2];
        int ksize = (int)cs.ofstab.size();
        const int* zyxtab = cs.coordtab.data();
        const int* ofstab = cs.ofstab.data();

        const float* inp = (const float*)inp_ + nc0*iplanesize;
        float* out = (float*)out_ + nc0*planesize;
        v_float32 z = vx_setzero_f32();
        v_float32 vscale0 = vx_setall_f32(1.f/ksize);

        for (int nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int z0 = 0; z0 < D; z0++) {
                int zi_ = z0*SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++, out += W*C0) {
                    int x0 = 0;
                    int x1 = z0 >= inner_z0 && z0 < inner_z1 &&
                        y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                    int yi_ = y0*SY - padY0;
                    for(;;) {
                        if (nlanes == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                v_float32 s0 = z;
                                int nitems = 0;
                                for (int k = 0; k < ksize; k++) {
                                    int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                    int yi = yi_ + zyxtab[k*MAX_POOL_DIMS+1];
                                    int xi = xi_ + zyxtab[k*MAX_POOL_DIMS+2];
                                    v_float32 v0;
                                    if ((unsigned)zi >= (unsigned)Di ||
                                        (unsigned)yi >= (unsigned)Hi ||
                                        (unsigned)xi >= (unsigned)Wi)
                                        continue;
                                    v0 = vx_load(inp + ((zi*Hi + yi)*Wi + xi)*C0);
                                    s0 = v_add(s0, v0);
                                    nitems++;
                                }
                                s0 = v_mul(s0, count_include_pad ? vscale0 : vx_setall_f32(1.f/nitems));
                                vx_store(out + x0*C0, s0);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*2) {
                                    v_float32 s0 = z, s1 = z;
                                    int nitems = 0;
                                    for (int k = 0; k < ksize; k++) {
                                        int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                        int yi = yi_ + zyxtab[k*MAX_POOL_DIMS+1];
                                        int xi = xi_ + zyxtab[k*MAX_POOL_DIMS+2];
                                        v_float32 v0, v1;
                                        if ((unsigned)zi >= (unsigned)Di ||
                                            (unsigned)yi >= (unsigned)Hi ||
                                            (unsigned)xi >= (unsigned)Wi)
                                            continue;
                                        int ofs_k = ((zi*Hi + yi)*Wi + xi)*C0 + c;
                                        v0 = vx_load(inp + ofs_k);
                                        v1 = vx_load(inp + ofs_k + nlanes);
                                        s0 = v_add(s0, v0);
                                        s1 = v_add(s1, v1);
                                    }
                                    v_float32 vscale = count_include_pad ? vscale0 : vx_setall_f32(1.f/nitems);
                                    s0 = v_mul(s0, vscale);
                                    s1 = v_mul(s1, vscale);
                                    vx_store(out + x0*C0 + c, s0);
                                    vx_store(out + x0*C0 + c + nlanes, s1);
                                }
                            }
                        }
                        if (x0 == W)
                            break;
                        x1 = inner_x1;
                        if (nlanes == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const float* inp_xi = inp + ((Hi*zi_ + Wi)*yi_ + xi_)*C0;
                                
                                v_float32 s0 = vx_load(inp_xi + ofstab[0]);
                                for (int k = 1; k < ksize; k++)
                                    s0 = v_add(s0, vx_load(inp_xi + ofstab[k]));
                                vx_store(out + x0*C0, v_mul(s0, vscale0));
                            }
                        } else if (nlanes*2 == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const float* inp_xi = inp + ((Hi*zi_ + Wi)*yi_ + xi_)*C0;
                                
                                int ofs_k = ofstab[0];
                                v_float32 s0 = vx_load(inp_xi + ofs_k);
                                v_float32 s1 = vx_load(inp_xi + ofs_k + nlanes);
                                for (int k = 1; k < ksize; k++) {
                                    ofs_k = ofstab[k];
                                    s0 = v_add(s0, vx_load(inp_xi + ofs_k));
                                    s1 = v_add(s1, vx_load(inp_xi + ofs_k + nlanes));
                                }
                                s0 = v_mul(s0, vscale0);
                                s1 = v_mul(s1, vscale0);
                                vx_store(out + x0*C0, s0);
                                vx_store(out + x0*C0 + nlanes, s1);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*4) {
                                    const float* inp_xi = inp + ((Hi*zi_ + Wi)*yi_ + xi_)*C0;
                                    
                                    int ofs_k = ofstab[0];
                                    v_float32 s0 = vx_load(inp_xi + ofs_k);
                                    v_float32 s1 = vx_load(inp_xi + ofs_k + nlanes);
                                    v_float32 s2 = vx_load(inp_xi + ofs_k + nlanes*2);
                                    v_float32 s3 = vx_load(inp_xi + ofs_k + nlanes*3);
                                    for (int k = 1; k < ksize; k++) {
                                        ofs_k = ofstab[k];
                                        s0 = v_add(s0, vx_load(inp_xi + ofs_k));
                                        s1 = v_add(s1, vx_load(inp_xi + ofs_k + nlanes));
                                        s2 = v_add(s2, vx_load(inp_xi + ofs_k + nlanes*2));
                                        s3 = v_add(s3, vx_load(inp_xi + ofs_k + nlanes*3));
                                    }
                                    s0 = v_mul(s0, vscale0);
                                    s1 = v_mul(s1, vscale0);
                                    s2 = v_mul(s2, vscale0);
                                    s3 = v_mul(s3, vscale0);
                                    vx_store(out + x0*C0 + c, s0);
                                    vx_store(out + x0*C0 + c + nlanes, s1);
                                    vx_store(out + x0*C0 + c + nlanes*2, s2);
                                    vx_store(out + x0*C0 + c + nlanes*3, s3);
                                }
                            }
                        }
                        x1 = W;
                    }
                }
            }
        }
    });
}

template<typename _Tp>
static void avgPool16xf(const _Tp* inp_, _Tp* out_,
                        const ConvState& cs, bool count_include_pad_)
{
    constexpr int MAX_POOL_DIMS = ConvState::MAX_CONV_DIMS;
    int C0_ = cs.inpshape.back();
    int NC = cs.inpshape[0]*cs.inpshape[1];
    int nlanes_ = VTraits<v_float32>::vlanes();

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.nspatialdims <= MAX_POOL_DIMS && MAX_POOL_DIMS == 3);
    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.dims == cs.outshape.dims);

    parallel_for_(Range(0, NC), [&](const Range& r) {
        bool count_include_pad = count_include_pad_;
        int sdims = cs.nspatialdims;
        int nc0 = r.start, nc1 = r.end;
        int nlanes = nlanes_, C0 = cs.inpshape.back();
        int Di = sdims > 2 ? cs.inpshape[sdims - 1] : 1;
        int Hi = sdims > 1 ? cs.inpshape[sdims] : 1;
        int Wi = cs.inpshape[sdims + 1];
        int D = sdims > 2 ? cs.outshape[sdims - 1] : 1;
        int H = sdims > 1 ? cs.outshape[sdims] : 1;
        int W = cs.outshape[sdims + 1];
        int iplanesize = Di*Hi*Wi*C0;
        int planesize = D*H*W*C0;
        int SZ = cs.strides[0], SY = cs.strides[1], SX = cs.strides[2];
        int padZ0 = cs.pads[0], padY0 = cs.pads[1], padX0 = cs.pads[2];
        int inner_z0 = cs.inner[0], inner_z1 = cs.inner[MAX_POOL_DIMS];
        int inner_y0 = cs.inner[1], inner_y1 = cs.inner[MAX_POOL_DIMS + 1];
        int inner_x0 = cs.inner[2], inner_x1 = cs.inner[MAX_POOL_DIMS + 2];
        int ksize = (int)cs.ofstab.size();
        const int* zyxtab = cs.coordtab.data();
        const int* ofstab = cs.ofstab.data();

        const _Tp* inp = (const _Tp*)inp_ + nc0*iplanesize;
        _Tp* out = (_Tp*)out_ + nc0*planesize;
        v_float32 z = vx_setzero_f32();
        v_float32 vscale0 = vx_setall_f32(1.f/ksize);

        for (int nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int z0 = 0; z0 < D; z0++) {
                int zi_ = z0*SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++, out += W*C0) {
                    int x0 = 0;
                    int x1 = z0 >= inner_z0 && z0 < inner_z1 &&
                        y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                    int yi_ = y0*SY - padY0;
                    for(;;) {
                        if (nlanes == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                v_float32 s0 = z;
                                int nitems = 0;
                                for (int k = 0; k < ksize; k++) {
                                    int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                    int yi = yi_ + zyxtab[k*MAX_POOL_DIMS+1];
                                    int xi = xi_ + zyxtab[k*MAX_POOL_DIMS+2];
                                    v_float32 v0;
                                    if ((unsigned)zi >= (unsigned)Di ||
                                        (unsigned)yi >= (unsigned)Hi ||
                                        (unsigned)xi >= (unsigned)Wi)
                                        continue;
                                    v0 = vx_load_expand(inp + ((zi*Hi + yi)*Wi + xi)*C0);
                                    s0 = v_add(s0, v0);
                                    nitems++;
                                }
                                s0 = v_mul(s0, count_include_pad ? vscale0 : vx_setall_f32(1.f/nitems));
                                v_pack_store(out + x0*C0, s0);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*2) {
                                    v_float32 s0 = z, s1 = z;
                                    int nitems = 0;
                                    for (int k = 0; k < ksize; k++) {
                                        int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                        int yi = yi_ + zyxtab[k*MAX_POOL_DIMS+1];
                                        int xi = xi_ + zyxtab[k*MAX_POOL_DIMS+2];
                                        v_float32 v0, v1;
                                        if ((unsigned)zi >= (unsigned)Di ||
                                            (unsigned)yi >= (unsigned)Hi ||
                                            (unsigned)xi >= (unsigned)Wi)
                                            continue;
                                        int ofs_k = ((zi*Hi + yi)*Wi + xi)*C0 + c;
                                        v0 = vx_load_expand(inp + ofs_k);
                                        v1 = vx_load_expand(inp + ofs_k + nlanes);
                                        s0 = v_add(s0, v0);
                                        s1 = v_add(s1, v1);
                                    }
                                    v_float32 vscale = count_include_pad ? vscale0 : vx_setall_f32(1.f/nitems);
                                    s0 = v_mul(s0, vscale);
                                    s1 = v_mul(s1, vscale);
                                    v_pack_store(out + x0*C0 + c, s0);
                                    v_pack_store(out + x0*C0 + c + nlanes, s1);
                                }
                            }
                        }
                        if (x0 == W)
                            break;
                        x1 = inner_x1;
                        if (nlanes == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const _Tp* inp_xi = inp + ((Hi*zi_ + Wi)*yi_ + xi_)*C0;
                                
                                v_float32 s0 = vx_load_expand(inp_xi + ofstab[0]);
                                for (int k = 1; k < ksize; k++)
                                    s0 = v_add(s0, vx_load_expand(inp_xi + ofstab[k]));
                                v_pack_store(out + x0*C0, v_mul(s0, vscale0));
                            }
                        } else if (nlanes*2 == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const _Tp* inp_xi = inp + ((Hi*zi_ + Wi)*yi_ + xi_)*C0;
                                
                                int ofs_k = ofstab[0];
                                v_float32 s0 = vx_load_expand(inp_xi + ofs_k);
                                v_float32 s1 = vx_load_expand(inp_xi + ofs_k + nlanes);
                                for (int k = 1; k < ksize; k++) {
                                    ofs_k = ofstab[k];
                                    s0 = v_add(s0, vx_load_expand(inp_xi + ofs_k));
                                    s1 = v_add(s1, vx_load_expand(inp_xi + ofs_k + nlanes));
                                }
                                s0 = v_mul(s0, vscale0);
                                s1 = v_mul(s1, vscale0);
                                v_pack_store(out + x0*C0, s0);
                                v_pack_store(out + x0*C0 + nlanes, s1);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*4) {
                                    const _Tp* inp_xi = inp + ((Hi*zi_ + Wi)*yi_ + xi_)*C0;
                                    
                                    int ofs_k = ofstab[0];
                                    v_float32 s0 = vx_load_expand(inp_xi + ofs_k);
                                    v_float32 s1 = vx_load_expand(inp_xi + ofs_k + nlanes);
                                    v_float32 s2 = vx_load_expand(inp_xi + ofs_k + nlanes*2);
                                    v_float32 s3 = vx_load_expand(inp_xi + ofs_k + nlanes*3);
                                    for (int k = 1; k < ksize; k++) {
                                        ofs_k = ofstab[k];
                                        s0 = v_add(s0, vx_load_expand(inp_xi + ofs_k));
                                        s1 = v_add(s1, vx_load_expand(inp_xi + ofs_k + nlanes));
                                        s2 = v_add(s2, vx_load_expand(inp_xi + ofs_k + nlanes*2));
                                        s3 = v_add(s3, vx_load_expand(inp_xi + ofs_k + nlanes*3));
                                    }
                                    s0 = v_mul(s0, vscale0);
                                    s1 = v_mul(s1, vscale0);
                                    s2 = v_mul(s2, vscale0);
                                    s3 = v_mul(s3, vscale0);
                                    v_pack_store(out + x0*C0 + c, s0);
                                    v_pack_store(out + x0*C0 + c + nlanes, s1);
                                    v_pack_store(out + x0*C0 + c + nlanes*2, s2);
                                    v_pack_store(out + x0*C0 + c + nlanes*3, s3);
                                }
                            }
                        }
                        x1 = W;
                    }
                }
            }
        }
    });
}

static void avgPool16f(const void* inp_, void* out_,
                       const ConvState& cs, bool countIncludePadding)
{
    avgPool16xf((const hfloat*)inp_, (hfloat*)out_, cs, countIncludePadding);
}

static void avgPool16bf(const void* inp_, void* out_,
                        const ConvState& cs, bool countIncludePadding)
{
    avgPool16xf((const bfloat*)inp_, (bfloat*)out_, cs, countIncludePadding);
}

typedef void (*AvgPoolFunc)(const void* inp, void* out,
                            const ConvState& cs, bool countIncludePadding);

class AveragePoolLayerImpl : public AveragePoolLayer
{
public:
    AveragePoolLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        kernel_shape = params.getVector<int>("kernel_size");
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        ceil_mode = params.get<bool>("ceil_mode", false);
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "kernel_size: [";
        for (size_t k = 0; k < kernel_shape.size(); k++)
            strm << (k > 0 ? ", " : "") << kernel_shape[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "dilation: [";
        for (size_t k = 0; k < dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pad: [";
        for (size_t k = 0; k < pads.size(); k++)
            strm << (k > 0 ? ", " : "") << pads[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "stride: [";
        for (size_t k = 0; k < strides.size(); k++)
            strm << (k > 0 ? ", " : "") << strides[k];
        strm << "],\n";

        return strm;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape> &inputs,
                             const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        int ksize = 1;
        for (auto sz: kernel_shape) ksize *= sz;
        return (int64_t)(inputs[0].total()*ksize);
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

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape> &outshapes,
                                 std::vector<MatShape> &tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs == 1);

        outshapes.assign(1, convInferShape(inpshapes[0], MatShape(),
                                           kernel_shape, 0, strides, dilations,
                                           pads, auto_pad, ceil_mode));
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                    std::vector<DataLayout>& desiredInputs,
                    const int requiredOutputs,
                    std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        CV_Assert(actualInputs.size() == 1u);
        desiredInputs.assign(1, DATA_LAYOUT_BLOCK);
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        return getNetImpl(this)->defaultC0;
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
        MatShape outshape = convInferShape(inpshape, MatShape(),
                                           kernel_shape, 0, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        ConvState cs;
        cs.initPooling(inpshape, outshape, kernel_shape, strides,
                       dilations, pads, auto_pad, ceil_mode);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            runOp(inp, outs[0], cs);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, inptype);
            Mat temp(outshape, inptype);
            runOp(inp, temp, cs);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, Mat& out, const ConvState& cs)
    {
        int inptype = inp.type();
        AvgPoolFunc func =
            inptype == CV_32F ? avgPool32f :
            inptype == CV_16F ? avgPool16f :
            inptype == CV_16BF ? avgPool16bf : nullptr;

        CV_Assert(func != nullptr && "AveragePool: unsupported data type");
        func(inp.data, out.data, cs, count_include_pad);
    }
};

Ptr<AveragePoolLayer> AveragePoolLayer::create(const LayerParams& params)
{
    return Ptr<AveragePoolLayer>(new AveragePoolLayerImpl(params));
}

}}
