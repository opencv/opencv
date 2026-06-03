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

static void maxPool32f(const void* inp_, void* out_, const ConvState& cs)
{
    int NC1 = cs.inpshape[0]*cs.inpshape[1];

    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.dims == cs.outshape.dims);

    parallel_for_(Range(0, NC1), [&](const Range& r) {
        constexpr int MAX_POOL_DIMS = ConvState::MAX_CONV_DIMS;

        CV_Assert(cs.nspatialdims <= MAX_POOL_DIMS && MAX_POOL_DIMS == 3);

        int sdims = cs.nspatialdims;
        int nc0 = r.start, nc1 = r.end;
        int C0 = cs.inpshape.back();
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
        const float INITVAL = -FLT_MAX;

    #if CV_SIMD
        int nlanes = VTraits<v_float32>::vlanes();
        v_float32 s_min = vx_setall_f32(INITVAL);
        CV_Assert(C0 == nlanes || C0 == nlanes*2 || C0 % (nlanes*4) == 0);
    #endif

        for (int nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int z0 = 0; z0 < D; z0++) {
                int zi_ = z0*SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++, out += W*C0) {
                    int x0 = 0;
                    int x1 = z0 >= inner_z0 && z0 < inner_z1 &&
                        y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                    int yi_ = y0*SY - padY0;

                #if !(CV_SIMD)
                    for (int c = 0; c < C0*W; c++)
                        out[c] = INITVAL;
                #endif

                    for(;;) {
                    #if CV_SIMD
                        if (nlanes == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                v_float32 s0 = s_min;
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
                                    s0 = v_max(s0, v0);
                                }
                                vx_store(out + x0*C0, s0);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*2) {
                                    v_float32 s0 = s_min, s1 = s_min;
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
                                        s0 = v_max(s0, v0);
                                        s1 = v_max(s1, v1);
                                    }
                                    vx_store(out + x0*C0 + c, s0);
                                    vx_store(out + x0*C0 + c + nlanes, s1);
                                }
                            }
                        }
                    #else
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - padX0;
                            for (int k = 0; k < ksize; k++) {
                                int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                int yi = yi_ + zyxtab[k*MAX_POOL_DIMS+1];
                                int xi = xi_ + zyxtab[k*MAX_POOL_DIMS+2];
                                if ((unsigned)zi >= (unsigned)Di ||
                                    (unsigned)yi >= (unsigned)Hi ||
                                    (unsigned)xi >= (unsigned)Wi)
                                    continue;
                                const float* inptr = inp + ((zi*Hi + yi)*Wi + xi)*C0;
                                for (int c = 0; c < C0; c++)
                                    out[x0*C0 + c] = std::max(out[x0*C0 + c], inptr[c]);
                            }
                        }
                    #endif
                        if (x0 == W)
                            break;
                        x1 = inner_x1;

                    #if CV_SIMD
                        if (nlanes == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                v_float32 s0 = vx_load(inp_xi + ofstab[0]);
                                for (int k = 1; k < ksize; k++)
                                    s0 = v_max(s0, vx_load(inp_xi + ofstab[k]));
                                vx_store(out + x0*C0, s0);
                            }
                        } else if (nlanes*2 == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                int ofs_k = ofstab[0];
                                v_float32 s0 = vx_load(inp_xi + ofs_k);
                                v_float32 s1 = vx_load(inp_xi + ofs_k + nlanes);
                                for (int k = 1; k < ksize; k++) {
                                    ofs_k = ofstab[k];
                                    s0 = v_max(s0, vx_load(inp_xi + ofs_k));
                                    s1 = v_max(s1, vx_load(inp_xi + ofs_k + nlanes));
                                }
                                vx_store(out + x0*C0, s0);
                                vx_store(out + x0*C0 + nlanes, s1);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*4) {
                                    const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0 + c;

                                    int ofs_k = ofstab[0];
                                    v_float32 s0 = vx_load(inp_xi + ofs_k);
                                    v_float32 s1 = vx_load(inp_xi + ofs_k + nlanes);
                                    v_float32 s2 = vx_load(inp_xi + ofs_k + nlanes*2);
                                    v_float32 s3 = vx_load(inp_xi + ofs_k + nlanes*3);
                                    for (int k = 1; k < ksize; k++) {
                                        ofs_k = ofstab[k];
                                        s0 = v_max(s0, vx_load(inp_xi + ofs_k));
                                        s1 = v_max(s1, vx_load(inp_xi + ofs_k + nlanes));
                                        s2 = v_max(s2, vx_load(inp_xi + ofs_k + nlanes*2));
                                        s3 = v_max(s3, vx_load(inp_xi + ofs_k + nlanes*3));
                                    }
                                    vx_store(out + x0*C0 + c, s0);
                                    vx_store(out + x0*C0 + c + nlanes, s1);
                                    vx_store(out + x0*C0 + c + nlanes*2, s2);
                                    vx_store(out + x0*C0 + c + nlanes*3, s3);
                                }
                            }
                        }
                    #else
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - padX0;
                            const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;
                            for (int k = 0; k < ksize; k++) {
                                const float* inptr = inp_xi + ofstab[k];
                                for (int c = 0; c < C0; c++)
                                    out[x0*C0 + c] = std::max(out[x0*C0 + c], inptr[c]);
                            }
                        }
                    #endif
                        x1 = W;
                    }
                }
            }
        }
    });
}

// temporarily exclude fp16/bf16 versions,
// since convolution and other layers don't support those types yet
#if 0
template<typename _Tp>
static void maxPool16xf(const _Tp* inp_, _Tp* out_, const ConvState& cs)
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
        v_float32 s_min = vx_setall_f32(-FLT_MAX);

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
                                v_float32 s0 = s_min;
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
                                    s0 = v_max(s0, v0);
                                }
                                v_pack_store(out + x0*C0, s0);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*2) {
                                    v_float32 s0 = s_min, s1 = s_min;
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
                                        s0 = v_max(s0, v0);
                                        s1 = v_max(s1, v1);
                                    }
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
                                const _Tp* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                v_float32 s0 = vx_load_expand(inp_xi + ofstab[0]);
                                for (int k = 1; k < ksize; k++)
                                    s0 = v_max(s0, vx_load_expand(inp_xi + ofstab[k]));
                                v_pack_store(out + x0*C0, s0);
                            }
                        } else if (nlanes*2 == C0) {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const _Tp* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                int ofs_k = ofstab[0];
                                v_float32 s0 = vx_load_expand(inp_xi + ofs_k);
                                v_float32 s1 = vx_load_expand(inp_xi + ofs_k + nlanes);
                                for (int k = 1; k < ksize; k++) {
                                    ofs_k = ofstab[k];
                                    s0 = v_max(s0, vx_load_expand(inp_xi + ofs_k));
                                    s1 = v_max(s1, vx_load_expand(inp_xi + ofs_k + nlanes));
                                }
                                v_pack_store(out + x0*C0, s0);
                                v_pack_store(out + x0*C0 + nlanes, s1);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*4) {
                                    const _Tp* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                    int ofs_k = ofstab[0];
                                    v_float32 s0 = vx_load_expand(inp_xi + ofs_k);
                                    v_float32 s1 = vx_load_expand(inp_xi + ofs_k + nlanes);
                                    v_float32 s2 = vx_load_expand(inp_xi + ofs_k + nlanes*2);
                                    v_float32 s3 = vx_load_expand(inp_xi + ofs_k + nlanes*3);
                                    for (int k = 1; k < ksize; k++) {
                                        ofs_k = ofstab[k];
                                        s0 = v_max(s0, vx_load_expand(inp_xi + ofs_k));
                                        s1 = v_max(s1, vx_load_expand(inp_xi + ofs_k + nlanes));
                                        s2 = v_max(s2, vx_load_expand(inp_xi + ofs_k + nlanes*2));
                                        s3 = v_max(s3, vx_load_expand(inp_xi + ofs_k + nlanes*3));
                                    }
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

static void maxPool16f(const void* inp_, void* out_, const ConvState& cs)
{
    maxPool16xf((const hfloat*)inp_, (hfloat*)out_, cs);
}

static void maxPool16bf(const void* inp_, void* out_, const ConvState& cs)
{
    maxPool16xf((const bfloat*)inp_, (bfloat*)out_, cs);
}
#endif

typedef void (*MaxPoolFunc)(const void* inp, void* out, const ConvState& cs);

// 2-output (values + ONNX-style int64 indices) NCHW scalar implementation.
static void maxPool32f_nchw_with_indices(const float* inp, float* out, int64_t* outIdx,
                                         int N, int C, int Hi, int Wi, int H, int W,
                                         int kH, int kW, int sH, int sW,
                                         int padH, int padW, int dilH, int dilW)
{
    int NC = N * C;
    int inHW = Hi * Wi;
    int outHW = H * W;
    parallel_for_(Range(0, NC), [&](const Range& r) {
        for (int nc = r.start; nc < r.end; nc++) {
            int c = nc % C;
            const float* inp_nc = inp + nc * inHW;
            float*       out_nc = out + nc * outHW;
            int64_t*     idx_nc = outIdx + nc * outHW;
            for (int yo = 0; yo < H; yo++) {
                for (int xo = 0; xo < W; xo++) {
                    float vmax = -FLT_MAX;
                    int64_t idxmax = -1;
                    for (int ky = 0; ky < kH; ky++) {
                        int yi = yo * sH - padH + ky * dilH;
                        if ((unsigned)yi >= (unsigned)Hi)
                            continue;
                        for (int kx = 0; kx < kW; kx++) {
                            int xi = xo * sW - padW + kx * dilW;
                            if ((unsigned)xi >= (unsigned)Wi)
                                continue;
                            float v = inp_nc[yi * Wi + xi];
                            if (v > vmax) {
                                vmax = v;
                                // ONNX storage_order=0: index = (c*Hi + yi)*Wi + xi
                                idxmax = (int64_t)(c * Hi + yi) * Wi + xi;
                            }
                        }
                    }
                    if (idxmax < 0) {
                        vmax = 0.f;
                        idxmax = (int64_t)(c * Hi + 0) * Wi + 0;
                    }
                    out_nc[yo * W + xo] = vmax;
                    idx_nc[yo * W + xo] = idxmax;
                }
            }
        }
    });
}

class MaxPoolLayerImpl : public MaxPoolLayer
{
public:
    MaxPoolLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        kernel_shape = params.getVector<int>("kernel_size");
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        ceil_mode = params.get<bool>("ceil_mode", false);
        storage_order = params.get<int>("storage_order", 0);
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

        if (outputs.size() > 1u) {
            prindent(strm, indent);
            strm << "storage_order: " << storage_order << ",\n";
        }
        return strm;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
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

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs == 1);

        outtypes.clear();
        outtypes.push_back(inferType(inptypes[0]));
        if (outputs.size() == 2u)
            outtypes.push_back(CV_64S); // ONNX MaxPool indices
        temptypes.clear();
    }

    bool isDataShuffling() const CV_OVERRIDE { return true; }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape> &outshapes,
                                 std::vector<MatShape> &tempshapes) const CV_OVERRIDE
    {
        CV_Assert(outputs.size() == 1u || outputs.size() == 2u);
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs == 1);

        MatShape outshape = convInferShape(inpshapes[0], MatShape(),
                                           kernel_shape, 0, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        outshapes.assign(outputs.size(), outshape);
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                    std::vector<DataLayout>& desiredInputs,
                    const int requiredOutputs,
                    std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        CV_Assert(actualInputs.size() == 1u);

        const bool wantsIndices = requiredOutputs == 2;
        const DataLayout layout = wantsIndices ? DATA_LAYOUT_NCHW : DATA_LAYOUT_BLOCK;
        desiredInputs.assign(1, layout);
        outputs.assign(requiredOutputs, layout);
        return wantsIndices ? 0 : getNetImpl(this)->defaultC0;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_Assert(inputs_arr.total() == 1);
        const int inptype = inputs_arr.type(0);
        const MatShape inpshape = inputs_arr.shape(0);

        const int outKind = outputs_arr.kind();
        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        const size_t noutputs = outputs.size();
        CV_Assert(noutputs == 1u || noutputs == 2u);
        const bool wantsIndices = (noutputs == 2u);

        if (wantsIndices) {
            CV_Assert(inptype == CV_32F && "MaxPool with indices currently supports CV_32F only");
            CV_Assert(inpshape.dims == 4 && "MaxPool with indices: only 4D (N,C,H,W) inputs supported");
            CV_Assert(inpshape.layout != DATA_LAYOUT_BLOCK &&
                      "MaxPool with indices does not run on a BLOCK-layout input");
        }

        const MatShape outshape = convInferShape(inpshape, MatShape(),
                                                 kernel_shape, 0, strides, dilations,
                                                 pads, auto_pad, ceil_mode);
        Mat inp = inputs_arr.getMat(0);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(noutputs);
            outs[0].fit(outshape, inptype);
            if (wantsIndices)
                outs[1].fit(outshape, CV_64S);
            runForward(inp, outs, inpshape, outshape);
        } else {
            // [TODO] more efficient OpenCL implementation
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(noutputs);
            outs[0].fit(outshape, inptype);
            if (wantsIndices)
                outs[1].fit(outshape, CV_64S);

            std::vector<Mat> tmp(noutputs);
            tmp[0].create(outshape, inptype);
            if (wantsIndices)
                tmp[1].create(outshape, CV_64S);
            runForward(inp, tmp, inpshape, outshape);
            for (size_t i = 0; i < noutputs; ++i)
                tmp[i].copyTo(outs[i]);
        }
    }

    void runForward(const Mat& inp, std::vector<Mat>& outs,
                    const MatShape& inpshape, const MatShape& outshape)
    {
        if (outs.size() == 1u) {
            ConvState cs;
            cs.initPooling(inpshape, outshape, kernel_shape, strides,
                           dilations, pads, auto_pad, ceil_mode);
            runOp(inp, outs[0], cs);
        } else {
            runOpWithIndices(inp, outs[0], outs[1]);
        }
    }

    void runOpWithIndices(const Mat& inp, Mat& outVal, Mat& outIdx)
    {
        const int N = inp.size[0], C = inp.size[1], Hi = inp.size[2], Wi = inp.size[3];
        const int H = outVal.size[2], W = outVal.size[3];
        const int kH = (int)kernel_shape[0];
        const int kW = (int)kernel_shape[1];
        const int sH   = strides.size() > 0 ? (int)strides[0] : 1;
        const int sW   = strides.size() > 1 ? (int)strides[1] : sH;
        const int dilH = dilations.size() > 0 ? (int)dilations[0] : 1;
        const int dilW = dilations.size() > 1 ? (int)dilations[1] : dilH;
        const int padH = pads.size() > 0 ? (int)pads[0] : 0;
        const int padW = pads.size() > 1 ? (int)pads[1] : padH;
        maxPool32f_nchw_with_indices(
            inp.ptr<float>(), outVal.ptr<float>(), outIdx.ptr<int64_t>(),
            N, C, Hi, Wi, H, W, kH, kW, sH, sW, padH, padW, dilH, dilW);
    }

    void runOp(const Mat& inp, Mat& out, const ConvState& cs)
    {
        int inptype = inp.type();
        MaxPoolFunc func =
            inptype == CV_32F ? maxPool32f :
            /*inptype == CV_16F ? maxPool16f :
            inptype == CV_16BF ? maxPool16bf :*/
            nullptr;

        CV_Assert(func != nullptr && "MaxPool: unsupported data type");
        func(inp.data, out.data, cs);
    }
};

Ptr<MaxPoolLayer> MaxPoolLayer::create(const LayerParams& params)
{
    return Ptr<MaxPoolLayer>(new MaxPoolLayerImpl(params));
}

}}
