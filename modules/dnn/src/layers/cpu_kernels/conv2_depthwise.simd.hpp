// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

// === dispatched calls (implemented here)

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

cv::dnn::ConvFunc getDepthwiseConvFunc_(int depth);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

// === implementation

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

static void depthwiseConv32f(const void* inp__, const void* residual__,
                             void* out__, const ConvState& cs,
                             const void* weights__, const float* scale__,
                             const float* bias__)
{
    int NC1 = cs.inpshape[0]*cs.inpshape[1];

    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.inpshape.dims == cs.outshape.dims);

    parallel_for_(Range(0, NC1), [&](const Range& range)
    {
        constexpr int MAX_CONV_DIMS = ConvState::MAX_CONV_DIMS;
        constexpr int C0 = 8;

        CV_Assert(cs.nspatialdims <= MAX_CONV_DIMS && MAX_CONV_DIMS == 3);
        CV_Assert(C0 == cs.inpshape.back());

        int sdims = cs.nspatialdims;
        int C = cs.inpshape.C;
        int C1 = cs.inpshape[1];
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
        int inner_z0 = cs.inner[0], inner_z1 = cs.inner[MAX_CONV_DIMS];
        int inner_y0 = cs.inner[1], inner_y1 = cs.inner[MAX_CONV_DIMS + 1];
        int inner_x0 = cs.inner[2], inner_x1 = cs.inner[MAX_CONV_DIMS + 2];
        int ksize = (int)cs.ofstab.size();
        const int* zyxtab = cs.coordtab.data();
        const int* ofstab = cs.ofstab.data();
        const float* scale_ = scale__;
        const float* bias_ = bias__;

        const float* inp = (const float*)inp__ + range.start*iplanesize;
        float* out = (float*)out__ + range.start*planesize;
        const float* residual = residual__ ? (const float*)residual__ + range.start*planesize : nullptr;

        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams.data();
        ActivationFunc activation = cs.activation;
        float maxval = FLT_MAX, defaultAlpha = 0.f;
        float scalebuf[C0], biasbuf[C0], alphabuf[C0];
        if (fastActivation == FAST_ACTIV_CLIP) {
            CV_Assert(cs.activParams.size() == 2u);
            maxval = activParams[1];
        } else if (fastActivation == FAST_ACTIV_RELU) {
            CV_Assert(!activParams);
        } else if (fastActivation == FAST_ACTIV_LEAKY_RELU) {
            CV_Assert(cs.activParams.size() == 1u);
            defaultAlpha = activParams[0];
        } else if (fastActivation == FAST_ACTIV_PRELU) {
            CV_Assert(cs.activParams.size() == size_t(C));
        } else {
            // FAST_ACTIV_NONE: activation (if any) is handled via function pointer
            defaultAlpha = 1.f;
        }

    #if CV_SIMD || CV_SIMD_SCALABLE
        v_float32 v_maxval = vx_setall_f32(maxval);
        v_float32 z = vx_setzero_f32();
        const int nlanes = VTraits<v_float32>::vlanes();
        CV_Assert(C0 == nlanes || C0 == nlanes*2 || C0 % (nlanes*4) == 0);
    #endif

        for (int nc1 = range.start; nc1 < range.end; nc1++, inp += iplanesize) {
            int n = nc1 / C1;
            int c_base = (nc1 - n*C1)*C0;
            int c_count = std::min(C0, C - c_base);
            const float* weights = (const float*)weights__ + (c_base/C0)*ksize*C0;

            {
                int c = 0;
                for (; c < c_count; c++) {
                    scalebuf[c] = scale_ ? scale_[c_base + c] : 1.f;
                    biasbuf[c] = bias_ ? bias_[c_base + c] : 0.f;
                    alphabuf[c] = fastActivation == FAST_ACTIV_PRELU ? activParams[c_base + c] : defaultAlpha;
                }
                for (; c < C0; c++) {
                    scalebuf[c] = 0.f;
                    biasbuf[c] = 0.f;
                    alphabuf[c] = 0.f;
                }
            }

            for (int z0 = 0; z0 < D; z0++) {
                int zi_ = z0*SZ - padZ0;
                for (int y0 = 0; y0 < H; y0++, out += W*C0,
                     residual += (residual ? W*C0 : 0)) {
                    //int64_t x0 = 0, x1 = W;
                    int x0 = 0;
                    int x1 = z0 >= inner_z0 && z0 < inner_z1 &&
                        y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                    int yi_ = y0*SY - padY0;

                #if !(CV_SIMD || CV_SIMD_SCALABLE)
                    memset(out, 0, W*C0*sizeof(out[0]));
                #endif

                    for(;;) {
                    #if CV_SIMD || CV_SIMD_SCALABLE
                        if (nlanes == C0) {
                            v_float32 sc0 = vx_load(scalebuf), b0 = vx_load(biasbuf);
                            v_float32 alpha0 = vx_load(alphabuf);
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                v_float32 s0 = z;
                                for (int k = 0; k < ksize; k++) {
                                    int zi = zi_ + zyxtab[k*MAX_CONV_DIMS];
                                    int yi = yi_ + zyxtab[k*MAX_CONV_DIMS + 1];
                                    int xi = xi_ + zyxtab[k*MAX_CONV_DIMS + 2];
                                    v_float32 v0, w0;
                                    if ((unsigned)zi >= (unsigned)Di ||
                                        (unsigned)yi >= (unsigned)Hi ||
                                        (unsigned)xi >= (unsigned)Wi)
                                        continue;
                                    v0 = vx_load(inp + ((zi*Hi + yi)*Wi + xi)*C0);
                                    w0 = vx_load(weights + k*C0);
                                    s0 = v_fma(v0, w0, s0);
                                }
                                s0 = v_fma(s0, sc0, b0);
                                if (residual)
                                    s0 = v_add(s0, vx_load(residual + x0*C0));
                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, alpha0)), v_maxval);
                                vx_store(out + x0*C0, s0);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*2) {
                                    v_float32 s0 = z, s1 = z;
                                    for (int k = 0; k < ksize; k++) {
                                        int zi = zi_ + zyxtab[k*MAX_CONV_DIMS];
                                        int yi = yi_ + zyxtab[k*MAX_CONV_DIMS + 1];
                                        int xi = xi_ + zyxtab[k*MAX_CONV_DIMS + 2];
                                        v_float32 v0, v1, w0, w1;
                                        if ((unsigned)zi >= (unsigned)Di ||
                                            (unsigned)yi >= (unsigned)Hi ||
                                            (unsigned)xi >= (unsigned)Wi)
                                            continue;
                                        int ofs_k = ((zi*Hi + yi)*Wi + xi)*C0 + c;
                                        int ofs_w = k*C0;
                                        v0 = vx_load(inp + ofs_k);
                                        v1 = vx_load(inp + ofs_k + nlanes);
                                        w0 = vx_load(weights + ofs_w);
                                        w1 = vx_load(weights + ofs_w + nlanes);
                                        s0 = v_fma(v0, w0, s0);
                                        s1 = v_fma(v1, w1, s1);
                                    }
                                    s0 = v_fma(s0, vx_load(scalebuf + c), vx_load(biasbuf + c));
                                    s1 = v_fma(s1, vx_load(scalebuf + c + nlanes), vx_load(biasbuf + c + nlanes));
                                    if (residual) {
                                        s0 = v_add(s0, vx_load(residual + x0*C0 + c));
                                        s1 = v_add(s1, vx_load(residual + x0*C0 + c + nlanes));
                                    }
                                    v_float32 alpha0 = vx_load(alphabuf + c);
                                    v_float32 alpha1 = vx_load(alphabuf + c + nlanes);

                                    s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, alpha0)), v_maxval);
                                    s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, alpha1)), v_maxval);
                                    vx_store(out + x0*C0 + c, s0);
                                    vx_store(out + x0*C0 + c + nlanes, s1);
                                }
                            }
                        }
                    #else
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - padX0;
                            for (int k = 0; k < ksize; k++) {
                                int zi = zi_ + zyxtab[k*MAX_CONV_DIMS];
                                int yi = yi_ + zyxtab[k*MAX_CONV_DIMS + 1];
                                int xi = xi_ + zyxtab[k*MAX_CONV_DIMS + 2];
                                if ((unsigned)zi >= (unsigned)Di ||
                                    (unsigned)yi >= (unsigned)Hi ||
                                    (unsigned)xi >= (unsigned)Wi)
                                    continue;
                                const float* inptr = inp + ((zi*Hi + yi)*Wi + xi)*C0;
                                for (int c = 0; c < C0; c++)
                                    out[x0*C0 + c] += inptr[c]*weights[k*C0 + c];
                            }
                        }
                    #endif

                        if (x0 == W)
                            break;
                        x1 = inner_x1;

                    #if CV_SIMD || CV_SIMD_SCALABLE
                        if (nlanes == C0) {
                            v_float32 sc0 = vx_load(scalebuf), b0 = vx_load(biasbuf), alpha0 = vx_load(alphabuf);
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                v_float32 s0 = z;
                                for (int k = 0; k < ksize; k++) {
                                    v_float32 v0 = vx_load(inp_xi + ofstab[k]);
                                    v_float32 w0 = vx_load(weights + k*C0);
                                    s0 = v_fma(v0, w0, s0);
                                }
                                s0 = v_fma(s0, sc0, b0);
                                if (residual)
                                    s0 = v_add(s0, vx_load(residual + x0*C0));
                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, alpha0)), v_maxval);
                                vx_store(out + x0*C0, s0);
                            }
                        } else if (nlanes*2 == C0) {
                            v_float32 sc0 = vx_load(scalebuf), sc1 = vx_load(scalebuf + nlanes);
                            v_float32 b0 = vx_load(biasbuf), b1 = vx_load(biasbuf + nlanes);
                            v_float32 alpha0 = vx_load(alphabuf), alpha1 = vx_load(alphabuf + nlanes);
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0;

                                v_float32 s0 = z, s1 = z;
                                for (int k = 0; k < ksize; k++) {
                                    int ofs_k = ofstab[k], ofs_w = k*C0;
                                    v_float32 v0 = vx_load(inp_xi + ofs_k);
                                    v_float32 v1 = vx_load(inp_xi + ofs_k + nlanes);
                                    v_float32 w0 = vx_load(weights + ofs_w);
                                    v_float32 w1 = vx_load(weights + ofs_w + nlanes);
                                    s0 = v_fma(v0, w0, s0);
                                    s1 = v_fma(v1, w1, s1);
                                }
                                s0 = v_fma(s0, sc0, b0);
                                s1 = v_fma(s1, sc1, b1);
                                if (residual) {
                                    s0 = v_add(s0, vx_load(residual + x0*C0));
                                    s1 = v_add(s1, vx_load(residual + x0*C0 + nlanes));
                                }
                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, alpha0)), v_maxval);
                                s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, alpha1)), v_maxval);
                                vx_store(out + x0*C0, s0);
                                vx_store(out + x0*C0 + nlanes, s1);
                            }
                        } else {
                            for (; x0 < x1; x0++) {
                                int xi_ = x0*SX - padX0;
                                for (int c = 0; c < C0; c += nlanes*4) {
                                    const float* inp_xi = inp + ((Hi*zi_ + yi_)*Wi + xi_)*C0 + c;
                                    v_float32 s0 = z, s1 = z, s2 = z, s3 = z;
                                    for (int k = 0; k < ksize; k++) {
                                        int ofs_k = ofstab[k], ofs_w = k*C0 + c;
                                        v_float32 v0 = vx_load(inp_xi + ofs_k);
                                        v_float32 v1 = vx_load(inp_xi + ofs_k + nlanes);
                                        v_float32 v2 = vx_load(inp_xi + ofs_k + nlanes*2);
                                        v_float32 v3 = vx_load(inp_xi + ofs_k + nlanes*3);
                                        v_float32 w0 = vx_load(weights + ofs_w);
                                        v_float32 w1 = vx_load(weights + ofs_w + nlanes);
                                        v_float32 w2 = vx_load(weights + ofs_w + nlanes*2);
                                        v_float32 w3 = vx_load(weights + ofs_w + nlanes*3);
                                        s0 = v_fma(v0, w0, s0);
                                        s1 = v_fma(v1, w1, s1);
                                        s2 = v_fma(v2, w2, s2);
                                        s3 = v_fma(v3, w3, s3);
                                    }
                                    s0 = v_fma(s0, vx_load(scalebuf + c),
                                               vx_load(biasbuf + c));
                                    s1 = v_fma(s1, vx_load(scalebuf + c + nlanes),
                                               vx_load(biasbuf + c + nlanes));
                                    s2 = v_fma(s2, vx_load(scalebuf + c + nlanes*2),
                                               vx_load(biasbuf + c + nlanes*2));
                                    s3 = v_fma(s3, vx_load(scalebuf + c + nlanes*3),
                                               vx_load(biasbuf + c + nlanes*3));

                                    if (residual) {
                                        s0 = v_add(s0, vx_load(residual + x0*C0 + c));
                                        s1 = v_add(s1, vx_load(residual + x0*C0 + c + nlanes));
                                        s2 = v_add(s2, vx_load(residual + x0*C0 + c + nlanes*2));
                                        s3 = v_add(s3, vx_load(residual + x0*C0 + c + nlanes*3));
                                    }

                                    v_float32 alpha0 = vx_load(alphabuf + c);
                                    v_float32 alpha1 = vx_load(alphabuf + c + nlanes);
                                    v_float32 alpha2 = vx_load(alphabuf + c + nlanes*2);
                                    v_float32 alpha3 = vx_load(alphabuf + c + nlanes*3);

                                    s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, alpha0)), v_maxval);
                                    s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, alpha1)), v_maxval);
                                    s2 = v_min(v_select(v_ge(s2, z), s2, v_mul(s2, alpha2)), v_maxval);
                                    s3 = v_min(v_select(v_ge(s3, z), s3, v_mul(s3, alpha3)), v_maxval);
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
                                    out[x0*C0 + c] += inptr[c]*weights[k*C0 + c];
                            }
                        }
                    #endif

                        x1 = W;
                    }

                #if !(CV_SIMD || CV_SIMD_SCALABLE)
                    if (residual) {
                        for (int x = 0; x < W*C0; x += C0) {
                            for (int c = 0; c < C0; c++) {
                                float v = out[x + c]*scalebuf[c] + biasbuf[c] + residual[x + c];
                                v = std::min(v*(v >= 0 ? 1.f : alphabuf[c]), maxval);
                                out[x + c] = v;
                            }
                        }
                    } else {
                        for (int x = 0; x < W*C0; x += C0) {
                            for (int c = 0; c < C0; c++) {
                                float v = out[x + c]*scalebuf[c] + biasbuf[c];
                                v = std::min(v*(v >= 0 ? 1.f : alphabuf[c]), maxval);
                                out[x + c] = v;
                            }
                        }
                    }
                #endif
                }

                if (activation) {
                    activation(out - planesize, out - planesize, planesize, activParams);
                }
            }
        }
    });
}

ConvFunc getDepthwiseConvFunc_(int depth)
{
    ConvFunc func = depth == CV_32F ? depthwiseConv32f : nullptr;
    return func;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#endif
