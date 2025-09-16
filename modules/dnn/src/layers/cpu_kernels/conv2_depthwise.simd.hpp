// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

// === dispatched calls (implemented here)

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

cv::dnn::DepthwiseConvFunc getDepthwiseConvFunc_(int depth);

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
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = cs.inpshape.back(), C1_ = cs.inpshape[1];
    int NC = cs.inpshape[0]*C1_;

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.nspatialdims == 2);

    parallel_for_(Range(0, NC), [&](const Range& r)
    {
        int nc0 = r.start, nc1 = r.end;
        int nlanes = nlanes_, C0 = C0_, C1 = C1_;
        int Hi = cs.inpshape[2], Wi = cs.inpshape[3];
        int H = cs.outshape[2], W = cs.outshape[3];
        int iplanesize = Hi*Wi*C0;
        int planesize = H*W*C0;
        int SY = cs.strides[0], SX = cs.strides[1];
        int pad_y0 = cs.pads[0], pad_x0 = cs.pads[1];
        int inner_y0 = cs.inner[0], inner_y1 = cs.inner[1];
        int inner_x0 = cs.inner[cs.nspatialdims], inner_x1 = cs.inner[cs.nspatialdims + 1];
        int ksize = (int)(cs.coordtab.size()/2);
        const int* yxtab = cs.coordtab.data();
        const int* ofstab = cs.ofstab.data();
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        AutoBuffer<float> buf(C0*3);
        float* sum = buf.data();
        float* scale = sum + C0;
        float* bias = scale + C0;

        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;
        v_float32 v_maxval = vx_setall_f32(maxval);
        v_float32 v_alpha = vx_setall_f32(alpha);

        const float* inp = (const float*)inp__ + nc0*iplanesize;
        float* out = (float*)out__ + nc0*planesize;
        const float* residual = residual__ ? (const float*)residual__ + nc0*planesize : nullptr;
        v_float32 z = vx_setzero_f32();

        for (int nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            int n = nc / C1;
            int c0 = (nc - n*C1)*C0;
            const float* weights = (const float*)weights__ + (c0/C0)*ksize*C0;

            for (int c = 0; c < C0; c++) {
                scale[c] = scale_ ? scale_[c0 + c] : 1.f;
                bias[c] = bias_ ? bias_[c0 + c] : 0.f;
            }

            for (int y0 = 0; y0 < H; y0++, out += W*C0, residual += (residual ? W*C0 : 0)) {
                //int64_t x0 = 0, x1 = W;
                int x0 = 0, x1 = y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                int yi_ = y0*SY - pad_y0;
                for(;;) {
                    if (nlanes == C0) {
                        v_float32 sc0 = vx_load(scale), b0 = vx_load(bias);
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - pad_x0;
                            v_float32 s0 = z;
                            for (int k = 0; k < ksize; k++) {
                                int yi = yi_ + yxtab[k*2];
                                int xi = xi_ + yxtab[k*2+1];
                                v_float32 v0, w0;
                                if ((unsigned)yi >= (unsigned)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                    continue;
                                v0 = vx_load(inp + (yi*Wi + xi)*C0);
                                w0 = vx_load(weights + k*C0);
                                s0 = v_fma(v0, w0, s0);
                            }
                            s0 = v_fma(s0, sc0, b0);
                            if (residual)
                                s0 = v_add(s0, v_load(residual + x0*C0));
                            s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                            vx_store(out + x0*C0, s0);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*2) {
                                v_float32 s0 = z, s1 = z;
                                for (int k = 0; k < ksize; k++) {
                                    int yi = yi_ + yxtab[k*2];
                                    int xi = xi_ + yxtab[k*2+1];
                                    v_float32 v0, v1, w0, w1;
                                    if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                        continue;
                                    int ofs_k = (yi*Wi + xi)*C0 + c;
                                    int ofs_w = k*C0;
                                    v0 = vx_load(inp + ofs_k);
                                    v1 = vx_load(inp + ofs_k + nlanes);
                                    w0 = vx_load(weights + ofs_w);
                                    w1 = vx_load(weights + ofs_w + nlanes);
                                    s0 = v_fma(v0, w0, s0);
                                    s1 = v_fma(v1, w1, s1);
                                }
                                s0 = v_fma(s0, vx_load(scale + c), vx_load(bias + c));
                                s1 = v_fma(s1, vx_load(scale + c + nlanes), vx_load(bias + c + nlanes));
                                if (residual) {
                                    s0 = v_add(s0, v_load(residual + x0*C0 + c));
                                    s1 = v_add(s1, v_load(residual + x0*C0 + c + nlanes));
                                }
                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                                s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, v_alpha)), v_maxval);
                                vx_store(out + x0*C0 + c, s0);
                                vx_store(out + x0*C0 + c + nlanes, s1);
                            }
                        }
                    }
                    if (x0 == W)
                        break;
                    x1 = inner_x1;
                    if (nlanes == C0) {
                        v_float32 sc0 = vx_load(scale), b0 = vx_load(bias);
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - pad_x0;
                            const float* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            v_float32 s0 = z;
                            for (int k = 0; k < ksize; k++) {
                                v_float32 v0 = vx_load(inp_xi + ofstab[k]);
                                v_float32 w0 = vx_load(weights + k*C0);
                                s0 = v_fma(v0, w0, s0);
                            }
                            s0 = v_fma(s0, sc0, b0);
                            if (residual)
                                s0 = v_add(s0, v_load(residual + x0*C0));
                            s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                            vx_store(out + x0*C0, s0);
                        }
                    } else if (nlanes*2 == C0) {
                        v_float32 sc0 = vx_load(scale), sc1 = vx_load(scale + nlanes);
                        v_float32 b0 = vx_load(bias), b1 = vx_load(bias + nlanes);
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - pad_x0;
                            const float* inp_xi = inp + (Wi*yi_ + xi_)*C0;

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
                                s0 = v_add(s0, v_load(residual + x0*C0));
                                s1 = v_add(s1, v_load(residual + x0*C0 + nlanes));
                            }
                            s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                            s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, v_alpha)), v_maxval);
                            vx_store(out + x0*C0, s0);
                            vx_store(out + x0*C0 + nlanes, s1);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*4) {
                                const float* inp_xi = inp + (Wi*yi_ + xi_)*C0 + c;
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
                                s0 = v_fma(s0, vx_load(scale + c), vx_load(bias + c));
                                s1 = v_fma(s1, vx_load(scale + c + nlanes), vx_load(bias + c + nlanes));
                                s2 = v_fma(s2, vx_load(scale + c + nlanes*2), vx_load(bias + c + nlanes*2));
                                s3 = v_fma(s3, vx_load(scale + c + nlanes*3), vx_load(bias + c + nlanes*3));

                                if (residual) {
                                    s0 = v_add(s0, v_load(residual + x0*C0 + c));
                                    s1 = v_add(s1, v_load(residual + x0*C0 + c + nlanes));
                                    s2 = v_add(s2, v_load(residual + x0*C0 + c + nlanes*2));
                                    s3 = v_add(s3, v_load(residual + x0*C0 + c + nlanes*3));
                                }

                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                                s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, v_alpha)), v_maxval);
                                s2 = v_min(v_select(v_ge(s2, z), s2, v_mul(s2, v_alpha)), v_maxval);
                                s3 = v_min(v_select(v_ge(s3, z), s3, v_mul(s3, v_alpha)), v_maxval);
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

            if (activation) {
                activation(out - planesize, out - planesize, planesize, activParams);
            }
        }
    });
}

DepthwiseConvFunc getDepthwiseConvFunc_(int depth)
{
    DepthwiseConvFunc func = depth == CV_32F ? depthwiseConv32f : nullptr;
    return func;
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#endif
