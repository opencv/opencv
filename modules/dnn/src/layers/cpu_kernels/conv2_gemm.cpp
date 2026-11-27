// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Ported from the im2col + MLAS path of #30291, with the core GEMM engine as the GEMM.

#include "../../precomp.hpp"
#include "conv2_gemm.hpp"
#include "opencv2/core/hal/intrin.hpp"

#if CV_NEON && CV_NEON_AARCH64

#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/private.hpp>
#include <opencv2/core/utils/configuration.private.hpp>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>

namespace cv { namespace dnn {

Conv2GemmMode conv2GemmMode()
{
    const size_t mode = utils::getConfigurationParameterSizeT("OPENCV_DNN_CONV_GEMM", 2);
    if (mode == 0)
        return CONV2_GEMM_OFF;
    if (mode == 1)
        return CONV2_GEMM_FORCED;
    static const bool sme = hal::gemm32fUsesSME();
    return sme && gemmEngineEnabled() ? CONV2_GEMM_AUTO : CONV2_GEMM_OFF;
}

void conv2GemmPackWeights(const Mat& wpack, const MatShape& wshape0, int C0, EngineGemmB& packed)
{
    packed.release();
    CV_Assert(wpack.type() == CV_32F && wpack.isContinuous() && wshape0.dims == 4);
    const int Cout = wshape0[0], Cin = wshape0[1];
    const int ksize = wshape0[2] * wshape0[3];
    const int Kdim = ksize * Cin;

    const ConvWeightPack pack = ConvWeightPack::forConv(wshape0, wpack.shape(), 1, C0);
    const size_t tapStride = pack.tapStride();
    const float* src = wpack.ptr<float>();
    Mat w2d(Cout, Kdim, CV_32F);
    parallel_for_(Range(0, Cout), [&](const Range& r) {
        for (int co = r.start; co < r.end; co++) {
            float* row = w2d.ptr<float>(co);
            for (int ci = 0; ci < Cin; ci++) {
                const float* w = src + pack.offset(co, ci);
                for (int t = 0; t < ksize; t++)
                    row[(size_t)t * Cin + ci] = w[t * tapStride];
            }
        }
    });
    packed.pack(w2d.ptr<float>(), w2d.step[0], Cout, Kdim, true);
}

// im2col rows [m0, m1) for output pixels starting at p0; padded taps are zero.
static void im2colNCHWc8(const float* inp, int C1, int H, int W, int Cin,
                         int Wo, int kh, int kw, int sy, int sx, int dy, int dx,
                         int pt, int pl, int p0, int m0, int m1, float* A, int Kdim)
{
    const size_t plane = (size_t)H * W * 8;
    const int fullC1 = Cin / 8, tail = Cin - fullC1 * 8;
    for (int mi = m0; mi < m1; mi++) {
        const int p = p0 + mi;
        const int y = p / Wo, x = p - y * Wo;
        const int iy0 = y * sy - pt, ix0 = x * sx - pl;
        float* row = A + (size_t)mi * Kdim;
        for (int ky = 0; ky < kh; ky++) {
            const int iy = iy0 + ky * dy;
            const bool yin = (unsigned)iy < (unsigned)H;
            for (int kx = 0; kx < kw; kx++) {
                const int ix = ix0 + kx * dx;
                float* dst = row + (size_t)(ky * kw + kx) * Cin;
                if (!yin || (unsigned)ix >= (unsigned)W) {
                    std::memset(dst, 0, (size_t)Cin * sizeof(float));
                    continue;
                }
                const float* s = inp + ((size_t)iy * W + ix) * 8;
                int c1 = 0;
                for (; c1 < fullC1; c1++, s += plane, dst += 8)
                    std::memcpy(dst, s, 8 * sizeof(float));
                if (tail)
                    std::memcpy(dst, s, (size_t)tail * sizeof(float));
            }
        }
    }
}

namespace {

// Output step, same as the Conv2 kernels (setupActivation/fillCoeffBufs in conv2_common.hpp).
struct ConvEpilogue
{
    const float* scale;
    const float* bias;
    FastActivation fastAct;
    const float* activParams;
    ActivationFunc activation;
    float maxval, defaultAlpha;
    int Cout;

    // Writes rows [m0, m1) of C for channel block k1 into the NCHWc8 output.
    void run(const float* C, int k1, int m0, int m1, float* dst, const float* res) const
    {
        const int kbase = k1 * 8, nvalid = std::min(8, Cout - kbase);
        float sc[8], bi[8], al[8];
        fillCoeffBufs(fastAct, activParams, defaultAlpha, nvalid, kbase, scale, bias, sc, bi, al, 8);
        const float* crow = C + kbase;
        int mi = m0;
#if CV_SIMD128
        const v_float32x4 vs0 = v_load(sc), vs1 = v_load(sc + 4);
        const v_float32x4 vb0 = v_load(bi), vb1 = v_load(bi + 4);
        const v_float32x4 va0 = v_load(al), va1 = v_load(al + 4);
        const v_float32x4 vmax = v_setall_f32(maxval), z = v_setzero_f32();
        float tmp[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (; mi < m1; mi++) {
            const float* c = crow + (size_t)mi * Cout;
            v_float32x4 a0, a1;
            if (nvalid == 8) {
                a0 = v_load(c); a1 = v_load(c + 4);
            } else {
                for (int c0 = 0; c0 < nvalid; c0++) tmp[c0] = c[c0];
                a0 = v_load(tmp); a1 = v_load(tmp + 4);
            }
            a0 = v_fma(a0, vs0, vb0);
            a1 = v_fma(a1, vs1, vb1);
            if (res) {
                a0 = v_add(a0, v_load(res + (size_t)mi * 8));
                a1 = v_add(a1, v_load(res + (size_t)mi * 8 + 4));
            }
            a0 = v_select(v_ge(a0, z), a0, v_mul(a0, va0));
            a1 = v_select(v_ge(a1, z), a1, v_mul(a1, va1));
            v_store(dst + (size_t)mi * 8, v_min(a0, vmax));
            v_store(dst + (size_t)mi * 8 + 4, v_min(a1, vmax));
        }
#endif
        for (; mi < m1; mi++) {
            const float* c = crow + (size_t)mi * Cout;
            for (int c0 = 0; c0 < 8; c0++) {
                float v = (c0 < nvalid ? c[c0] : 0.f) * sc[c0] + bi[c0];
                if (res) v += res[(size_t)mi * 8 + c0];
                v = v >= 0.f ? v : v * al[c0];
                dst[(size_t)mi * 8 + c0] = std::min(v, maxval);
            }
        }
        if (activation && m1 > m0)
            activation(dst + (size_t)m0 * 8, dst + (size_t)m0 * 8,
                       (size_t)(m1 - m0) * 8, activParams);
    }
};

}  // namespace

void conv2GemmForward(const Mat& inp, Mat& out, const float* residual, const ConvState& cs,
                      const EngineGemmB& packed, int Cout, int Cin, const float* scale, const float* bias,
                      Mat& scratchA, Mat& scratchC)
{
    const MatShape& is = inp.shape();
    const MatShape& os = out.shape();
    CV_Assert(is.dims == 5 && is.back() == 8 && os.back() == 8);
    CV_Assert(inp.type() == CV_32F && out.type() == CV_32F && !packed.empty());
    CV_Assert(packed.n == Cout && packed.k == cs.kshape[1] * cs.kshape[2] * Cin);
    const int N = is[0], C1 = is[1], H = is[2], W = is[3];
    const int K1 = os[1], Ho = os[2], Wo = os[3];
    const int HWo = Ho * Wo;
    CV_Assert(C1 == (Cin + 7) / 8 && K1 == (Cout + 7) / 8);

    const int kh = cs.kshape[1], kw = cs.kshape[2];
    const int sy = cs.strides[1], sx = cs.strides[2];
    const int dy = cs.dilations[1], dx = cs.dilations[2];
    const int pt = cs.pads[1], pl = cs.pads[2];
    const int Kdim = kh * kw * Cin;

    ConvEpilogue ep;
    ep.scale = scale;
    ep.bias = bias;
    ep.Cout = Cout;
    setupActivation(cs, Cout, ep.fastAct, ep.activParams, ep.activation, ep.maxval, ep.defaultAlpha);

    const int nthreads = std::max(1, getNumThreads());
    const size_t inpStride = (size_t)C1 * H * W * 8;
    const size_t outStride = (size_t)K1 * HWo * 8;

    // Many output pixels: each thread does im2col + GEMM + output for its own block of rows.
    // Blocks are a multiple of 32 rows and small enough to stay in L2.
    const size_t segBudget = (size_t)512 << 10;
    int seg = (int)std::max<size_t>(32, segBudget / ((size_t)Kdim * sizeof(float)));
    seg = std::min(seg, 256) & ~31;
    seg = std::max(seg, 32);
    const int nseg = (HWo + seg - 1) / seg;

    if ((int64_t)N * nseg >= nthreads) {
        const int ntasks = N * nseg;
        parallel_for_(Range(0, ntasks), [&](const Range& r) {
            // Not zero-filled: im2col and the GEMM write every element that is read.
            AutoBuffer<float> Abuf((size_t)seg * Kdim), Cbuf((size_t)seg * Cout);
            for (int t = r.start; t < r.end; t++) {
                const int n = t / nseg, si = t - n * nseg;
                const int p0 = si * seg, M = std::min(seg, HWo - p0);
                const float* inp_n = inp.ptr<float>() + n * inpStride;
                float* out_n = out.ptr<float>() + n * outStride;
                const float* res_n = residual ? residual + n * outStride : nullptr;

                im2colNCHWc8(inp_n, C1, H, W, Cin, Wo, kh, kw, sy, sx, dy, dx,
                             pt, pl, p0, 0, M, Abuf.data(), Kdim);
                packed.gemm(0, Abuf.data(), (size_t)Kdim * sizeof(float), false, Cbuf.data(),
                            (size_t)Cout * sizeof(float), M, 1.f, 0.f, false);
                for (int k1 = 0; k1 < K1; k1++) {
                    const size_t ofs = ((size_t)k1 * HWo + p0) * 8;
                    ep.run(Cbuf.data(), k1, 0, M, out_n + ofs, res_n ? res_n + ofs : nullptr);
                }
            }
        }, N * nseg);
        return;
    }

    // Few output pixels (deep layers): one big GEMM that the engine splits over threads.
    const size_t budget = (size_t)4 << 20;
    const int chunk = (int)std::min<size_t>((size_t)HWo,
                          std::max<size_t>(64, budget / ((size_t)Kdim * sizeof(float))));
    if ((size_t)scratchA.total() < (size_t)chunk * Kdim)
        scratchA.create(1, chunk * Kdim, CV_32F);
    if ((size_t)scratchC.total() < (size_t)chunk * Cout)
        scratchC.create(1, chunk * Cout, CV_32F);
    float* A = scratchA.ptr<float>();
    float* C = scratchC.ptr<float>();

    for (int n = 0; n < N; n++) {
        const float* inp_n = inp.ptr<float>() + n * inpStride;
        float* out_n = out.ptr<float>() + n * outStride;
        const float* res_n = residual ? residual + n * outStride : nullptr;

        for (int p0 = 0; p0 < HWo; p0 += chunk) {
            const int M = std::min(chunk, HWo - p0);

            parallel_for_(Range(0, M), [&](const Range& r) {
                im2colNCHWc8(inp_n, C1, H, W, Cin, Wo, kh, kw, sy, sx, dy, dx,
                             pt, pl, p0, r.start, r.end, A, Kdim);
            }, std::min(M, nthreads * 4));

            packed.gemm(0, A, (size_t)Kdim * sizeof(float), false, C, (size_t)Cout * sizeof(float), M, 1.f, 0.f, true);

            const int S = std::max(1, std::min(std::max(1, M / 32), (nthreads * 2 + K1 - 1) / K1));
            parallel_for_(Range(0, K1 * S), [&](const Range& r) {
                for (int t = r.start; t < r.end; t++) {
                    const int k1 = t / S, s = t - k1 * S;
                    const int m0 = (int)((int64_t)M * s / S), m1 = (int)((int64_t)M * (s + 1) / S);
                    const size_t ofs = ((size_t)k1 * HWo + p0) * 8;
                    ep.run(C, k1, m0, m1, out_n + ofs, res_n ? res_n + ofs : nullptr);
                }
            }, K1 * S);
        }
    }
}

}}  // cv::dnn

#endif  // CV_NEON && CV_NEON_AARCH64
