// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"

namespace cv { namespace rvv_hal { namespace dnn {

#if CV_HAL_RVV_1P0_ENABLED

// Blocked NCHWc depthwise convolution, RVV, CV_32F. Mirrors the built-in depthwiseConv32f
// (modules/dnn/src/layers/cpu_kernels/conv2_depthwise.simd.hpp) but computes only the
// block-plane range [task_start, task_end), and applies the same fused epilogue
//   out = min( s >= 0 ? s : s*alpha, maxval ),  s = in*W + bias, then optional residual add,
// with alpha = prelu_slope[c] (if provided) else default_alpha.
//
// Each output row is split into border pixels ([0,inner_x0) and [inner_x1,W): a tap may fall
// in padding, so every tap is bounds-checked via the coordinate table) and interior pixels
// ([inner_x0,inner_x1): no tap touches padding, so the precomputed flat offset table ofstab[]
// is used with no checks).
//
// The channel block is C0-narrow (C0 == 8), so one e32m2 group holds a whole C0 vector at any
// VLEN >= 128. This uses native vsetvl and never trips the C0-vs-VLMAX assert that forces the
// universal-intrinsic path to run scalar on RVV. Two strategies share the kernel:
//
//   * per-pixel (m2): the correctness floor. Borders, stride>1, narrow VLEN and small planes
//     all use it; it processes one output pixel's C0 channels per vector op (interior blocks 4
//     output columns for ILP). Beats the shipped scalar path at every VLEN, no regression.
//   * register-fill (m8): for stride-1 interiors on large planes at wide VLEN (>= 512), where
//     the per-pixel path uses only C0 of the wide register's lanes. It packs P = VLMAX/C0
//     consecutive output pixels into one wide vector, reusing per-tap weights that are tiled
//     across the P pixels ONCE per channel-block via vrgather (into a small scratch buffer).
//     This reaches ~full register utilisation; the per-block tiling build is amortised over the
//     plane, so it is gated to sizes where that pays (see use_fill below).
//
// The two produce identical results; use_fill only selects the faster path per call. Because the
// per-pixel path is always available, a mis-tuned guard costs speed, never correctness.

int depthwise_conv32f(const float* inp_data, const float* residual_data,
                      float* out_data, const float* weights_all,
                      const float* scale_all, const float* bias_all,
                      int C, int C0, int C1,
                      const int* insize, const int* outsize, const int* strides,
                      const int* pads, const int* inner, const int* coordtab,
                      const int* ofstab, int ksize,
                      float maxval, float default_alpha, const float* prelu_slope,
                      int task_start, int task_end)
{
    constexpr int MAX_CONV_DIMS = 3;   // coordtab/pads/inner use a fixed Z,Y,X frame
    constexpr int MAX_C0 = 8;          // engine's fixed f32 channel block
    if (C0 > MAX_C0)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const int Di = insize[0], Hi = insize[1], Wi = insize[2];
    const int D  = outsize[0], H = outsize[1], W = outsize[2];
    const int64_t iplanesize = (int64_t)Di*Hi*Wi*C0;
    const int64_t planesize  = (int64_t)D*H*W*C0;
    const int SZ = strides[0], SY = strides[1], SX = strides[2];
    const int padZ0 = pads[0], padY0 = pads[1], padX0 = pads[2];
    const int inner_z0 = inner[0], inner_z1 = inner[MAX_CONV_DIMS];
    const int inner_y0 = inner[1], inner_y1 = inner[MAX_CONV_DIMS + 1];
    const int inner_x0 = inner[2], inner_x1 = inner[MAX_CONV_DIMS + 2];
    const int* zyxtab = coordtab;

    float scalebuf[MAX_C0], biasbuf[MAX_C0], alphabuf[MAX_C0];

    // Register-fill selection. fillw = P*C0 pixels-worth of lanes in one e32m8 group. Enable
    // only where the fill's per-block vrgather tiling pays off: a wide register (VLEN >= 512,
    // else per-pixel already uses most lanes), a stride-1 interior (contiguous input), the
    // benchmarked small-kernel regime (ksize <= 3x3), and a plane large enough to amortise the
    // build (>= 2*fillw, the measured VLEN-1024 crossover). Everything else uses per-pixel.
    const int fillw = (int)((__riscv_vsetvlmax_e32m8() / C0) * C0);
    const bool use_fill = (__riscv_vlenb() * 8 >= 512) && (SX == 1) && (ksize <= 9)
                          && ((int64_t)D * H * W >= (int64_t)2 * fillw);
    cv::AutoBuffer<float> fillscratch(use_fill ? (size_t)(ksize + 3) * fillw : 0);
    float* tw  = fillscratch.data();               // ksize * fillw : tiled per-tap weights
    float* tsc = tw  + (int64_t)ksize * fillw;     // fillw : tiled scale
    float* tbi = tsc + fillw;                       // fillw : tiled bias
    float* tal = tbi + fillw;                       // fillw : tiled alpha

    // Fused epilogue on one C0-lane accumulator: acc = acc*scale + bias (+ residual), then
    // acc = min(acc >= 0 ? acc : acc*alpha, maxval). rp_c is the residual pointer already
    // advanced to +c, or nullptr.
    auto finish = [&](vfloat32m2_t acc, int c, size_t vl, const float* rp_c) -> vfloat32m2_t {
        vfloat32m2_t sc = __riscv_vle32_v_f32m2(scalebuf + c, vl);
        vfloat32m2_t bi = __riscv_vle32_v_f32m2(biasbuf + c, vl);
        acc = __riscv_vfmadd_vv_f32m2(acc, sc, bi, vl);          // acc*scale + bias
        if (rp_c)
            acc = __riscv_vfadd_vv_f32m2(acc, __riscv_vle32_v_f32m2(rp_c, vl), vl);
        vfloat32m2_t al  = __riscv_vle32_v_f32m2(alphabuf + c, vl);
        vfloat32m2_t neg = __riscv_vfmul_vv_f32m2(acc, al, vl);  // acc*alpha (leaky/prelu branch)
        vbool16_t m = __riscv_vmfge_vf_f32m2_b16(acc, 0.f, vl);  // m: acc >= 0
        acc = __riscv_vmerge_vvm_f32m2(neg, acc, m, vl);         // m ? acc : acc*alpha
        return __riscv_vfmin_vf_f32m2(acc, maxval, vl);
    };

    for (int nc = task_start; nc < task_end; nc++) {
        const int n = nc / C1;
        const int c_base = (nc - n*C1) * C0;
        int c_count = C - c_base;
        if (c_count > C0) c_count = C0;
        if (c_count < 0)  c_count = 0;

        const float* inp = inp_data + nc * iplanesize;
        float* outp = out_data + nc * planesize;
        const float* resp = residual_data ? residual_data + nc * planesize : nullptr;
        const float* wblk = weights_all + (int64_t)(c_base / C0) * ksize * C0;

        for (int c = 0; c < C0; c++) {
            if (c < c_count) {
                scalebuf[c] = scale_all ? scale_all[c_base + c] : 1.f;
                biasbuf[c]  = bias_all  ? bias_all[c_base + c]  : 0.f;
                alphabuf[c] = prelu_slope ? prelu_slope[c_base + c] : default_alpha;
            } else {
                scalebuf[c] = 0.f; biasbuf[c] = 0.f; alphabuf[c] = 0.f;
            }
        }

        // Tile this block's per-tap weights (and scale/bias/alpha) across the P pixels of a fill
        // group ONCE, so the wide interior loop just reloads them from L1: t[k][j] = src[k*C0 + j%C0].
        if (use_fill) {
            size_t vl = (size_t)fillw;
            vuint32m8_t idx = __riscv_vand_vx_u32m8(__riscv_vid_v_u32m8(vl), C0 - 1, vl);
            for (int k = 0; k < ksize; k++)
                __riscv_vse32_v_f32m8(tw + (int64_t)k*fillw,
                    __riscv_vrgather_vv_f32m8(__riscv_vle32_v_f32m8(wblk + k*C0, C0), idx, vl), vl);
            __riscv_vse32_v_f32m8(tsc, __riscv_vrgather_vv_f32m8(__riscv_vle32_v_f32m8(scalebuf, C0), idx, vl), vl);
            __riscv_vse32_v_f32m8(tbi, __riscv_vrgather_vv_f32m8(__riscv_vle32_v_f32m8(biasbuf,  C0), idx, vl), vl);
            __riscv_vse32_v_f32m8(tal, __riscv_vrgather_vv_f32m8(__riscv_vle32_v_f32m8(alphabuf, C0), idx, vl), vl);
        }

        for (int z0 = 0; z0 < D; z0++) {
            const int zi_ = z0*SZ - padZ0;
            const bool z_in = (z0 >= inner_z0 && z0 < inner_z1);
            for (int y0 = 0; y0 < H; y0++) {
                const int yi_ = y0*SY - padY0;
                float* out_row = outp + ((int64_t)(z0*H + y0)*W)*C0;
                const float* res_row = resp ? resp + ((int64_t)(z0*H + y0)*W)*C0 : nullptr;

                // border pixels: every tap bounds-checked against the coordinate table
                auto do_border = [&](int xa, int xb) {
                    for (int x0 = xa; x0 < xb; x0++) {
                        const int xi_ = x0*SX - padX0;
                        float* op = out_row + (int64_t)x0*C0;
                        const float* rp = res_row ? res_row + (int64_t)x0*C0 : nullptr;
                        for (int c = 0; c < C0; ) {
                            size_t vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.f, vl);
                            for (int k = 0; k < ksize; k++) {
                                int zi = zi_ + zyxtab[k*MAX_CONV_DIMS];
                                int yi = yi_ + zyxtab[k*MAX_CONV_DIMS + 1];
                                int xi = xi_ + zyxtab[k*MAX_CONV_DIMS + 2];
                                if ((unsigned)zi >= (unsigned)Di ||
                                    (unsigned)yi >= (unsigned)Hi ||
                                    (unsigned)xi >= (unsigned)Wi) continue;
                                const float* iptr = inp + (((int64_t)zi*Hi + yi)*Wi + xi)*C0 + c;
                                vfloat32m2_t vi = __riscv_vle32_v_f32m2(iptr, vl);
                                vfloat32m2_t vw = __riscv_vle32_v_f32m2(wblk + k*C0 + c, vl);
                                acc = __riscv_vfmacc_vv_f32m2(acc, vi, vw, vl);
                            }
                            acc = finish(acc, c, vl, rp ? rp + c : nullptr);
                            __riscv_vse32_v_f32m2(op + c, acc, vl);
                            c += (int)vl;
                        }
                    }
                };

                if (!(z_in && y0 >= inner_y0 && y0 < inner_y1)) {
                    do_border(0, W);
                    continue;
                }
                do_border(0, inner_x0);

                const int64_t rowbase = ((int64_t)(Hi*zi_ + yi_)*Wi)*C0;
                if (use_fill) {
                    // register-fill: pack P = fillw/C0 stride-1 output pixels into one wide (m8)
                    // vector; tiled weights come from the per-block scratch built above.
                    int xi = inner_x0;
                    while (xi < inner_x1) {
                        int gp = inner_x1 - xi;
                        if (gp > fillw / C0) gp = fillw / C0;
                        size_t vl = (size_t)gp * C0;
                        const float* base = inp + rowbase + (int64_t)(xi - padX0)*C0;   // SX == 1
                        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.f, vl);
                        for (int k = 0; k < ksize; k++)
                            acc = __riscv_vfmacc_vv_f32m8(acc, __riscv_vle32_v_f32m8(base + ofstab[k], vl),
                                                          __riscv_vle32_v_f32m8(tw + (int64_t)k*fillw, vl), vl);
                        acc = __riscv_vfmadd_vv_f32m8(acc, __riscv_vle32_v_f32m8(tsc, vl),
                                                      __riscv_vle32_v_f32m8(tbi, vl), vl);
                        if (res_row)
                            acc = __riscv_vfadd_vv_f32m8(acc, __riscv_vle32_v_f32m8(res_row + (int64_t)xi*C0, vl), vl);
                        vfloat32m8_t neg = __riscv_vfmul_vv_f32m8(acc, __riscv_vle32_v_f32m8(tal, vl), vl);
                        vbool4_t m = __riscv_vmfge_vf_f32m8_b4(acc, 0.f, vl);
                        acc = __riscv_vmerge_vvm_f32m8(neg, acc, m, vl);
                        acc = __riscv_vfmin_vf_f32m8(acc, maxval, vl);
                        __riscv_vse32_v_f32m8(out_row + (int64_t)xi*C0, acc, vl);
                        xi += gp;
                    }
                    do_border(inner_x1, W);
                    continue;
                }

                int x0 = inner_x0;
                // interior, 4 output columns at a time: weight loaded once per tap, reused x4
                for (; x0 + 4 <= inner_x1; x0 += 4) {
                    const float* b0 = inp + rowbase + (int64_t)((x0  )*SX - padX0)*C0;
                    const float* b1 = inp + rowbase + (int64_t)((x0+1)*SX - padX0)*C0;
                    const float* b2 = inp + rowbase + (int64_t)((x0+2)*SX - padX0)*C0;
                    const float* b3 = inp + rowbase + (int64_t)((x0+3)*SX - padX0)*C0;
                    float* o0 = out_row + (int64_t)x0*C0;
                    const float* r0 = res_row ? res_row + (int64_t)x0*C0 : nullptr;
                    for (int c = 0; c < C0; ) {
                        size_t vl = __riscv_vsetvl_e32m2(C0 - c);
                        vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
                        vfloat32m2_t a1 = a0, a2 = a0, a3 = a0;
                        for (int k = 0; k < ksize; k++) {
                            vfloat32m2_t vw = __riscv_vle32_v_f32m2(wblk + k*C0 + c, vl);
                            const int o = ofstab[k] + c;
                            a0 = __riscv_vfmacc_vv_f32m2(a0, __riscv_vle32_v_f32m2(b0 + o, vl), vw, vl);
                            a1 = __riscv_vfmacc_vv_f32m2(a1, __riscv_vle32_v_f32m2(b1 + o, vl), vw, vl);
                            a2 = __riscv_vfmacc_vv_f32m2(a2, __riscv_vle32_v_f32m2(b2 + o, vl), vw, vl);
                            a3 = __riscv_vfmacc_vv_f32m2(a3, __riscv_vle32_v_f32m2(b3 + o, vl), vw, vl);
                        }
                        a0 = finish(a0, c, vl, r0 ? r0 + c : nullptr);
                        a1 = finish(a1, c, vl, r0 ? r0 + (int64_t)C0   + c : nullptr);
                        a2 = finish(a2, c, vl, r0 ? r0 + (int64_t)2*C0 + c : nullptr);
                        a3 = finish(a3, c, vl, r0 ? r0 + (int64_t)3*C0 + c : nullptr);
                        __riscv_vse32_v_f32m2(o0 + c,               a0, vl);
                        __riscv_vse32_v_f32m2(o0 + (int64_t)C0   + c, a1, vl);
                        __riscv_vse32_v_f32m2(o0 + (int64_t)2*C0 + c, a2, vl);
                        __riscv_vse32_v_f32m2(o0 + (int64_t)3*C0 + c, a3, vl);
                        c += (int)vl;
                    }
                }
                for (; x0 < inner_x1; x0++) {
                    const float* w = inp + rowbase + (int64_t)(x0*SX - padX0)*C0;
                    float* op = out_row + (int64_t)x0*C0;
                    const float* rp = res_row ? res_row + (int64_t)x0*C0 : nullptr;
                    for (int c = 0; c < C0; ) {
                        size_t vl = __riscv_vsetvl_e32m2(C0 - c);
                        vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.f, vl);
                        for (int k = 0; k < ksize; k++) {
                            vfloat32m2_t vw = __riscv_vle32_v_f32m2(wblk + k*C0 + c, vl);
                            vfloat32m2_t vi = __riscv_vle32_v_f32m2(w + ofstab[k] + c, vl);
                            acc = __riscv_vfmacc_vv_f32m2(acc, vi, vw, vl);
                        }
                        acc = finish(acc, c, vl, rp ? rp + c : nullptr);
                        __riscv_vse32_v_f32m2(op + c, acc, vl);
                        c += (int)vl;
                    }
                }
                do_border(inner_x1, W);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

#endif // CV_HAL_RVV_1P0_ENABLED

}}} // cv::rvv_hal::dnn
