// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "rvv_hal.hpp"

#if CV_HAL_RVV_1P0_ENABLED
#include <cfloat>
#endif

namespace cv { namespace rvv_hal { namespace dnn {

#if CV_HAL_RVV_1P0_ENABLED

// Blocked NCDHWc pooling, RVV, CV_32F. Mirrors the built-in maxPool32f/avgPool32f
// semantics (modules/dnn/src/layers/{max,avg}pool_layer.cpp) but computes only the
// block-plane range [task_start, task_end).
//
// Each output row is split like the engine into border pixels ([0,inner_x0) and
// [inner_x1,W): a tap may fall in padding, so every tap is bounds-checked via the
// coordinate table) and interior pixels ([inner_x0,inner_x1): no tap touches
// padding, so the precomputed flat offset table ofstab[] is used with no checks).
//
// Interior strategy:
//   * stride-x == 1: consecutive output pixels' tap-k data are contiguous in memory,
//     so we pack VLMAX/C0 output pixels into ONE wide (LMUL m8) vector and stream the
//     whole interior — reaching ~100% register utilisation at any VLEN. This is the
//     win the C0=8-blocked universal-intrinsic path (capped to C0 lanes) cannot get.
//   * stride-x > 1: pixels' tap data are strided; we fall back to processing C0
//     channels per vector, register-blocked 4 output columns at a time for ILP.

int maxpool3d32f(const float* inp_data, float* out_data, int C0,
               const int* insize, const int* outsize, const int* strides,
               const int* pads, const int* inner, const int* coordtab,
               const int* ofstab, int ksize, int task_start, int task_end)
{
    constexpr int MAX_POOL_DIMS = 3;   // coordtab/pads/inner use a fixed Z,Y,X frame
    const int Di = insize[0], Hi = insize[1], Wi = insize[2];
    const int D  = outsize[0], H = outsize[1], W = outsize[2];
    const int64_t iplanesize = (int64_t)Di*Hi*Wi*C0;
    const int64_t planesize  = (int64_t)D*H*W*C0;
    const int SZ = strides[0], SY = strides[1], SX = strides[2];
    const int padZ0 = pads[0], padY0 = pads[1], padX0 = pads[2];
    const int inner_z0 = inner[0], inner_z1 = inner[MAX_POOL_DIMS];
    const int inner_y0 = inner[1], inner_y1 = inner[MAX_POOL_DIMS + 1];
    const int inner_x0 = inner[2], inner_x1 = inner[MAX_POOL_DIMS + 2];
    const int* zyxtab = coordtab;

    for (int nc = task_start; nc < task_end; nc++) {
        const float* inp = inp_data + nc * iplanesize;
        float* outp = out_data + nc * planesize;
        for (int z0 = 0; z0 < D; z0++) {
            const int zi_ = z0*SZ - padZ0;
            const bool z_in = (z0 >= inner_z0 && z0 < inner_z1);
            for (int y0 = 0; y0 < H; y0++) {
                const int yi_ = y0*SY - padY0;
                float* out_row = outp + ((int64_t)(z0*H + y0)*W)*C0;

                auto do_border = [&](int xa, int xb) {
                    for (int x0 = xa; x0 < xb; x0++) {
                        const int xi_ = x0*SX - padX0;
                        float* op = out_row + (int64_t)x0*C0;
                        for (int c = 0; c < C0; ) {
                            int vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(-FLT_MAX, vl);
                            for (int k = 0; k < ksize; k++) {
                                int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                int yi = yi_ + zyxtab[k*MAX_POOL_DIMS + 1];
                                int xi = xi_ + zyxtab[k*MAX_POOL_DIMS + 2];
                                if ((unsigned)zi >= (unsigned)Di ||
                                    (unsigned)yi >= (unsigned)Hi ||
                                    (unsigned)xi >= (unsigned)Wi) continue;
                                const float* iptr = inp + (((int64_t)zi*Hi + yi)*Wi + xi)*C0 + c;
                                acc = __riscv_vfmax_vv_f32m2(acc, __riscv_vle32_v_f32m2(iptr, vl), vl);
                            }
                            __riscv_vse32_v_f32m2(op + c, acc, vl);
                            c += vl;
                        }
                    }
                };

                if (!(z_in && y0 >= inner_y0 && y0 < inner_y1)) {
                    do_border(0, W);
                    continue;
                }
                do_border(0, inner_x0);

                const int64_t rowbase = ((int64_t)(Hi*zi_ + yi_)*Wi)*C0;
                if (SX == 1) {
                    // register-filling: pack VLMAX/C0 output pixels per wide (m8) vector
                    int xi = inner_x0;
                    while (xi < inner_x1) {
                        const float* w = inp + rowbase + (int64_t)(xi - padX0)*C0;
                        size_t vl = __riscv_vsetvl_e32m8((size_t)(inner_x1 - xi)*C0);
                        int gotpix = (int)(vl / C0); vl = (size_t)gotpix * C0;
                        vfloat32m8_t acc = __riscv_vle32_v_f32m8(w + ofstab[0], vl);
                        for (int k = 1; k < ksize; k++)
                            acc = __riscv_vfmax_vv_f32m8(acc, __riscv_vle32_v_f32m8(w + ofstab[k], vl), vl);
                        __riscv_vse32_v_f32m8(out_row + (int64_t)xi*C0, acc, vl);
                        xi += gotpix;
                    }
                } else {
                    // stride>1: C0 per vector, register-blocked 4 output columns
                    int x0 = inner_x0;
                    for (; x0 + 4 <= inner_x1; x0 += 4) {
                        const float* w0 = inp + rowbase + (int64_t)((x0  )*SX - padX0)*C0;
                        const float* w1 = inp + rowbase + (int64_t)((x0+1)*SX - padX0)*C0;
                        const float* w2 = inp + rowbase + (int64_t)((x0+2)*SX - padX0)*C0;
                        const float* w3 = inp + rowbase + (int64_t)((x0+3)*SX - padX0)*C0;
                        float* o0 = out_row + (int64_t)x0*C0;
                        for (int c = 0; c < C0; ) {
                            int vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t a0 = __riscv_vle32_v_f32m2(w0 + ofstab[0] + c, vl);
                            vfloat32m2_t a1 = __riscv_vle32_v_f32m2(w1 + ofstab[0] + c, vl);
                            vfloat32m2_t a2 = __riscv_vle32_v_f32m2(w2 + ofstab[0] + c, vl);
                            vfloat32m2_t a3 = __riscv_vle32_v_f32m2(w3 + ofstab[0] + c, vl);
                            for (int k = 1; k < ksize; k++) {
                                const int o = ofstab[k] + c;
                                a0 = __riscv_vfmax_vv_f32m2(a0, __riscv_vle32_v_f32m2(w0 + o, vl), vl);
                                a1 = __riscv_vfmax_vv_f32m2(a1, __riscv_vle32_v_f32m2(w1 + o, vl), vl);
                                a2 = __riscv_vfmax_vv_f32m2(a2, __riscv_vle32_v_f32m2(w2 + o, vl), vl);
                                a3 = __riscv_vfmax_vv_f32m2(a3, __riscv_vle32_v_f32m2(w3 + o, vl), vl);
                            }
                            __riscv_vse32_v_f32m2(o0 + c,             a0, vl);
                            __riscv_vse32_v_f32m2(o0 + (int64_t)C0  + c, a1, vl);
                            __riscv_vse32_v_f32m2(o0 + (int64_t)2*C0 + c, a2, vl);
                            __riscv_vse32_v_f32m2(o0 + (int64_t)3*C0 + c, a3, vl);
                            c += vl;
                        }
                    }
                    for (; x0 < inner_x1; x0++) {
                        const float* w = inp + rowbase + (int64_t)(x0*SX - padX0)*C0;
                        float* op = out_row + (int64_t)x0*C0;
                        for (int c = 0; c < C0; ) {
                            int vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t acc = __riscv_vle32_v_f32m2(w + ofstab[0] + c, vl);
                            for (int k = 1; k < ksize; k++)
                                acc = __riscv_vfmax_vv_f32m2(acc, __riscv_vle32_v_f32m2(w + ofstab[k] + c, vl), vl);
                            __riscv_vse32_v_f32m2(op + c, acc, vl);
                            c += vl;
                        }
                    }
                }
                do_border(inner_x1, W);
            }
        }
    }
    return CV_HAL_ERROR_OK;
}

int avgpool3d32f(const float* inp_data, float* out_data, int C0,
               const int* insize, const int* outsize, const int* strides,
               const int* pads, const int* inner, const int* coordtab,
               const int* ofstab, int ksize, int count_include_pad,
               int task_start, int task_end)
{
    constexpr int MAX_POOL_DIMS = 3;   // coordtab/pads/inner use a fixed Z,Y,X frame
    const int Di = insize[0], Hi = insize[1], Wi = insize[2];
    const int D  = outsize[0], H = outsize[1], W = outsize[2];
    const int64_t iplanesize = (int64_t)Di*Hi*Wi*C0;
    const int64_t planesize  = (int64_t)D*H*W*C0;
    const int SZ = strides[0], SY = strides[1], SX = strides[2];
    const int padZ0 = pads[0], padY0 = pads[1], padX0 = pads[2];
    const int padZ1 = pads[MAX_POOL_DIMS], padY1 = pads[MAX_POOL_DIMS + 1],
              padX1 = pads[MAX_POOL_DIMS + 2];
    const int inner_z0 = inner[0], inner_z1 = inner[MAX_POOL_DIMS];
    const int inner_y0 = inner[1], inner_y1 = inner[MAX_POOL_DIMS + 1];
    const int inner_x0 = inner[2], inner_x1 = inner[MAX_POOL_DIMS + 2];
    const int* zyxtab = coordtab;
    const float iksize = 1.f/ksize;   // interior has no padding: denom == ksize either way

    for (int nc = task_start; nc < task_end; nc++) {
        const float* inp = inp_data + nc * iplanesize;
        float* outp = out_data + nc * planesize;
        for (int z0 = 0; z0 < D; z0++) {
            const int zi_ = z0*SZ - padZ0;
            const bool z_in = (z0 >= inner_z0 && z0 < inner_z1);
            for (int y0 = 0; y0 < H; y0++) {
                const int yi_ = y0*SY - padY0;
                float* out_row = outp + ((int64_t)(z0*H + y0)*W)*C0;

                auto do_border = [&](int xa, int xb) {
                    for (int x0 = xa; x0 < xb; x0++) {
                        const int xi_ = x0*SX - padX0;
                        float* op = out_row + (int64_t)x0*C0;
                        int nitems = 0, npadded = 0;
                        for (int k = 0; k < ksize; k++) {
                            int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                            int yi = yi_ + zyxtab[k*MAX_POOL_DIMS + 1];
                            int xi = xi_ + zyxtab[k*MAX_POOL_DIMS + 2];
                            npadded += (zi >= -padZ0 && zi < Di + padZ1 &&
                                        yi >= -padY0 && yi < Hi + padY1 &&
                                        xi >= -padX0 && xi < Wi + padX1);
                            if ((unsigned)zi >= (unsigned)Di ||
                                (unsigned)yi >= (unsigned)Hi ||
                                (unsigned)xi >= (unsigned)Wi) continue;
                            nitems++;
                        }
                        const int denom = count_include_pad ? npadded : nitems;
                        const float scale = denom ? 1.f/denom : 0.f;
                        for (int c = 0; c < C0; ) {
                            int vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.f, vl);
                            for (int k = 0; k < ksize; k++) {
                                int zi = zi_ + zyxtab[k*MAX_POOL_DIMS];
                                int yi = yi_ + zyxtab[k*MAX_POOL_DIMS + 1];
                                int xi = xi_ + zyxtab[k*MAX_POOL_DIMS + 2];
                                if ((unsigned)zi >= (unsigned)Di ||
                                    (unsigned)yi >= (unsigned)Hi ||
                                    (unsigned)xi >= (unsigned)Wi) continue;
                                const float* iptr = inp + (((int64_t)zi*Hi + yi)*Wi + xi)*C0 + c;
                                acc = __riscv_vfadd_vv_f32m2(acc, __riscv_vle32_v_f32m2(iptr, vl), vl);
                            }
                            __riscv_vse32_v_f32m2(op + c, __riscv_vfmul_vf_f32m2(acc, scale, vl), vl);
                            c += vl;
                        }
                    }
                };

                if (!(z_in && y0 >= inner_y0 && y0 < inner_y1)) {
                    do_border(0, W);
                    continue;
                }
                do_border(0, inner_x0);

                const int64_t rowbase = ((int64_t)(Hi*zi_ + yi_)*Wi)*C0;
                if (SX == 1) {
                    int xi = inner_x0;
                    while (xi < inner_x1) {
                        const float* w = inp + rowbase + (int64_t)(xi - padX0)*C0;
                        size_t vl = __riscv_vsetvl_e32m8((size_t)(inner_x1 - xi)*C0);
                        int gotpix = (int)(vl / C0); vl = (size_t)gotpix * C0;
                        vfloat32m8_t acc = __riscv_vle32_v_f32m8(w + ofstab[0], vl);
                        for (int k = 1; k < ksize; k++)
                            acc = __riscv_vfadd_vv_f32m8(acc, __riscv_vle32_v_f32m8(w + ofstab[k], vl), vl);
                        acc = __riscv_vfmul_vf_f32m8(acc, iksize, vl);
                        __riscv_vse32_v_f32m8(out_row + (int64_t)xi*C0, acc, vl);
                        xi += gotpix;
                    }
                } else {
                    int x0 = inner_x0;
                    for (; x0 + 4 <= inner_x1; x0 += 4) {
                        const float* w0 = inp + rowbase + (int64_t)((x0  )*SX - padX0)*C0;
                        const float* w1 = inp + rowbase + (int64_t)((x0+1)*SX - padX0)*C0;
                        const float* w2 = inp + rowbase + (int64_t)((x0+2)*SX - padX0)*C0;
                        const float* w3 = inp + rowbase + (int64_t)((x0+3)*SX - padX0)*C0;
                        float* o0 = out_row + (int64_t)x0*C0;
                        for (int c = 0; c < C0; ) {
                            int vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t a0 = __riscv_vle32_v_f32m2(w0 + ofstab[0] + c, vl);
                            vfloat32m2_t a1 = __riscv_vle32_v_f32m2(w1 + ofstab[0] + c, vl);
                            vfloat32m2_t a2 = __riscv_vle32_v_f32m2(w2 + ofstab[0] + c, vl);
                            vfloat32m2_t a3 = __riscv_vle32_v_f32m2(w3 + ofstab[0] + c, vl);
                            for (int k = 1; k < ksize; k++) {
                                const int o = ofstab[k] + c;
                                a0 = __riscv_vfadd_vv_f32m2(a0, __riscv_vle32_v_f32m2(w0 + o, vl), vl);
                                a1 = __riscv_vfadd_vv_f32m2(a1, __riscv_vle32_v_f32m2(w1 + o, vl), vl);
                                a2 = __riscv_vfadd_vv_f32m2(a2, __riscv_vle32_v_f32m2(w2 + o, vl), vl);
                                a3 = __riscv_vfadd_vv_f32m2(a3, __riscv_vle32_v_f32m2(w3 + o, vl), vl);
                            }
                            a0 = __riscv_vfmul_vf_f32m2(a0, iksize, vl);
                            a1 = __riscv_vfmul_vf_f32m2(a1, iksize, vl);
                            a2 = __riscv_vfmul_vf_f32m2(a2, iksize, vl);
                            a3 = __riscv_vfmul_vf_f32m2(a3, iksize, vl);
                            __riscv_vse32_v_f32m2(o0 + c,             a0, vl);
                            __riscv_vse32_v_f32m2(o0 + (int64_t)C0  + c, a1, vl);
                            __riscv_vse32_v_f32m2(o0 + (int64_t)2*C0 + c, a2, vl);
                            __riscv_vse32_v_f32m2(o0 + (int64_t)3*C0 + c, a3, vl);
                            c += vl;
                        }
                    }
                    for (; x0 < inner_x1; x0++) {
                        const float* w = inp + rowbase + (int64_t)(x0*SX - padX0)*C0;
                        float* op = out_row + (int64_t)x0*C0;
                        for (int c = 0; c < C0; ) {
                            int vl = __riscv_vsetvl_e32m2(C0 - c);
                            vfloat32m2_t acc = __riscv_vle32_v_f32m2(w + ofstab[0] + c, vl);
                            for (int k = 1; k < ksize; k++)
                                acc = __riscv_vfadd_vv_f32m2(acc, __riscv_vle32_v_f32m2(w + ofstab[k] + c, vl), vl);
                            __riscv_vse32_v_f32m2(op + c, __riscv_vfmul_vf_f32m2(acc, iksize, vl), vl);
                            c += vl;
                        }
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
