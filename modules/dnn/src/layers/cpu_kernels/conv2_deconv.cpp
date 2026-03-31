// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "../conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

static void deconvBlock32f(const void* inp__, const void* /*residual*/,
                           void* out__, const ConvState& cs,
                           const void* weights__, const float* /*scale*/,
                           const float* bias__)
{
    CV_Assert(cs.inpshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.outshape.layout == DATA_LAYOUT_BLOCK);
    CV_Assert(cs.wshape.dims == 5);

    enum { MAX_DIMS = ConvState::MAX_CONV_DIMS };
    const int sdims = cs.nspatialdims;
    const int C0    = cs.inpshape.back();
    const int K0    = C0;

    const int N  = cs.inpshape[0];
    const int C1 = cs.inpshape[1];
    const int K1 = cs.outshape[1];
    const int C  = cs.inpshape.C;
    const int K  = cs.outshape.C;
    const int ngroups = cs.ngroups;
    const int Kg  = K / ngroups;
    const int Cg  = C / ngroups;
    const int Kblk  = cs.wshape[1];
    const int ksize = cs.wshape[2];
    const int C1Max = cs.wshape[3];

    int ispatial = 1, ospatial = 1;
    for (int i = 0; i < sdims; i++) {
        ispatial *= cs.inpshape[2 + i];
        ospatial *= cs.outshape[2 + i];
    }

    int oDims[MAX_DIMS], iDims[MAX_DIMS];
    for (int i = 0; i < MAX_DIMS; i++) oDims[i] = iDims[i] = 1;
    for (int i = 0; i < sdims; i++) {
        oDims[MAX_DIMS - sdims + i] = cs.outshape[2 + i];
        iDims[MAX_DIMS - sdims + i] = cs.inpshape[2 + i];
    }

    std::vector<std::array<int, MAX_DIMS>> kcoords_tab(ksize);
    for (int ks = 0; ks < ksize; ks++) {
        int ktmp = ks;
        for (int i = sdims - 1; i >= 0; i--) {
            int di = MAX_DIMS - sdims + i;
            kcoords_tab[ks][di] = ktmp % cs.kshape[di];
            ktmp /= cs.kshape[di];
        }
    }

    const float* wdata = (const float*)weights__;
    const float* bias  = bias__;

    const int NK1 = N * K1;
    parallel_for_(Range(0, NK1), [&](const Range& range) {
        for (int nk1 = range.start; nk1 < range.end; nk1++) {
            const int n  = nk1 / K1;
            const int k1 = nk1 % K1;

            const int k_base = k1 * C0;
            const int currK0 = std::min(C0, K - k_base);
            if (currK0 <= 0) continue;

            float* out_k1 = (float*)out__ + ((int64_t)n * K1 + k1) * ospatial * C0;

            for (int opos = 0; opos < ospatial; opos++) {
                float* p = out_k1 + opos * C0;
                if (bias) {
                    for (int k0 = 0; k0 < currK0; k0++) p[k0] = bias[k_base + k0];
                } else {
                    for (int k0 = 0; k0 < currK0; k0++) p[k0] = 0.f;
                }
                for (int k0 = currK0; k0 < C0; k0++) p[k0] = 0.f;
            }

            const float* inp_n = (const float*)inp__ + (int64_t)n * C1 * ispatial * C0;

            for (int opos_flat = 0; opos_flat < ospatial; opos_flat++) {
                float* out_ptr = out_k1 + opos_flat * C0;

                int ocoords[MAX_DIMS];
                {
                    int tmp = opos_flat;
                    for (int i = sdims - 1; i >= 0; i--) {
                        int di = MAX_DIMS - sdims + i;
                        ocoords[di] = tmp % oDims[di];
                        tmp /= oDims[di];
                    }
                }

                for (int ks = 0; ks < ksize; ks++) {
                    bool valid = true;
                    int ipos_flat = 0;
                    for (int i = 0; i < sdims; i++) {
                        int di = MAX_DIMS - sdims + i;
                        int raw = ocoords[di] + cs.pads[di]
                                  - kcoords_tab[ks][di] * cs.dilations[di];
                        if (raw < 0 || raw % cs.strides[di] != 0) {
                            valid = false; break;
                        }
                        int ic = raw / cs.strides[di];
                        if (ic >= iDims[di]) { valid = false; break; }
                        ipos_flat = ipos_flat * iDims[di] + ic;
                    }
                    if (!valid) continue;

                    for (int k0 = 0; k0 < currK0; k0++) {
                        const int k    = k_base + k0;
                        const int g    = k / Kg;
                        const int kin  = k - g * Kg;
                        const int kblk = kin / K0;
                        const int k0l  = kin & (K0 - 1);

                        const float* w_base = wdata +
                            ((int64_t)(g * Kblk + kblk) * ksize + ks) * C1Max * C0 * K0;

                        const int c_start     = g * Cg;
                        const int c1_abs_base = c_start / C0;

                        float sum = 0.f;
                        for (int c1p = 0; c1p < C1Max; c1p++) {
                            const int c1_abs = c1_abs_base + c1p;
                            if (c1_abs >= C1) break;

                            const float* inp_ptr = inp_n +
                                (int64_t)(c1_abs * ispatial + ipos_flat) * C0;
                            const float* w_c1p = w_base + (int64_t)c1p * C0 * K0;

                            for (int c0 = 0; c0 < C0; c0++)
                                sum += w_c1p[c0 * K0 + k0l] * inp_ptr[c0];
                        }
                        out_ptr[k0] += sum;
                    }
                }
            }
        }
    });
}

DeconvFunc getDeconvFunc(int depth)
{
    if (depth == CV_32F)
        return deconvBlock32f;
    return nullptr;
}

CV__DNN_INLINE_NS_END
}}
