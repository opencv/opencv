// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "gfluidimgproc_func.hpp"

#include "gfluidutils.hpp"

#include <opencv2/core/hal/intrin.hpp>

#include <cmath>
#include <cstdlib>

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif

namespace cv {
namespace gapi {
namespace fluid {

//---------------------
//
// Fluid kernels: Sobel
//
//---------------------

// Sobel 3x3: vertical pass
template<bool noscale, typename DST>
void run_sobel3x3_vert(DST out[], int length, const float ky[],
         float scale, float delta, const int r[], float *buf[])
{
    float ky0 = ky[0],
          ky1 = ky[1],
          ky2 = ky[2];

    int r0 = r[0],
        r1 = r[1],
        r2 = r[2];

#if CV_SIMD
    // for floating-point output,
    // manual vectoring may be not better than compiler's optimization
#define EXPLICIT_SIMD_32F 0  // 1=vectorize 32f case explicitly, 0=don't
#if     EXPLICIT_SIMD_32F
    if (std::is_same<DST, float>::value && length >= v_int16::nlanes)
    {
        constexpr static int nlanes = v_float32::nlanes;

        for (int l=0; l < length; )
        {
            for (; l <= length - nlanes; l += nlanes)
            {
                v_float32 sum = vx_load(&buf[r0][l]) * vx_setall_f32(ky0);
                    sum = v_fma(vx_load(&buf[r1][l]),  vx_setall_f32(ky1), sum);
                    sum = v_fma(vx_load(&buf[r2][l]),  vx_setall_f32(ky2), sum);

                if (!noscale)
                {
                    sum = v_fma(sum, vx_setall_f32(scale), vx_setall_f32(delta));
                }

                v_store(reinterpret_cast<float*>(&out[l]), sum);
            }

            if (l < length)
            {
                // tail: recalculate last pixels
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }

        return;
    }
#endif

    if ((std::is_same<DST, short>::value || std::is_same<DST, ushort>::value)
        && length >= v_int16::nlanes)
    {
        constexpr static int nlanes = v_int16::nlanes;

        for (int l=0; l < length; )
        {
            for (; l <= length - nlanes; l += nlanes)
            {
                v_float32 sum0 = vx_load(&buf[r0][l])            * vx_setall_f32(ky0);
                    sum0 = v_fma(vx_load(&buf[r1][l]),             vx_setall_f32(ky1), sum0);
                    sum0 = v_fma(vx_load(&buf[r2][l]),             vx_setall_f32(ky2), sum0);

                v_float32 sum1 = vx_load(&buf[r0][l + nlanes/2]) * vx_setall_f32(ky0);
                    sum1 = v_fma(vx_load(&buf[r1][l + nlanes/2]),  vx_setall_f32(ky1), sum1);
                    sum1 = v_fma(vx_load(&buf[r2][l + nlanes/2]),  vx_setall_f32(ky2), sum1);

                if (!noscale)
                {
                    sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                    sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
                }

                v_int32 isum0 = v_round(sum0),
                        isum1 = v_round(sum1);

                if (std::is_same<DST, short>::value)
                {
                    // signed short
                    v_int16 res = v_pack(isum0, isum1);
                    v_store(reinterpret_cast<short*>(&out[l]), res);
                } else
                {
                    // unsigned short
                    v_uint16 res = v_pack_u(isum0, isum1);
                    v_store(reinterpret_cast<ushort*>(&out[l]), res);
                }
            }

            if (l < length)
            {
                // tail: recalculate last pixels
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }

        return;
    }

    if (std::is_same<DST, uchar>::value && length >= v_uint8::nlanes)
    {
        constexpr static int nlanes = v_uint8::nlanes;

        for (int l=0; l < length; )
        {
            for (; l <= length - nlanes; l += nlanes)
            {
                v_float32 sum0 = vx_load(&buf[r0][l])              * vx_setall_f32(ky0);
                    sum0 = v_fma(vx_load(&buf[r1][l]),               vx_setall_f32(ky1), sum0);
                    sum0 = v_fma(vx_load(&buf[r2][l]),               vx_setall_f32(ky2), sum0);

                v_float32 sum1 = vx_load(&buf[r0][l +   nlanes/4]) * vx_setall_f32(ky0);
                    sum1 = v_fma(vx_load(&buf[r1][l +   nlanes/4]),  vx_setall_f32(ky1), sum1);
                    sum1 = v_fma(vx_load(&buf[r2][l +   nlanes/4]),  vx_setall_f32(ky2), sum1);

                v_float32 sum2 = vx_load(&buf[r0][l + 2*nlanes/4]) * vx_setall_f32(ky0);
                    sum2 = v_fma(vx_load(&buf[r1][l + 2*nlanes/4]),  vx_setall_f32(ky1), sum2);
                    sum2 = v_fma(vx_load(&buf[r2][l + 2*nlanes/4]),  vx_setall_f32(ky2), sum2);

                v_float32 sum3 = vx_load(&buf[r0][l + 3*nlanes/4]) * vx_setall_f32(ky0);
                    sum3 = v_fma(vx_load(&buf[r1][l + 3*nlanes/4]),  vx_setall_f32(ky1), sum3);
                    sum3 = v_fma(vx_load(&buf[r2][l + 3*nlanes/4]),  vx_setall_f32(ky2), sum3);

                if (!noscale)
                {
                    sum0 = v_fma(sum0, vx_setall_f32(scale), vx_setall_f32(delta));
                    sum1 = v_fma(sum1, vx_setall_f32(scale), vx_setall_f32(delta));
                    sum2 = v_fma(sum2, vx_setall_f32(scale), vx_setall_f32(delta));
                    sum3 = v_fma(sum3, vx_setall_f32(scale), vx_setall_f32(delta));
                }

                v_int32 isum0 = v_round(sum0),
                        isum1 = v_round(sum1),
                        isum2 = v_round(sum2),
                        isum3 = v_round(sum3);

                v_int16 ires0 = v_pack(isum0, isum1),
                        ires1 = v_pack(isum2, isum3);

                v_uint8 res = v_pack_u(ires0, ires1);
                v_store(reinterpret_cast<uchar*>(&out[l]), res);
            }

            if (l < length)
            {
                // tail: recalculate last pixels
                GAPI_DbgAssert(length >= nlanes);
                l = length - nlanes;
            }
        }

        return;
    }
#endif

    // reference code
    for (int l=0; l < length; l++)
    {
        float sum = buf[r0][l]*ky0 + buf[r1][l]*ky1 + buf[r2][l]*ky2;

        if (!noscale)
        {
            sum = sum*scale + delta;
        }

        out[l] = saturate<DST>(sum, rintf);
    }
}

template<typename DST, typename SRC>
void run_sobel_impl(DST out[], const SRC *in[], int width, int chan,
                    const float kx[], const float ky[], int border,
                    float scale, float delta, float *buf[],
                    int y, int y0)
{
    int r[3];
    r[0] = (y - y0)     % 3;  // buf[r[0]]: previous
    r[1] = (y - y0 + 1) % 3;  //            this
    r[2] = (y - y0 + 2) % 3;  //            next row

    int length = width * chan;

    // horizontal pass

    // full horizontal pass is needed only if very 1st row in ROI;
    // for 2nd and further rows, it is enough to convolve only the
    // "next" row - as we can reuse buffers from previous calls to
    // this kernel (note that Fluid processes rows consequently)
    int k0 = (y == y0)? 0: 2;

    for (int k = k0; k < 3; k++)
    {
        //                             previous, this , next pixel
        const SRC *s[3] = {in[k] - border*chan , in[k], in[k] + border*chan};

        // rely on compiler vectoring
        for (int l=0; l < length; l++)
        {
            buf[r[k]][l] = s[0][l]*kx[0] + s[1][l]*kx[1] + s[2][l]*kx[2];
        }
    }

    // vertical pass
    if (scale == 1 && delta == 0)
    {
        constexpr static bool noscale = true;  // omit scaling
        run_sobel3x3_vert<noscale, DST>(out, length, ky, scale, delta, r, buf);
    } else
    {
        constexpr static bool noscale = false;  // do scaling
        run_sobel3x3_vert<noscale, DST>(out, length, ky, scale, delta, r, buf);
    }
}

#define INSTANTIATE(DST, SRC)                                                 \
template void run_sobel_impl(DST out[], const SRC *in[], int width, int chan, \
                             const float kx[], const float ky[], int border,  \
                             float scale, float delta, float *buf[],          \
                             int y, int y0);

INSTANTIATE(uchar , uchar )
INSTANTIATE(ushort, ushort)
INSTANTIATE( short, uchar )
INSTANTIATE( short, ushort)
INSTANTIATE( short,  short)
INSTANTIATE( float, uchar )
INSTANTIATE( float, ushort)
INSTANTIATE( float,  short)
INSTANTIATE( float,  float)

#undef INSTANTIATE

} // namespace fliud
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)
