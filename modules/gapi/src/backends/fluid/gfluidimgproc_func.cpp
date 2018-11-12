// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#if !defined(GAPI_STANDALONE)

#include "gfluidimgproc_func.hpp"

#include "gfluidutils.hpp"

#include <opencv2/core/hal/intrin.hpp>

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
    // for floating-point output, manual vectoring is not better than compiler optimization
#if 0
    if (std::is_same<DST, float>::value && length >= v_int16::nlanes)
    {
        constexpr static int nlanes = v_float32::nlanes;

        for (int l=0; l < length; )
        {
            for (; l <= length - nlanes; l += nlanes)
            {
                v_float32 sum = v_load(&buf[r0][l]) * v_setall_f32(ky0);
                    sum = v_fma(v_load(&buf[r1][l]),  v_setall_f32(ky1), sum);
                    sum = v_fma(v_load(&buf[r2][l]),  v_setall_f32(ky2), sum);

                if (!noscale)
                {
                    sum = v_fma(sum, v_setall_f32(scale), v_setall_f32(delta));
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
                v_float32 sum0 = v_load(&buf[r0][l])            * v_setall_f32(ky0);
                    sum0 = v_fma(v_load(&buf[r1][l]),             v_setall_f32(ky1), sum0);
                    sum0 = v_fma(v_load(&buf[r2][l]),             v_setall_f32(ky2), sum0);

                v_float32 sum1 = v_load(&buf[r0][l + nlanes/2]) * v_setall_f32(ky0);
                    sum1 = v_fma(v_load(&buf[r1][l + nlanes/2]),  v_setall_f32(ky1), sum1);
                    sum1 = v_fma(v_load(&buf[r2][l + nlanes/2]),  v_setall_f32(ky2), sum1);

                if (!noscale)
                {
                    sum0 = v_fma(sum0, v_setall_f32(scale), v_setall_f32(delta));
                    sum1 = v_fma(sum1, v_setall_f32(scale), v_setall_f32(delta));
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
                v_float32 sum0 = v_load(&buf[r0][l])              * v_setall_f32(ky0);
                    sum0 = v_fma(v_load(&buf[r1][l]),               v_setall_f32(ky1), sum0);
                    sum0 = v_fma(v_load(&buf[r2][l]),               v_setall_f32(ky2), sum0);

                v_float32 sum1 = v_load(&buf[r0][l +   nlanes/4]) * v_setall_f32(ky0);
                    sum1 = v_fma(v_load(&buf[r1][l +   nlanes/4]),  v_setall_f32(ky1), sum1);
                    sum1 = v_fma(v_load(&buf[r2][l +   nlanes/4]),  v_setall_f32(ky2), sum1);

                v_float32 sum2 = v_load(&buf[r0][l + 2*nlanes/4]) * v_setall_f32(ky0);
                    sum2 = v_fma(v_load(&buf[r1][l + 2*nlanes/4]),  v_setall_f32(ky1), sum2);
                    sum2 = v_fma(v_load(&buf[r2][l + 2*nlanes/4]),  v_setall_f32(ky2), sum2);

                v_float32 sum3 = v_load(&buf[r0][l + 3*nlanes/4]) * v_setall_f32(ky0);
                    sum3 = v_fma(v_load(&buf[r1][l + 3*nlanes/4]),  v_setall_f32(ky1), sum3);
                    sum3 = v_fma(v_load(&buf[r2][l + 3*nlanes/4]),  v_setall_f32(ky2), sum3);

                if (!noscale)
                {
                    sum0 = v_fma(sum0, v_setall_f32(scale), v_setall_f32(delta));
                    sum1 = v_fma(sum1, v_setall_f32(scale), v_setall_f32(delta));
                    sum2 = v_fma(sum2, v_setall_f32(scale), v_setall_f32(delta));
                    sum3 = v_fma(sum3, v_setall_f32(scale), v_setall_f32(delta));
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

#define INSTANTIATE(noscale, DST) \
template void run_sobel3x3_vert<noscale, DST>(DST out[], int length, const float ky[], \
                                float scale, float delta, const int r[], float *buf[]);

INSTANTIATE(true , uchar )
INSTANTIATE(false, uchar )
INSTANTIATE(true , ushort)
INSTANTIATE(false, ushort)
INSTANTIATE(true ,  short)
INSTANTIATE(false,  short)
INSTANTIATE(true ,  float)
INSTANTIATE(false,  float)

#undef INSTANTIATE

} // namespace fliud
} // namespace gapi
} // namespace cv

#endif // !defined(GAPI_STANDALONE)
