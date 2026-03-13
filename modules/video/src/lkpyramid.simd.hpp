// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.
#include "precomp.hpp"
#include "lkpyramid.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace detail {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// SIMD-optimized implementation
void ScharrDerivInvoker_SIMD(const Mat& src, Mat& dst, const Range& range);

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // namespace detail
} // namespace cv

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace detail {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void ScharrDerivInvoker_SIMD(const Mat& src, Mat& dst, const Range& range)
{
    typedef short deriv_type;
    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols*cn;

    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta*2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

#if (CV_SIMD)
    const int vlanes = VTraits<v_int16>::vlanes();
    v_int16 c3 = vx_setall_s16(3), c10 = vx_setall_s16(10);
#endif

    for( y = range.start; y < range.end; y++ )
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y-1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows-1 ? y+1 : rows > 1 ? rows-2 : 0);
        deriv_type* drow = (deriv_type *)dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;
#if (CV_SIMD)
        {
            for( ; x <= colsn - vlanes; x += vlanes )
            {
                v_int16 s0 = v_reinterpret_as_s16(vx_load_expand(srow0 + x));
                v_int16 s1 = v_reinterpret_as_s16(vx_load_expand(srow1 + x));
                v_int16 s2 = v_reinterpret_as_s16(vx_load_expand(srow2 + x));

                v_int16 t1 = v_sub(s2, s0);
                v_int16 t0 = v_add(v_mul_wrap(v_add(s0, s2), c3), v_mul_wrap(s1, c10));

                v_store(trow0 + x, t0);
                v_store(trow1 + x, t1);
            }
        }
#endif

        for( ; x < colsn; x++ )
        {
            int t0 = (srow0[x] + srow2[x])*3 + srow1[x]*10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols-2 : 0)*cn;
        for( int k = 0; k < cn; k++ )
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
#if (CV_SIMD)
        {
            for( ; x <= colsn - vlanes; x += vlanes )
            {
                v_int16 s0 = vx_load(trow0 + x - cn);
                v_int16 s1 = vx_load(trow0 + x + cn);
                v_int16 s2 = vx_load(trow1 + x - cn);
                v_int16 s3 = vx_load(trow1 + x);
                v_int16 s4 = vx_load(trow1 + x + cn);

                v_int16 t0 = v_sub(s1, s0);
                v_int16 t1 = v_add(v_mul_wrap(v_add(s2, s4), c3), v_mul_wrap(s3, c10));

                v_store_interleave((drow + x*2), t0, t1);
            }
        }
#endif
        for( ; x < colsn; x++ )
        {
            deriv_type t0 = (deriv_type)(trow0[x+cn] - trow0[x-cn]);
            deriv_type t1 = (deriv_type)((trow1[x+cn] + trow1[x-cn])*3 + trow1[x]*10);
            drow[x*2] = t0; drow[x*2+1] = t1;
        }
    }
}

CV_CPU_OPTIMIZATION_NAMESPACE_END

} // namespace detail
} // namespace cv

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
