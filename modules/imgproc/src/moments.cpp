/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"

namespace cv
{

// The function calculates center of gravity and the central second order moments
static void completeMomentState( Moments* moments )
{
    double cx = 0, cy = 0;
    double mu20, mu11, mu02;
    double inv_m00 = 0.0;
    assert( moments != 0 );

    if( fabs(moments->m00) > DBL_EPSILON )
    {
        inv_m00 = 1. / moments->m00;
        cx = moments->m10 * inv_m00;
        cy = moments->m01 * inv_m00;
    }

    // mu20 = m20 - m10*cx
    mu20 = moments->m20 - moments->m10 * cx;
    // mu11 = m11 - m10*cy
    mu11 = moments->m11 - moments->m10 * cy;
    // mu02 = m02 - m01*cy
    mu02 = moments->m02 - moments->m01 * cy;

    moments->mu20 = mu20;
    moments->mu11 = mu11;
    moments->mu02 = mu02;

    // mu30 = m30 - cx*(3*mu20 + cx*m10)
    moments->mu30 = moments->m30 - cx * (3 * mu20 + cx * moments->m10);
    mu11 += mu11;
    // mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20
    moments->mu21 = moments->m21 - cx * (mu11 + cx * moments->m01) - cy * mu20;
    // mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02
    moments->mu12 = moments->m12 - cy * (mu11 + cy * moments->m10) - cx * mu02;
    // mu03 = m03 - cy*(3*mu02 + cy*m01)
    moments->mu03 = moments->m03 - cy * (3 * mu02 + cy * moments->m01);


    double inv_sqrt_m00 = std::sqrt(std::abs(inv_m00));
    double s2 = inv_m00*inv_m00, s3 = s2*inv_sqrt_m00;

    moments->nu20 = moments->mu20*s2; moments->nu11 = moments->mu11*s2; moments->nu02 = moments->mu02*s2;
    moments->nu30 = moments->mu30*s3; moments->nu21 = moments->mu21*s3; moments->nu12 = moments->mu12*s3; moments->nu03 = moments->mu03*s3;

}


static Moments contourMoments( const Mat& contour )
{
    Moments m;
    int lpt = contour.checkVector(2);
    int is_float = contour.depth() == CV_32F;
    const Point* ptsi = contour.ptr<Point>();
    const Point2f* ptsf = contour.ptr<Point2f>();

    CV_Assert( contour.depth() == CV_32S || contour.depth() == CV_32F );

    if( lpt == 0 )
        return m;

    double a00 = 0, a10 = 0, a01 = 0, a20 = 0, a11 = 0, a02 = 0, a30 = 0, a21 = 0, a12 = 0, a03 = 0;
    double xi, yi, xi2, yi2, xi_1, yi_1, xi_12, yi_12, dxy, xii_1, yii_1;

    if( !is_float )
    {
        xi_1 = ptsi[lpt-1].x;
        yi_1 = ptsi[lpt-1].y;
    }
    else
    {
        xi_1 = ptsf[lpt-1].x;
        yi_1 = ptsf[lpt-1].y;
    }

    xi_12 = xi_1 * xi_1;
    yi_12 = yi_1 * yi_1;

    for( int i = 0; i < lpt; i++ )
    {
        if( !is_float )
        {
            xi = ptsi[i].x;
            yi = ptsi[i].y;
        }
        else
        {
            xi = ptsf[i].x;
            yi = ptsf[i].y;
        }

        xi2 = xi * xi;
        yi2 = yi * yi;
        dxy = xi_1 * yi - xi * yi_1;
        xii_1 = xi_1 + xi;
        yii_1 = yi_1 + yi;

        a00 += dxy;
        a10 += dxy * xii_1;
        a01 += dxy * yii_1;
        a20 += dxy * (xi_1 * xii_1 + xi2);
        a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
        a02 += dxy * (yi_1 * yii_1 + yi2);
        a30 += dxy * xii_1 * (xi_12 + xi2);
        a03 += dxy * yii_1 * (yi_12 + yi2);
        a21 += dxy * (xi_12 * (3 * yi_1 + yi) + 2 * xi * xi_1 * yii_1 +
                   xi2 * (yi_1 + 3 * yi));
        a12 += dxy * (yi_12 * (3 * xi_1 + xi) + 2 * yi * yi_1 * xii_1 +
                   yi2 * (xi_1 + 3 * xi));
        xi_1 = xi;
        yi_1 = yi;
        xi_12 = xi2;
        yi_12 = yi2;
    }

    if( fabs(a00) > FLT_EPSILON )
    {
        double db1_2, db1_6, db1_12, db1_24, db1_20, db1_60;

        if( a00 > 0 )
        {
            db1_2 = 0.5;
            db1_6 = 0.16666666666666666666666666666667;
            db1_12 = 0.083333333333333333333333333333333;
            db1_24 = 0.041666666666666666666666666666667;
            db1_20 = 0.05;
            db1_60 = 0.016666666666666666666666666666667;
        }
        else
        {
            db1_2 = -0.5;
            db1_6 = -0.16666666666666666666666666666667;
            db1_12 = -0.083333333333333333333333333333333;
            db1_24 = -0.041666666666666666666666666666667;
            db1_20 = -0.05;
            db1_60 = -0.016666666666666666666666666666667;
        }

        // spatial moments
        m.m00 = a00 * db1_2;
        m.m10 = a10 * db1_6;
        m.m01 = a01 * db1_6;
        m.m20 = a20 * db1_12;
        m.m11 = a11 * db1_24;
        m.m02 = a02 * db1_12;
        m.m30 = a30 * db1_20;
        m.m21 = a21 * db1_60;
        m.m12 = a12 * db1_60;
        m.m03 = a03 * db1_20;

        completeMomentState( &m );
    }
    return m;
}


/****************************************************************************************\
*                                Spatial Raster Moments                                  *
\****************************************************************************************/

template<typename T, typename WT, typename MT>
struct MomentsInTile_SIMD
{
    int operator() (const T *, int, WT &, WT &, WT &, MT &)
    {
        return 0;
    }
};

#if CV_SSE2

template <>
struct MomentsInTile_SIMD<uchar, int, int>
{
    MomentsInTile_SIMD()
    {
        useSIMD = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const uchar * ptr, int len, int & x0, int & x1, int & x2, int & x3)
    {
        int x = 0;

        if( useSIMD )
        {
            __m128i dx = _mm_set1_epi16(8);
            __m128i z = _mm_setzero_si128(), qx0 = z, qx1 = z, qx2 = z, qx3 = z, qx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);

            for( ; x <= len - 8; x += 8 )
            {
                __m128i p = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(ptr + x)), z);
                __m128i sx = _mm_mullo_epi16(qx, qx);

                qx0 = _mm_add_epi16(qx0, p);
                qx1 = _mm_add_epi32(qx1, _mm_madd_epi16(p, qx));
                qx2 = _mm_add_epi32(qx2, _mm_madd_epi16(p, sx));
                qx3 = _mm_add_epi32(qx3, _mm_madd_epi16( _mm_mullo_epi16(p, qx), sx));

                qx = _mm_add_epi16(qx, dx);
            }

            __m128i qx01_lo = _mm_unpacklo_epi32(qx0, qx1);
            __m128i qx23_lo = _mm_unpacklo_epi32(qx2, qx3);
            __m128i qx01_hi = _mm_unpackhi_epi32(qx0, qx1);
            __m128i qx23_hi = _mm_unpackhi_epi32(qx2, qx3);
            qx01_lo = _mm_add_epi32(qx01_lo, qx01_hi);
            qx23_lo = _mm_add_epi32(qx23_lo, qx23_hi);
            __m128i qx0123_lo = _mm_unpacklo_epi64(qx01_lo, qx23_lo);
            __m128i qx0123_hi = _mm_unpackhi_epi64(qx01_lo, qx23_lo);
            qx0123_lo = _mm_add_epi32(qx0123_lo, qx0123_hi);
            _mm_store_si128((__m128i*)buf, qx0123_lo);

            x0 = (buf[0] & 0xffff) + (buf[0] >> 16);
            x1 = buf[1];
            x2 = buf[2];
            x3 = buf[3];
        }

        return x;
    }

    int CV_DECL_ALIGNED(16) buf[4];
    bool useSIMD;
};

#elif CV_NEON

template <>
struct MomentsInTile_SIMD<uchar, int, int>
{
    MomentsInTile_SIMD()
    {
        ushort CV_DECL_ALIGNED(8) init[4] = { 0, 1, 2, 3 };
        qx_init = vld1_u16(init);
        v_step = vdup_n_u16(4);
    }

    int operator() (const uchar * ptr, int len, int & x0, int & x1, int & x2, int & x3)
    {
        int x = 0;

        uint32x4_t v_z = vdupq_n_u32(0), v_x0 = v_z, v_x1 = v_z,
            v_x2 = v_z, v_x3 = v_z;
        uint16x4_t qx = qx_init;

        for( ; x <= len - 8; x += 8 )
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(ptr + x));

            // first part
            uint32x4_t v_qx = vmovl_u16(qx);
            uint16x4_t v_p = vget_low_u16(v_src);
            uint32x4_t v_px = vmull_u16(qx, v_p);

            v_x0 = vaddw_u16(v_x0, v_p);
            v_x1 = vaddq_u32(v_x1, v_px);
            v_px = vmulq_u32(v_px, v_qx);
            v_x2 = vaddq_u32(v_x2, v_px);
            v_x3 = vaddq_u32(v_x3, vmulq_u32(v_px, v_qx));
            qx = vadd_u16(qx, v_step);

            // second part
            v_qx = vmovl_u16(qx);
            v_p = vget_high_u16(v_src);
            v_px = vmull_u16(qx, v_p);

            v_x0 = vaddw_u16(v_x0, v_p);
            v_x1 = vaddq_u32(v_x1, v_px);
            v_px = vmulq_u32(v_px, v_qx);
            v_x2 = vaddq_u32(v_x2, v_px);
            v_x3 = vaddq_u32(v_x3, vmulq_u32(v_px, v_qx));

            qx = vadd_u16(qx, v_step);
        }

        vst1q_u32(buf, v_x0);
        x0 = buf[0] + buf[1] + buf[2] + buf[3];
        vst1q_u32(buf, v_x1);
        x1 = buf[0] + buf[1] + buf[2] + buf[3];
        vst1q_u32(buf, v_x2);
        x2 = buf[0] + buf[1] + buf[2] + buf[3];
        vst1q_u32(buf, v_x3);
        x3 = buf[0] + buf[1] + buf[2] + buf[3];

        return x;
    }

    uint CV_DECL_ALIGNED(16) buf[4];
    uint16x4_t qx_init, v_step;
};

#endif

#if CV_SSE4_1

template <>
struct MomentsInTile_SIMD<ushort, int, int64>
{
    MomentsInTile_SIMD()
    {
        useSIMD = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator() (const ushort * ptr, int len, int & x0, int & x1, int & x2, int64 & x3)
    {
        int x = 0;

        if (useSIMD)
        {
            __m128i v_delta = _mm_set1_epi32(4), v_zero = _mm_setzero_si128(), v_x0 = v_zero,
                v_x1 = v_zero, v_x2 = v_zero, v_x3 = v_zero, v_ix0 = _mm_setr_epi32(0, 1, 2, 3);

            for( ; x <= len - 4; x += 4 )
            {
                __m128i v_src = _mm_loadl_epi64((const __m128i *)(ptr + x));
                v_src = _mm_unpacklo_epi16(v_src, v_zero);

                v_x0 = _mm_add_epi32(v_x0, v_src);
                v_x1 = _mm_add_epi32(v_x1, _mm_mullo_epi32(v_src, v_ix0));

                __m128i v_ix1 = _mm_mullo_epi32(v_ix0, v_ix0);
                v_x2 = _mm_add_epi32(v_x2, _mm_mullo_epi32(v_src, v_ix1));

                v_ix1 = _mm_mullo_epi32(v_ix0, v_ix1);
                v_src = _mm_mullo_epi32(v_src, v_ix1);
                v_x3 = _mm_add_epi64(v_x3, _mm_add_epi64(_mm_unpacklo_epi32(v_src, v_zero), _mm_unpackhi_epi32(v_src, v_zero)));

                v_ix0 = _mm_add_epi32(v_ix0, v_delta);
            }

            __m128i v_x01_lo = _mm_unpacklo_epi32(v_x0, v_x1);
            __m128i v_x22_lo = _mm_unpacklo_epi32(v_x2, v_x2);
            __m128i v_x01_hi = _mm_unpackhi_epi32(v_x0, v_x1);
            __m128i v_x22_hi = _mm_unpackhi_epi32(v_x2, v_x2);
            v_x01_lo = _mm_add_epi32(v_x01_lo, v_x01_hi);
            v_x22_lo = _mm_add_epi32(v_x22_lo, v_x22_hi);
            __m128i v_x0122_lo = _mm_unpacklo_epi64(v_x01_lo, v_x22_lo);
            __m128i v_x0122_hi = _mm_unpackhi_epi64(v_x01_lo, v_x22_lo);
            v_x0122_lo = _mm_add_epi32(v_x0122_lo, v_x0122_hi);
            _mm_store_si128((__m128i*)buf64, v_x3);
            _mm_store_si128((__m128i*)buf, v_x0122_lo);

            x0 = buf[0];
            x1 = buf[1];
            x2 = buf[2];
            x3 = buf64[0] + buf64[1];
        }

        return x;
    }

    int CV_DECL_ALIGNED(16) buf[4];
    int64 CV_DECL_ALIGNED(16) buf64[2];
    bool useSIMD;
};

#endif

template<typename T, typename WT, typename MT>
#if defined __GNUC__ && __GNUC__ == 4 && __GNUC_MINOR__ >= 5 && __GNUC_MINOR__ < 9
// Workaround for http://gcc.gnu.org/bugzilla/show_bug.cgi?id=60196
__attribute__((optimize("no-tree-vectorize")))
#endif
static void momentsInTile( const Mat& img, double* moments )
{
    Size size = img.size();
    int x, y;
    MT mom[10] = {0,0,0,0,0,0,0,0,0,0};
    MomentsInTile_SIMD<T, WT, MT> vop;

    for( y = 0; y < size.height; y++ )
    {
        const T* ptr = img.ptr<T>(y);
        WT x0 = 0, x1 = 0, x2 = 0;
        MT x3 = 0;
        x = vop(ptr, size.width, x0, x1, x2, x3);

        for( ; x < size.width; x++ )
        {
            WT p = ptr[x];
            WT xp = x * p, xxp;

            x0 += p;
            x1 += xp;
            xxp = xp * x;
            x2 += xxp;
            x3 += xxp * x;
        }

        WT py = y * x0, sy = y*y;

        mom[9] += ((MT)py) * sy;  // m03
        mom[8] += ((MT)x1) * sy;  // m12
        mom[7] += ((MT)x2) * y;  // m21
        mom[6] += x3;             // m30
        mom[5] += x0 * sy;        // m02
        mom[4] += x1 * y;         // m11
        mom[3] += x2;             // m20
        mom[2] += py;             // m01
        mom[1] += x1;             // m10
        mom[0] += x0;             // m00
    }

    for( x = 0; x < 10; x++ )
        moments[x] = (double)mom[x];
}

typedef void (*MomentsInTileFunc)(const Mat& img, double* moments);

Moments::Moments()
{
    m00 = m10 = m01 = m20 = m11 = m02 = m30 = m21 = m12 = m03 =
    mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 =
    nu20 = nu11 = nu02 = nu30 = nu21 = nu12 = nu03 = 0.;
}

Moments::Moments( double _m00, double _m10, double _m01, double _m20, double _m11,
                  double _m02, double _m30, double _m21, double _m12, double _m03 )
{
    m00 = _m00; m10 = _m10; m01 = _m01;
    m20 = _m20; m11 = _m11; m02 = _m02;
    m30 = _m30; m21 = _m21; m12 = _m12; m03 = _m03;

    double cx = 0, cy = 0, inv_m00 = 0;
    if( std::abs(m00) > DBL_EPSILON )
    {
        inv_m00 = 1./m00;
        cx = m10*inv_m00; cy = m01*inv_m00;
    }

    mu20 = m20 - m10*cx;
    mu11 = m11 - m10*cy;
    mu02 = m02 - m01*cy;

    mu30 = m30 - cx*(3*mu20 + cx*m10);
    mu21 = m21 - cx*(2*mu11 + cx*m01) - cy*mu20;
    mu12 = m12 - cy*(2*mu11 + cy*m10) - cx*mu02;
    mu03 = m03 - cy*(3*mu02 + cy*m01);

    double inv_sqrt_m00 = std::sqrt(std::abs(inv_m00));
    double s2 = inv_m00*inv_m00, s3 = s2*inv_sqrt_m00;

    nu20 = mu20*s2; nu11 = mu11*s2; nu02 = mu02*s2;
    nu30 = mu30*s3; nu21 = mu21*s3; nu12 = mu12*s3; nu03 = mu03*s3;
}

#ifdef HAVE_OPENCL

static bool ocl_moments( InputArray _src, Moments& m, bool binary)
{
    const int TILE_SIZE = 32;
    const int K = 10;

    ocl::Kernel k = ocl::Kernel("moments", ocl::imgproc::moments_oclsrc,
        format("-D TILE_SIZE=%d%s",
        TILE_SIZE,
        binary ? " -D OP_MOMENTS_BINARY" : ""));

    if( k.empty() )
        return false;

    UMat src = _src.getUMat();
    Size sz = src.size();
    int xtiles = (sz.width + TILE_SIZE-1)/TILE_SIZE;
    int ytiles = (sz.height + TILE_SIZE-1)/TILE_SIZE;
    int ntiles = xtiles*ytiles;
    UMat umbuf(1, ntiles*K, CV_32S);

    size_t globalsize[] = {(size_t)xtiles, std::max((size_t)TILE_SIZE, (size_t)sz.height)};
    size_t localsize[] = {1, TILE_SIZE};
    bool ok = k.args(ocl::KernelArg::ReadOnly(src),
                     ocl::KernelArg::PtrWriteOnly(umbuf),
                     xtiles).run(2, globalsize, localsize, true);
    if(!ok)
        return false;
    Mat mbuf = umbuf.getMat(ACCESS_READ);
    for( int i = 0; i < ntiles; i++ )
    {
        double x = (i % xtiles)*TILE_SIZE, y = (i / xtiles)*TILE_SIZE;
        const int* mom = mbuf.ptr<int>() + i*K;
        double xm = x * mom[0], ym = y * mom[0];

        // accumulate moments computed in each tile

        // + m00 ( = m00' )
        m.m00 += mom[0];

        // + m10 ( = m10' + x*m00' )
        m.m10 += mom[1] + xm;

        // + m01 ( = m01' + y*m00' )
        m.m01 += mom[2] + ym;

        // + m20 ( = m20' + 2*x*m10' + x*x*m00' )
        m.m20 += mom[3] + x * (mom[1] * 2 + xm);

        // + m11 ( = m11' + x*m01' + y*m10' + x*y*m00' )
        m.m11 += mom[4] + x * (mom[2] + ym) + y * mom[1];

        // + m02 ( = m02' + 2*y*m01' + y*y*m00' )
        m.m02 += mom[5] + y * (mom[2] * 2 + ym);

        // + m30 ( = m30' + 3*x*m20' + 3*x*x*m10' + x*x*x*m00' )
        m.m30 += mom[6] + x * (3. * mom[3] + x * (3. * mom[1] + xm));

        // + m21 ( = m21' + x*(2*m11' + 2*y*m10' + x*m01' + x*y*m00') + y*m20')
        m.m21 += mom[7] + x * (2 * (mom[4] + y * mom[1]) + x * (mom[2] + ym)) + y * mom[3];

        // + m12 ( = m12' + y*(2*m11' + 2*x*m01' + y*m10' + x*y*m00') + x*m02')
        m.m12 += mom[8] + y * (2 * (mom[4] + x * mom[2]) + y * (mom[1] + xm)) + x * mom[5];

        // + m03 ( = m03' + 3*y*m02' + 3*y*y*m01' + y*y*y*m00' )
        m.m03 += mom[9] + y * (3. * mom[5] + y * (3. * mom[2] + ym));
    }

    completeMomentState( &m );

    return true;
}

#endif

#ifdef HAVE_IPP
typedef IppStatus (CV_STDCALL * ippiMoments)(const void* pSrc, int srcStep, IppiSize roiSize, IppiMomentState_64f* pCtx);

static bool ipp_moments(Mat &src, Moments &m )
{
#if IPP_VERSION_X100 >= 900
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 < 201801
    // Degradations for CV_8UC1
    if(src.type() == CV_8UC1)
        return false;
#endif

    IppiSize  roi      = { src.cols, src.rows };
    IppiPoint point    = { 0, 0 };
    int       type     = src.type();
    IppStatus ippStatus;

    IppAutoBuffer<IppiMomentState_64f> state;
    int stateSize = 0;

    ippiMoments ippiMoments64f =
        (type == CV_8UC1)?(ippiMoments)ippiMoments64f_8u_C1R:
        (type == CV_16UC1)?(ippiMoments)ippiMoments64f_16u_C1R:
        (type == CV_32FC1)?(ippiMoments)ippiMoments64f_32f_C1R:
        NULL;
    if(!ippiMoments64f)
        return false;

    ippStatus = ippiMomentGetStateSize_64f(ippAlgHintAccurate, &stateSize);
    if(ippStatus < 0)
        return false;

    if(!state.allocate(stateSize) && stateSize)
        return false;

    ippStatus = ippiMomentInit_64f(state, ippAlgHintAccurate);
    if(ippStatus < 0)
        return false;

    ippStatus = CV_INSTRUMENT_FUN_IPP(ippiMoments64f, src.ptr<Ipp8u>(), (int)src.step, roi, state);
    if(ippStatus < 0)
        return false;

    ippStatus = ippiGetSpatialMoment_64f(state, 0, 0, 0, point, &m.m00);
    if(ippStatus < 0)
        return false;
    ippiGetSpatialMoment_64f(state, 1, 0, 0, point, &m.m10);
    ippiGetSpatialMoment_64f(state, 0, 1, 0, point, &m.m01);
    ippiGetSpatialMoment_64f(state, 2, 0, 0, point, &m.m20);
    ippiGetSpatialMoment_64f(state, 1, 1, 0, point, &m.m11);
    ippiGetSpatialMoment_64f(state, 0, 2, 0, point, &m.m02);
    ippiGetSpatialMoment_64f(state, 3, 0, 0, point, &m.m30);
    ippiGetSpatialMoment_64f(state, 2, 1, 0, point, &m.m21);
    ippiGetSpatialMoment_64f(state, 1, 2, 0, point, &m.m12);
    ippiGetSpatialMoment_64f(state, 0, 3, 0, point, &m.m03);

    ippStatus = ippiGetCentralMoment_64f(state, 2, 0, 0, &m.mu20);
    if(ippStatus < 0)
        return false;
    ippiGetCentralMoment_64f(state, 1, 1, 0, &m.mu11);
    ippiGetCentralMoment_64f(state, 0, 2, 0, &m.mu02);
    ippiGetCentralMoment_64f(state, 3, 0, 0, &m.mu30);
    ippiGetCentralMoment_64f(state, 2, 1, 0, &m.mu21);
    ippiGetCentralMoment_64f(state, 1, 2, 0, &m.mu12);
    ippiGetCentralMoment_64f(state, 0, 3, 0, &m.mu03);

    ippStatus = ippiGetNormalizedCentralMoment_64f(state, 2, 0, 0, &m.nu20);
    if(ippStatus < 0)
        return false;
    ippiGetNormalizedCentralMoment_64f(state, 1, 1, 0, &m.nu11);
    ippiGetNormalizedCentralMoment_64f(state, 0, 2, 0, &m.nu02);
    ippiGetNormalizedCentralMoment_64f(state, 3, 0, 0, &m.nu30);
    ippiGetNormalizedCentralMoment_64f(state, 2, 1, 0, &m.nu21);
    ippiGetNormalizedCentralMoment_64f(state, 1, 2, 0, &m.nu12);
    ippiGetNormalizedCentralMoment_64f(state, 0, 3, 0, &m.nu03);

    return true;
#else
    CV_UNUSED(src); CV_UNUSED(m);
    return false;
#endif
}
#endif

}

cv::Moments cv::moments( InputArray _src, bool binary )
{
    CV_INSTRUMENT_REGION()

    const int TILE_SIZE = 32;
    MomentsInTileFunc func = 0;
    uchar nzbuf[TILE_SIZE*TILE_SIZE];
    Moments m;
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Size size = _src.size();

    if( size.width <= 0 || size.height <= 0 )
        return m;

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(type == CV_8UC1 && _src.isUMat(), ocl_moments(_src, m, binary), m);
#endif

    Mat mat = _src.getMat();
    if( mat.checkVector(2) >= 0 && (depth == CV_32F || depth == CV_32S))
        return contourMoments(mat);

    if( cn > 1 )
        CV_Error( CV_StsBadArg, "Invalid image type (must be single-channel)" );

    CV_IPP_RUN(!binary, ipp_moments(mat, m), m);

    if( binary || depth == CV_8U )
        func = momentsInTile<uchar, int, int>;
    else if( depth == CV_16U )
        func = momentsInTile<ushort, int, int64>;
    else if( depth == CV_16S )
        func = momentsInTile<short, int, int64>;
    else if( depth == CV_32F )
        func = momentsInTile<float, double, double>;
    else if( depth == CV_64F )
        func = momentsInTile<double, double, double>;
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    Mat src0(mat);

    for( int y = 0; y < size.height; y += TILE_SIZE )
    {
        Size tileSize;
        tileSize.height = std::min(TILE_SIZE, size.height - y);

        for( int x = 0; x < size.width; x += TILE_SIZE )
        {
            tileSize.width = std::min(TILE_SIZE, size.width - x);
            Mat src(src0, cv::Rect(x, y, tileSize.width, tileSize.height));

            if( binary )
            {
                cv::Mat tmp(tileSize, CV_8U, nzbuf);
                cv::compare( src, 0, tmp, CV_CMP_NE );
                src = tmp;
            }

            double mom[10];
            func( src, mom );

            if(binary)
            {
                double s = 1./255;
                for( int k = 0; k < 10; k++ )
                    mom[k] *= s;
            }

            double xm = x * mom[0], ym = y * mom[0];

            // accumulate moments computed in each tile

            // + m00 ( = m00' )
            m.m00 += mom[0];

            // + m10 ( = m10' + x*m00' )
            m.m10 += mom[1] + xm;

            // + m01 ( = m01' + y*m00' )
            m.m01 += mom[2] + ym;

            // + m20 ( = m20' + 2*x*m10' + x*x*m00' )
            m.m20 += mom[3] + x * (mom[1] * 2 + xm);

            // + m11 ( = m11' + x*m01' + y*m10' + x*y*m00' )
            m.m11 += mom[4] + x * (mom[2] + ym) + y * mom[1];

            // + m02 ( = m02' + 2*y*m01' + y*y*m00' )
            m.m02 += mom[5] + y * (mom[2] * 2 + ym);

            // + m30 ( = m30' + 3*x*m20' + 3*x*x*m10' + x*x*x*m00' )
            m.m30 += mom[6] + x * (3. * mom[3] + x * (3. * mom[1] + xm));

            // + m21 ( = m21' + x*(2*m11' + 2*y*m10' + x*m01' + x*y*m00') + y*m20')
            m.m21 += mom[7] + x * (2 * (mom[4] + y * mom[1]) + x * (mom[2] + ym)) + y * mom[3];

            // + m12 ( = m12' + y*(2*m11' + 2*x*m01' + y*m10' + x*y*m00') + x*m02')
            m.m12 += mom[8] + y * (2 * (mom[4] + x * mom[2]) + y * (mom[1] + xm)) + x * mom[5];

            // + m03 ( = m03' + 3*y*m02' + 3*y*y*m01' + y*y*y*m00' )
            m.m03 += mom[9] + y * (3. * mom[5] + y * (3. * mom[2] + ym));
        }
    }

    completeMomentState( &m );
    return m;
}


void cv::HuMoments( const Moments& m, double hu[7] )
{
    CV_INSTRUMENT_REGION()

    double t0 = m.nu30 + m.nu12;
    double t1 = m.nu21 + m.nu03;

    double q0 = t0 * t0, q1 = t1 * t1;

    double n4 = 4 * m.nu11;
    double s = m.nu20 + m.nu02;
    double d = m.nu20 - m.nu02;

    hu[0] = s;
    hu[1] = d * d + n4 * m.nu11;
    hu[3] = q0 + q1;
    hu[5] = d * (q0 - q1) + n4 * t0 * t1;

    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;

    q0 = m.nu30 - 3 * m.nu12;
    q1 = 3 * m.nu21 - m.nu03;

    hu[2] = q0 * q0 + q1 * q1;
    hu[4] = q0 * t0 + q1 * t1;
    hu[6] = q1 * t0 - q0 * t1;
}

void cv::HuMoments( const Moments& m, OutputArray _hu )
{
    CV_INSTRUMENT_REGION()

    _hu.create(7, 1, CV_64F);
    Mat hu = _hu.getMat();
    CV_Assert( hu.isContinuous() );
    HuMoments(m, hu.ptr<double>());
}


CV_IMPL void cvMoments( const CvArr* arr, CvMoments* moments, int binary )
{
    const IplImage* img = (const IplImage*)arr;
    cv::Mat src;
    if( CV_IS_IMAGE(arr) && img->roi && img->roi->coi > 0 )
        cv::extractImageCOI(arr, src, img->roi->coi-1);
    else
        src = cv::cvarrToMat(arr);
    cv::Moments m = cv::moments(src, binary != 0);
    CV_Assert( moments != 0 );
    *moments = m;
}


CV_IMPL double cvGetSpatialMoment( CvMoments * moments, int x_order, int y_order )
{
    int order = x_order + y_order;

    if( !moments )
        CV_Error( CV_StsNullPtr, "" );
    if( (x_order | y_order) < 0 || order > 3 )
        CV_Error( CV_StsOutOfRange, "" );

    return (&(moments->m00))[order + (order >> 1) + (order > 2) * 2 + y_order];
}


CV_IMPL double cvGetCentralMoment( CvMoments * moments, int x_order, int y_order )
{
    int order = x_order + y_order;

    if( !moments )
        CV_Error( CV_StsNullPtr, "" );
    if( (x_order | y_order) < 0 || order > 3 )
        CV_Error( CV_StsOutOfRange, "" );

    return order >= 2 ? (&(moments->m00))[4 + order * 3 + y_order] :
    order == 0 ? moments->m00 : 0;
}


CV_IMPL double cvGetNormalizedCentralMoment( CvMoments * moments, int x_order, int y_order )
{
    int order = x_order + y_order;

    double mu = cvGetCentralMoment( moments, x_order, y_order );
    double m00s = moments->inv_sqrt_m00;

    while( --order >= 0 )
        mu *= m00s;
    return mu * m00s * m00s;
}


CV_IMPL void cvGetHuMoments( CvMoments * mState, CvHuMoments * HuState )
{
    if( !mState || !HuState )
        CV_Error( CV_StsNullPtr, "" );

    double m00s = mState->inv_sqrt_m00, m00 = m00s * m00s, s2 = m00 * m00, s3 = s2 * m00s;

    double nu20 = mState->mu20 * s2,
    nu11 = mState->mu11 * s2,
    nu02 = mState->mu02 * s2,
    nu30 = mState->mu30 * s3,
    nu21 = mState->mu21 * s3, nu12 = mState->mu12 * s3, nu03 = mState->mu03 * s3;

    double t0 = nu30 + nu12;
    double t1 = nu21 + nu03;

    double q0 = t0 * t0, q1 = t1 * t1;

    double n4 = 4 * nu11;
    double s = nu20 + nu02;
    double d = nu20 - nu02;

    HuState->hu1 = s;
    HuState->hu2 = d * d + n4 * nu11;
    HuState->hu4 = q0 + q1;
    HuState->hu6 = d * (q0 - q1) + n4 * t0 * t1;

    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;

    q0 = nu30 - 3 * nu12;
    q1 = 3 * nu21 - nu03;

    HuState->hu3 = q0 * q0 + q1 * q1;
    HuState->hu5 = q0 * t0 + q1 * t1;
    HuState->hu7 = q1 * t0 - q0 * t1;
}


/* End of file. */
