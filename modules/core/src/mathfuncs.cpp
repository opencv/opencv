/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include "opencl_kernels_core.hpp"
#include <limits>
#include <iostream>

namespace cv
{

typedef void (*MathFunc)(const void* src, void* dst, int len);

#ifdef HAVE_OPENCL

enum { OCL_OP_LOG=0, OCL_OP_EXP=1, OCL_OP_MAG=2, OCL_OP_PHASE_DEGREES=3, OCL_OP_PHASE_RADIANS=4 };

static const char* oclop2str[] = { "OP_LOG", "OP_EXP", "OP_MAG", "OP_PHASE_DEGREES", "OP_PHASE_RADIANS", 0 };

static bool ocl_math_op(InputArray _src1, InputArray _src2, OutputArray _dst, int oclop)
{
    int type = _src1.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    int kercn = oclop == OCL_OP_PHASE_DEGREES ||
            oclop == OCL_OP_PHASE_RADIANS ? 1 : ocl::predictOptimalVectorWidth(_src1, _src2, _dst);

    const ocl::Device d = ocl::Device::getDefault();
    bool double_support = d.doubleFPConfig() > 0;
    if (!double_support && depth == CV_64F)
        return false;
    int rowsPerWI = d.isIntel() ? 4 : 1;

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D %s -D %s -D dstT=%s -D rowsPerWI=%d%s", _src2.empty() ? "UNARY_OP" : "BINARY_OP",
                         oclop2str[oclop], ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)), rowsPerWI,
                         double_support ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2 = _src2.getUMat();
    _dst.create(src1.size(), type);
    UMat dst = _dst.getUMat();

    ocl::KernelArg src1arg = ocl::KernelArg::ReadOnlyNoSize(src1),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    if (src2.empty())
        k.args(src1arg, dstarg);
    else
        k.args(src1arg, src2arg, dstarg);

    size_t globalsize[] = { (size_t)src1.cols * cn / kercn, ((size_t)src1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, 0, false);
}

#endif

/* ************************************************************************** *\
   Fast cube root by Ken Turkowski
   (http://www.worldserver.com/turk/computergraphics/papers.html)
\* ************************************************************************** */
float  cubeRoot( float value )
{
    float fr;
    Cv32suf v, m;
    int ix, s;
    int ex, shx;

    v.f = value;
    ix = v.i & 0x7fffffff;
    s = v.i & 0x80000000;
    ex = (ix >> 23) - 127;
    shx = ex % 3;
    shx -= shx >= 0 ? 3 : 0;
    ex = (ex - shx) / 3; /* exponent of cube root */
    v.i = (ix & ((1<<23)-1)) | ((shx + 127)<<23);
    fr = v.f;

    /* 0.125 <= fr < 1.0 */
    /* Use quartic rational polynomial with error < 2^(-24) */
    fr = (float)(((((45.2548339756803022511987494 * fr +
    192.2798368355061050458134625) * fr +
    119.1654824285581628956914143) * fr +
    13.43250139086239872172837314) * fr +
    0.1636161226585754240958355063)/
    ((((14.80884093219134573786480845 * fr +
    151.9714051044435648658557668) * fr +
    168.5254414101568283957668343) * fr +
    33.9905941350215598754191872) * fr +
    1.0));

    /* fr *= 2^ex * sign */
    m.f = value;
    v.f = fr;
    v.i = (v.i + (ex << 23) + s) & (m.i*2 != 0 ? -1 : 0);
    return v.f;
}

/****************************************************************************************\
*                                  Cartezian -> Polar                                    *
\****************************************************************************************/

void magnitude( InputArray src1, InputArray src2, OutputArray dst )
{
    int type = src1.type(), depth = src1.depth(), cn = src1.channels();
    CV_Assert( src1.size() == src2.size() && type == src2.type() && (depth == CV_32F || depth == CV_64F));

    CV_OCL_RUN(dst.isUMat() && src1.dims() <= 2 && src2.dims() <= 2,
               ocl_math_op(src1, src2, dst, OCL_OP_MAG))

    Mat X = src1.getMat(), Y = src2.getMat();
    dst.create(X.dims, X.size, X.type());
    Mat Mag = dst.getMat();

    const Mat* arrays[] = {&X, &Y, &Mag, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size*cn;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        if( depth == CV_32F )
        {
            const float *x = (const float*)ptrs[0], *y = (const float*)ptrs[1];
            float *mag = (float*)ptrs[2];
            hal::magnitude32f( x, y, mag, len );
        }
        else
        {
            const double *x = (const double*)ptrs[0], *y = (const double*)ptrs[1];
            double *mag = (double*)ptrs[2];
            hal::magnitude64f( x, y, mag, len );
        }
    }
}

void phase( InputArray src1, InputArray src2, OutputArray dst, bool angleInDegrees )
{
    int type = src1.type(), depth = src1.depth(), cn = src1.channels();
    CV_Assert( src1.size() == src2.size() && type == src2.type() && (depth == CV_32F || depth == CV_64F));

    CV_OCL_RUN(dst.isUMat() && src1.dims() <= 2 && src2.dims() <= 2,
               ocl_math_op(src1, src2, dst, angleInDegrees ? OCL_OP_PHASE_DEGREES : OCL_OP_PHASE_RADIANS))

    Mat X = src1.getMat(), Y = src2.getMat();
    dst.create( X.dims, X.size, type );
    Mat Angle = dst.getMat();

    const Mat* arrays[] = {&X, &Y, &Angle, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)(it.size*cn), blockSize = total;
    size_t esz1 = X.elemSize1();
    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);
            if( depth == CV_32F )
            {
                const float *x = (const float*)ptrs[0], *y = (const float*)ptrs[1];
                float *angle = (float*)ptrs[2];
                hal::fastAtan32f( y, x, angle, len, angleInDegrees );
            }
            else
            {
                const double *x = (const double*)ptrs[0], *y = (const double*)ptrs[1];
                double *angle = (double*)ptrs[2];
                hal::fastAtan64f(y, x, angle, len, angleInDegrees);
            }
            ptrs[0] += len*esz1;
            ptrs[1] += len*esz1;
            ptrs[2] += len*esz1;
        }
    }
}

#ifdef HAVE_OPENCL

static bool ocl_cartToPolar( InputArray _src1, InputArray _src2,
                             OutputArray _dst1, OutputArray _dst2, bool angleInDegrees )
{
    const ocl::Device & d = ocl::Device::getDefault();
    int type = _src1.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = d.isIntel() ? 4 : 1;
    bool doubleSupport = d.doubleFPConfig() > 0;

    if ( !(_src1.dims() <= 2 && _src2.dims() <= 2 &&
           (depth == CV_32F || depth == CV_64F) && type == _src2.type()) ||
         (depth == CV_64F && !doubleSupport) )
        return false;

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D BINARY_OP -D dstT=%s -D depth=%d -D rowsPerWI=%d -D OP_CTP_%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, 1)),
                         depth, rowsPerWI, angleInDegrees ? "AD" : "AR",
                         doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2 = _src2.getUMat();
    Size size = src1.size();
    CV_Assert( size == src2.size() );

    _dst1.create(size, type);
    _dst2.create(size, type);
    UMat dst1 = _dst1.getUMat(), dst2 = _dst2.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(src1),
           ocl::KernelArg::ReadOnlyNoSize(src2),
           ocl::KernelArg::WriteOnly(dst1, cn),
           ocl::KernelArg::WriteOnlyNoSize(dst2));

    size_t globalsize[2] = { (size_t)dst1.cols * cn, ((size_t)dst1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

void cartToPolar( InputArray src1, InputArray src2,
                  OutputArray dst1, OutputArray dst2, bool angleInDegrees )
{
    CV_OCL_RUN(dst1.isUMat() && dst2.isUMat(),
            ocl_cartToPolar(src1, src2, dst1, dst2, angleInDegrees))

    Mat X = src1.getMat(), Y = src2.getMat();
    int type = X.type(), depth = X.depth(), cn = X.channels();
    CV_Assert( X.size == Y.size && type == Y.type() && (depth == CV_32F || depth == CV_64F));
    dst1.create( X.dims, X.size, type );
    dst2.create( X.dims, X.size, type );
    Mat Mag = dst1.getMat(), Angle = dst2.getMat();

    const Mat* arrays[] = {&X, &Y, &Mag, &Angle, 0};
    uchar* ptrs[4];
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)(it.size*cn), blockSize = std::min(total, ((BLOCK_SIZE+cn-1)/cn)*cn);
    size_t esz1 = X.elemSize1();

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);
            if( depth == CV_32F )
            {
                const float *x = (const float*)ptrs[0], *y = (const float*)ptrs[1];
                float *mag = (float*)ptrs[2], *angle = (float*)ptrs[3];
                hal::magnitude32f( x, y, mag, len );
                hal::fastAtan32f( y, x, angle, len, angleInDegrees );
            }
            else
            {
                const double *x = (const double*)ptrs[0], *y = (const double*)ptrs[1];
                double *angle = (double*)ptrs[3];
                hal::magnitude64f(x, y, (double*)ptrs[2], len);
                hal::fastAtan64f(y, x, angle, len, angleInDegrees);
            }
            ptrs[0] += len*esz1;
            ptrs[1] += len*esz1;
            ptrs[2] += len*esz1;
            ptrs[3] += len*esz1;
        }
    }
}


/****************************************************************************************\
*                                  Polar -> Cartezian                                    *
\****************************************************************************************/

static void SinCos_32f( const float *angle, float *sinval, float* cosval,
                        int len, int angle_in_degrees )
{
    const int N = 64;

    static const double sin_table[] =
    {
     0.00000000000000000000,     0.09801714032956060400,
     0.19509032201612825000,     0.29028467725446233000,
     0.38268343236508978000,     0.47139673682599764000,
     0.55557023301960218000,     0.63439328416364549000,
     0.70710678118654746000,     0.77301045336273699000,
     0.83146961230254524000,     0.88192126434835494000,
     0.92387953251128674000,     0.95694033573220894000,
     0.98078528040323043000,     0.99518472667219682000,
     1.00000000000000000000,     0.99518472667219693000,
     0.98078528040323043000,     0.95694033573220894000,
     0.92387953251128674000,     0.88192126434835505000,
     0.83146961230254546000,     0.77301045336273710000,
     0.70710678118654757000,     0.63439328416364549000,
     0.55557023301960218000,     0.47139673682599786000,
     0.38268343236508989000,     0.29028467725446239000,
     0.19509032201612861000,     0.09801714032956082600,
     0.00000000000000012246,    -0.09801714032956059000,
    -0.19509032201612836000,    -0.29028467725446211000,
    -0.38268343236508967000,    -0.47139673682599764000,
    -0.55557023301960196000,    -0.63439328416364527000,
    -0.70710678118654746000,    -0.77301045336273666000,
    -0.83146961230254524000,    -0.88192126434835494000,
    -0.92387953251128652000,    -0.95694033573220882000,
    -0.98078528040323032000,    -0.99518472667219693000,
    -1.00000000000000000000,    -0.99518472667219693000,
    -0.98078528040323043000,    -0.95694033573220894000,
    -0.92387953251128663000,    -0.88192126434835505000,
    -0.83146961230254546000,    -0.77301045336273688000,
    -0.70710678118654768000,    -0.63439328416364593000,
    -0.55557023301960218000,    -0.47139673682599792000,
    -0.38268343236509039000,    -0.29028467725446250000,
    -0.19509032201612872000,    -0.09801714032956050600,
    };

    static const double k2 = (2*CV_PI)/N;

    static const double sin_a0 = -0.166630293345647*k2*k2*k2;
    static const double sin_a2 = k2;

    static const double cos_a0 = -0.499818138450326*k2*k2;
    /*static const double cos_a2 =  1;*/

    double k1;
    int i = 0;

    if( !angle_in_degrees )
        k1 = N/(2*CV_PI);
    else
        k1 = N/360.;

#if CV_AVX2
    if (USE_AVX2)
    {
        __m128d v_k1 = _mm_set1_pd(k1);
        __m128d v_1 = _mm_set1_pd(1);
        __m128i v_N1 = _mm_set1_epi32(N - 1);
        __m128i v_N4 = _mm_set1_epi32(N >> 2);
        __m128d v_sin_a0 = _mm_set1_pd(sin_a0);
        __m128d v_sin_a2 = _mm_set1_pd(sin_a2);
        __m128d v_cos_a0 = _mm_set1_pd(cos_a0);

        for ( ; i <= len - 4; i += 4)
        {
            __m128 v_angle = _mm_loadu_ps(angle + i);

            // 0-1
            __m128d v_t = _mm_mul_pd(_mm_cvtps_pd(v_angle), v_k1);
            __m128i v_it = _mm_cvtpd_epi32(v_t);
            v_t = _mm_sub_pd(v_t, _mm_cvtepi32_pd(v_it));

            __m128i v_sin_idx = _mm_and_si128(v_it, v_N1);
            __m128i v_cos_idx = _mm_and_si128(_mm_sub_epi32(v_N4, v_sin_idx), v_N1);

            __m128d v_t2 = _mm_mul_pd(v_t, v_t);
            __m128d v_sin_b = _mm_mul_pd(_mm_add_pd(_mm_mul_pd(v_sin_a0, v_t2), v_sin_a2), v_t);
            __m128d v_cos_b = _mm_add_pd(_mm_mul_pd(v_cos_a0, v_t2), v_1);

            __m128d v_sin_a = _mm_i32gather_pd(sin_table, v_sin_idx, 8);
            __m128d v_cos_a = _mm_i32gather_pd(sin_table, v_cos_idx, 8);

            __m128d v_sin_val_0 = _mm_add_pd(_mm_mul_pd(v_sin_a, v_cos_b),
                                             _mm_mul_pd(v_cos_a, v_sin_b));
            __m128d v_cos_val_0 = _mm_sub_pd(_mm_mul_pd(v_cos_a, v_cos_b),
                                             _mm_mul_pd(v_sin_a, v_sin_b));

            // 2-3
            v_t = _mm_mul_pd(_mm_cvtps_pd(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_angle), 8))), v_k1);
            v_it = _mm_cvtpd_epi32(v_t);
            v_t = _mm_sub_pd(v_t, _mm_cvtepi32_pd(v_it));

            v_sin_idx = _mm_and_si128(v_it, v_N1);
            v_cos_idx = _mm_and_si128(_mm_sub_epi32(v_N4, v_sin_idx), v_N1);

            v_t2 = _mm_mul_pd(v_t, v_t);
            v_sin_b = _mm_mul_pd(_mm_add_pd(_mm_mul_pd(v_sin_a0, v_t2), v_sin_a2), v_t);
            v_cos_b = _mm_add_pd(_mm_mul_pd(v_cos_a0, v_t2), v_1);

            v_sin_a = _mm_i32gather_pd(sin_table, v_sin_idx, 8);
            v_cos_a = _mm_i32gather_pd(sin_table, v_cos_idx, 8);

            __m128d v_sin_val_1 = _mm_add_pd(_mm_mul_pd(v_sin_a, v_cos_b),
                                             _mm_mul_pd(v_cos_a, v_sin_b));
            __m128d v_cos_val_1 = _mm_sub_pd(_mm_mul_pd(v_cos_a, v_cos_b),
                                             _mm_mul_pd(v_sin_a, v_sin_b));

            _mm_storeu_ps(sinval + i, _mm_movelh_ps(_mm_cvtpd_ps(v_sin_val_0),
                                                    _mm_cvtpd_ps(v_sin_val_1)));
            _mm_storeu_ps(cosval + i, _mm_movelh_ps(_mm_cvtpd_ps(v_cos_val_0),
                                                    _mm_cvtpd_ps(v_cos_val_1)));
        }
    }
#endif

    for( ; i < len; i++ )
    {
        double t = angle[i]*k1;
        int it = cvRound(t);
        t -= it;
        int sin_idx = it & (N - 1);
        int cos_idx = (N/4 - sin_idx) & (N - 1);

        double sin_b = (sin_a0*t*t + sin_a2)*t;
        double cos_b = cos_a0*t*t + 1;

        double sin_a = sin_table[sin_idx];
        double cos_a = sin_table[cos_idx];

        double sin_val = sin_a*cos_b + cos_a*sin_b;
        double cos_val = cos_a*cos_b - sin_a*sin_b;

        sinval[i] = (float)sin_val;
        cosval[i] = (float)cos_val;
    }
}


#ifdef HAVE_OPENCL

static bool ocl_polarToCart( InputArray _mag, InputArray _angle,
                             OutputArray _dst1, OutputArray _dst2, bool angleInDegrees )
{
    const ocl::Device & d = ocl::Device::getDefault();
    int type = _angle.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = d.isIntel() ? 4 : 1;
    bool doubleSupport = d.doubleFPConfig() > 0;

    if ( !doubleSupport && depth == CV_64F )
        return false;

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D dstT=%s -D rowsPerWI=%d -D depth=%d -D BINARY_OP -D OP_PTC_%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, 1)), rowsPerWI,
                         depth, angleInDegrees ? "AD" : "AR",
                         doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat mag = _mag.getUMat(), angle = _angle.getUMat();
    Size size = angle.size();
    CV_Assert(mag.size() == size);

    _dst1.create(size, type);
    _dst2.create(size, type);
    UMat dst1 = _dst1.getUMat(), dst2 = _dst2.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(mag), ocl::KernelArg::ReadOnlyNoSize(angle),
           ocl::KernelArg::WriteOnly(dst1, cn), ocl::KernelArg::WriteOnlyNoSize(dst2));

    size_t globalsize[2] = { (size_t)dst1.cols * cn, ((size_t)dst1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

void polarToCart( InputArray src1, InputArray src2,
                  OutputArray dst1, OutputArray dst2, bool angleInDegrees )
{
    int type = src2.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert((depth == CV_32F || depth == CV_64F) && (src1.empty() || src1.type() == type));

    CV_OCL_RUN(!src1.empty() && src2.dims() <= 2 && dst1.isUMat() && dst2.isUMat(),
               ocl_polarToCart(src1, src2, dst1, dst2, angleInDegrees))

    Mat Mag = src1.getMat(), Angle = src2.getMat();
    CV_Assert( Mag.empty() || Angle.size == Mag.size);
    dst1.create( Angle.dims, Angle.size, type );
    dst2.create( Angle.dims, Angle.size, type );
    Mat X = dst1.getMat(), Y = dst2.getMat();

#if defined(HAVE_IPP)
    CV_IPP_CHECK()
    {
        if (Mag.isContinuous() && Angle.isContinuous() && X.isContinuous() && Y.isContinuous() && !angleInDegrees)
        {
            typedef IppStatus (CV_STDCALL * ippsPolarToCart)(const void * pSrcMagn, const void * pSrcPhase,
                                                             void * pDstRe, void * pDstIm, int len);
            ippsPolarToCart ippFunc =
            depth == CV_32F ? (ippsPolarToCart)ippsPolarToCart_32f :
            depth == CV_64F ? (ippsPolarToCart)ippsPolarToCart_64f : 0;
            CV_Assert(ippFunc != 0);

            IppStatus status = ippFunc(Mag.ptr(), Angle.ptr(), X.ptr(), Y.ptr(), static_cast<int>(cn * X.total()));
            if (status >= 0)
            {
                CV_IMPL_ADD(CV_IMPL_IPP);
                return;
            }
            setIppErrorStatus();
        }
    }
#endif

    const Mat* arrays[] = {&Mag, &Angle, &X, &Y, 0};
    uchar* ptrs[4];
    NAryMatIterator it(arrays, ptrs);
    cv::AutoBuffer<float> _buf;
    float* buf[2] = {0, 0};
    int j, k, total = (int)(it.size*cn), blockSize = std::min(total, ((BLOCK_SIZE+cn-1)/cn)*cn);
    size_t esz1 = Angle.elemSize1();

    if( depth == CV_64F )
    {
        _buf.allocate(blockSize*2);
        buf[0] = _buf;
        buf[1] = buf[0] + blockSize;
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);
            if( depth == CV_32F )
            {
                const float *mag = (const float*)ptrs[0], *angle = (const float*)ptrs[1];
                float *x = (float*)ptrs[2], *y = (float*)ptrs[3];

                SinCos_32f( angle, y, x, len, angleInDegrees );
                if( mag )
                {
                    k = 0;

                    #if CV_NEON
                    for( ; k <= len - 4; k += 4 )
                    {
                        float32x4_t v_m = vld1q_f32(mag + k);
                        vst1q_f32(x + k, vmulq_f32(vld1q_f32(x + k), v_m));
                        vst1q_f32(y + k, vmulq_f32(vld1q_f32(y + k), v_m));
                    }
                    #elif CV_SSE2
                    if (USE_SSE2)
                    {
                        for( ; k <= len - 4; k += 4 )
                        {
                            __m128 v_m = _mm_loadu_ps(mag + k);
                            _mm_storeu_ps(x + k, _mm_mul_ps(_mm_loadu_ps(x + k), v_m));
                            _mm_storeu_ps(y + k, _mm_mul_ps(_mm_loadu_ps(y + k), v_m));
                        }
                    }
                    #endif

                    for( ; k < len; k++ )
                    {
                        float m = mag[k];
                        x[k] *= m; y[k] *= m;
                    }
                }
            }
            else
            {
                const double *mag = (const double*)ptrs[0], *angle = (const double*)ptrs[1];
                double *x = (double*)ptrs[2], *y = (double*)ptrs[3];

                for( k = 0; k < len; k++ )
                    buf[0][k] = (float)angle[k];

                SinCos_32f( buf[0], buf[1], buf[0], len, angleInDegrees );
                if( mag )
                    for( k = 0; k < len; k++ )
                    {
                        double m = mag[k];
                        x[k] = buf[0][k]*m; y[k] = buf[1][k]*m;
                    }
                else
                {
                    std::memcpy(x, buf[0], sizeof(float) * len);
                    std::memcpy(y, buf[1], sizeof(float) * len);
                }
            }

            if( ptrs[0] )
                ptrs[0] += len*esz1;
            ptrs[1] += len*esz1;
            ptrs[2] += len*esz1;
            ptrs[3] += len*esz1;
        }
    }
}

/****************************************************************************************\
*                                          E X P                                         *
\****************************************************************************************/

#ifdef HAVE_IPP
static void Exp_32f_ipp(const float *x, float *y, int n)
{
    CV_IPP_CHECK()
    {
        if (0 <= ippsExp_32f_A21(x, y, n))
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
    hal::exp32f(x, y, n);
}

static void Exp_64f_ipp(const double *x, double *y, int n)
{
    CV_IPP_CHECK()
    {
        if (0 <= ippsExp_64f_A50(x, y, n))
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
    hal::exp64f(x, y, n);
}

#define Exp_32f Exp_32f_ipp
#define Exp_64f Exp_64f_ipp
#else
#define Exp_32f hal::exp32f
#define Exp_64f hal::exp64f
#endif


void exp( InputArray _src, OutputArray _dst )
{
    int type = _src.type(), depth = _src.depth(), cn = _src.channels();
    CV_Assert( depth == CV_32F || depth == CV_64F );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_math_op(_src, noArray(), _dst, OCL_OP_EXP))

    Mat src = _src.getMat();
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        if( depth == CV_32F )
            Exp_32f((const float*)ptrs[0], (float*)ptrs[1], len);
        else
            Exp_64f((const double*)ptrs[0], (double*)ptrs[1], len);
    }
}


/****************************************************************************************\
*                                          L O G                                         *
\****************************************************************************************/

#ifdef HAVE_IPP
static void Log_32f_ipp(const float *x, float *y, int n)
{
    CV_IPP_CHECK()
    {
        if (0 <= ippsLn_32f_A21(x, y, n))
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
    hal::log32f(x, y, n);
}

static void Log_64f_ipp(const double *x, double *y, int n)
{
    CV_IPP_CHECK()
    {
        if (0 <= ippsLn_64f_A50(x, y, n))
        {
            CV_IMPL_ADD(CV_IMPL_IPP);
            return;
        }
        setIppErrorStatus();
    }
    hal::log64f(x, y, n);
}

#define Log_32f Log_32f_ipp
#define Log_64f Log_64f_ipp
#else
#define Log_32f hal::log32f
#define Log_64f hal::log64f
#endif

void log( InputArray _src, OutputArray _dst )
{
    int type = _src.type(), depth = _src.depth(), cn = _src.channels();
    CV_Assert( depth == CV_32F || depth == CV_64F );

    CV_OCL_RUN( _dst.isUMat() && _src.dims() <= 2,
                ocl_math_op(_src, noArray(), _dst, OCL_OP_LOG))

    Mat src = _src.getMat();
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        if( depth == CV_32F )
            Log_32f( (const float*)ptrs[0], (float*)ptrs[1], len );
        else
            Log_64f( (const double*)ptrs[0], (double*)ptrs[1], len );
    }
}

/****************************************************************************************\
*                                    P O W E R                                           *
\****************************************************************************************/

template <typename T, typename WT>
struct iPow_SIMD
{
    int operator() ( const T *, T *, int, int)
    {
        return 0;
    }
};

#if CV_SIMD128

template <>
struct iPow_SIMD<uchar, int>
{
    int operator() ( const uchar * src, uchar * dst, int len, int power )
    {
        int i = 0;
        v_uint32x4 v_1 = v_setall_u32(1u);

        for ( ; i <= len - 8; i += 8)
        {
            v_uint32x4 v_a1 = v_1, v_a2 = v_1;
            v_uint16x8 v = v_load_expand(src + i);
            v_uint32x4 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v = v_pack(v_a1, v_a2);
            v_pack_store(dst + i, v);
        }

        return i;
    }
};

template <>
struct iPow_SIMD<schar, int>
{
    int operator() ( const schar * src, schar * dst, int len, int power)
    {
        int i = 0;
        v_int32x4 v_1 = v_setall_s32(1);

        for ( ; i <= len - 8; i += 8)
        {
            v_int32x4 v_a1 = v_1, v_a2 = v_1;
            v_int16x8 v = v_load_expand(src + i);
            v_int32x4 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v = v_pack(v_a1, v_a2);
            v_pack_store(dst + i, v);
        }

        return i;
    }
};

template <>
struct iPow_SIMD<ushort, int>
{
    int operator() ( const ushort * src, ushort * dst, int len, int power)
    {
        int i = 0;
        v_uint32x4 v_1 = v_setall_u32(1u);

        for ( ; i <= len - 8; i += 8)
        {
            v_uint32x4 v_a1 = v_1, v_a2 = v_1;
            v_uint16x8 v = v_load(src + i);
            v_uint32x4 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v = v_pack(v_a1, v_a2);
            v_store(dst + i, v);
        }

        return i;
    }
};

template <>
struct iPow_SIMD<short, int>
{
    int operator() ( const short * src, short * dst, int len, int power)
    {
        int i = 0;
        v_int32x4 v_1 = v_setall_s32(1);

        for ( ; i <= len - 8; i += 8)
        {
            v_int32x4 v_a1 = v_1, v_a2 = v_1;
            v_int16x8 v = v_load(src + i);
            v_int32x4 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v = v_pack(v_a1, v_a2);
            v_store(dst + i, v);
        }

        return i;
    }
};

template <>
struct iPow_SIMD<int, int>
{
    int operator() ( const int * src, int * dst, int len, int power)
    {
        int i = 0;
        v_int32x4 v_1 = v_setall_s32(1);

        for ( ; i <= len - 8; i += 8)
        {
            v_int32x4 v_a1 = v_1, v_a2 = v_1;
            v_int32x4 v_b1 = v_load(src + i), v_b2 = v_load(src + i + 4);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v_store(dst + i, v_a1);
            v_store(dst + i + 4, v_a2);
        }

        return i;
    }
};

template <>
struct iPow_SIMD<float, float>
{
    int operator() ( const float * src, float * dst, int len, int power)
    {
        int i = 0;
        v_float32x4 v_1 = v_setall_f32(1.f);

        for ( ; i <= len - 8; i += 8)
        {
            v_float32x4 v_a1 = v_1, v_a2 = v_1;
            v_float32x4 v_b1 = v_load(src + i), v_b2 = v_load(src + i + 4);
            int p = std::abs(power);
            if( power < 0 )
            {
                v_b1 = v_1 / v_b1;
                v_b2 = v_1 / v_b2;
            }

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v_store(dst + i, v_a1);
            v_store(dst + i + 4, v_a2);
        }

        return i;
    }
};

#if CV_SIMD128_64F
template <>
struct iPow_SIMD<double, double>
{
    int operator() ( const double * src, double * dst, int len, int power)
    {
        int i = 0;
        v_float64x2 v_1 = v_setall_f64(1.);

        for ( ; i <= len - 4; i += 4)
        {
            v_float64x2 v_a1 = v_1, v_a2 = v_1;
            v_float64x2 v_b1 = v_load(src + i), v_b2 = v_load(src + i + 2);
            int p = std::abs(power);
            if( power < 0 )
            {
                v_b1 = v_1 / v_b1;
                v_b2 = v_1 / v_b2;
            }

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 *= v_b1;
                    v_a2 *= v_b2;
                }
                v_b1 *= v_b1;
                v_b2 *= v_b2;
                p >>= 1;
            }

            v_a1 *= v_b1;
            v_a2 *= v_b2;

            v_store(dst + i, v_a1);
            v_store(dst + i + 2, v_a2);
        }

        return i;
    }
};
#endif

#endif

template<typename T, typename WT>
static void
iPow_i( const T* src, T* dst, int len, int power )
{
    if( power < 0 )
    {
        T tab[5] =
        {
            saturate_cast<T>(power == -1 ? -1 : 0), saturate_cast<T>((power & 1) ? -1 : 1),
            std::numeric_limits<T>::max(), 1, saturate_cast<T>(power == -1 ? 1 : 0)
        };
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            dst[i] = cv_abs(val) <= 2 ? tab[val + 2] : (T)0;
        }
    }
    else
    {
        iPow_SIMD<T, WT> vop;
        int i = vop(src, dst, len, power);

        for( ; i < len; i++ )
        {
            WT a = 1, b = src[i];
            int p = power;
            while( p > 1 )
            {
                if( p & 1 )
                    a *= b;
                b *= b;
                p >>= 1;
            }

            a *= b;
            dst[i] = saturate_cast<T>(a);
        }
    }
}

template<typename T>
static void
iPow_f( const T* src, T* dst, int len, int power0 )
{
    iPow_SIMD<T, T> vop;
    int i = vop(src, dst, len, power0);
    int power = std::abs(power0);

    for( ; i < len; i++ )
    {
        T a = 1, b = src[i];
        int p = power;
        if( power0 < 0 )
            b = 1/b;

        while( p > 1 )
        {
            if( p & 1 )
                a *= b;
            b *= b;
            p >>= 1;
        }

        a *= b;
        dst[i] = a;
    }
}

static void iPow8u(const uchar* src, uchar* dst, int len, int power)
{
    iPow_i<uchar, unsigned>(src, dst, len, power);
}

static void iPow8s(const schar* src, schar* dst, int len, int power)
{
    iPow_i<schar, int>(src, dst, len, power);
}

static void iPow16u(const ushort* src, ushort* dst, int len, int power)
{
    iPow_i<ushort, unsigned>(src, dst, len, power);
}

static void iPow16s(const short* src, short* dst, int len, int power)
{
    iPow_i<short, int>(src, dst, len, power);
}

static void iPow32s(const int* src, int* dst, int len, int power)
{
    iPow_i<int, int>(src, dst, len, power);
}

static void iPow32f(const float* src, float* dst, int len, int power)
{
    iPow_f<float>(src, dst, len, power);
}

static void iPow64f(const double* src, double* dst, int len, int power)
{
    iPow_f<double>(src, dst, len, power);
}


typedef void (*IPowFunc)( const uchar* src, uchar* dst, int len, int power );

static IPowFunc ipowTab[] =
{
    (IPowFunc)iPow8u, (IPowFunc)iPow8s, (IPowFunc)iPow16u, (IPowFunc)iPow16s,
    (IPowFunc)iPow32s, (IPowFunc)iPow32f, (IPowFunc)iPow64f, 0
};

#ifdef HAVE_OPENCL

static bool ocl_pow(InputArray _src, double power, OutputArray _dst,
                    bool is_ipower, int ipower)
{
    const ocl::Device & d = ocl::Device::getDefault();
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = d.isIntel() ? 4 : 1;
    bool doubleSupport = d.doubleFPConfig() > 0;

    _dst.createSameSize(_src, type);
    if (is_ipower)
    {
        if (ipower == 0)
        {
            _dst.setTo(Scalar::all(1));
            return true;
        }
        if (ipower == 1)
        {
            _src.copyTo(_dst);
            return true;
        }
        if( ipower < 0 )
        {
            if( depth == CV_32F || depth == CV_64F )
                is_ipower = false;
            else
                return false;
        }
    }

    if (depth == CV_64F && !doubleSupport)
        return false;

    bool issqrt = std::abs(power - 0.5) < DBL_EPSILON;
    const char * const op = issqrt ? "OP_SQRT" : is_ipower ? "OP_POWN" : "OP_POW";

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D dstT=%s -D depth=%d -D rowsPerWI=%d -D %s -D UNARY_OP%s",
                         ocl::typeToStr(depth), depth,  rowsPerWI, op,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(src.size(), type);
    UMat dst = _dst.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn);

    if (issqrt)
        k.args(srcarg, dstarg);
    else if (is_ipower)
        k.args(srcarg, dstarg, ipower);
    else
    {
        if (depth == CV_32F)
            k.args(srcarg, dstarg, (float)power);
        else
            k.args(srcarg, dstarg, power);
    }

    size_t globalsize[2] = { (size_t)dst.cols *  cn, ((size_t)dst.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

static void InvSqrt_32f(const float* src, float* dst, int n) { hal::invSqrt32f(src, dst, n); }
static void InvSqrt_64f(const double* src, double* dst, int n) { hal::invSqrt64f(src, dst, n); }
static void Sqrt_32f(const float* src, float* dst, int n) { hal::sqrt32f(src, dst, n); }
static void Sqrt_64f(const double* src, double* dst, int n) { hal::sqrt64f(src, dst, n); }

void pow( InputArray _src, double power, OutputArray _dst )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type),
            cn = CV_MAT_CN(type), ipower = cvRound(power);
    bool is_ipower = fabs(ipower - power) < DBL_EPSILON;
#ifdef HAVE_OPENCL
    bool useOpenCL = _dst.isUMat() && _src.dims() <= 2;
#endif

    if( is_ipower
#ifdef HAVE_OPENCL
            && !(useOpenCL && ocl::Device::getDefault().isIntel() && depth != CV_64F)
#endif
      )
    {
        switch( ipower )
        {
        case 0:
            _dst.createSameSize(_src, type);
            _dst.setTo(Scalar::all(1));
            return;
        case 1:
            _src.copyTo(_dst);
            return;
        case 2:
            multiply(_src, _src, _dst);
            return;
        }
    }
    else
        CV_Assert( depth == CV_32F || depth == CV_64F );

    CV_OCL_RUN(useOpenCL, ocl_pow(_src, power, _dst, is_ipower, ipower))

    Mat src = _src.getMat();
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);

    if( is_ipower )
    {
        IPowFunc func = ipowTab[depth];
        CV_Assert( func != 0 );

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], ptrs[1], len, ipower );
    }
    else if( fabs(fabs(power) - 0.5) < DBL_EPSILON )
    {
        MathFunc func = power < 0 ?
            (depth == CV_32F ? (MathFunc)InvSqrt_32f : (MathFunc)InvSqrt_64f) :
            (depth == CV_32F ? (MathFunc)Sqrt_32f : (MathFunc)Sqrt_64f);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func( ptrs[0], ptrs[1], len );
    }
    else
    {
        int j, k, blockSize = std::min(len, ((BLOCK_SIZE + cn-1)/cn)*cn);
        size_t esz1 = src.elemSize1();
        AutoBuffer<uchar> buf;
        Cv32suf inf32, nan32;
        Cv64suf inf64, nan64;
        float* fbuf = 0;
        double* dbuf = 0;
        inf32.i = 0x7f800000;
        nan32.i = 0x7fffffff;
        inf64.i = CV_BIG_INT(0x7FF0000000000000);
        nan64.i = CV_BIG_INT(0x7FFFFFFFFFFFFFFF);

        if( src.ptr() == dst.ptr() )
        {
            buf.allocate(blockSize*esz1);
            fbuf = (float*)(uchar*)buf;
            dbuf = (double*)(uchar*)buf;
        }

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( j = 0; j < len; j += blockSize )
            {
                int bsz = std::min(len - j, blockSize);

            #if defined(HAVE_IPP)
                CV_IPP_CHECK()
                {
                    IppStatus status = depth == CV_32F ?
                    ippsPowx_32f_A21((const float*)ptrs[0], (float)power, (float*)ptrs[1], bsz) :
                    ippsPowx_64f_A50((const double*)ptrs[0], (double)power, (double*)ptrs[1], bsz);

                    if (status >= 0)
                    {
                        CV_IMPL_ADD(CV_IMPL_IPP);
                        ptrs[0] += bsz*esz1;
                        ptrs[1] += bsz*esz1;
                        continue;
                    }
                    setIppErrorStatus();
                }
            #endif

                if( depth == CV_32F )
                {
                    float* x0 = (float*)ptrs[0];
                    float* x = fbuf ? fbuf : x0;
                    float* y = (float*)ptrs[1];

                    if( x != x0 )
                        memcpy(x, x0, bsz*esz1);

                    Log_32f(x, y, bsz);
                    for( k = 0; k < bsz; k++ )
                        y[k] = (float)(y[k]*power);
                    Exp_32f(y, y, bsz);
                    for( k = 0; k < bsz; k++ )
                    {
                        if( x0[k] <= 0 )
                        {
                            if( x0[k] == 0.f )
                            {
                                if( power < 0 )
                                    y[k] = inf32.f;
                            }
                            else
                                y[k] = nan32.f;
                        }
                    }
                }
                else
                {
                    double* x0 = (double*)ptrs[0];
                    double* x = dbuf ? dbuf : x0;
                    double* y = (double*)ptrs[1];

                    if( x != x0 )
                        memcpy(x, x0, bsz*esz1);

                    Log_64f(x, y, bsz);
                    for( k = 0; k < bsz; k++ )
                        y[k] *= power;
                    Exp_64f(y, y, bsz);

                    for( k = 0; k < bsz; k++ )
                    {
                        if( x0[k] <= 0 )
                        {
                            if( x0[k] == 0. )
                            {
                                if( power < 0 )
                                    y[k] = inf64.f;
                            }
                            else
                                y[k] = nan64.f;
                        }
                    }
                }
                ptrs[0] += bsz*esz1;
                ptrs[1] += bsz*esz1;
            }
        }
    }
}

void sqrt(InputArray a, OutputArray b)
{
    cv::pow(a, 0.5, b);
}

/************************** CheckArray for NaN's, Inf's *********************************/

template<int cv_mat_type> struct mat_type_assotiations{};

template<> struct mat_type_assotiations<CV_8U>
{
    typedef unsigned char type;
    static const type min_allowable = 0x0;
    static const type max_allowable = 0xFF;
};

template<> struct mat_type_assotiations<CV_8S>
{
    typedef signed char type;
    static const type min_allowable = SCHAR_MIN;
    static const type max_allowable = SCHAR_MAX;
};

template<> struct mat_type_assotiations<CV_16U>
{
    typedef unsigned short type;
    static const type min_allowable = 0x0;
    static const type max_allowable = USHRT_MAX;
};
template<> struct mat_type_assotiations<CV_16S>
{
    typedef signed short type;
    static const type min_allowable = SHRT_MIN;
    static const type max_allowable = SHRT_MAX;
};

template<> struct mat_type_assotiations<CV_32S>
{
    typedef int type;
    static const type min_allowable = (-INT_MAX - 1);
    static const type max_allowable = INT_MAX;
};

template<int depth>
static bool checkIntegerRange(cv::Mat src, Point& bad_pt, int minVal, int maxVal)
{
    typedef mat_type_assotiations<depth> type_ass;

    if (minVal < type_ass::min_allowable && maxVal > type_ass::max_allowable)
    {
        return true;
    }
    else if (minVal > type_ass::max_allowable || maxVal < type_ass::min_allowable || maxVal < minVal)
    {
        bad_pt = cv::Point(0,0);
        return false;
    }
    cv::Mat as_one_channel = src.reshape(1,0);

    for (int j = 0; j < as_one_channel.rows; ++j)
        for (int i = 0; i < as_one_channel.cols; ++i)
        {
            typename type_ass::type v = as_one_channel.at<typename type_ass::type>(j ,i);
            if (v < minVal || v > maxVal)
            {
                bad_pt.y = j;
                bad_pt.x = i / src.channels();
                return false;
            }
        }

    return true;
}

typedef bool (*check_range_function)(cv::Mat src, Point& bad_pt, int minVal, int maxVal);

check_range_function check_range_functions[] =
{
    &checkIntegerRange<CV_8U>,
    &checkIntegerRange<CV_8S>,
    &checkIntegerRange<CV_16U>,
    &checkIntegerRange<CV_16S>,
    &checkIntegerRange<CV_32S>
};

bool checkRange(InputArray _src, bool quiet, Point* pt, double minVal, double maxVal)
{
    Mat src = _src.getMat();

    if ( src.dims > 2 )
    {
        CV_Assert(pt == NULL); // no way to provide location info

        const Mat* arrays[] = {&src, 0};
        Mat planes[1];
        NAryMatIterator it(arrays, planes);

        for ( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            if (!checkRange( it.planes[0], quiet, NULL, minVal, maxVal ))
            {
                return false;
            }
        }
        return true;
    }

    int depth = src.depth();
    Point badPt(-1, -1);

    if (depth < CV_32F)
    {
        int minVali = minVal <= INT_MIN ? INT_MIN : cvFloor(minVal);
        int maxVali = maxVal > INT_MAX ? INT_MAX : cvCeil(maxVal) - 1;

        (check_range_functions[depth])(src, badPt, minVali, maxVali);
    }
    else
    {
        int i, loc = 0;
        int cn = src.channels();
        Size size = getContinuousSize( src, cn );

        if( depth == CV_32F )
        {
            Cv32suf a, b;
            int ia, ib;
            const int* isrc = src.ptr<int>();
            size_t step = src.step/sizeof(isrc[0]);

            a.f = (float)std::max(minVal, (double)-FLT_MAX);
            b.f = (float)std::min(maxVal, (double)FLT_MAX);

            ia = CV_TOGGLE_FLT(a.i);
            ib = CV_TOGGLE_FLT(b.i);

            for( ; badPt.x < 0 && size.height--; loc += size.width, isrc += step )
            {
                for( i = 0; i < size.width; i++ )
                {
                    int val = isrc[i];
                    val = CV_TOGGLE_FLT(val);

                    if( val < ia || val >= ib )
                    {
                        int pixelId = (loc + i) / cn;
                        badPt = Point(pixelId % src.cols, pixelId / src.cols);
                        break;
                    }
                }
            }
        }
        else
        {
            Cv64suf a, b;
            int64 ia, ib;
            const int64* isrc = src.ptr<int64>();
            size_t step = src.step/sizeof(isrc[0]);

            a.f = minVal;
            b.f = maxVal;

            ia = CV_TOGGLE_DBL(a.i);
            ib = CV_TOGGLE_DBL(b.i);

            for( ; badPt.x < 0 && size.height--; loc += size.width, isrc += step )
            {
                for( i = 0; i < size.width; i++ )
                {
                    int64 val = isrc[i];
                    val = CV_TOGGLE_DBL(val);

                    if( val < ia || val >= ib )
                    {
                        int pixelId = (loc + i) / cn;
                        badPt = Point(pixelId % src.cols, pixelId / src.cols);
                        break;
                    }
                }
            }
        }
    }

    if( badPt.x >= 0 )
    {
        if( pt )
            *pt = badPt;
        if( !quiet )
        {
            cv::String value_str;
            value_str << src(cv::Range(badPt.y, badPt.y + 1), cv::Range(badPt.x, badPt.x + 1));
            CV_Error_( CV_StsOutOfRange,
            ("the value at (%d, %d)=%s is out of range [%f, %f)", badPt.x, badPt.y, value_str.c_str(), minVal, maxVal));
        }
        return false;
    }
    return true;
}

#ifdef HAVE_OPENCL

static bool ocl_patchNaNs( InputOutputArray _a, float value )
{
    int rowsPerWI = ocl::Device::getDefault().isIntel() ? 4 : 1;
    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                     format("-D UNARY_OP -D OP_PATCH_NANS -D dstT=float -D rowsPerWI=%d",
                            rowsPerWI));
    if (k.empty())
        return false;

    UMat a = _a.getUMat();
    int cn = a.channels();

    k.args(ocl::KernelArg::ReadOnlyNoSize(a),
           ocl::KernelArg::WriteOnly(a, cn), (float)value);

    size_t globalsize[2] = { (size_t)a.cols * cn, ((size_t)a.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

void patchNaNs( InputOutputArray _a, double _val )
{
    CV_Assert( _a.depth() == CV_32F );

    CV_OCL_RUN(_a.isUMat() && _a.dims() <= 2,
               ocl_patchNaNs(_a, (float)_val))

    Mat a = _a.getMat();
    const Mat* arrays[] = {&a, 0};
    int* ptrs[1];
    NAryMatIterator it(arrays, (uchar**)ptrs);
    size_t len = it.size*a.channels();
    Cv32suf val;
    val.f = (float)_val;

#if CV_SSE2
    __m128i v_mask1 = _mm_set1_epi32(0x7fffffff), v_mask2 = _mm_set1_epi32(0x7f800000);
    __m128i v_val = _mm_set1_epi32(val.i);
#elif CV_NEON
    int32x4_t v_mask1 = vdupq_n_s32(0x7fffffff), v_mask2 = vdupq_n_s32(0x7f800000),
        v_val = vdupq_n_s32(val.i);
#endif

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        int* tptr = ptrs[0];
        size_t j = 0;

#if CV_SSE2
        if (USE_SSE2)
        {
            for ( ; j + 4 <= len; j += 4)
            {
                __m128i v_src = _mm_loadu_si128((__m128i const *)(tptr + j));
                __m128i v_cmp_mask = _mm_cmplt_epi32(v_mask2, _mm_and_si128(v_src, v_mask1));
                __m128i v_res = _mm_or_si128(_mm_andnot_si128(v_cmp_mask, v_src), _mm_and_si128(v_cmp_mask, v_val));
                _mm_storeu_si128((__m128i *)(tptr + j), v_res);
            }
        }
#elif CV_NEON
        for ( ; j + 4 <= len; j += 4)
        {
            int32x4_t v_src = vld1q_s32(tptr + j);
            uint32x4_t v_cmp_mask = vcltq_s32(v_mask2, vandq_s32(v_src, v_mask1));
            int32x4_t v_dst = vbslq_s32(v_cmp_mask, v_val, v_src);
            vst1q_s32(tptr + j, v_dst);
        }
#endif

        for( ; j < len; j++ )
            if( (tptr[j] & 0x7fffffff) > 0x7f800000 )
                tptr[j] = val.i;
    }
}

}

CV_IMPL float cvCbrt(float value) { return cv::cubeRoot(value); }
CV_IMPL float cvFastArctan(float y, float x) { return cv::fastAtan2(y, x); }

CV_IMPL void
cvCartToPolar( const CvArr* xarr, const CvArr* yarr,
               CvArr* magarr, CvArr* anglearr,
               int angle_in_degrees )
{
    cv::Mat X = cv::cvarrToMat(xarr), Y = cv::cvarrToMat(yarr), Mag, Angle;
    if( magarr )
    {
        Mag = cv::cvarrToMat(magarr);
        CV_Assert( Mag.size() == X.size() && Mag.type() == X.type() );
    }
    if( anglearr )
    {
        Angle = cv::cvarrToMat(anglearr);
        CV_Assert( Angle.size() == X.size() && Angle.type() == X.type() );
    }
    if( magarr )
    {
        if( anglearr )
            cv::cartToPolar( X, Y, Mag, Angle, angle_in_degrees != 0 );
        else
            cv::magnitude( X, Y, Mag );
    }
    else
        cv::phase( X, Y, Angle, angle_in_degrees != 0 );
}

CV_IMPL void
cvPolarToCart( const CvArr* magarr, const CvArr* anglearr,
               CvArr* xarr, CvArr* yarr, int angle_in_degrees )
{
    cv::Mat X, Y, Angle = cv::cvarrToMat(anglearr), Mag;
    if( magarr )
    {
        Mag = cv::cvarrToMat(magarr);
        CV_Assert( Mag.size() == Angle.size() && Mag.type() == Angle.type() );
    }
    if( xarr )
    {
        X = cv::cvarrToMat(xarr);
        CV_Assert( X.size() == Angle.size() && X.type() == Angle.type() );
    }
    if( yarr )
    {
        Y = cv::cvarrToMat(yarr);
        CV_Assert( Y.size() == Angle.size() && Y.type() == Angle.type() );
    }

    cv::polarToCart( Mag, Angle, X, Y, angle_in_degrees != 0 );
}

CV_IMPL void cvExp( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() && src.size == dst.size );
    cv::exp( src, dst );
}

CV_IMPL void cvLog( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() && src.size == dst.size );
    cv::log( src, dst );
}

CV_IMPL void cvPow( const CvArr* srcarr, CvArr* dstarr, double power )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() && src.size == dst.size );
    cv::pow( src, power, dst );
}

CV_IMPL int cvCheckArr( const CvArr* arr, int flags,
                        double minVal, double maxVal )
{
    if( (flags & CV_CHECK_RANGE) == 0 )
        minVal = -DBL_MAX, maxVal = DBL_MAX;
    return cv::checkRange(cv::cvarrToMat(arr), (flags & CV_CHECK_QUIET) != 0, 0, minVal, maxVal );
}


/*
  Finds real roots of cubic, quadratic or linear equation.
  The original code has been taken from Ken Turkowski web page
  (http://www.worldserver.com/turk/opensource/) and adopted for OpenCV.
  Here is the copyright notice.

  -----------------------------------------------------------------------
  Copyright (C) 1978-1999 Ken Turkowski. <turk@computer.org>

    All rights reserved.

    Warranty Information
      Even though I have reviewed this software, I make no warranty
      or representation, either express or implied, with respect to this
      software, its quality, accuracy, merchantability, or fitness for a
      particular purpose.  As a result, this software is provided "as is,"
      and you, its user, are assuming the entire risk as to its quality
      and accuracy.

    This code may be used and freely distributed as long as it includes
    this copyright notice and the above warranty information.
  -----------------------------------------------------------------------
*/

int cv::solveCubic( InputArray _coeffs, OutputArray _roots )
{
    const int n0 = 3;
    Mat coeffs = _coeffs.getMat();
    int ctype = coeffs.type();

    CV_Assert( ctype == CV_32F || ctype == CV_64F );
    CV_Assert( (coeffs.size() == Size(n0, 1) ||
                coeffs.size() == Size(n0+1, 1) ||
                coeffs.size() == Size(1, n0) ||
                coeffs.size() == Size(1, n0+1)) );

    _roots.create(n0, 1, ctype, -1, true, _OutputArray::DEPTH_MASK_FLT);
    Mat roots = _roots.getMat();

    int i = -1, n = 0;
    double a0 = 1., a1, a2, a3;
    double x0 = 0., x1 = 0., x2 = 0.;
    int ncoeffs = coeffs.rows + coeffs.cols - 1;

    if( ctype == CV_32FC1 )
    {
        if( ncoeffs == 4 )
            a0 = coeffs.at<float>(++i);

        a1 = coeffs.at<float>(i+1);
        a2 = coeffs.at<float>(i+2);
        a3 = coeffs.at<float>(i+3);
    }
    else
    {
        if( ncoeffs == 4 )
            a0 = coeffs.at<double>(++i);

        a1 = coeffs.at<double>(i+1);
        a2 = coeffs.at<double>(i+2);
        a3 = coeffs.at<double>(i+3);
    }

    if( a0 == 0 )
    {
        if( a1 == 0 )
        {
            if( a2 == 0 )
                n = a3 == 0 ? -1 : 0;
            else
            {
                // linear equation
                x0 = -a3/a2;
                n = 1;
            }
        }
        else
        {
            // quadratic equation
            double d = a2*a2 - 4*a1*a3;
            if( d >= 0 )
            {
                d = std::sqrt(d);
                double q1 = (-a2 + d) * 0.5;
                double q2 = (a2 + d) * -0.5;
                if( fabs(q1) > fabs(q2) )
                {
                    x0 = q1 / a1;
                    x1 = a3 / q1;
                }
                else
                {
                    x0 = q2 / a1;
                    x1 = a3 / q2;
                }
                n = d > 0 ? 2 : 1;
            }
        }
    }
    else
    {
        a0 = 1./a0;
        a1 *= a0;
        a2 *= a0;
        a3 *= a0;

        double Q = (a1 * a1 - 3 * a2) * (1./9);
        double R = (2 * a1 * a1 * a1 - 9 * a1 * a2 + 27 * a3) * (1./54);
        double Qcubed = Q * Q * Q;
        double d = Qcubed - R * R;

        if( d > 0 )
        {
            double theta = acos(R / sqrt(Qcubed));
            double sqrtQ = sqrt(Q);
            double t0 = -2 * sqrtQ;
            double t1 = theta * (1./3);
            double t2 = a1 * (1./3);
            x0 = t0 * cos(t1) - t2;
            x1 = t0 * cos(t1 + (2.*CV_PI/3)) - t2;
            x2 = t0 * cos(t1 + (4.*CV_PI/3)) - t2;
            n = 3;
        }
        else if( d == 0 )
        {
            if(R >= 0)
            {
                x0 = -2*pow(R, 1./3) - a1/3;
                x1 = pow(R, 1./3) - a1/3;
            }
            else
            {
                x0 = 2*pow(-R, 1./3) - a1/3;
                x1 = -pow(-R, 1./3) - a1/3;
            }
            x2 = 0;
            n = x0 == x1 ? 1 : 2;
            x1 = x0 == x1 ? 0 : x1;
        }
        else
        {
            double e;
            d = sqrt(-d);
            e = pow(d + fabs(R), 1./3);
            if( R > 0 )
                e = -e;
            x0 = (e + Q / e) - a1 * (1./3);
            n = 1;
        }
    }

    if( roots.type() == CV_32FC1 )
    {
        roots.at<float>(0) = (float)x0;
        roots.at<float>(1) = (float)x1;
        roots.at<float>(2) = (float)x2;
    }
    else
    {
        roots.at<double>(0) = x0;
        roots.at<double>(1) = x1;
        roots.at<double>(2) = x2;
    }

    return n;
}

/* finds complex roots of a polynomial using Durand-Kerner method:
   http://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method */
double cv::solvePoly( InputArray _coeffs0, OutputArray _roots0, int maxIters )
{
    typedef Complex<double> C;

    double maxDiff = 0;
    int iter, i, j;
    Mat coeffs0 = _coeffs0.getMat();
    int ctype = _coeffs0.type();
    int cdepth = CV_MAT_DEPTH(ctype);

    CV_Assert( CV_MAT_DEPTH(ctype) >= CV_32F && CV_MAT_CN(ctype) <= 2 );
    CV_Assert( coeffs0.rows == 1 || coeffs0.cols == 1 );

    int n0 = coeffs0.cols + coeffs0.rows - 2, n = n0;

    _roots0.create(n, 1, CV_MAKETYPE(cdepth, 2), -1, true, _OutputArray::DEPTH_MASK_FLT);
    Mat roots0 = _roots0.getMat();

    AutoBuffer<C> buf(n*2+2);
    C *coeffs = buf, *roots = coeffs + n + 1;
    Mat coeffs1(coeffs0.size(), CV_MAKETYPE(CV_64F, coeffs0.channels()), coeffs0.channels() == 2 ? coeffs : roots);
    coeffs0.convertTo(coeffs1, coeffs1.type());
    if( coeffs0.channels() == 1 )
    {
        const double* rcoeffs = (const double*)roots;
        for( i = 0; i <= n; i++ )
            coeffs[i] = C(rcoeffs[i], 0);
    }

    for( ; n > 1; n-- )
    {
        if( std::abs(coeffs[n].re) + std::abs(coeffs[n].im) > DBL_EPSILON )
            break;
    }

    C p(1, 0), r(1, 1);

    for( i = 0; i < n; i++ )
    {
        roots[i] = p;
        p = p * r;
    }

    maxIters = maxIters <= 0 ? 1000 : maxIters;
    for( iter = 0; iter < maxIters; iter++ )
    {
        maxDiff = 0;
        for( i = 0; i < n; i++ )
        {
            p = roots[i];
            C num = coeffs[n], denom = coeffs[n];
            int num_same_root = 1;
            for( j = 0; j < n; j++ )
            {
                num = num*p + coeffs[n-j-1];
                if( j != i )
                {
                    if ( (p - roots[j]).re != 0 || (p - roots[j]).im != 0 )
                        denom = denom * (p - roots[j]);
                    else
                        num_same_root++;
                }
            }
            num /= denom;
            if( num_same_root > 1)
            {
                double old_num_re = num.re;
                double old_num_im = num.im;
                int square_root_times = num_same_root % 2 == 0 ? num_same_root / 2 : num_same_root / 2 - 1;

                for( j = 0; j < square_root_times; j++)
                {
                    num.re = old_num_re*old_num_re + old_num_im*old_num_im;
                    num.re = sqrt(num.re);
                    num.re += old_num_re;
                    num.im = num.re - old_num_re;
                    num.re /= 2;
                    num.re = sqrt(num.re);

                    num.im /= 2;
                    num.im = sqrt(num.im);
                    if( old_num_re < 0 ) num.im = -num.im;
                }

                if( num_same_root % 2 != 0){
                    Mat cube_coefs(4, 1, CV_64FC1);
                    Mat cube_roots(3, 1, CV_64FC2);
                    cube_coefs.at<double>(3) = -(pow(old_num_re, 3));
                    cube_coefs.at<double>(2) = -(15*pow(old_num_re, 2) + 27*pow(old_num_im, 2));
                    cube_coefs.at<double>(1) = -48*old_num_re;
                    cube_coefs.at<double>(0) = 64;
                    solveCubic(cube_coefs, cube_roots);

                    if(cube_roots.at<double>(0) >= 0) num.re = pow(cube_roots.at<double>(0), 1./3);
                    else num.re = -pow(-cube_roots.at<double>(0), 1./3);
                    num.im = sqrt(pow(num.re, 2) / 3 - old_num_re / (3*num.re));
                }
            }
            roots[i] = p - num;
            maxDiff = std::max(maxDiff, cv::abs(num));
        }
        if( maxDiff <= 0 )
            break;
    }

    if( coeffs0.channels() == 1 )
    {
        const double verySmallEps = 1e-100;
        for( i = 0; i < n; i++ )
            if( fabs(roots[i].im) < verySmallEps )
                roots[i].im = 0;
    }

    for( ; n < n0; n++ )
        roots[n+1] = roots[n];

    Mat(roots0.size(), CV_64FC2, roots).convertTo(roots0, roots0.type());
    return maxDiff;
}


CV_IMPL int
cvSolveCubic( const CvMat* coeffs, CvMat* roots )
{
    cv::Mat _coeffs = cv::cvarrToMat(coeffs), _roots = cv::cvarrToMat(roots), _roots0 = _roots;
    int nroots = cv::solveCubic(_coeffs, _roots);
    CV_Assert( _roots.data == _roots0.data ); // check that the array of roots was not reallocated
    return nroots;
}


void cvSolvePoly(const CvMat* a, CvMat *r, int maxiter, int)
{
    cv::Mat _a = cv::cvarrToMat(a);
    cv::Mat _r = cv::cvarrToMat(r);
    cv::Mat _r0 = _r;
    cv::solvePoly(_a, _r, maxiter);
    CV_Assert( _r.data == _r0.data ); // check that the array of roots was not reallocated
}


/* End of file. */
