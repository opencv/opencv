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
#include <atomic>
#include <limits>
#include <iostream>
#include "mathfuncs.hpp"

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
                  format("-D %s -D %s -D dstT=%s -D DEPTH_dst=%d -D rowsPerWI=%d%s", _src2.empty() ? "UNARY_OP" : "BINARY_OP",
                         oclop2str[oclop], ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)), depth, rowsPerWI,
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
    CV_INSTRUMENT_REGION();

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
    CV_INSTRUMENT_REGION();

    int type = src1.type(), depth = src1.depth(), cn = src1.channels();
    CV_Assert( src1.size() == src2.size() && type == src2.type() && (depth == CV_32F || depth == CV_64F));

    CV_OCL_RUN(dst.isUMat() && src1.dims() <= 2 && src2.dims() <= 2,
               ocl_math_op(src1, src2, dst, OCL_OP_MAG))

    Mat X = src1.getMat(), Y = src2.getMat();
    dst.create(X.dims, X.size, X.type());
    Mat Mag = dst.getMat();

    const Mat* arrays[] = {&X, &Y, &Mag, 0};
    uchar* ptrs[3] = {};
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
    CV_INSTRUMENT_REGION();

    int type = src1.type(), depth = src1.depth(), cn = src1.channels();
    CV_Assert( src1.size() == src2.size() && type == src2.type() && (depth == CV_32F || depth == CV_64F));

    CV_OCL_RUN(dst.isUMat() && src1.dims() <= 2 && src2.dims() <= 2,
               ocl_math_op(src1, src2, dst, angleInDegrees ? OCL_OP_PHASE_DEGREES : OCL_OP_PHASE_RADIANS))

    Mat X = src1.getMat(), Y = src2.getMat();
    dst.create( X.dims, X.size, type );
    Mat Angle = dst.getMat();

    const Mat* arrays[] = {&X, &Y, &Angle, 0};
    uchar* ptrs[3] = {};
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

    const bool _src1IsDstMag = (_src1.getObj() == _dst1.getObj());
    const bool _src1IsDstAngle = (_src1.getObj() == _dst2.getObj());
    const bool _src2IsDstMag = (_src2.getObj() == _dst1.getObj());
    const bool _src2IsDstAngle = (_src2.getObj() == _dst2.getObj());

    if ( !(_src1.dims() <= 2 && _src2.dims() <= 2 &&
           (depth == CV_32F || depth == CV_64F) && type == _src2.type()) ||
         (depth == CV_64F && !doubleSupport) )
        return false;

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D BINARY_OP -D dstT=%s -D DEPTH_dst=%d -D rowsPerWI=%d -D OP_CTP_%s%s%s%s%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, 1)), depth,
                         rowsPerWI, angleInDegrees ? "AD" : "AR",
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src1IsDstMag   ? " -D SRC1_IS_DST_MAG" : "",
                         _src1IsDstAngle ? " -D SRC1_IS_DST_ANGLE" : "",
                         _src2IsDstMag   ? " -D SRC2_IS_DST_MAG" : "",
                         _src2IsDstAngle ? " -D SRC2_IS_DST_ANGLE" : ""
                         ));
    if (k.empty())
        return false;

    UMat src1 = _src1.getUMat(), src2 = _src2.getUMat();
    Size size = src1.size();
    CV_Assert( size == src2.size() );

    _dst1.create(size, type);
    _dst2.create(size, type);
    UMat dst1 = _dst1.getUMat(), dst2 = _dst2.getUMat();

    k.args(_src1IsDstMag || _src1IsDstAngle ? ocl::KernelArg::ReadWriteNoSize(src1) : ocl::KernelArg::ReadOnlyNoSize(src1),
           _src2IsDstMag || _src2IsDstAngle ? ocl::KernelArg::ReadWriteNoSize(src2) : ocl::KernelArg::ReadOnlyNoSize(src2),
           ocl::KernelArg::WriteOnly(dst1, cn),
           ocl::KernelArg::WriteOnlyNoSize(dst2));

    size_t globalsize[2] = { (size_t)dst1.cols * cn, ((size_t)dst1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

void cartToPolar( InputArray src1, InputArray src2,
                  OutputArray dst1, OutputArray dst2, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(dst1.getObj() != dst2.getObj());

    CV_OCL_RUN(dst1.isUMat() && dst2.isUMat(),
            ocl_cartToPolar(src1, src2, dst1, dst2, angleInDegrees))

    Mat X = src1.getMat(), Y = src2.getMat();
    int type = X.type(), depth = X.depth(), cn = X.channels();
    CV_Assert( X.size == Y.size && type == Y.type() && (depth == CV_32F || depth == CV_64F));
    dst1.create( X.dims, X.size, type );
    dst2.create( X.dims, X.size, type );
    Mat Mag = dst1.getMat(), Angle = dst2.getMat();

    const Mat* arrays[] = {&X, &Y, &Mag, &Angle, 0};
    uchar* ptrs[4] = {};
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
                hal::cartToPolar32f( x, y, mag, angle, len, angleInDegrees );
            }
            else
            {
                const double *x = (const double*)ptrs[0], *y = (const double*)ptrs[1];
                double *mag = (double*)ptrs[2], *angle = (double*)ptrs[3];
                hal::cartToPolar64f(x, y, mag, angle, len, angleInDegrees);
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

#ifdef HAVE_OPENCL

static bool ocl_polarToCart( InputArray _mag, InputArray _angle,
                             OutputArray _dst1, OutputArray _dst2, bool angleInDegrees )
{
    const ocl::Device & d = ocl::Device::getDefault();
    int type = _angle.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            rowsPerWI = d.isIntel() ? 4 : 1;
    bool doubleSupport = d.doubleFPConfig() > 0;

    const bool _src1IsDstX = (_mag.getObj() == _dst1.getObj());
    const bool _src1IsDstY = (_mag.getObj() == _dst2.getObj());
    const bool _src2IsDstX = (_angle.getObj() == _dst1.getObj());
    const bool _src2IsDstY = (_angle.getObj() == _dst2.getObj());

    if ( !doubleSupport && depth == CV_64F )
        return false;

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D dstT=%s -D DEPTH_dst=%d -D rowsPerWI=%d -D BINARY_OP -D OP_PTC_%s%s%s%s%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, 1)), depth,
                         rowsPerWI,
                         angleInDegrees ? "AD" : "AR",
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src1IsDstX   ? " -D SRC1_IS_DST_X" : "",
                         _src1IsDstY ? " -D SRC1_IS_DST_Y" : "",
                         _src2IsDstX   ? " -D SRC2_IS_DST_X" : "",
                         _src2IsDstY ? " -D SRC2_IS_DST_Y" : ""));
    if (k.empty())
        return false;

    UMat mag = _mag.getUMat(), angle = _angle.getUMat();
    Size size = angle.size();
    CV_Assert(mag.size() == size);

    _dst1.create(size, type);
    _dst2.create(size, type);
    UMat dst1 = _dst1.getUMat(), dst2 = _dst2.getUMat();

    k.args(_src1IsDstX || _src1IsDstY ? ocl::KernelArg::ReadWriteNoSize(mag) : ocl::KernelArg::ReadOnlyNoSize(mag),
           _src2IsDstX || _src2IsDstY  ? ocl::KernelArg::ReadWriteNoSize(angle) : ocl::KernelArg::ReadOnlyNoSize(angle),
           ocl::KernelArg::WriteOnly(dst1, cn),
           ocl::KernelArg::WriteOnlyNoSize(dst2));

    size_t globalsize[2] = { (size_t)dst1.cols * cn, ((size_t)dst1.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

void polarToCart( InputArray src1, InputArray src2,
                  OutputArray dst1, OutputArray dst2, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(dst1.getObj() != dst2.getObj());

    int type = src2.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert((depth == CV_32F || depth == CV_64F) && (src1.empty() || src1.type() == type));

    CV_OCL_RUN(!src1.empty() && src2.dims() <= 2 && dst1.isUMat() && dst2.isUMat(),
               ocl_polarToCart(src1, src2, dst1, dst2, angleInDegrees))

    Mat Mag = src1.getMat(), Angle = src2.getMat();
    CV_Assert( Mag.empty() || Angle.size == Mag.size);
    dst1.create( Angle.dims, Angle.size, type );
    dst2.create( Angle.dims, Angle.size, type );
    Mat X = dst1.getMat(), Y = dst2.getMat();

    const Mat* arrays[] = {&Mag, &Angle, &X, &Y, 0};
    uchar* ptrs[4] = {};
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)(it.size*cn), blockSize = std::min(total, ((BLOCK_SIZE+cn-1)/cn)*cn);
    size_t esz1 = Angle.elemSize1();

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int len = std::min(total - j, blockSize);
            if ( depth == CV_32F )
            {
                const float *mag = (const float*)ptrs[0], *angle = (const float*)ptrs[1];
                float *x = (float*)ptrs[2], *y = (float*)ptrs[3];
                hal::polarToCart32f( mag, angle, x, y, len, angleInDegrees );
            }
            else
            {
                const double *mag = (const double*)ptrs[0], *angle = (const double*)ptrs[1];
                double *x = (double*)ptrs[2], *y = (double*)ptrs[3];
                hal::polarToCart64f( mag, angle, x, y, len, angleInDegrees );
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

void exp( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), depth = _src.depth(), cn = _src.channels();
    CV_Assert( depth == CV_32F || depth == CV_64F );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_math_op(_src, noArray(), _dst, OCL_OP_EXP))

    Mat src = _src.getMat();
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        if( depth == CV_32F )
            hal::exp32f((const float*)ptrs[0], (float*)ptrs[1], len);
        else
            hal::exp64f((const double*)ptrs[0], (double*)ptrs[1], len);
    }
}


/****************************************************************************************\
*                                          L O G                                         *
\****************************************************************************************/

void log( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), depth = _src.depth(), cn = _src.channels();
    CV_Assert( depth == CV_32F || depth == CV_64F );

    CV_OCL_RUN( _dst.isUMat() && _src.dims() <= 2,
                ocl_math_op(_src, noArray(), _dst, OCL_OP_LOG))

    Mat src = _src.getMat();
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        if( depth == CV_32F )
            hal::log32f( (const float*)ptrs[0], (float*)ptrs[1], len );
        else
            hal::log64f( (const double*)ptrs[0], (double*)ptrs[1], len );
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

#if (CV_SIMD || CV_SIMD_SCALABLE)

template <>
struct iPow_SIMD<uchar, unsigned>
{
    int operator() ( const uchar * src, uchar * dst, int len, int power )
    {
        int i = 0;
        v_uint32 v_1 = vx_setall_u32(1u);

        for ( ; i <= len - VTraits<v_uint16>::vlanes(); i += VTraits<v_uint16>::vlanes())
        {
            v_uint32 v_a1 = v_1, v_a2 = v_1;
            v_uint16 v = vx_load_expand(src + i);
            v_uint32 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v = v_pack(v_a1, v_a2);
            v_pack_store(dst + i, v);
        }
        vx_cleanup();

        return i;
    }
};

template <>
struct iPow_SIMD<schar, int>
{
    int operator() ( const schar * src, schar * dst, int len, int power)
    {
        int i = 0;
        v_int32 v_1 = vx_setall_s32(1);

        for ( ; i <= len - VTraits<v_int16>::vlanes(); i += VTraits<v_int16>::vlanes())
        {
            v_int32 v_a1 = v_1, v_a2 = v_1;
            v_int16 v = vx_load_expand(src + i);
            v_int32 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v = v_pack(v_a1, v_a2);
            v_pack_store(dst + i, v);
        }
        vx_cleanup();

        return i;
    }
};

template <>
struct iPow_SIMD<ushort, unsigned>
{
    int operator() ( const ushort * src, ushort * dst, int len, int power)
    {
        int i = 0;
        v_uint32 v_1 = vx_setall_u32(1u);

        for ( ; i <= len - VTraits<v_uint16>::vlanes(); i += VTraits<v_uint16>::vlanes())
        {
            v_uint32 v_a1 = v_1, v_a2 = v_1;
            v_uint16 v = vx_load(src + i);
            v_uint32 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v = v_pack(v_a1, v_a2);
            v_store(dst + i, v);
        }
        vx_cleanup();

        return i;
    }
};

template <>
struct iPow_SIMD<short, int>
{
    int operator() ( const short * src, short * dst, int len, int power)
    {
        int i = 0;
        v_int32 v_1 = vx_setall_s32(1);

        for ( ; i <= len - VTraits<v_int16>::vlanes(); i += VTraits<v_int16>::vlanes())
        {
            v_int32 v_a1 = v_1, v_a2 = v_1;
            v_int16 v = vx_load(src + i);
            v_int32 v_b1, v_b2;
            v_expand(v, v_b1, v_b2);
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v = v_pack(v_a1, v_a2);
            v_store(dst + i, v);
        }
        vx_cleanup();

        return i;
    }
};

template <>
struct iPow_SIMD<int, int>
{
    int operator() ( const int * src, int * dst, int len, int power)
    {
        int i = 0;
        v_int32 v_1 = vx_setall_s32(1);

        for ( ; i <= len - VTraits<v_int32>::vlanes()*2; i += VTraits<v_int32>::vlanes()*2)
        {
            v_int32 v_a1 = v_1, v_a2 = v_1;
            v_int32 v_b1 = vx_load(src + i), v_b2 = vx_load(src + i + VTraits<v_int32>::vlanes());
            int p = power;

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v_store(dst + i, v_a1);
            v_store(dst + i + VTraits<v_int32>::vlanes(), v_a2);
        }
        vx_cleanup();

        return i;
    }
};

template <>
struct iPow_SIMD<float, float>
{
    int operator() ( const float * src, float * dst, int len, int power)
    {
        int i = 0;
        v_float32 v_1 = vx_setall_f32(1.f);

        for ( ; i <= len - VTraits<v_float32>::vlanes()*2; i += VTraits<v_float32>::vlanes()*2)
        {
            v_float32 v_a1 = v_1, v_a2 = v_1;
            v_float32 v_b1 = vx_load(src + i), v_b2 = vx_load(src + i + VTraits<v_float32>::vlanes());
            int p = std::abs(power);
            if( power < 0 )
            {
                v_b1 = v_div(v_1, v_b1);
                v_b2 = v_div(v_1, v_b2);
            }

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v_store(dst + i, v_a1);
            v_store(dst + i + VTraits<v_float32>::vlanes(), v_a2);
        }
        vx_cleanup();

        return i;
    }
};

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template <>
struct iPow_SIMD<double, double>
{
    int operator() ( const double * src, double * dst, int len, int power)
    {
        int i = 0;
        v_float64 v_1 = vx_setall_f64(1.);

        for ( ; i <= len - VTraits<v_float64>::vlanes()*2; i += VTraits<v_float64>::vlanes()*2)
        {
            v_float64 v_a1 = v_1, v_a2 = v_1;
            v_float64 v_b1 = vx_load(src + i), v_b2 = vx_load(src + i + VTraits<v_float64>::vlanes());
            int p = std::abs(power);
            if( power < 0 )
            {
                v_b1 = v_div(v_1, v_b1);
                v_b2 = v_div(v_1, v_b2);
            }

            while( p > 1 )
            {
                if (p & 1)
                {
                    v_a1 = v_mul(v_a1, v_b1);
                    v_a2 = v_mul(v_a2, v_b2);
                }
                v_b1 = v_mul(v_b1, v_b1);
                v_b2 = v_mul(v_b2, v_b2);
                p >>= 1;
            }

            v_a1 = v_mul(v_a1, v_b1);
            v_a2 = v_mul(v_a2, v_b2);

            v_store(dst + i, v_a1);
            v_store(dst + i + VTraits<v_float64>::vlanes(), v_a2);
        }
        vx_cleanup();

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

static IPowFunc ipowTab[CV_DEPTH_MAX] =
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

    // Note: channels are unrolled

    std::string extra_opts ="";
    if (is_ipower)
    {
        int wdepth = CV_32F;
        if (depth == CV_64F)
            wdepth = CV_64F;
        else if (depth == CV_16F)
            wdepth = CV_16F;

        char cvt[2][50];
        extra_opts = format(
            " -D srcT1=%s -DsrcT1_C1=%s"
            " -D srcT2=int -D workST=int"
            " -D workT=%s -D wdepth=%d -D convertToWT1=%s"
            " -D convertToDT=%s"
            " -D workT1=%s",
            ocl::typeToStr(CV_MAKE_TYPE(depth, 1)),
            ocl::typeToStr(CV_MAKE_TYPE(depth, 1)),
            ocl::typeToStr(CV_MAKE_TYPE(wdepth, 1)),
            wdepth,
            ocl::convertTypeStr(depth, wdepth, 1, cvt[0], sizeof(cvt[0])),
            ocl::convertTypeStr(wdepth, depth, 1, cvt[1], sizeof(cvt[1])),
            ocl::typeToStr(wdepth)
        );
    }

    ocl::Kernel k("KF", ocl::core::arithm_oclsrc,
                  format("-D cn=%d -D dstT=%s -D dstT_C1=%s -D DEPTH_dst=%d -D rowsPerWI=%d -D %s%s%s%s",
                         1,
                         ocl::typeToStr(depth), ocl::typeToStr(depth), depth, rowsPerWI, op,
                         " -D UNARY_OP=1",
                         extra_opts.empty() ? "" : extra_opts.c_str(),
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

void pow( InputArray _src, double power, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), depth = CV_MAT_DEPTH(type),
            cn = CV_MAT_CN(type), ipower = cvRound(power);
    bool is_ipower = fabs(ipower - power) < DBL_EPSILON;
#ifdef HAVE_OPENCL
    bool useOpenCL = _dst.isUMat() && _src.dims() <= 2;
#endif

    if (is_ipower)
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

    CV_OCL_RUN(useOpenCL, ocl_pow(_src, power, _dst, is_ipower, ipower))

    Mat src = _src.getMat();
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
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
            (depth == CV_32F ? (MathFunc)hal::invSqrt32f : (MathFunc)hal::invSqrt64f) :
            (depth == CV_32F ? (MathFunc)hal::sqrt32f : (MathFunc)hal::sqrt64f);

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
#ifndef __EMSCRIPTEN__
        inf32.i = 0x7f800000;
        nan32.i = 0x7fffffff;
        inf64.i = CV_BIG_INT(0x7FF0000000000000);
        nan64.i = CV_BIG_INT(0x7FFFFFFFFFFFFFFF);
#else
        inf32.f = std::numeric_limits<float>::infinity();
        nan32.f = std::numeric_limits<float>::quiet_NaN();
        inf64.f = std::numeric_limits<double>::infinity();
        nan64.f = std::numeric_limits<double>::quiet_NaN();
#endif

        if( src.ptr() == dst.ptr() )
        {
            buf.allocate(blockSize*esz1);
            fbuf = (float*)buf.data();
            dbuf = (double*)buf.data();
        }

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            for( j = 0; j < len; j += blockSize )
            {
                int bsz = std::min(len - j, blockSize);

                if( depth == CV_32F )
                {
                    float* x0 = (float*)ptrs[0];
                    float* x = fbuf ? fbuf : x0;
                    float* y = (float*)ptrs[1];

                    if( x != x0 )
                        memcpy(x, x0, bsz*esz1);

                    hal::log32f(x, y, bsz);
                    for( k = 0; k < bsz; k++ )
                        y[k] = (float)(y[k]*power);
                    hal::exp32f(y, y, bsz);
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

                    hal::log64f(x, y, bsz);
                    for( k = 0; k < bsz; k++ )
                        y[k] *= power;
                    hal::exp64f(y, y, bsz);

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
    CV_INSTRUMENT_REGION();

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
    CV_INSTRUMENT_REGION();

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
        Size size = getContinuousSize2D(src, cn);

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
            CV_Error_( cv::Error::StsOutOfRange,
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
                     format("-D UNARY_OP -D OP_PATCH_NANS -D dstT=float -D DEPTH_dst=%d -D rowsPerWI=%d",
                            CV_32F, rowsPerWI));
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
    CV_INSTRUMENT_REGION();

    CV_Assert( _a.depth() == CV_32F );

    CV_OCL_RUN(_a.isUMat() && _a.dims() <= 2,
               ocl_patchNaNs(_a, (float)_val))

    Mat a = _a.getMat();
    const Mat* arrays[] = {&a, 0};
    int* ptrs[1] = {};
    NAryMatIterator it(arrays, (uchar**)ptrs);
    int len = (int)(it.size*a.channels());
    Cv32suf val;
    val.f = (float)_val;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        int* tptr = ptrs[0];
        int j = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_int32 v_pos_mask = vx_setall_s32(0x7fffffff), v_exp_mask = vx_setall_s32(0x7f800000);
        v_int32 v_val = vx_setall_s32(val.i);

        int cWidth = VTraits<v_int32>::vlanes();
        for (; j < len - cWidth * 2 + 1; j += cWidth * 2)
        {
            v_int32 v_src0 = vx_load(tptr + j);
            v_int32 v_src1 = vx_load(tptr + j + cWidth);

            v_int32 v_cmp_mask0 = v_lt(v_exp_mask, v_and(v_src0, v_pos_mask));
            v_int32 v_cmp_mask1 = v_lt(v_exp_mask, v_and(v_src1, v_pos_mask));

            if (v_check_any(v_or(v_cmp_mask0, v_cmp_mask1)))
            {
                v_int32 v_dst0 = v_select(v_cmp_mask0, v_val, v_src0);
                v_int32 v_dst1 = v_select(v_cmp_mask1, v_val, v_src1);

                v_store(tptr + j, v_dst0);
                v_store(tptr + j + cWidth, v_dst1);
            }
        }
#endif

        for( ; j < len; j++ )
            if( (tptr[j] & 0x7fffffff) > 0x7f800000 )
                tptr[j] = val.i;
    }
}

}


#ifndef OPENCV_EXCLUDE_C_API

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

#endif  // OPENCV_EXCLUDE_C_API

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
    CV_INSTRUMENT_REGION();

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
    CV_INSTRUMENT_REGION();

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
    C *coeffs = buf.data(), *roots = coeffs + n + 1;
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


#ifndef OPENCV_EXCLUDE_C_API

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

#endif  // OPENCV_EXCLUDE_C_API


// Common constants for dispatched code
namespace cv { namespace details {

#define EXPTAB_SCALE 6
#define EXPTAB_MASK  ((1 << EXPTAB_SCALE) - 1)

#define EXPPOLY_32F_A0 .9670371139572337719125840413672004409288e-2

static const double CV_DECL_ALIGNED(64) expTab[EXPTAB_MASK + 1] = {
    1.0 * EXPPOLY_32F_A0,
    1.0108892860517004600204097905619 * EXPPOLY_32F_A0,
    1.0218971486541166782344801347833 * EXPPOLY_32F_A0,
    1.0330248790212284225001082839705 * EXPPOLY_32F_A0,
    1.0442737824274138403219664787399 * EXPPOLY_32F_A0,
    1.0556451783605571588083413251529 * EXPPOLY_32F_A0,
    1.0671404006768236181695211209928 * EXPPOLY_32F_A0,
    1.0787607977571197937406800374385 * EXPPOLY_32F_A0,
    1.0905077326652576592070106557607 * EXPPOLY_32F_A0,
    1.1023825833078409435564142094256 * EXPPOLY_32F_A0,
    1.1143867425958925363088129569196 * EXPPOLY_32F_A0,
    1.126521618608241899794798643787 * EXPPOLY_32F_A0,
    1.1387886347566916537038302838415 * EXPPOLY_32F_A0,
    1.151189229952982705817759635202 * EXPPOLY_32F_A0,
    1.1637248587775775138135735990922 * EXPPOLY_32F_A0,
    1.1763969916502812762846457284838 * EXPPOLY_32F_A0,
    1.1892071150027210667174999705605 * EXPPOLY_32F_A0,
    1.2021567314527031420963969574978 * EXPPOLY_32F_A0,
    1.2152473599804688781165202513388 * EXPPOLY_32F_A0,
    1.2284805361068700056940089577928 * EXPPOLY_32F_A0,
    1.2418578120734840485936774687266 * EXPPOLY_32F_A0,
    1.2553807570246910895793906574423 * EXPPOLY_32F_A0,
    1.2690509571917332225544190810323 * EXPPOLY_32F_A0,
    1.2828700160787782807266697810215 * EXPPOLY_32F_A0,
    1.2968395546510096659337541177925 * EXPPOLY_32F_A0,
    1.3109612115247643419229917863308 * EXPPOLY_32F_A0,
    1.3252366431597412946295370954987 * EXPPOLY_32F_A0,
    1.3396675240533030053600306697244 * EXPPOLY_32F_A0,
    1.3542555469368927282980147401407 * EXPPOLY_32F_A0,
    1.3690024229745906119296011329822 * EXPPOLY_32F_A0,
    1.3839098819638319548726595272652 * EXPPOLY_32F_A0,
    1.3989796725383111402095281367152 * EXPPOLY_32F_A0,
    1.4142135623730950488016887242097 * EXPPOLY_32F_A0,
    1.4296133383919700112350657782751 * EXPPOLY_32F_A0,
    1.4451808069770466200370062414717 * EXPPOLY_32F_A0,
    1.4609177941806469886513028903106 * EXPPOLY_32F_A0,
    1.476826145939499311386907480374 * EXPPOLY_32F_A0,
    1.4929077282912648492006435314867 * EXPPOLY_32F_A0,
    1.5091644275934227397660195510332 * EXPPOLY_32F_A0,
    1.5255981507445383068512536895169 * EXPPOLY_32F_A0,
    1.5422108254079408236122918620907 * EXPPOLY_32F_A0,
    1.5590044002378369670337280894749 * EXPPOLY_32F_A0,
    1.5759808451078864864552701601819 * EXPPOLY_32F_A0,
    1.5931421513422668979372486431191 * EXPPOLY_32F_A0,
    1.6104903319492543081795206673574 * EXPPOLY_32F_A0,
    1.628027421857347766848218522014 * EXPPOLY_32F_A0,
    1.6457554781539648445187567247258 * EXPPOLY_32F_A0,
    1.6636765803267364350463364569764 * EXPPOLY_32F_A0,
    1.6817928305074290860622509524664 * EXPPOLY_32F_A0,
    1.7001063537185234695013625734975 * EXPPOLY_32F_A0,
    1.7186192981224779156293443764563 * EXPPOLY_32F_A0,
    1.7373338352737062489942020818722 * EXPPOLY_32F_A0,
    1.7562521603732994831121606193753 * EXPPOLY_32F_A0,
    1.7753764925265212525505592001993 * EXPPOLY_32F_A0,
    1.7947090750031071864277032421278 * EXPPOLY_32F_A0,
    1.8142521755003987562498346003623 * EXPPOLY_32F_A0,
    1.8340080864093424634870831895883 * EXPPOLY_32F_A0,
    1.8539791250833855683924530703377 * EXPPOLY_32F_A0,
    1.8741676341102999013299989499544 * EXPPOLY_32F_A0,
    1.8945759815869656413402186534269 * EXPPOLY_32F_A0,
    1.9152065613971472938726112702958 * EXPPOLY_32F_A0,
    1.9360617934922944505980559045667 * EXPPOLY_32F_A0,
    1.9571441241754002690183222516269 * EXPPOLY_32F_A0,
    1.9784560263879509682582499181312 * EXPPOLY_32F_A0,
};

const double* getExpTab64f()
{
    return expTab;
}

const float* getExpTab32f()
{
    static float CV_DECL_ALIGNED(64) expTab_f[EXPTAB_MASK+1];
    static std::atomic<bool> expTab_f_initialized(false);
    if (!expTab_f_initialized.load())
    {
        for( int j = 0; j <= EXPTAB_MASK; j++ )
            expTab_f[j] = (float)expTab[j];
        expTab_f_initialized = true;
    }
    return expTab_f;
}



#define LOGTAB_SCALE        8
#define LOGTAB_MASK         ((1 << LOGTAB_SCALE) - 1)

static const double CV_DECL_ALIGNED(64) logTab[(LOGTAB_MASK+1)*2] = {
    0.0000000000000000000000000000000000000000,    1.000000000000000000000000000000000000000,
    .00389864041565732288852075271279318258166,    .9961089494163424124513618677042801556420,
    .00778214044205494809292034119607706088573,    .9922480620155038759689922480620155038760,
    .01165061721997527263705585198749759001657,    .9884169884169884169884169884169884169884,
    .01550418653596525274396267235488267033361,    .9846153846153846153846153846153846153846,
    .01934296284313093139406447562578250654042,    .9808429118773946360153256704980842911877,
    .02316705928153437593630670221500622574241,    .9770992366412213740458015267175572519084,
    .02697658769820207233514075539915211265906,    .9733840304182509505703422053231939163498,
    .03077165866675368732785500469617545604706,    .9696969696969696969696969696969696969697,
    .03455238150665972812758397481047722976656,    .9660377358490566037735849056603773584906,
    .03831886430213659461285757856785494368522,    .9624060150375939849624060150375939849624,
    .04207121392068705056921373852674150839447,    .9588014981273408239700374531835205992509,
    .04580953603129420126371940114040626212953,    .9552238805970149253731343283582089552239,
    .04953393512227662748292900118940451648088,    .9516728624535315985130111524163568773234,
    .05324451451881227759255210685296333394944,    .9481481481481481481481481481481481481481,
    .05694137640013842427411105973078520037234,    .9446494464944649446494464944649446494465,
    .06062462181643483993820353816772694699466,    .9411764705882352941176470588235294117647,
    .06429435070539725460836422143984236754475,    .9377289377289377289377289377289377289377,
    .06795066190850773679699159401934593915938,    .9343065693430656934306569343065693430657,
    .07159365318700880442825962290953611955044,    .9309090909090909090909090909090909090909,
    .07522342123758751775142172846244648098944,    .9275362318840579710144927536231884057971,
    .07884006170777602129362549021607264876369,    .9241877256317689530685920577617328519856,
    .08244366921107458556772229485432035289706,    .9208633093525179856115107913669064748201,
    .08603433734180314373940490213499288074675,    .9175627240143369175627240143369175627240,
    .08961215868968712416897659522874164395031,    .9142857142857142857142857142857142857143,
    .09317722485418328259854092721070628613231,    .9110320284697508896797153024911032028470,
    .09672962645855109897752299730200320482256,    .9078014184397163120567375886524822695035,
    .10026945316367513738597949668474029749630,    .9045936395759717314487632508833922261484,
    .10379679368164355934833764649738441221420,    .9014084507042253521126760563380281690141,
    .10731173578908805021914218968959175981580,    .8982456140350877192982456140350877192982,
    .11081436634029011301105782649756292812530,    .8951048951048951048951048951048951048951,
    .11430477128005862852422325204315711744130,    .8919860627177700348432055749128919860627,
    .11778303565638344185817487641543266363440,    .8888888888888888888888888888888888888889,
    .12124924363286967987640707633545389398930,    .8858131487889273356401384083044982698962,
    .12470347850095722663787967121606925502420,    .8827586206896551724137931034482758620690,
    .12814582269193003360996385708858724683530,    .8797250859106529209621993127147766323024,
    .13157635778871926146571524895989568904040,    .8767123287671232876712328767123287671233,
    .13499516453750481925766280255629681050780,    .8737201365187713310580204778156996587031,
    .13840232285911913123754857224412262439730,    .8707482993197278911564625850340136054422,
    .14179791186025733629172407290752744302150,    .8677966101694915254237288135593220338983,
    .14518200984449788903951628071808954700830,    .8648648648648648648648648648648648648649,
    .14855469432313711530824207329715136438610,    .8619528619528619528619528619528619528620,
    .15191604202584196858794030049466527998450,    .8590604026845637583892617449664429530201,
    .15526612891112392955683674244937719777230,    .8561872909698996655518394648829431438127,
    .15860503017663857283636730244325008243330,    .8533333333333333333333333333333333333333,
    .16193282026931324346641360989451641216880,    .8504983388704318936877076411960132890365,
    .16524957289530714521497145597095368430010,    .8476821192052980132450331125827814569536,
    .16855536102980664403538924034364754334090,    .8448844884488448844884488448844884488449,
    .17185025692665920060697715143760433420540,    .8421052631578947368421052631578947368421,
    .17513433212784912385018287750426679849630,    .8393442622950819672131147540983606557377,
    .17840765747281828179637841458315961062910,    .8366013071895424836601307189542483660131,
    .18167030310763465639212199675966985523700,    .8338762214983713355048859934853420195440,
    .18492233849401198964024217730184318497780,    .8311688311688311688311688311688311688312,
    .18816383241818296356839823602058459073300,    .8284789644012944983818770226537216828479,
    .19139485299962943898322009772527962923050,    .8258064516129032258064516129032258064516,
    .19461546769967164038916962454095482826240,    .8231511254019292604501607717041800643087,
    .19782574332991986754137769821682013571260,    .8205128205128205128205128205128205128205,
    .20102574606059073203390141770796617493040,    .8178913738019169329073482428115015974441,
    .20421554142869088876999228432396193966280,    .8152866242038216560509554140127388535032,
    .20739519434607056602715147164417430758480,    .8126984126984126984126984126984126984127,
    .21056476910734961416338251183333341032260,    .8101265822784810126582278481012658227848,
    .21372432939771812687723695489694364368910,    .8075709779179810725552050473186119873817,
    .21687393830061435506806333251006435602900,    .8050314465408805031446540880503144654088,
    .22001365830528207823135744547471404075630,    .8025078369905956112852664576802507836991,
    .22314355131420973710199007200571941211830,    .8000000000000000000000000000000000000000,
    .22626367865045338145790765338460914790630,    .7975077881619937694704049844236760124611,
    .22937410106484582006380890106811420992010,    .7950310559006211180124223602484472049689,
    .23247487874309405442296849741978803649550,    .7925696594427244582043343653250773993808,
    .23556607131276688371634975283086532726890,    .7901234567901234567901234567901234567901,
    .23864773785017498464178231643018079921600,    .7876923076923076923076923076923076923077,
    .24171993688714515924331749374687206000090,    .7852760736196319018404907975460122699387,
    .24478272641769091566565919038112042471760,    .7828746177370030581039755351681957186544,
    .24783616390458124145723672882013488560910,    .7804878048780487804878048780487804878049,
    .25088030628580937353433455427875742316250,    .7781155015197568389057750759878419452888,
    .25391520998096339667426946107298135757450,    .7757575757575757575757575757575757575758,
    .25694093089750041913887912414793390780680,    .7734138972809667673716012084592145015106,
    .25995752443692604627401010475296061486000,    .7710843373493975903614457831325301204819,
    .26296504550088134477547896494797896593800,    .7687687687687687687687687687687687687688,
    .26596354849713793599974565040611196309330,    .7664670658682634730538922155688622754491,
    .26895308734550393836570947314612567424780,    .7641791044776119402985074626865671641791,
    .27193371548364175804834985683555714786050,    .7619047619047619047619047619047619047619,
    .27490548587279922676529508862586226314300,    .7596439169139465875370919881305637982196,
    .27786845100345625159121709657483734190480,    .7573964497041420118343195266272189349112,
    .28082266290088775395616949026589281857030,    .7551622418879056047197640117994100294985,
    .28376817313064456316240580235898960381750,    .7529411764705882352941176470588235294118,
    .28670503280395426282112225635501090437180,    .7507331378299120234604105571847507331378,
    .28963329258304265634293983566749375313530,    .7485380116959064327485380116959064327485,
    .29255300268637740579436012922087684273730,    .7463556851311953352769679300291545189504,
    .29546421289383584252163927885703742504130,    .7441860465116279069767441860465116279070,
    .29836697255179722709783618483925238251680,    .7420289855072463768115942028985507246377,
    .30126133057816173455023545102449133992200,    .7398843930635838150289017341040462427746,
    .30414733546729666446850615102448500692850,    .7377521613832853025936599423631123919308,
    .30702503529491181888388950937951449304830,    .7356321839080459770114942528735632183908,
    .30989447772286465854207904158101882785550,    .7335243553008595988538681948424068767908,
    .31275571000389684739317885942000430077330,    .7314285714285714285714285714285714285714,
    .31560877898630329552176476681779604405180,    .7293447293447293447293447293447293447293,
    .31845373111853458869546784626436419785030,    .7272727272727272727272727272727272727273,
    .32129061245373424782201254856772720813750,    .7252124645892351274787535410764872521246,
    .32411946865421192853773391107097268104550,    .7231638418079096045197740112994350282486,
    .32694034499585328257253991068864706903700,    .7211267605633802816901408450704225352113,
    .32975328637246797969240219572384376078850,    .7191011235955056179775280898876404494382,
    .33255833730007655635318997155991382896900,    .7170868347338935574229691876750700280112,
    .33535554192113781191153520921943709254280,    .7150837988826815642458100558659217877095,
    .33814494400871636381467055798566434532400,    .7130919220055710306406685236768802228412,
    .34092658697059319283795275623560883104800,    .7111111111111111111111111111111111111111,
    .34370051385331840121395430287520866841080,    .7091412742382271468144044321329639889197,
    .34646676734620857063262633346312213689100,    .7071823204419889502762430939226519337017,
    .34922538978528827602332285096053965389730,    .7052341597796143250688705234159779614325,
    .35197642315717814209818925519357435405250,    .7032967032967032967032967032967032967033,
    .35471990910292899856770532096561510115850,    .7013698630136986301369863013698630136986,
    .35745588892180374385176833129662554711100,    .6994535519125683060109289617486338797814,
    .36018440357500774995358483465679455548530,    .6975476839237057220708446866485013623978,
    .36290549368936841911903457003063522279280,    .6956521739130434782608695652173913043478,
    .36561919956096466943762379742111079394830,    .6937669376693766937669376693766937669377,
    .36832556115870762614150635272380895912650,    .6918918918918918918918918918918918918919,
    .37102461812787262962487488948681857436900,    .6900269541778975741239892183288409703504,
    .37371640979358405898480555151763837784530,    .6881720430107526881720430107526881720430,
    .37640097516425302659470730759494472295050,    .6863270777479892761394101876675603217158,
    .37907835293496944251145919224654790014030,    .6844919786096256684491978609625668449198,
    .38174858149084833769393299007788300514230,    .6826666666666666666666666666666666666667,
    .38441169891033200034513583887019194662580,    .6808510638297872340425531914893617021277,
    .38706774296844825844488013899535872042180,    .6790450928381962864721485411140583554377,
    .38971675114002518602873692543653305619950,    .6772486772486772486772486772486772486772,
    .39235876060286384303665840889152605086580,    .6754617414248021108179419525065963060686,
    .39499380824086893770896722344332374632350,    .6736842105263157894736842105263157894737,
    .39762193064713846624158577469643205404280,    .6719160104986876640419947506561679790026,
    .40024316412701266276741307592601515352730,    .6701570680628272251308900523560209424084,
    .40285754470108348090917615991202183067800,    .6684073107049608355091383812010443864230,
    .40546510810816432934799991016916465014230,    .6666666666666666666666666666666666666667,
    .40806588980822172674223224930756259709600,    .6649350649350649350649350649350649350649,
    .41065992498526837639616360320360399782650,    .6632124352331606217616580310880829015544,
    .41324724855021932601317757871584035456180,    .6614987080103359173126614987080103359173,
    .41582789514371093497757669865677598863850,    .6597938144329896907216494845360824742268,
    .41840189913888381489925905043492093682300,    .6580976863753213367609254498714652956298,
    .42096929464412963239894338585145305842150,    .6564102564102564102564102564102564102564,
    .42353011550580327293502591601281892508280,    .6547314578005115089514066496163682864450,
    .42608439531090003260516141381231136620050,    .6530612244897959183673469387755102040816,
    .42863216738969872610098832410585600882780,    .6513994910941475826972010178117048346056,
    .43117346481837132143866142541810404509300,    .6497461928934010152284263959390862944162,
    .43370832042155937902094819946796633303180,    .6481012658227848101265822784810126582278,
    .43623676677491801667585491486534010618930,    .6464646464646464646464646464646464646465,
    .43875883620762790027214350629947148263450,    .6448362720403022670025188916876574307305,
    .44127456080487520440058801796112675219780,    .6432160804020100502512562814070351758794,
    .44378397241030093089975139264424797147500,    .6416040100250626566416040100250626566416,
    .44628710262841947420398014401143882423650,    .6400000000000000000000000000000000000000,
    .44878398282700665555822183705458883196130,    .6384039900249376558603491271820448877805,
    .45127464413945855836729492693848442286250,    .6368159203980099502487562189054726368159,
    .45375911746712049854579618113348260521900,    .6352357320099255583126550868486352357320,
    .45623743348158757315857769754074979573500,    .6336633663366336633663366336633663366337,
    .45870962262697662081833982483658473938700,    .6320987654320987654320987654320987654321,
    .46117571512217014895185229761409573256980,    .6305418719211822660098522167487684729064,
    .46363574096303250549055974261136725544930,    .6289926289926289926289926289926289926290,
    .46608972992459918316399125615134835243230,    .6274509803921568627450980392156862745098,
    .46853771156323925639597405279346276074650,    .6259168704156479217603911980440097799511,
    .47097971521879100631480241645476780831830,    .6243902439024390243902439024390243902439,
    .47341577001667212165614273544633761048330,    .6228710462287104622871046228710462287105,
    .47584590486996386493601107758877333253630,    .6213592233009708737864077669902912621359,
    .47827014848147025860569669930555392056700,    .6198547215496368038740920096852300242131,
    .48068852934575190261057286988943815231330,    .6183574879227053140096618357487922705314,
    .48310107575113581113157579238759353756900,    .6168674698795180722891566265060240963855,
    .48550781578170076890899053978500887751580,    .6153846153846153846153846153846153846154,
    .48790877731923892879351001283794175833480,    .6139088729016786570743405275779376498801,
    .49030398804519381705802061333088204264650,    .6124401913875598086124401913875598086124,
    .49269347544257524607047571407747454941280,    .6109785202863961813842482100238663484487,
    .49507726679785146739476431321236304938800,    .6095238095238095238095238095238095238095,
    .49745538920281889838648226032091770321130,    .6080760095011876484560570071258907363420,
    .49982786955644931126130359189119189977650,    .6066350710900473933649289099526066350711,
    .50219473456671548383667413872899487614650,    .6052009456264775413711583924349881796690,
    .50455601075239520092452494282042607665050,    .6037735849056603773584905660377358490566,
    .50691172444485432801997148999362252652650,    .6023529411764705882352941176470588235294,
    .50926190178980790257412536448100581765150,    .6009389671361502347417840375586854460094,
    .51160656874906207391973111953120678663250,    .5995316159250585480093676814988290398126,
    .51394575110223428282552049495279788970950,    .5981308411214953271028037383177570093458,
    .51627947444845445623684554448118433356300,    .5967365967365967365967365967365967365967,
    .51860776420804555186805373523384332656850,    .5953488372093023255813953488372093023256,
    .52093064562418522900344441950437612831600,    .5939675174013921113689095127610208816705,
    .52324814376454775732838697877014055848100,    .5925925925925925925925925925925925925926,
    .52556028352292727401362526507000438869000,    .5912240184757505773672055427251732101617,
    .52786708962084227803046587723656557500350,    .5898617511520737327188940092165898617512,
    .53016858660912158374145519701414741575700,    .5885057471264367816091954022988505747126,
    .53246479886947173376654518506256863474850,    .5871559633027522935779816513761467889908,
    .53475575061602764748158733709715306758900,    .5858123569794050343249427917620137299771,
    .53704146589688361856929077475797384977350,    .5844748858447488584474885844748858447489,
    .53932196859560876944783558428753167390800,    .5831435079726651480637813211845102505695,
    .54159728243274429804188230264117009937750,    .5818181818181818181818181818181818181818,
    .54386743096728351609669971367111429572100,    .5804988662131519274376417233560090702948,
    .54613243759813556721383065450936555862450,    .5791855203619909502262443438914027149321,
    .54839232556557315767520321969641372561450,    .5778781038374717832957110609480812641084,
    .55064711795266219063194057525834068655950,    .5765765765765765765765765765765765765766,
    .55289683768667763352766542084282264113450,    .5752808988764044943820224719101123595506,
    .55514150754050151093110798683483153581600,    .5739910313901345291479820627802690582960,
    .55738115013400635344709144192165695130850,    .5727069351230425055928411633109619686801,
    .55961578793542265941596269840374588966350,    .5714285714285714285714285714285714285714,
    .56184544326269181269140062795486301183700,    .5701559020044543429844097995545657015590,
    .56407013828480290218436721261241473257550,    .5688888888888888888888888888888888888889,
    .56628989502311577464155334382667206227800,    .5676274944567627494456762749445676274945,
    .56850473535266865532378233183408156037350,    .5663716814159292035398230088495575221239,
    .57071468100347144680739575051120482385150,    .5651214128035320088300220750551876379691,
    .57291975356178548306473885531886480748650,    .5638766519823788546255506607929515418502,
    .57511997447138785144460371157038025558000,    .5626373626373626373626373626373626373626,
    .57731536503482350219940144597785547375700,    .5614035087719298245614035087719298245614,
    .57950594641464214795689713355386629700650,    .5601750547045951859956236323851203501094,
    .58169173963462239562716149521293118596100,    .5589519650655021834061135371179039301310,
    .58387276558098266665552955601015128195300,    .5577342047930283224400871459694989106754,
    .58604904500357812846544902640744112432000,    .5565217391304347826086956521739130434783,
    .58822059851708596855957011939608491957200,    .5553145336225596529284164859002169197397,
    .59038744660217634674381770309992134571100,    .5541125541125541125541125541125541125541,
    .59254960960667157898740242671919986605650,    .5529157667386609071274298056155507559395,
    .59470710774669277576265358220553025603300,    .5517241379310344827586206896551724137931,
    .59685996110779382384237123915227130055450,    .5505376344086021505376344086021505376344,
    .59900818964608337768851242799428291618800,    .5493562231759656652360515021459227467811,
    .60115181318933474940990890900138765573500,    .5481798715203426124197002141327623126338,
    .60329085143808425240052883964381180703650,    .5470085470085470085470085470085470085470,
    .60542532396671688843525771517306566238400,    .5458422174840085287846481876332622601279,
    .60755525022454170969155029524699784815300,    .5446808510638297872340425531914893617021,
    .60968064953685519036241657886421307921400,    .5435244161358811040339702760084925690021,
    .61180154110599282990534675263916142284850,    .5423728813559322033898305084745762711864,
    .61391794401237043121710712512140162289150,    .5412262156448202959830866807610993657505,
    .61602987721551394351138242200249806046500,    .5400843881856540084388185654008438818565,
    .61813735955507864705538167982012964785100,    .5389473684210526315789473684210526315789,
    .62024040975185745772080281312810257077200,    .5378151260504201680672268907563025210084,
    .62233904640877868441606324267922900617100,    .5366876310272536687631027253668763102725,
    .62443328801189346144440150965237990021700,    .5355648535564853556485355648535564853556,
    .62652315293135274476554741340805776417250,    .5344467640918580375782881002087682672234,
    .62860865942237409420556559780379757285100,    .5333333333333333333333333333333333333333,
    .63068982562619868570408243613201193511500,    .5322245322245322245322245322245322245322,
    .63276666957103777644277897707070223987100,    .5311203319502074688796680497925311203320,
    .63483920917301017716738442686619237065300,    .5300207039337474120082815734989648033126,
    .63690746223706917739093569252872839570050,    .5289256198347107438016528925619834710744,
    .63897144645792069983514238629140891134750,    .5278350515463917525773195876288659793814,
    .64103117942093124081992527862894348800200,    .5267489711934156378600823045267489711934,
    .64308667860302726193566513757104985415950,    .5256673511293634496919917864476386036961,
    .64513796137358470073053240412264131009600,    .5245901639344262295081967213114754098361,
    .64718504499530948859131740391603671014300,    .5235173824130879345603271983640081799591,
    .64922794662510974195157587018911726772800,    .5224489795918367346938775510204081632653,
    .65126668331495807251485530287027359008800,    .5213849287169042769857433808553971486762,
    .65330127201274557080523663898929953575150,    .5203252032520325203252032520325203252033,
    .65533172956312757406749369692988693714150,    .5192697768762677484787018255578093306288,
    .65735807270835999727154330685152672231200,    .5182186234817813765182186234817813765182,
    .65938031808912778153342060249997302889800,    .5171717171717171717171717171717171717172,
    .66139848224536490484126716182800009846700,    .5161290322580645161290322580645161290323,
    .66341258161706617713093692145776003599150,    .5150905432595573440643863179074446680080,
    .66542263254509037562201001492212526500250,    .5140562248995983935742971887550200803213,
    .66742865127195616370414654738851822912700,    .5130260521042084168336673346693386773547,
    .66943065394262923906154583164607174694550,    .5120000000000000000000000000000000000000,
    .67142865660530226534774556057527661323550,    .5109780439121756487025948103792415169661,
    .67342267521216669923234121597488410770900,    .5099601593625498007968127490039840637450,
    .67541272562017662384192817626171745359900,    .5089463220675944333996023856858846918489,
    .67739882359180603188519853574689477682100,    .5079365079365079365079365079365079365079,
    .67938098479579733801614338517538271844400,    .5069306930693069306930693069306930693069,
    .68135922480790300781450241629499942064300,    .5059288537549407114624505928853754940711,
    .68333355911162063645036823800182901322850,    .5049309664694280078895463510848126232742,
    .68530400309891936760919861626462079584600,    .5039370078740157480314960629921259842520,
    .68727057207096020619019327568821609020250,    .5029469548133595284872298624754420432220,
    .68923328123880889251040571252815425395950,    .5019607843137254901960784313725490196078,
    .69314718055994530941723212145818, 5.0e-01,
};

const double* getLogTab64f()
{
    return logTab;
}

const float* getLogTab32f()
{
    static float CV_DECL_ALIGNED(64) logTab_f[(LOGTAB_MASK+1)*2];
    static std::atomic<bool> logTab_f_initialized(false);
    if (!logTab_f_initialized.load())
    {
        for (int j = 0; j < (LOGTAB_MASK+1)*2; j++)
            logTab_f[j] = (float)logTab[j];
        logTab_f_initialized = true;
    }
    return logTab_f;
}

}} // namespace

/* End of file. */
