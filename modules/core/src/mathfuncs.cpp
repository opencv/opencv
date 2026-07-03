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
#include <algorithm>
#include <cmath>
#include "mathfuncs.hpp"
#include "arithm_expr.hpp"      // the element-wise engine: getMathFunc + TExpr for cv::exp/log/sqrt

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
    _dst.createSameSize(src1, type);
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
CV_DISABLE_UBSAN float cubeRoot( float value )
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
    dst.create(X.size, X.type());
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
    dst.create( X.size, type );
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
    dst1.create( X.size, type );
    dst2.create( X.size, type );
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
    dst1.create( Angle.size, type );
    dst2.create( Angle.size, type );
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

// The master function of the unary math family (cv::exp/log/sqrt - the analogue of arithm_op):
// same-shape same-type output over the four float depths, computed by the element-wise engine's
// kernels (math.simd.hpp). Two tiers:
//  - SMALL and continuous (a common pattern - exp() over one image row as a lookup substitute):
//    call the kernel DIRECTLY over the flattened elements. No TExpr, no broadcastOp, no
//    parallel_for machinery - their setup dominates at these sizes.
//  - everything else (large arrays - worth parallelizing; ROIs - need real steps): the usual
//    1-instruction program via compile()/exec().
enum { MATH_OP_SMALL = 100000 };     // elements; tune with a benchmark if the crossover moves

static void math_op(ew::TOp op, InputArray _src, OutputArray _dst)
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    CV_Assert(depth == CV_16F || depth == CV_16BF || depth == CV_32F || depth == CV_64F);

    Mat src = _src.getMat();
    _dst.create(src.dims, src.size.p, type);
    Mat dst = _dst.getMat();
    if (src.empty())
        return;

    const size_t total = src.total() * src.channels();
    if (src.isContinuous() && dst.isContinuous() && total <= (size_t)MATH_OP_SMALL)
    {
        ew::TKernel k = ew::getMathFunc(op, depth);
        CV_Assert(k.fptr);
        static const double noparams[4] = {};
        k.fptr(src.data, 0, 1, nullptr, 0, 0, nullptr, 0, 0,
               dst.data, 0, (int)total, 1, noparams, k.flags, k.userdata);
        return;
    }

    ew::TExpr p;
    const int a = p.addInput(depth);
    const int out = p.addOutput(depth);
    p.moveToOutput(p.emitUnary(op, a, depth), out);
    p.compile();
    const Mat* inputs[] = { &src };
    p.exec(inputs, &dst);
}

void exp( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int depth = _src.depth();
    CV_UNUSED(depth);   // consumed by CV_OCL_RUN only - a no-op without OpenCL

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && (depth == CV_32F || depth == CV_64F),
               ocl_math_op(_src, noArray(), _dst, OCL_OP_EXP))

    math_op(ew::OP_EXP, _src, _dst);
}


/****************************************************************************************\
*                                          L O G                                         *
\****************************************************************************************/

void log( InputArray _src, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int depth = _src.depth();
    CV_UNUSED(depth);   // consumed by CV_OCL_RUN only - a no-op without OpenCL

    CV_OCL_RUN( _dst.isUMat() && _src.dims() <= 2 && (depth == CV_32F || depth == CV_64F),
                ocl_math_op(_src, noArray(), _dst, OCL_OP_LOG))

    math_op(ew::OP_LOG, _src, _dst);
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
    _dst.createSameSize(src, type);
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
    _dst.create( src.size, type );
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

    if (b.isUMat() && a.dims() <= 2)     // the OpenCL route (via ocl_pow) is unchanged
    {
        cv::pow(a, 0.5, b);
        return;
    }
    math_op(ew::OP_SQRT, a, b);
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

} // namespace cv

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

    // Fix for Issue #27748: Normalize coefficients to avoid overflow/underflow
    // and correctly detect negligible cubic terms.
    double max_coeff = std::max({std::abs(a0), std::abs(a1), std::abs(a2), std::abs(a3)});

    // If max_coeff is effectively zero, the equation is 0=0
    if (max_coeff < std::numeric_limits<double>::epsilon())
        return -1;

    // Check if the cubic term is negligible relative to the largest coefficient
    if( std::abs(a0) < std::numeric_limits<double>::epsilon() * max_coeff )
    {
        if( std::abs(a1) < std::numeric_limits<double>::epsilon() * max_coeff )
        {
            if( std::abs(a2) < std::numeric_limits<double>::epsilon() * max_coeff )
                n = std::abs(a3) < std::numeric_limits<double>::epsilon() * max_coeff ? -1 : 0;
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
                if( std::abs(q1) > std::abs(q2) )
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
        // cubic equation
        // Normalize coefficients in-place to prevent overflow/instability
        double scale = 1.0 / max_coeff;
        a0 *= scale;
        a1 *= scale;
        a2 *= scale;
        a3 *= scale;

        a0 = 1./a0;
        a1 *= a0;
        a2 *= a0;
        a3 *= a0;

        double Q = (a1 * a1 - 3 * a2) * (1./9);
        double R = (a1 * (2 * a1 * a1 - 9 * a2) + 27 * a3) * (1./54);
        double Qcubed = Q * Q * Q;

        /*
          Here we expand expression `Qcubed - R * R` for `d` variable
          to reduce common terms `a1^6 / 729` and `-a1^4 * a2 / 81`
          and thus decrease rounding error (in case of quite big coefficients).

          And then we additionally group terms to further reduce rounding error.
        */
        double d = (a1 * a1 * (a2 * a2 - 4 * a1 * a3) + 2 * a2 * (9 * a1 * a3 - 2 * a2 * a2) - 27 * a3 * a3) * (1./108);

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
            x0 = -2*std::cbrt(R) - a1/3;
            x1 = std::cbrt(R) - a1/3;
            x2 = 0;
            n = x0 == x1 ? 1 : 2;
            x1 = x0 == x1 ? 0 : x1;
        }
        else
        {
            double e;
            d = sqrt(-d);
            e = std::cbrt(d + fabs(R));
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

    // Related issue: https://github.com/opencv/opencv/issues/23644,
    // This the initialization scheme of "Initial approximations in Durand-Kerner's root finding method" by Guggenheimer.
    // https://link.springer.com/article/10.1007/BF01935059
    // We put the initial points equidistantly on a circle on the complex plane. This code computes the circle radius as in the paper.
    Mat absCoeffs(n + 1, 1, CV_64F);
    for( i = 0; i <= n; i++ )
        absCoeffs.at<double>(i) = abs(coeffs[i]);

    int nonZeroCoeffs = 0;
    Mat u(n, 1, CV_64F, Scalar(0)), v(n, 1, CV_64F, Scalar(0));
    for( i = 0; i <= n; i++ )
    {
        double coeff = absCoeffs.at<double>(i);
        if( coeff > DBL_EPSILON )
        {
            if( i != n )
                u.at<double>(i) = 2.0 * pow(coeff / absCoeffs.at<double>(n), 1.0 / (n - i));
            if( i != 0 )
                v.at<double>(i - 1) = 0.5 * pow(absCoeffs.at<double>(0) / coeff, 1.0 / i);
            nonZeroCoeffs++;
        }
    }
    double scale = 1;
    if( nonZeroCoeffs > 2 )
    {
        Point maxU, minV;
        minMaxLoc(u, nullptr, nullptr, nullptr, &maxU);
        minMaxLoc(v, nullptr, nullptr, &minV);
        u.at<double>(maxU) = 0;
        v.at<double>(minV) = 0;
        scale = (sum(u).val[0] + sum(v).val[0]) / (2 * nonZeroCoeffs - 2);
    }

    C p(scale, 0), r(cos(CV_2PI / n), sin(CV_2PI / n));

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
                    cube_coefs.at<double>(3) = -(std::pow(old_num_re, 3));
                    cube_coefs.at<double>(2) = -(15*std::pow(old_num_re, 2) + 27*std::pow(old_num_im, 2));
                    cube_coefs.at<double>(1) = -48*old_num_re;
                    cube_coefs.at<double>(0) = 64;
                    solveCubic(cube_coefs, cube_roots);

                    num.re = std::cbrt(cube_roots.at<double>(0));
                    num.im = sqrt(std::pow(num.re, 2) / 3 - old_num_re / (3*num.re));
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


/* End of file. */
