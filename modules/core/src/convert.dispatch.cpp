// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "opencl_kernels_core.hpp"

#include "convert.simd.hpp"
#include "convert.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

/* [TODO] Recover IPP calls
#if defined(HAVE_IPP)
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_RUN(src && dst, CV_INSTRUMENT_FUN_IPP(ippiConvert_##ippFavor, src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height)) >= 0) \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CVT_FUNC_F2(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    CV_IPP_RUN(src && dst, CV_INSTRUMENT_FUN_IPP(ippiConvert_##ippFavor, src, (int)sstep, dst, (int)dstep, ippiSize(size.width, size.height), ippRndFinancial, 0) >= 0) \
    cvt_(src, sstep, dst, dstep, size); \
}
#else
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}
#define DEF_CVT_FUNC_F2 DEF_CVT_FUNC_F
#endif

#define DEF_CVT_FUNC(suffix, stype, dtype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         dtype* dst, size_t dstep, Size size, double*) \
{ \
    cvt_(src, sstep, dst, dstep, size); \
}

#define DEF_CPY_FUNC(suffix, stype) \
static void cvt##suffix( const stype* src, size_t sstep, const uchar*, size_t, \
                         stype* dst, size_t dstep, Size size, double*) \
{ \
    cpy_(src, sstep, dst, dstep, size); \
}

DEF_CPY_FUNC(8u,     uchar)
DEF_CVT_FUNC_F(8s8u,   schar, uchar, 8s8u_C1Rs)
DEF_CVT_FUNC_F(16u8u,  ushort, uchar, 16u8u_C1R)
DEF_CVT_FUNC_F(16s8u,  short, uchar, 16s8u_C1R)
DEF_CVT_FUNC_F(32s8u,  int, uchar, 32s8u_C1R)
DEF_CVT_FUNC_F2(32f8u,  float, uchar, 32f8u_C1RSfs)
DEF_CVT_FUNC(64f8u,  double, uchar)

DEF_CVT_FUNC_F2(8u8s,   uchar, schar, 8u8s_C1RSfs)
DEF_CVT_FUNC_F2(16u8s,  ushort, schar, 16u8s_C1RSfs)
DEF_CVT_FUNC_F2(16s8s,  short, schar, 16s8s_C1RSfs)
DEF_CVT_FUNC_F(32s8s,  int, schar, 32s8s_C1R)
DEF_CVT_FUNC_F2(32f8s,  float, schar, 32f8s_C1RSfs)
DEF_CVT_FUNC(64f8s,  double, schar)

DEF_CVT_FUNC_F(8u16u,  uchar, ushort, 8u16u_C1R)
DEF_CVT_FUNC_F(8s16u,  schar, ushort, 8s16u_C1Rs)
DEF_CPY_FUNC(16u,    ushort)
DEF_CVT_FUNC_F(16s16u, short, ushort, 16s16u_C1Rs)
DEF_CVT_FUNC_F2(32s16u, int, ushort, 32s16u_C1RSfs)
DEF_CVT_FUNC_F2(32f16u, float, ushort, 32f16u_C1RSfs)
DEF_CVT_FUNC(64f16u, double, ushort)

DEF_CVT_FUNC_F(8u16s,  uchar, short, 8u16s_C1R)
DEF_CVT_FUNC_F(8s16s,  schar, short, 8s16s_C1R)
DEF_CVT_FUNC_F2(16u16s, ushort, short, 16u16s_C1RSfs)
DEF_CVT_FUNC_F2(32s16s, int, short, 32s16s_C1RSfs)
DEF_CVT_FUNC(32f16s, float, short)
DEF_CVT_FUNC(64f16s, double, short)

DEF_CVT_FUNC_F(8u32s,  uchar, int, 8u32s_C1R)
DEF_CVT_FUNC_F(8s32s,  schar, int, 8s32s_C1R)
DEF_CVT_FUNC_F(16u32s, ushort, int, 16u32s_C1R)
DEF_CVT_FUNC_F(16s32s, short, int, 16s32s_C1R)
DEF_CPY_FUNC(32s,    int)
DEF_CVT_FUNC_F2(32f32s, float, int, 32f32s_C1RSfs)
DEF_CVT_FUNC(64f32s, double, int)

DEF_CVT_FUNC_F(8u32f,  uchar, float, 8u32f_C1R)
DEF_CVT_FUNC_F(8s32f,  schar, float, 8s32f_C1R)
DEF_CVT_FUNC_F(16u32f, ushort, float, 16u32f_C1R)
DEF_CVT_FUNC_F(16s32f, short, float, 16s32f_C1R)
DEF_CVT_FUNC_F(32s32f, int, float, 32s32f_C1R)
DEF_CVT_FUNC(64f32f, double, float)

DEF_CVT_FUNC(8u64f,  uchar, double)
DEF_CVT_FUNC(8s64f,  schar, double)
DEF_CVT_FUNC(16u64f, ushort, double)
DEF_CVT_FUNC(16s64f, short, double)
DEF_CVT_FUNC(32s64f, int, double)
DEF_CVT_FUNC(32f64f, float, double)
DEF_CPY_FUNC(64s,    int64)
*/

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getConvertFunc, (sdepth, ddepth),
        CV_CPU_DISPATCH_MODES_ALL);
}

static BinaryFunc get_cvt32f16f()
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(get_cvt32f16f, (),
        CV_CPU_DISPATCH_MODES_ALL);
}
static BinaryFunc get_cvt16f32f()
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(get_cvt16f32f, (),
        CV_CPU_DISPATCH_MODES_ALL);
}

#ifdef HAVE_OPENCL
static bool ocl_convertFp16( InputArray _src, OutputArray _dst, int sdepth, int ddepth )
{
    int type = _src.type(), cn = CV_MAT_CN(type);

    _dst.createSameSize( _src, CV_MAKETYPE(ddepth, cn) );
    int kercn = 1;
    int rowsPerWI = 1;
    String build_opt = format("-D HALF_SUPPORT -D srcT=%s -D dstT=%s -D rowsPerWI=%d%s",
                              sdepth == CV_32F ? "float" : "half",
                              sdepth == CV_32F ? "half" : "float",
                              rowsPerWI,
                              sdepth == CV_32F ? " -D FLOAT_TO_HALF " : "");
    ocl::Kernel k(sdepth == CV_32F ? "convertFp16_FP32_to_FP16" : "convertFp16_FP16_to_FP32", ocl::core::halfconvert_oclsrc, build_opt);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    UMat dst = _dst.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
    dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    k.args(srcarg, dstarg);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}
#endif

void Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    CV_INSTRUMENT_REGION();

    if( empty() )
    {
        _dst.release();
        return;
    }

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if( _type < 0 )
        _type = _dst.fixedType() ? _dst.type() : type();
    else
        _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = CV_MAT_DEPTH(_type);
    if( sdepth == ddepth && noScale )
    {
        copyTo(_dst);
        return;
    }

    Mat src = *this;
    if( dims <= 2 )
        _dst.create( size(), _type );
    else
        _dst.create( dims, size, _type );
    Mat dst = _dst.getMat();

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    CV_Assert( func != 0 );

    if( dims <= 2 )
    {
        Size sz = getContinuousSize2D(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, scale );
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

//==================================================================================================

void convertFp16(InputArray _src, OutputArray _dst)
{
    CV_INSTRUMENT_REGION();

    int sdepth = _src.depth(), ddepth = 0;
    BinaryFunc func = 0;

    switch( sdepth )
    {
    case CV_32F:
        if(_dst.fixedType())
        {
            ddepth = _dst.depth();
            CV_Assert(ddepth == CV_16S /*|| ddepth == CV_16F*/);
            CV_Assert(_dst.channels() == _src.channels());
        }
        else
            ddepth =  CV_16S;
        func = get_cvt32f16f();
        break;
    case CV_16S:
    //case CV_16F:
        ddepth = CV_32F;
        func = get_cvt16f32f();
        break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "Unsupported input depth");
        return;
    }

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_convertFp16(_src, _dst, sdepth, ddepth))

    Mat src = _src.getMat();

    int type = CV_MAKETYPE(ddepth, src.channels());
    _dst.create( src.dims, src.size, type );
    Mat dst = _dst.getMat();
    int cn = src.channels();

    CV_Assert( func != 0 );

    if( src.dims <= 2 )
    {
        Size sz = getContinuousSize2D(src, dst, cn);
        func( src.data, src.step, 0, 0, dst.data, dst.step, sz, 0);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size*cn), 1);

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], 0, 0, 0, ptrs[1], 0, sz, 0);
    }
}

} // namespace cv
