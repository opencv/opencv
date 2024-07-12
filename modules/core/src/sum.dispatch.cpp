// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "stat.hpp"

#include "sum.simd.hpp"
#include "sum.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

#ifndef OPENCV_IPP_SUM
#undef HAVE_IPP
#undef CV_IPP_RUN_FAST
#define CV_IPP_RUN_FAST(f, ...)
#undef CV_IPP_RUN
#define CV_IPP_RUN(c, f, ...)
#endif // OPENCV_IPP_SUM

namespace cv
{

SumFunc getSumFunc(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getSumFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

#ifdef HAVE_OPENCL

bool ocl_sum( InputArray _src, Scalar & res, int sum_op, InputArray _mask,
                     InputArray _src2, bool calc2, const Scalar & res2 )
{
    CV_Assert(sum_op == OCL_OP_SUM || sum_op == OCL_OP_SUM_ABS || sum_op == OCL_OP_SUM_SQR);

    const ocl::Device & dev = ocl::Device::getDefault();
    bool doubleSupport = dev.doubleFPConfig() > 0,
        haveMask = _mask.kind() != _InputArray::NONE,
        haveSrc2 = _src2.kind() != _InputArray::NONE;
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            kercn = cn == 1 && !haveMask ? ocl::predictOptimalVectorWidth(_src, _src2) : 1,
            mcn = std::max(cn, kercn);
    CV_Assert(!haveSrc2 || _src2.type() == type);
    int convert_cn = haveSrc2 ? mcn : cn;

    if ( (!doubleSupport && depth == CV_64F) || cn > 4 )
        return false;

    if (depth >= CV_16F)
        return false;

    int ngroups = dev.maxComputeUnits(), dbsize = ngroups * (calc2 ? 2 : 1);
    size_t wgs = dev.maxWorkGroupSize();

    int ddepth = std::max(sum_op == OCL_OP_SUM_SQR ? CV_32F : CV_32S, depth),
            dtype = CV_MAKE_TYPE(ddepth, cn);
    CV_Assert(!haveMask || _mask.type() == CV_8UC1 || _mask.type() == CV_Bool);

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    static const char * const opMap[3] = { "OP_SUM", "OP_SUM_ABS", "OP_SUM_SQR" };
    char cvt[2][50];
    String opts = format("-D srcT=%s -D srcT1=%s -D dstT=%s -D dstTK=%s -D dstT1=%s -D ddepth=%d -D cn=%d"
                         " -D convertToDT=%s -D %s -D WGS=%d -D WGS2_ALIGNED=%d%s%s%s%s -D kercn=%d%s%s%s -D convertFromU=%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, mcn)), ocl::typeToStr(depth),
                         ocl::typeToStr(dtype), ocl::typeToStr(CV_MAKE_TYPE(ddepth, mcn)),
                         ocl::typeToStr(ddepth), ddepth, cn,
                         ocl::convertTypeStr(depth, ddepth, mcn, cvt[0], sizeof(cvt[0])),
                         opMap[sum_op], (int)wgs, wgs2_aligned,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         haveMask ? " -D HAVE_MASK" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : "",
                         haveMask && _mask.isContinuous() ? " -D HAVE_MASK_CONT" : "", kercn,
                         haveSrc2 ? " -D HAVE_SRC2" : "", calc2 ? " -D OP_CALC2" : "",
                         haveSrc2 && _src2.isContinuous() ? " -D HAVE_SRC2_CONT" : "",
                         depth <= CV_32S && ddepth == CV_32S ? ocl::convertTypeStr(CV_8U, ddepth, convert_cn, cvt[1], sizeof(cvt[1])) : "noconvert");

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), src2 = _src2.getUMat(),
        db(1, dbsize, dtype), mask = _mask.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dbarg = ocl::KernelArg::PtrWriteOnly(db),
            maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2);

    if (haveMask)
    {
        if (haveSrc2)
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, maskarg, src2arg);
        else
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, maskarg);
    }
    else
    {
        if (haveSrc2)
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, src2arg);
        else
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg);
    }

    size_t globalsize = ngroups * wgs;
    if (k.run(1, &globalsize, &wgs, true))
    {
        typedef Scalar (*part_sum)(Mat m);
        part_sum funcs[3] = { ocl_part_sum<int>, ocl_part_sum<float>, ocl_part_sum<double> },
                func = funcs[ddepth - CV_32S];

        Mat mres = db.getMat(ACCESS_READ);
        if (calc2)
            const_cast<Scalar &>(res2) = func(mres.colRange(ngroups, dbsize));

        res = func(mres.colRange(0, ngroups));
        return true;
    }
    return false;
}

#endif

#ifdef HAVE_IPP
static bool ipp_sum(Mat &src, Scalar &_res)
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 >= 700
    int cn = src.channels();
    if (cn > 4)
        return false;
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims <= 2 || (src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        typedef IppStatus (CV_STDCALL* ippiSumFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
        typedef IppStatus (CV_STDCALL* ippiSumFuncNoHint)(const void*, int, IppiSize, double *);
        ippiSumFuncHint ippiSumHint =
            type == CV_32FC1 ? (ippiSumFuncHint)ippiSum_32f_C1R :
            type == CV_32FC3 ? (ippiSumFuncHint)ippiSum_32f_C3R :
            type == CV_32FC4 ? (ippiSumFuncHint)ippiSum_32f_C4R :
            0;
        ippiSumFuncNoHint ippiSum =
            type == CV_8UC1 ? (ippiSumFuncNoHint)ippiSum_8u_C1R :
            type == CV_8UC3 ? (ippiSumFuncNoHint)ippiSum_8u_C3R :
            type == CV_8UC4 ? (ippiSumFuncNoHint)ippiSum_8u_C4R :
            type == CV_16UC1 ? (ippiSumFuncNoHint)ippiSum_16u_C1R :
            type == CV_16UC3 ? (ippiSumFuncNoHint)ippiSum_16u_C3R :
            type == CV_16UC4 ? (ippiSumFuncNoHint)ippiSum_16u_C4R :
            type == CV_16SC1 ? (ippiSumFuncNoHint)ippiSum_16s_C1R :
            type == CV_16SC3 ? (ippiSumFuncNoHint)ippiSum_16s_C3R :
            type == CV_16SC4 ? (ippiSumFuncNoHint)ippiSum_16s_C4R :
            0;
        CV_Assert(!ippiSumHint || !ippiSum);
        if( ippiSumHint || ippiSum )
        {
            Ipp64f res[4];
            IppStatus ret = ippiSumHint ?
                            CV_INSTRUMENT_FUN_IPP(ippiSumHint, src.ptr(), (int)src.step[0], sz, res, ippAlgHintAccurate) :
                            CV_INSTRUMENT_FUN_IPP(ippiSum, src.ptr(), (int)src.step[0], sz, res);
            if( ret >= 0 )
            {
                for( int i = 0; i < cn; i++ )
                    _res[i] = res[i];
                return true;
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(_res);
#endif
    return false;
}
#endif

Scalar sum(InputArray _src)
{
    CV_INSTRUMENT_REGION();

#if defined HAVE_OPENCL || defined HAVE_IPP
    Scalar _res;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_sum(_src, _res, OCL_OP_SUM),
                _res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_sum(src, _res), _res);

    int k, cn = src.channels(), depth = src.depth();
    SumFunc func = getSumFunc(depth);
    if (func == nullptr) {
        if (depth == CV_Bool && cn == 1)
            return Scalar((double)countNonZero(src));
        CV_Error(Error::StsNotImplemented, "");
    }
    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1] = {};
    NAryMatIterator it(arrays, ptrs);
    Scalar s;
    int total = (int)it.size, blockSize = total, partialBlockSize = 0;
    int j, count = 0;
    int _buf[CV_CN_MAX];
    int* buf = (int*)&s[0];
    size_t esz = 0;
    bool partialSumIsInt = depth < CV_32S;
    bool blockSum = partialSumIsInt || depth == CV_16F || depth == CV_16BF;

    if( blockSum )
    {
        partialBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, partialBlockSize);
        buf = _buf;
        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], 0, (uchar*)buf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= partialBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                if (partialSumIsInt) {
                    for( k = 0; k < cn; k++ )
                    {
                        s[k] += buf[k];
                        buf[k] = 0;
                    }
                } else {
                    for( k = 0; k < cn; k++ )
                    {
                        s[k] += ((float*)buf)[k];
                        buf[k] = 0;
                    }
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
        }
    }
    return s;
}

} // namespace
