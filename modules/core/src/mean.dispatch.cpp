// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "opencv2/core/openvx/ovx_defs.hpp"
#include "stat.hpp"

#ifndef OPENCV_IPP_MEAN
#undef HAVE_IPP
#undef CV_IPP_RUN_FAST
#define CV_IPP_RUN_FAST(f, ...)
#undef CV_IPP_RUN
#define CV_IPP_RUN(c, f, ...)
#endif // OPENCV_IPP_MEAN

#include "mean.simd.hpp"
#include "mean.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

#ifndef OPENCV_IPP_MEAN
#undef HAVE_IPP
#undef CV_IPP_RUN_FAST
#define CV_IPP_RUN_FAST(f, ...)
#undef CV_IPP_RUN
#define CV_IPP_RUN(c, f, ...)
#endif // OPENCV_IPP_MEAN

namespace cv {

#if defined HAVE_IPP
static bool ipp_mean( Mat &src, Mat &mask, Scalar &ret )
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 >= 700
    size_t total_size = src.total();
    int cn = src.channels();
    if (cn > 4)
        return false;
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskMeanFuncC1 ippiMean_C1MR =
            type == CV_8UC1 ? (ippiMaskMeanFuncC1)ippiMean_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMeanFuncC1)ippiMean_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMeanFuncC1)ippiMean_32f_C1MR :
            0;
            if( ippiMean_C1MR )
            {
                Ipp64f res;
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_C1MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, &res) >= 0 )
                {
                    ret = Scalar(res);
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskMeanFuncC3 ippiMean_C3MR =
            type == CV_8UC3 ? (ippiMaskMeanFuncC3)ippiMean_8u_C3CMR :
            type == CV_16UC3 ? (ippiMaskMeanFuncC3)ippiMean_16u_C3CMR :
            type == CV_32FC3 ? (ippiMaskMeanFuncC3)ippiMean_32f_C3CMR :
            0;
            if( ippiMean_C3MR )
            {
                Ipp64f res1, res2, res3;
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 1, &res1) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 2, &res2) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 3, &res3) >= 0 )
                {
                    ret = Scalar(res1, res2, res3);
                    return true;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
            typedef IppStatus (CV_STDCALL* ippiMeanFuncNoHint)(const void*, int, IppiSize, double *);
            ippiMeanFuncHint ippiMeanHint =
                type == CV_32FC1 ? (ippiMeanFuncHint)ippiMean_32f_C1R :
                type == CV_32FC3 ? (ippiMeanFuncHint)ippiMean_32f_C3R :
                type == CV_32FC4 ? (ippiMeanFuncHint)ippiMean_32f_C4R :
                0;
            ippiMeanFuncNoHint ippiMean =
                type == CV_8UC1 ? (ippiMeanFuncNoHint)ippiMean_8u_C1R :
                type == CV_8UC3 ? (ippiMeanFuncNoHint)ippiMean_8u_C3R :
                type == CV_8UC4 ? (ippiMeanFuncNoHint)ippiMean_8u_C4R :
                type == CV_16UC1 ? (ippiMeanFuncNoHint)ippiMean_16u_C1R :
                type == CV_16UC3 ? (ippiMeanFuncNoHint)ippiMean_16u_C3R :
                type == CV_16UC4 ? (ippiMeanFuncNoHint)ippiMean_16u_C4R :
                type == CV_16SC1 ? (ippiMeanFuncNoHint)ippiMean_16s_C1R :
                type == CV_16SC3 ? (ippiMeanFuncNoHint)ippiMean_16s_C3R :
                type == CV_16SC4 ? (ippiMeanFuncNoHint)ippiMean_16s_C4R :
                0;
            // Make sure only zero or one version of the function pointer is valid
            CV_Assert(!ippiMeanHint || !ippiMean);
            if( ippiMeanHint || ippiMean )
            {
                Ipp64f res[4];
                IppStatus status = ippiMeanHint ? CV_INSTRUMENT_FUN_IPP(ippiMeanHint, src.ptr(), (int)src.step[0], sz, res, ippAlgHintAccurate) :
                                CV_INSTRUMENT_FUN_IPP(ippiMean, src.ptr(), (int)src.step[0], sz, res);
                if( status >= 0 )
                {
                    for( int i = 0; i < cn; i++ )
                        ret[i] = res[i];
                    return true;
                }
            }
        }
    }
    return false;
#else
    return false;
#endif
}
#endif

Scalar mean(InputArray _src, InputArray _mask)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_Assert( mask.empty() || mask.type() == CV_8U );

    int k, cn = src.channels(), depth = src.depth();
    Scalar s;

    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_mean(src, mask, s), s)

    SumFunc func = getSumFunc(depth);

    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0;
    AutoBuffer<int> _buf;
    int* buf = (int*)&s[0];
    bool blockSum = depth <= CV_16S;
    size_t esz = 0, nz0 = 0;

    if( blockSum )
    {
        intSumBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        _buf.allocate(cn);
        buf = _buf.data();

        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            int nz = func( ptrs[0], ptrs[1], (uchar*)buf, bsz, cn );
            count += nz;
            nz0 += nz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += buf[k];
                    buf[k] = 0;
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }
    return s*(nz0 ? 1./nz0 : 0);
}

static SumSqrFunc getSumSqrFunc(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getSumSqrFunc, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

#ifdef HAVE_OPENCL
static bool ocl_meanStdDev( InputArray _src, OutputArray _mean, OutputArray _sdv, InputArray _mask )
{
    CV_INSTRUMENT_REGION_OPENCL();

    bool haveMask = _mask.kind() != _InputArray::NONE;
    int nz = haveMask ? -1 : (int)_src.total();
    Scalar mean(0), stddev(0);
    const int cn = _src.channels();
    if (cn > 4)
        return false;

    {
        int type = _src.type(), depth = CV_MAT_DEPTH(type);
        bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0,
                isContinuous = _src.isContinuous(),
                isMaskContinuous = _mask.isContinuous();
        const ocl::Device &defDev = ocl::Device::getDefault();
        int groups = defDev.maxComputeUnits();
        if (defDev.isIntel())
        {
            static const int subSliceEUCount = 10;
            groups = (groups / subSliceEUCount) * 2;
        }
        size_t wgs = defDev.maxWorkGroupSize();

        int ddepth = std::max(CV_32S, depth), sqddepth = std::max(CV_32F, depth),
                dtype = CV_MAKE_TYPE(ddepth, cn),
                sqdtype = CV_MAKETYPE(sqddepth, cn);
        CV_Assert(!haveMask || _mask.type() == CV_8UC1);

        int wgs2_aligned = 1;
        while (wgs2_aligned < (int)wgs)
            wgs2_aligned <<= 1;
        wgs2_aligned >>= 1;

        if ( (!doubleSupport && depth == CV_64F) )
            return false;

        char cvt[2][50];
        String opts = format("-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D sqddepth=%d"
                             " -D sqdstT=%s -D sqdstT1=%s -D convertToSDT=%s -D cn=%d%s%s"
                             " -D convertToDT=%s -D WGS=%d -D WGS2_ALIGNED=%d%s%s",
                             ocl::typeToStr(type), ocl::typeToStr(depth),
                             ocl::typeToStr(dtype), ocl::typeToStr(ddepth), sqddepth,
                             ocl::typeToStr(sqdtype), ocl::typeToStr(sqddepth),
                             ocl::convertTypeStr(depth, sqddepth, cn, cvt[0], sizeof(cvt[0])),
                             cn, isContinuous ? " -D HAVE_SRC_CONT" : "",
                             isMaskContinuous ? " -D HAVE_MASK_CONT" : "",
                             ocl::convertTypeStr(depth, ddepth, cn, cvt[1], sizeof(cvt[1])),
                             (int)wgs, wgs2_aligned, haveMask ? " -D HAVE_MASK" : "",
                             doubleSupport ? " -D DOUBLE_SUPPORT" : "");

        ocl::Kernel k("meanStdDev", ocl::core::meanstddev_oclsrc, opts);
        if (k.empty())
            return false;

        int dbsize = groups * ((haveMask ? CV_ELEM_SIZE1(CV_32S) : 0) +
                               CV_ELEM_SIZE(sqdtype) + CV_ELEM_SIZE(dtype));
        UMat src = _src.getUMat(), db(1, dbsize, CV_8UC1), mask = _mask.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
                dbarg = ocl::KernelArg::PtrWriteOnly(db),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask);

        if (haveMask)
            k.args(srcarg, src.cols, (int)src.total(), groups, dbarg, maskarg);
        else
            k.args(srcarg, src.cols, (int)src.total(), groups, dbarg);

        size_t globalsize = groups * wgs;

        if(!k.run(1, &globalsize, &wgs, false))
            return false;

        typedef Scalar (* part_sum)(Mat m);
        part_sum funcs[3] = { ocl_part_sum<int>, ocl_part_sum<float>, ocl_part_sum<double> };
        Mat dbm = db.getMat(ACCESS_READ);

        mean = funcs[ddepth - CV_32S](Mat(1, groups, dtype, dbm.ptr()));
        stddev = funcs[sqddepth - CV_32S](Mat(1, groups, sqdtype, dbm.ptr() + groups * CV_ELEM_SIZE(dtype)));

        if (haveMask)
            nz = saturate_cast<int>(funcs[0](Mat(1, groups, CV_32SC1, dbm.ptr() +
                                                 groups * (CV_ELEM_SIZE(dtype) +
                                                           CV_ELEM_SIZE(sqdtype))))[0]);
    }

    double total = nz != 0 ? 1.0 / nz : 0;
    int k, j;
    for (int i = 0; i < cn; ++i)
    {
        mean[i] *= total;
        stddev[i] = std::sqrt(std::max(stddev[i] * total - mean[i] * mean[i] , 0.));
    }

    for( j = 0; j < 2; j++ )
    {
        const double * const sptr = j == 0 ? &mean[0] : &stddev[0];
        _OutputArray _dst = j == 0 ? _mean : _sdv;
        if( !_dst.needed() )
            continue;

        if( !_dst.fixedSize() )
            _dst.create(cn, 1, CV_64F, -1, true);
        Mat dst = _dst.getMat();
        int dcn = (int)dst.total();
        CV_Assert( dst.type() == CV_64F && dst.isContinuous() &&
                   (dst.cols == 1 || dst.rows == 1) && dcn >= cn );
        double* dptr = dst.ptr<double>();
        for( k = 0; k < cn; k++ )
            dptr[k] = sptr[k];
        for( ; k < dcn; k++ )
            dptr[k] = 0;
    }

    return true;
}
#endif

#ifdef HAVE_OPENVX
    static bool openvx_meanStdDev(Mat& src, OutputArray _mean, OutputArray _sdv, Mat& mask)
    {
        size_t total_size = src.total();
        int rows = src.size[0], cols = rows ? (int)(total_size / rows) : 0;
        if (src.type() != CV_8UC1|| !mask.empty() ||
               (src.dims != 2 && !(src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size))
           )
        return false;

        try
        {
            ivx::Context ctx = ovx::getOpenVXContext();
#ifndef VX_VERSION_1_1
            if (ctx.vendorID() == VX_ID_KHRONOS)
                return false; // Do not use OpenVX meanStdDev estimation for sample 1.0.1 implementation due to lack of accuracy
#endif

            ivx::Image
                ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                    ivx::Image::createAddressing(cols, rows, 1, (vx_int32)(src.step[0])), src.ptr());

            vx_float32 mean_temp, stddev_temp;
            ivx::IVX_CHECK_STATUS(vxuMeanStdDev(ctx, ia, &mean_temp, &stddev_temp));

            if (_mean.needed())
            {
                if (!_mean.fixedSize())
                    _mean.create(1, 1, CV_64F, -1, true);
                Mat mean = _mean.getMat();
                CV_Assert(mean.type() == CV_64F && mean.isContinuous() &&
                    (mean.cols == 1 || mean.rows == 1) && mean.total() >= 1);
                double *pmean = mean.ptr<double>();
                pmean[0] = mean_temp;
                for (int c = 1; c < (int)mean.total(); c++)
                    pmean[c] = 0;
            }

            if (_sdv.needed())
            {
                if (!_sdv.fixedSize())
                    _sdv.create(1, 1, CV_64F, -1, true);
                Mat stddev = _sdv.getMat();
                CV_Assert(stddev.type() == CV_64F && stddev.isContinuous() &&
                    (stddev.cols == 1 || stddev.rows == 1) && stddev.total() >= 1);
                double *pstddev = stddev.ptr<double>();
                pstddev[0] = stddev_temp;
                for (int c = 1; c < (int)stddev.total(); c++)
                    pstddev[c] = 0;
            }
        }
        catch (const ivx::RuntimeError & e)
        {
            VX_DbgThrow(e.what());
        }
        catch (const ivx::WrapperError & e)
        {
            VX_DbgThrow(e.what());
        }

        return true;
    }
#endif

#ifdef HAVE_IPP
static bool ipp_meanStdDev(Mat& src, OutputArray _mean, OutputArray _sdv, Mat& mask)
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 >= 700
    int cn = src.channels();

#if IPP_VERSION_X100 < 201801
    // IPP_DISABLE: C3C functions can read outside of allocated memory
    if (cn > 1)
        return false;
#endif
#if IPP_VERSION_X100 >= 201900 && IPP_VERSION_X100 < 201901
    // IPP_DISABLE: 32f C3C functions can read outside of allocated memory
    if (cn > 1 && src.depth() == CV_32F)
        return false;

    // SSE4.2 buffer overrun
#if defined(_WIN32) && !defined(_WIN64)
    // IPPICV doesn't have AVX2 in 32-bit builds
    // However cv::ipp::getIppTopFeatures() may return AVX2 value on AVX2 capable H/W
    // details #12959
#else
    if (cv::ipp::getIppTopFeatures() == ippCPUID_SSE42) // Linux x64 + OPENCV_IPP=SSE42 is affected too
#endif
    {
        if (src.depth() == CV_32F && src.dims > 1 && src.size[src.dims - 1] == 6)
            return false;
    }
#endif

    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        Ipp64f mean_temp[3];
        Ipp64f stddev_temp[3];
        Ipp64f *pmean = &mean_temp[0];
        Ipp64f *pstddev = &stddev_temp[0];
        Mat mean, stddev;
        int dcn_mean = -1;
        if( _mean.needed() )
        {
            if( !_mean.fixedSize() )
                _mean.create(cn, 1, CV_64F, -1, true);
            mean = _mean.getMat();
            dcn_mean = (int)mean.total();
            pmean = mean.ptr<Ipp64f>();
        }
        int dcn_stddev = -1;
        if( _sdv.needed() )
        {
            if( !_sdv.fixedSize() )
                _sdv.create(cn, 1, CV_64F, -1, true);
            stddev = _sdv.getMat();
            dcn_stddev = (int)stddev.total();
            pstddev = stddev.ptr<Ipp64f>();
        }
        for( int c = cn; c < dcn_mean; c++ )
            pmean[c] = 0;
        for( int c = cn; c < dcn_stddev; c++ )
            pstddev[c] = 0;
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC1 ippiMean_StdDev_C1MR =
            type == CV_8UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_32f_C1MR :
            0;
            if( ippiMean_StdDev_C1MR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C1MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, pmean, pstddev) >= 0 )
                {
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC3 ippiMean_StdDev_C3CMR =
            type == CV_8UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CMR :
            type == CV_16UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CMR :
            type == CV_32FC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CMR :
            0;
            if( ippiMean_StdDev_C3CMR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                {
                    return true;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC1)(const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC1 ippiMean_StdDev_C1R =
            type == CV_8UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_8u_C1R :
            type == CV_16UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_16u_C1R :
#if (IPP_VERSION_X100 >= 810)
            type == CV_32FC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_32f_C1R ://Aug 2013: bug in IPP 7.1, 8.0
#endif
            0;
            if( ippiMean_StdDev_C1R )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C1R, src.ptr(), (int)src.step[0], sz, pmean, pstddev) >= 0 )
                {
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC3)(const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC3 ippiMean_StdDev_C3CR =
            type == CV_8UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CR :
            type == CV_16UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CR :
            type == CV_32FC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CR :
            0;
            if( ippiMean_StdDev_C3CR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src.ptr(), (int)src.step[0], sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src.ptr(), (int)src.step[0], sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src.ptr(), (int)src.step[0], sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                {
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(_mean); CV_UNUSED(_sdv); CV_UNUSED(mask);
#endif
    return false;
}
#endif

void meanStdDev(InputArray _src, OutputArray _mean, OutputArray _sdv, InputArray _mask)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_src.empty());
    CV_Assert( _mask.empty() || _mask.type() == CV_8UC1 );

    CV_OCL_RUN(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
               ocl_meanStdDev(_src, _mean, _sdv, _mask))

    Mat src = _src.getMat(), mask = _mask.getMat();

    CV_Assert(mask.empty() || src.size == mask.size);

    CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_MEAN_STDDEV>(src.cols, src.rows),
               openvx_meanStdDev(src, _mean, _sdv, mask))

    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_meanStdDev(src, _mean, _sdv, mask));

    int k, cn = src.channels(), depth = src.depth();
    Mat mean_mat, stddev_mat;

    if(_mean.needed())
    {
        if( !_mean.fixedSize() )
            _mean.create(cn, 1, CV_64F, -1, true);

        mean_mat = _mean.getMat();
        int dcn = (int)mean_mat.total();
        CV_Assert( mean_mat.type() == CV_64F && mean_mat.isContinuous() &&
                   (mean_mat.cols == 1 || mean_mat.rows == 1) && dcn >= cn );

        double* dptr = mean_mat.ptr<double>();
        for(k = cn ; k < dcn; k++ )
            dptr[k] = 0;
    }

    if (_sdv.needed())
    {
        if( !_sdv.fixedSize() )
            _sdv.create(cn, 1, CV_64F, -1, true);

        stddev_mat = _sdv.getMat();
        int dcn = (int)stddev_mat.total();
        CV_Assert( stddev_mat.type() == CV_64F && stddev_mat.isContinuous() &&
                   (stddev_mat.cols == 1 || stddev_mat.rows == 1) && dcn >= cn );

        double* dptr = stddev_mat.ptr<double>();
        for(k = cn ; k < dcn; k++ )
            dptr[k] = 0;

    }

    if (src.isContinuous() && mask.isContinuous())
    {
        CALL_HAL(meanStdDev, cv_hal_meanStdDev, src.data, 0, src.total(), static_cast<size_t>(1),
                 src.type(),
                 _mean.needed() ? mean_mat.ptr<double>() : nullptr,
                 _sdv.needed() ? stddev_mat.ptr<double>() : nullptr,
                 mask.data, static_cast<size_t>(0));
    }
    else
    {
        if (src.dims <= 2)
        {
            CALL_HAL(meanStdDev, cv_hal_meanStdDev, src.data, src.step, static_cast<size_t>(src.cols), static_cast<size_t>(src.rows), src.type(),
                     _mean.needed() ? mean_mat.ptr<double>() : nullptr,
                     _sdv.needed() ? stddev_mat.ptr<double>() : nullptr,
                     mask.data, mask.step);
        }
    }

    SumSqrFunc func = getSumSqrFunc(depth);

    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0, nz0 = 0;
    AutoBuffer<double> _buf(cn*4);
    double *s = (double*)_buf.data(), *sq = s + cn;
    int *sbuf = (int*)s, *sqbuf = (int*)sq;
    bool blockSum = depth <= CV_16S, blockSqSum = depth <= CV_8S;
    size_t esz = 0;

    for( k = 0; k < cn; k++ )
        s[k] = sq[k] = 0;

    if( blockSum )
    {
        intSumBlockSize = 1 << 15;
        blockSize = std::min(blockSize, intSumBlockSize);
        sbuf = (int*)(sq + cn);
        if( blockSqSum )
            sqbuf = sbuf + cn;
        for( k = 0; k < cn; k++ )
            sbuf[k] = sqbuf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            int nz = func( ptrs[0], ptrs[1], (uchar*)sbuf, (uchar*)sqbuf, bsz, cn );
            count += nz;
            nz0 += nz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += sbuf[k];
                    sbuf[k] = 0;
                }
                if( blockSqSum )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        sq[k] += sqbuf[k];
                        sqbuf[k] = 0;
                    }
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }

    double scale = nz0 ? 1./nz0 : 0.;
    for( k = 0; k < cn; k++ )
    {
        s[k] *= scale;
        sq[k] = std::sqrt(std::max(sq[k]*scale - s[k]*s[k], 0.));
    }

    if (_mean.needed())
    {
        const double* sptr = s;
        double* dptr = mean_mat.ptr<double>();
        for( k = 0; k < cn; k++ )
            dptr[k] = sptr[k];
    }

    if (_sdv.needed())
    {
        const double* sptr = sq;
        double* dptr = stddev_mat.ptr<double>();
        for( k = 0; k < cn; k++ )
            dptr[k] = sptr[k];
    }
}

} // namespace
