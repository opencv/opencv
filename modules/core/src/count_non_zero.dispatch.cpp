// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "stat.hpp"

#include "count_non_zero.simd.hpp"
#include "count_non_zero.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

static CountNonZeroFunc getCountNonZeroTab(int depth)
{
    CV_INSTRUMENT_REGION();
    CV_CPU_DISPATCH(getCountNonZeroTab, (depth),
        CV_CPU_DISPATCH_MODES_ALL);
}

#ifdef HAVE_OPENCL
static bool ocl_countNonZero( InputArray _src, int & res )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), kercn = ocl::predictOptimalVectorWidth(_src);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (depth == CV_64F && !doubleSupport)
        return false;

    int dbsize = ocl::Device::getDefault().maxComputeUnits();
    size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc,
                  format("-D srcT=%s -D srcT1=%s -D cn=1 -D OP_COUNT_NON_ZERO"
                         " -D WGS=%d -D kercn=%d -D WGS2_ALIGNED=%d%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(depth), (int)wgs, kercn,
                         wgs2_aligned, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : ""));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), db(1, dbsize, CV_32SC1);
    k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
           dbsize, ocl::KernelArg::PtrWriteOnly(db));

    size_t globalsize = dbsize * wgs;
    if (k.run(1, &globalsize, &wgs, true))
        return res = saturate_cast<int>(cv::sum(db.getMat(ACCESS_READ))[0]), true;
    return false;
}
#endif

#if defined HAVE_IPP
static bool ipp_countNonZero( Mat &src, int &res )
{
    CV_INSTRUMENT_REGION_IPP();

#if defined __APPLE__ || (defined _MSC_VER && defined _M_IX86)
    // see https://github.com/opencv/opencv/issues/17453
    if (src.dims <= 2 && src.step > 520000)
        return false;
#endif

#if IPP_VERSION_X100 < 201801
    // Poor performance of SSE42
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

    Ipp32s  count = 0;
    int     depth = src.depth();

    if(src.dims <= 2)
    {
        IppStatus status;
        IppiSize  size = {src.cols*src.channels(), src.rows};

        if(depth == CV_8U)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, (const Ipp8u *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else if(depth == CV_32F)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, (const Ipp32f *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else
            return false;

        if(status < 0)
            return false;

        res = size.width*size.height - count;
    }
    else
    {
        IppStatus       status;
        const Mat      *arrays[] = {&src, NULL};
        Mat            planes[1];
        NAryMatIterator it(arrays, planes, 1);
        IppiSize        size  = {(int)it.size*src.channels(), 1};
        res = 0;
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            if(depth == CV_8U)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, it.planes->ptr<Ipp8u>(), (int)it.planes->step, size, &count, 0, 0);
            else if(depth == CV_32F)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, it.planes->ptr<Ipp32f>(), (int)it.planes->step, size, &count, 0, 0);
            else
                return false;

            if(status < 0 || (int)it.planes->total()*src.channels() < count)
                return false;

            res += (int)it.planes->total()*src.channels() - count;
        }
    }

    return true;
}
#endif

int countNonZero(InputArray _src)
{
    CV_INSTRUMENT_REGION();

    int type = _src.type(), cn = CV_MAT_CN(type);
    CV_Assert( cn == 1 );

#if defined HAVE_OPENCL || defined HAVE_IPP
    int res = -1;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_countNonZero(_src, res),
                res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN_FAST(ipp_countNonZero(src, res), res);

    CountNonZeroFunc func = getCountNonZeroTab(src.depth());
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1] = {};
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, nz = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        nz += func( ptrs[0], total );

    return nz;
}

void findNonZero(InputArray _src, OutputArray _idx)
{
    Mat src = _src.getMat();
    CV_Assert( src.channels() == 1 && src.dims == 2 );

    int depth = src.depth();
    std::vector<Point> idxvec;
    int rows = src.rows, cols = src.cols;
    AutoBuffer<int> buf_(cols + 1);
    int* buf = buf_.data();

    for( int i = 0; i < rows; i++ )
    {
        int j, k = 0;
        const uchar* ptr8 = src.ptr(i);
        if( depth == CV_8U || depth == CV_8S )
        {
            for( j = 0; j < cols; j++ )
                if( ptr8[j] != 0 ) buf[k++] = j;
        }
        else if( depth == CV_16U || depth == CV_16S )
        {
            const ushort* ptr16 = (const ushort*)ptr8;
            for( j = 0; j < cols; j++ )
                if( ptr16[j] != 0 ) buf[k++] = j;
        }
        else if( depth == CV_32S )
        {
            const int* ptr32s = (const int*)ptr8;
            for( j = 0; j < cols; j++ )
                if( ptr32s[j] != 0 ) buf[k++] = j;
        }
        else if( depth == CV_32F )
        {
            const float* ptr32f = (const float*)ptr8;
            for( j = 0; j < cols; j++ )
                if( ptr32f[j] != 0 ) buf[k++] = j;
        }
        else
        {
            const double* ptr64f = (const double*)ptr8;
            for( j = 0; j < cols; j++ )
                if( ptr64f[j] != 0 ) buf[k++] = j;
        }

        if( k > 0 )
        {
            size_t sz = idxvec.size();
            idxvec.resize(sz + k);
            for( j = 0; j < k; j++ )
                idxvec[sz + j] = Point(buf[j], i);
        }
    }

    if( idxvec.empty() || (_idx.kind() == _InputArray::MAT && !_idx.getMatRef().isContinuous()) )
        _idx.release();

    if( !idxvec.empty() )
        Mat(idxvec).copyTo(_idx);
}

} // namespace
