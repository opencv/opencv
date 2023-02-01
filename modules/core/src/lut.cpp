// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"
#include "opencv2/core/openvx/ovx_defs.hpp"

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
LUT8u_( const uchar* src, const T* lut, T* dst, int len, int cn, int lutcn )
{
    if( lutcn == 1 )
    {
        for( int i = 0; i < len*cn; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        for( int i = 0; i < len*cn; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn+k];
    }
}

static void LUT8u_8u( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_8s( const uchar* src, const schar* lut, schar* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16u( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_16s( const uchar* src, const short* lut, short* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32s( const uchar* src, const int* lut, int* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_32f( const uchar* src, const float* lut, float* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

static void LUT8u_64f( const uchar* src, const double* lut, double* dst, int len, int cn, int lutcn )
{
    LUT8u_( src, lut, dst, len, cn, lutcn );
}

typedef void (*LUTFunc)( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

static LUTFunc lutTab[] =
{
    (LUTFunc)LUT8u_8u, (LUTFunc)LUT8u_8s, (LUTFunc)LUT8u_16u, (LUTFunc)LUT8u_16s,
    (LUTFunc)LUT8u_32s, (LUTFunc)LUT8u_32f, (LUTFunc)LUT8u_64f, 0
};

#ifdef HAVE_OPENCL

static bool ocl_LUT(InputArray _src, InputArray _lut, OutputArray _dst)
{
    int lcn = _lut.channels(), dcn = _src.channels(), ddepth = _lut.depth();

    UMat src = _src.getUMat(), lut = _lut.getUMat();
    _dst.create(src.size(), CV_MAKETYPE(ddepth, dcn));
    UMat dst = _dst.getUMat();
    int kercn = lcn == 1 ? std::min(4, ocl::predictOptimalVectorWidth(_src, _dst)) : dcn;

    ocl::Kernel k("LUT", ocl::core::lut_oclsrc,
                  format("-D dcn=%d -D lcn=%d -D srcT=%s -D dstT=%s", kercn, lcn,
                         ocl::typeToStr(src.depth()), ocl::memopTypeToStr(ddepth)));
    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::ReadOnlyNoSize(lut),
        ocl::KernelArg::WriteOnly(dst, dcn, kercn));

    size_t globalSize[2] = { (size_t)dst.cols * dcn / kercn, ((size_t)dst.rows + 3) / 4 };
    return k.run(2, globalSize, NULL, false);
}

#endif

#ifdef HAVE_OPENVX
static bool openvx_LUT(Mat src, Mat dst, Mat _lut)
{
    if (src.type() != CV_8UC1 || dst.type() != src.type() || _lut.type() != src.type() || !_lut.isContinuous())
        return false;

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();

        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(src.cols, src.rows, 1, (vx_int32)(src.step)), src.data),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(dst.cols, dst.rows, 1, (vx_int32)(dst.step)), dst.data);

        ivx::LUT lut = ivx::LUT::create(ctx);
        lut.copyFrom(_lut);
        ivx::IVX_CHECK_STATUS(vxuTableLookup(ctx, ia, lut, ib));
    }
    catch (const ivx::RuntimeError& e)
    {
        VX_DbgThrow(e.what());
    }
    catch (const ivx::WrapperError& e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
#endif

#if defined(HAVE_IPP)
#if !IPP_DISABLE_PERF_LUT // there are no performance benefits (PR #2653)
namespace ipp {

class IppLUTParallelBody_LUTC1 : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    int width;
    size_t elemSize1;

    IppLUTParallelBody_LUTC1(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        width = dst.cols * dst.channels();
        elemSize1 = CV_ELEM_SIZE1(dst.depth());

        CV_DbgAssert(elemSize1 == 1 || elemSize1 == 4);
        *ok = true;
    }

    void operator()( const cv::Range& range ) const
    {
        if (!*ok)
            return;

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        IppiSize sz = { width, dst.rows };

        if (elemSize1 == 1)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u_C1R, (const Ipp8u*)src.data, (int)src.step[0], dst.data, (int)dst.step[0], sz, lut_.data, 8) >= 0)
                return;
        }
        else if (elemSize1 == 4)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u32u_C1R, (const Ipp8u*)src.data, (int)src.step[0], (Ipp32u*)dst.data, (int)dst.step[0], sz, (Ipp32u*)lut_.data, 8) >= 0)
                return;
        }
        *ok = false;
    }
private:
    IppLUTParallelBody_LUTC1(const IppLUTParallelBody_LUTC1&);
    IppLUTParallelBody_LUTC1& operator=(const IppLUTParallelBody_LUTC1&);
};

class IppLUTParallelBody_LUTCN : public ParallelLoopBody
{
public:
    bool *ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    int lutcn;

    uchar* lutBuffer;
    uchar* lutTable[4];

    IppLUTParallelBody_LUTCN(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst), lutBuffer(NULL)
    {
        lutcn = lut.channels();
        IppiSize sz256 = {256, 1};

        size_t elemSize1 = dst.elemSize1();
        CV_DbgAssert(elemSize1 == 1);
        lutBuffer = (uchar*)CV_IPP_MALLOC(256 * (int)elemSize1 * 4);
        lutTable[0] = lutBuffer + 0;
        lutTable[1] = lutBuffer + 1 * 256 * elemSize1;
        lutTable[2] = lutBuffer + 2 * 256 * elemSize1;
        lutTable[3] = lutBuffer + 3 * 256 * elemSize1;

        CV_DbgAssert(lutcn == 3 || lutcn == 4);
        if (lutcn == 3)
        {
            IppStatus status = CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C3P3R, lut.ptr(), (int)lut.step[0], lutTable, (int)lut.step[0], sz256);
            if (status < 0)
                return;
        }
        else if (lutcn == 4)
        {
            IppStatus status = CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C4P4R, lut.ptr(), (int)lut.step[0], lutTable, (int)lut.step[0], sz256);
            if (status < 0)
                return;
        }

        *ok = true;
    }

    ~IppLUTParallelBody_LUTCN()
    {
        if (lutBuffer != NULL)
            ippFree(lutBuffer);
        lutBuffer = NULL;
        lutTable[0] = NULL;
    }

    void operator()( const cv::Range& range ) const
    {
        if (!*ok)
            return;

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        if (lutcn == 3)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u_C3R, src.ptr(), (int)src.step[0], dst.ptr(), (int)dst.step[0], ippiSize(dst.size()), lutTable, 8) >= 0)
                return;
        }
        else if (lutcn == 4)
        {
            if (CV_INSTRUMENT_FUN_IPP(ippiLUTPalette_8u_C4R, src.ptr(), (int)src.step[0], dst.ptr(), (int)dst.step[0], ippiSize(dst.size()), lutTable, 8) >= 0)
                return;
        }
        *ok = false;
    }
private:
    IppLUTParallelBody_LUTCN(const IppLUTParallelBody_LUTCN&);
    IppLUTParallelBody_LUTCN& operator=(const IppLUTParallelBody_LUTCN&);
};
} // namespace ipp

static bool ipp_lut(Mat &src, Mat &lut, Mat &dst)
{
    CV_INSTRUMENT_REGION_IPP();

    int lutcn = lut.channels();

    if(src.dims > 2)
        return false;

    bool ok = false;
    Ptr<ParallelLoopBody> body;

    size_t elemSize1 = CV_ELEM_SIZE1(dst.depth());

    if (lutcn == 1)
    {
        ParallelLoopBody* p = new ipp::IppLUTParallelBody_LUTC1(src, lut, dst, &ok);
        body.reset(p);
    }
    else if ((lutcn == 3 || lutcn == 4) && elemSize1 == 1)
    {
        ParallelLoopBody* p = new ipp::IppLUTParallelBody_LUTCN(src, lut, dst, &ok);
        body.reset(p);
    }

    if (body != NULL && ok)
    {
        Range all(0, dst.rows);
        if (dst.total()>>18)
            parallel_for_(all, *body, (double)std::max((size_t)1, dst.total()>>16));
        else
            (*body)(all);
        if (ok)
            return true;
    }

    return false;
}

#endif
#endif // IPP

class LUTParallelBody : public ParallelLoopBody
{
public:
    bool* ok;
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    LUTFunc func;

    LUTParallelBody(const Mat& src, const Mat& lut, Mat& dst, bool* _ok)
        : ok(_ok), src_(src), lut_(lut), dst_(dst)
    {
        func = lutTab[lut.depth()];
        *ok = (func != NULL);
    }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
        CV_DbgAssert(*ok);

        const int row0 = range.start;
        const int row1 = range.end;

        Mat src = src_.rowRange(row0, row1);
        Mat dst = dst_.rowRange(row0, row1);

        int cn = src.channels();
        int lutcn = lut_.channels();

        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2] = {};
        NAryMatIterator it(arrays, ptrs);
        int len = (int)it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
            func(ptrs[0], lut_.ptr(), ptrs[1], len, cn, lutcn);
    }
private:
    LUTParallelBody(const LUTParallelBody&);
    LUTParallelBody& operator=(const LUTParallelBody&);
};

} // cv::

void cv::LUT( InputArray _src, InputArray _lut, OutputArray _dst )
{
    CV_INSTRUMENT_REGION();

    int cn = _src.channels(), depth = _src.depth();
    int lutcn = _lut.channels();

    CV_Assert( (lutcn == cn || lutcn == 1) &&
        _lut.total() == 256 && _lut.isContinuous() &&
        (depth == CV_8U || depth == CV_8S) );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_LUT(_src, _lut, _dst))

    Mat src = _src.getMat(), lut = _lut.getMat();
    _dst.create(src.dims, src.size, CV_MAKETYPE(_lut.depth(), cn));
    Mat dst = _dst.getMat();

    CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_TABLE_LOOKUP>(src.cols, src.rows),
               openvx_LUT(src, dst, lut))

#if !IPP_DISABLE_PERF_LUT
    CV_IPP_RUN(_src.dims() <= 2, ipp_lut(src, lut, dst));
#endif

    if (_src.dims() <= 2)
    {
        bool ok = false;
        Ptr<ParallelLoopBody> body;

        if (body == NULL || ok == false)
        {
            ok = false;
            ParallelLoopBody* p = new LUTParallelBody(src, lut, dst, &ok);
            body.reset(p);
        }
        if (body != NULL && ok)
        {
            Range all(0, dst.rows);
            if (dst.total()>>18)
                parallel_for_(all, *body, (double)std::max((size_t)1, dst.total()>>16));
            else
                (*body)(all);
            if (ok)
                return;
        }
    }

    LUTFunc func = lutTab[lut.depth()];
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], lut.ptr(), ptrs[1], len, cn, lutcn);
}
