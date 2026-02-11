// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "opencl_kernels_core.hpp"
#include "convert.hpp"
#include <sys/types.h>

/****************************************************************************************\
*                                    LUT Transform                                       *
\****************************************************************************************/

namespace cv
{

template<typename Ti, typename T> static void
LUT_( const Ti* src, const T* lut, T* dst, const int len, const int cn, const int lutcn )
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

typedef void (*LUTFunc)( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );

static LUTFunc getLUTFunc(const int srcDepth, const int dstDepth)
{
    LUTFunc ret = nullptr;
    if((srcDepth == CV_8U) || (srcDepth == CV_8S))
    {
        switch(dstDepth)
        {
            case CV_8U:   ret = (LUTFunc)LUT_<uint8_t, uint8_t>;   break;
            case CV_8S:   ret = (LUTFunc)LUT_<uint8_t, int8_t>;    break;
            case CV_16U:  ret = (LUTFunc)LUT_<uint8_t, uint16_t>;  break;
            case CV_16S:  ret = (LUTFunc)LUT_<uint8_t, int16_t>;   break;
            case CV_32S:  ret = (LUTFunc)LUT_<uint8_t, int32_t>;   break;
            case CV_32F:  ret = (LUTFunc)LUT_<uint8_t, int32_t>;   break; // float
            case CV_64F:  ret = (LUTFunc)LUT_<uint8_t, int64_t>;   break; // double
            case CV_16F:  ret = (LUTFunc)LUT_<uint8_t, int16_t>;   break; // hfloat
            default:      ret = nullptr;                           break;
        }
    }
    else if((srcDepth == CV_16U) || (srcDepth == CV_16S))
    {
        switch(dstDepth)
        {
            case CV_8U:   ret = (LUTFunc)LUT_<uint16_t, uint8_t>;  break;
            case CV_8S:   ret = (LUTFunc)LUT_<uint16_t, int8_t>;   break;
            case CV_16U:  ret = (LUTFunc)LUT_<uint16_t, uint16_t>; break;
            case CV_16S:  ret = (LUTFunc)LUT_<uint16_t, int16_t>;  break;
            case CV_32S:  ret = (LUTFunc)LUT_<uint16_t, int32_t>;  break;
            case CV_32F:  ret = (LUTFunc)LUT_<uint16_t, int32_t>;  break; // float
            case CV_64F:  ret = (LUTFunc)LUT_<uint16_t, int64_t>;  break; // double
            case CV_16F:  ret = (LUTFunc)LUT_<uint16_t, int16_t>;  break; // hfloat
            default:      ret = nullptr;                           break;
        }
    }

    CV_CheckTrue(ret != nullptr, "An unexpected type combination was specified.");
    return ret;
}

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

class LUTParallelBody : public ParallelLoopBody
{
public:
    const Mat& src_;
    const Mat& lut_;
    Mat& dst_;

    LUTFunc func_;

    LUTParallelBody(const Mat& src, const Mat& lut, Mat& dst, LUTFunc func)
        : src_(src), lut_(lut), dst_(dst), func_(func)
    {
    }

    void operator()( const cv::Range& range ) const CV_OVERRIDE
    {
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
            func_(ptrs[0], lut_.ptr(), ptrs[1], len, cn, lutcn);
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
    const size_t lut_size = _lut.total();

    CV_Assert( (lutcn == cn || lutcn == 1) && _lut.isContinuous() &&
        (
            ((lut_size == 256) && ((depth == CV_8U)||(depth == CV_8S))) ||
            ((lut_size == 65536) && ((depth == CV_16U)||(depth == CV_16S)))
        )
    );

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && (lut_size == 256),
               ocl_LUT(_src, _lut, _dst))

    Mat src = _src.getMat(), lut = _lut.getMat();
    _dst.create(src.dims, src.size, CV_MAKETYPE(_lut.depth(), cn));
    Mat dst = _dst.getMat();

    if(lut_size == 256)
    {
        CALL_HAL(LUT, cv_hal_lut, src.data, src.step, src.type(), lut.data,
                 lut.elemSize1(), lutcn, dst.data, dst.step, src.cols, src.rows);
    }
    else
    {
        CALL_HAL(LUT16, cv_hal_lut16, src.ptr<ushort>(), src.step, src.type(), lut.data,
                 lut.elemSize1(), lutcn, dst.data, dst.step, src.cols, src.rows);
    }

    const LUTFunc func = getLUTFunc(src.depth(), dst.depth());
    CV_Assert( func != nullptr );

    if (_src.dims() <= 2)
    {
        LUTParallelBody body(src, lut, dst, func);
        Range all(0, dst.rows);
        if (dst.total() >= (size_t)(1<<18))
            parallel_for_(all, body, (double)std::max((size_t)1, dst.total()>>16));
        else
            body(all);

        return;
    }

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], lut.ptr(), ptrs[1], len, cn, lutcn);
}
