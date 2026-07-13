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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
/
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
#include "opencl_kernels_imgproc.hpp"
#include "opencv2/core/hal/intrin.hpp"
#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "accum.simd.hpp"
#include "accum.simd_declarations.hpp"
#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv
{

typedef void(*AccFunc)(const uchar*, uchar*, const uchar*, int, int);
typedef void(*AccProdFunc)(const uchar*, const uchar*, uchar*, const uchar*, int, int);
typedef void(*AccWFunc)(const uchar*, uchar*, const uchar*, int, int, double);

static AccFunc accTab[] =
{
    (AccFunc)acc_8u32f, (AccFunc)acc_8u64f,
    (AccFunc)acc_16u32f, (AccFunc)acc_16u64f,
    (AccFunc)acc_32f, (AccFunc)acc_32f64f,
    (AccFunc)acc_64f
};

static AccFunc accSqrTab[] =
{
    (AccFunc)accSqr_8u32f, (AccFunc)accSqr_8u64f,
    (AccFunc)accSqr_16u32f, (AccFunc)accSqr_16u64f,
    (AccFunc)accSqr_32f, (AccFunc)accSqr_32f64f,
    (AccFunc)accSqr_64f
};

static AccProdFunc accProdTab[] =
{
    (AccProdFunc)accProd_8u32f, (AccProdFunc)accProd_8u64f,
    (AccProdFunc)accProd_16u32f, (AccProdFunc)accProd_16u64f,
    (AccProdFunc)accProd_32f, (AccProdFunc)accProd_32f64f,
    (AccProdFunc)accProd_64f
};

static AccWFunc accWTab[] =
{
    (AccWFunc)accW_8u32f, (AccWFunc)accW_8u64f,
    (AccWFunc)accW_16u32f, (AccWFunc)accW_16u64f,
    (AccWFunc)accW_32f, (AccWFunc)accW_32f64f,
    (AccWFunc)accW_64f
};

inline int getAccTabIdx(int sdepth, int ddepth)
{
    return sdepth == CV_8U && ddepth == CV_32F ? 0 :
           sdepth == CV_8U && ddepth == CV_64F ? 1 :
           sdepth == CV_16U && ddepth == CV_32F ? 2 :
           sdepth == CV_16U && ddepth == CV_64F ? 3 :
           sdepth == CV_32F && ddepth == CV_32F ? 4 :
           sdepth == CV_32F && ddepth == CV_64F ? 5 :
           sdepth == CV_64F && ddepth == CV_64F ? 6 : -1;
}

#ifdef HAVE_OPENCL

enum
{
    ACCUMULATE = 0,
    ACCUMULATE_SQUARE = 1,
    ACCUMULATE_PRODUCT = 2,
    ACCUMULATE_WEIGHTED = 3
};

static bool ocl_accumulate( InputArray _src, InputArray _src2, InputOutputArray _dst, double alpha,
                            InputArray _mask, int op_type )
{
    CV_Assert(op_type == ACCUMULATE || op_type == ACCUMULATE_SQUARE ||
              op_type == ACCUMULATE_PRODUCT || op_type == ACCUMULATE_WEIGHTED);

    const ocl::Device & dev = ocl::Device::getDefault();
    bool haveMask = !_mask.empty(), doubleSupport = dev.doubleFPConfig() > 0;
    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype), ddepth = _dst.depth();
    int kercn = haveMask ? cn : ocl::predictOptimalVectorWidthMax(_src, _src2, _dst), rowsPerWI = dev.isIntel() ? 4 : 1;

    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

    const char * const opMap[4] = { "ACCUMULATE", "ACCUMULATE_SQUARE", "ACCUMULATE_PRODUCT",
                                   "ACCUMULATE_WEIGHTED" };

    char cvt[50];
    ocl::Kernel k("accumulate", ocl::imgproc::accumulate_oclsrc,
                  format("-D %s%s -D srcT1=%s -D cn=%d -D dstT1=%s%s -D rowsPerWI=%d -D convertToDT=%s",
                         opMap[op_type], haveMask ? " -D HAVE_MASK" : "",
                         ocl::typeToStr(sdepth), kercn, ocl::typeToStr(ddepth),
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "", rowsPerWI,
                         ocl::convertTypeStr(sdepth, ddepth, 1, cvt, sizeof(cvt))));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), src2 = _src2.getUMat(), dst = _dst.getUMat(), mask = _mask.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2),
            dstarg = ocl::KernelArg::ReadWrite(dst, cn, kercn),
            maskarg = ocl::KernelArg::ReadOnlyNoSize(mask);

    int argidx = k.set(0, srcarg);
    if (op_type == ACCUMULATE_PRODUCT)
        argidx = k.set(argidx, src2arg);
    argidx = k.set(argidx, dstarg);
    if (op_type == ACCUMULATE_WEIGHTED)
    {
        if (ddepth == CV_32F)
            argidx = k.set(argidx, (float)alpha);
        else
            argidx = k.set(argidx, alpha);
    }
    if (haveMask)
        k.set(argidx, maskarg);

    size_t globalsize[2] = { (size_t)src.cols * cn / kercn, ((size_t)src.rows + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

#endif

}

#ifdef HAVE_OPENVX
namespace cv
{
enum
{
    VX_ACCUMULATE_OP = 0,
    VX_ACCUMULATE_SQUARE_OP = 1,
    VX_ACCUMULATE_WEIGHTED_OP = 2
};

namespace ovx {
    template <> inline bool skipSmallImages<VX_KERNEL_ACCUMULATE>(int w, int h) { return w*h < 120 * 60; }
}
static bool openvx_accumulate(InputArray _src, InputOutputArray _dst, InputArray _mask, double _weight, int opType)
{
    Mat srcMat = _src.getMat(), dstMat = _dst.getMat();
    if (ovx::skipSmallImages<VX_KERNEL_ACCUMULATE>(srcMat.cols, srcMat.rows))
        return false;
    if(!_mask.empty() ||
       (opType == VX_ACCUMULATE_WEIGHTED_OP && dstMat.type() != CV_8UC1  ) ||
       (opType != VX_ACCUMULATE_WEIGHTED_OP && dstMat.type() != CV_16SC1 ) ||
       srcMat.type() != CV_8UC1)
    {
        return false;
    }
    //TODO: handle different number of channels (channel extract && channel combine)
    //TODO: handle mask (threshold mask to 0xff && bitwise AND with src)
    //(both things can be done by creating a graph)

    try
    {
        ivx::Context context = ovx::getOpenVXContext();
        ivx::Image srcImage = ivx::Image::createFromHandle(context, ivx::Image::matTypeToFormat(srcMat.type()),
                                                           ivx::Image::createAddressing(srcMat), srcMat.data);
        ivx::Image dstImage = ivx::Image::createFromHandle(context, ivx::Image::matTypeToFormat(dstMat.type()),
                                                           ivx::Image::createAddressing(dstMat), dstMat.data);
        ivx::Scalar shift = ivx::Scalar::create<VX_TYPE_UINT32>(context, 0);
        ivx::Scalar alpha = ivx::Scalar::create<VX_TYPE_FLOAT32>(context, _weight);

        switch (opType)
        {
        case VX_ACCUMULATE_OP:
            ivx::IVX_CHECK_STATUS(vxuAccumulateImage(context, srcImage, dstImage));
            break;
        case VX_ACCUMULATE_SQUARE_OP:
            ivx::IVX_CHECK_STATUS(vxuAccumulateSquareImage(context, srcImage, shift, dstImage));
            break;
        case VX_ACCUMULATE_WEIGHTED_OP:
            ivx::IVX_CHECK_STATUS(vxuAccumulateWeightedImage(context, srcImage, alpha, dstImage));
            break;
        default:
            break;
        }

#ifdef VX_VERSION_1_1
        //we should take user memory back before release
        //(it's not done automatically according to standard)
        srcImage.swapHandle(); dstImage.swapHandle();
#endif
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
}
#endif

void cv::accumulate( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src, noArray(), _dst, 0.0, _mask, ACCUMULATE))

    {
        Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
        if (src.dims <= 2 || (src.isContinuous() && dst.isContinuous() && (mask.empty() || mask.isContinuous())))
        {
            Size size = src.size();
            size_t srcstep = src.step, dststep = dst.step, maskstep = mask.step;
            if (src.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                srcstep = src.total() * src.elemSize();
                dststep = dst.total() * dst.elemSize();
                maskstep = mask.total() * mask.elemSize();
                size.width = (int)src.total();
                size.height = 1;
            }
            CALL_HAL(accumulate, cv_hal_accumulate, src.data, srcstep, dst.data, dststep,
                     mask.empty() ? nullptr : mask.data, maskstep, size.width, size.height, stype, dtype);
        }
    }

    CV_OVX_RUN(_src.dims() <= 2,
               openvx_accumulate(_src, _dst, _mask, 0.0, VX_ACCUMULATE_OP))

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();


    int fidx = getAccTabIdx(sdepth, ddepth);
    AccFunc func = fidx >= 0 ? accTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, scn);
}

void cv::accumulateSquare( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src, noArray(), _dst, 0.0, _mask, ACCUMULATE_SQUARE))

    {
        Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
        if (src.dims <= 2 || (src.isContinuous() && dst.isContinuous() && (mask.empty() || mask.isContinuous())))
        {
            Size size = src.size();
            size_t srcstep = src.step, dststep = dst.step, maskstep = mask.step;
            if (src.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                srcstep = src.total() * src.elemSize();
                dststep = dst.total() * dst.elemSize();
                maskstep = mask.total() * mask.elemSize();
                size.width = (int)src.total();
                size.height = 1;
            }
            CALL_HAL(accumulateSquare, cv_hal_accumulateSquare, src.data, srcstep, dst.data, dststep,
                     mask.empty() ? nullptr : mask.data, maskstep, size.width, size.height, stype, dtype);
        }
    }

    CV_OVX_RUN(_src.dims() <= 2,
               openvx_accumulate(_src, _dst, _mask, 0.0, VX_ACCUMULATE_SQUARE_OP))

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    int fidx = getAccTabIdx(sdepth, ddepth);
    AccFunc func = fidx >= 0 ? accSqrTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, scn);
}



void cv::accumulateProduct( InputArray _src1, InputArray _src2,
                            InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    int stype = _src1.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src1.sameSize(_src2) && stype == _src2.type() );
    CV_Assert( _src1.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src1.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src1.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src1, _src2, _dst, 0.0, _mask, ACCUMULATE_PRODUCT))

    {
        Mat src1 = _src1.getMat(), src2 = _src2.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
        if (src1.dims <= 2 || (src1.isContinuous() && src2.isContinuous() && dst.isContinuous()))
        {
            Size size = src1.size();
            size_t src1step = src1.step, src2step = src2.step, dststep = dst.step, maskstep = mask.step;
            if (src1.isContinuous() && src2.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                src1step = src1.total() * src1.elemSize();
                src2step = src2.total() * src2.elemSize();
                dststep = dst.total() * dst.elemSize();
                maskstep = mask.total() * mask.elemSize();
                size.width = (int)src1.total();
                size.height = 1;
            }
            CALL_HAL(accumulateProduct, cv_hal_accumulateProduct, src1.data, src1step, src2.data, src2step,
                     dst.data, dststep, mask.empty() ? nullptr : mask.data, maskstep, size.width, size.height, stype, dtype);
        }
    }

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), dst = _dst.getMat(), mask = _mask.getMat();

    int fidx = getAccTabIdx(sdepth, ddepth);
    AccProdFunc func = fidx >= 0 ? accProdTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src1, &src2, &dst, &mask, 0};
    uchar* ptrs[4] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], ptrs[3], len, scn);
}

void cv::accumulateWeighted( InputArray _src, InputOutputArray _dst,
                             double alpha, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && _mask.type() == CV_8U) );

    CV_OCL_RUN(_src.dims() <= 2 && _dst.isUMat(),
               ocl_accumulate(_src, noArray(), _dst, alpha, _mask, ACCUMULATE_WEIGHTED))

    {
        Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();
        if (src.dims <= 2 || (src.isContinuous() && dst.isContinuous() && mask.isContinuous()))
        {
            Size size = src.size();
            size_t srcstep = src.step, dststep = dst.step, maskstep = mask.step;
            if (src.isContinuous() && dst.isContinuous() && mask.isContinuous())
            {
                srcstep = src.total() * src.elemSize();
                dststep = dst.total() * dst.elemSize();
                maskstep = mask.total() * mask.elemSize();
                size.width = (int)src.total();
                size.height = 1;
            }
            CALL_HAL(accumulateWeighted, cv_hal_accumulateWeighted, src.data, srcstep, dst.data, dststep,
                     mask.empty() ? nullptr : mask.data, maskstep, size.width, size.height, stype, dtype, alpha);
        }
    }

    CV_OVX_RUN(_src.dims() <= 2,
               openvx_accumulate(_src, _dst, _mask, alpha, VX_ACCUMULATE_WEIGHTED_OP))

    Mat src = _src.getMat(), dst = _dst.getMat(), mask = _mask.getMat();


    int fidx = getAccTabIdx(sdepth, ddepth);
    AccWFunc func = fidx >= 0 ? accWTab[fidx] : 0;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, &mask, 0};
    uchar* ptrs[3] = {};
    NAryMatIterator it(arrays, ptrs);
    int len = (int)it.size;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        func(ptrs[0], ptrs[1], ptrs[2], len, scn, alpha);
}


CV_IMPL void
cvAcc( const void* arr, void* sumarr, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulate( src, dst, mask );
}

CV_IMPL void
cvSquareAcc( const void* arr, void* sumarr, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateSquare( src, dst, mask );
}

CV_IMPL void
cvMultiplyAcc( const void* arr1, const void* arr2,
               void* sumarr, const void* maskarr )
{
    cv::Mat src1 = cv::cvarrToMat(arr1), src2 = cv::cvarrToMat(arr2);
    cv::Mat dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateProduct( src1, src2, dst, mask );
}

CV_IMPL void
cvRunningAvg( const void* arr, void* sumarr, double alpha, const void* maskarr )
{
    cv::Mat src = cv::cvarrToMat(arr), dst = cv::cvarrToMat(sumarr), mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    cv::accumulateWeighted( src, dst, alpha, mask );
}

/* End of file. */
