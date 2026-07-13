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

namespace cv
{

typedef void(*AccFunc)(const uchar*, uchar*, const uchar*, int, int);
typedef void(*AccProdFunc)(const uchar*, const uchar*, uchar*, const uchar*, int, int);
typedef void(*AccWFunc)(const uchar*, uchar*, const uchar*, int, int, double);

static AccFunc accTab[CV_DEPTH_MAX] =
{
    (AccFunc)acc_8u32f, (AccFunc)acc_8u64f,
    (AccFunc)acc_16u32f, (AccFunc)acc_16u64f,
    (AccFunc)acc_32f, (AccFunc)acc_32f64f,
    (AccFunc)acc_64f
};

static AccFunc accSqrTab[CV_DEPTH_MAX] =
{
    (AccFunc)accSqr_8u32f, (AccFunc)accSqr_8u64f,
    (AccFunc)accSqr_16u32f, (AccFunc)accSqr_16u64f,
    (AccFunc)accSqr_32f, (AccFunc)accSqr_32f64f,
    (AccFunc)accSqr_64f
};

static AccProdFunc accProdTab[CV_DEPTH_MAX] =
{
    (AccProdFunc)accProd_8u32f, (AccProdFunc)accProd_8u64f,
    (AccProdFunc)accProd_16u32f, (AccProdFunc)accProd_16u64f,
    (AccProdFunc)accProd_32f, (AccProdFunc)accProd_32f64f,
    (AccProdFunc)accProd_64f
};

static AccWFunc accWTab[CV_DEPTH_MAX] =
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

void cv::accumulate( InputArray _src, InputOutputArray _dst, InputArray _mask )
{
    CV_INSTRUMENT_REGION();

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), scn = CV_MAT_CN(stype);
    int dtype = _dst.type(), ddepth = CV_MAT_DEPTH(dtype), dcn = CV_MAT_CN(dtype);

    CV_Assert( _src.sameSize(_dst) && dcn == scn );
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && (_mask.type() == CV_8U || _mask.type() == CV_Bool)) );

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
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && (_mask.type() == CV_8U || _mask.type() == CV_Bool)) );

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
    CV_Assert( _mask.empty() || (_src1.sameSize(_mask) && (_mask.type() == CV_8U || _mask.type() == CV_Bool)) );

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
    CV_Assert( _mask.empty() || (_src.sameSize(_mask) && (_mask.type() == CV_8U || _mask.type() == CV_Bool)) );

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



/* End of file. */
