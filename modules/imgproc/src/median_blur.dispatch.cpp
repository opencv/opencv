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
// Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include <vector>

#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_imgproc.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

#include "median_blur.simd.hpp"
#include "median_blur.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv {

#ifdef HAVE_OPENCL

#define DIVUP(total, grain) ((total + grain - 1) / (grain))

static bool ocl_medianFilter(InputArray _src, OutputArray _dst, int m)
{
    size_t localsize[2] = { 16, 16 };
    size_t globalsize[2];
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if ( !((depth == CV_8U || depth == CV_16U || depth == CV_16S || depth == CV_32F) && cn <= 4 && (m == 3 || m == 5)) )
        return false;

    Size imgSize = _src.size();
    bool useOptimized = (1 == cn) &&
                        (size_t)imgSize.width >= localsize[0] * 8  &&
                        (size_t)imgSize.height >= localsize[1] * 8 &&
                        imgSize.width % 4 == 0 &&
                        imgSize.height % 4 == 0 &&
                        (ocl::Device::getDefault().isIntel());

    cv::String kname = format( useOptimized ? "medianFilter%d_u" : "medianFilter%d", m) ;
    cv::String kdefs = useOptimized ?
                         format("-D T=%s -D T1=%s -D T4=%s%d -D cn=%d -D USE_4OPT", ocl::typeToStr(type),
                         ocl::typeToStr(depth), ocl::typeToStr(depth), cn*4, cn)
                         :
                         format("-D T=%s -D T1=%s -D cn=%d", ocl::typeToStr(type), ocl::typeToStr(depth), cn) ;

    ocl::Kernel k(kname.c_str(), ocl::imgproc::medianFilter_oclsrc, kdefs.c_str() );

    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(src.size(), type);
    UMat dst = _dst.getUMat();

    k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst));

    if( useOptimized )
    {
        globalsize[0] = DIVUP(src.cols / 4, localsize[0]) * localsize[0];
        globalsize[1] = DIVUP(src.rows / 4, localsize[1]) * localsize[1];
    }
    else
    {
        globalsize[0] = (src.cols + localsize[0] + 2) / localsize[0] * localsize[0];
        globalsize[1] = (src.rows + localsize[1] - 1) / localsize[1] * localsize[1];
    }

    return k.run(2, globalsize, localsize, false);
}

#undef DIVUP

#endif

#ifdef HAVE_OPENVX
namespace ovx {
    template <> inline bool skipSmallImages<VX_KERNEL_MEDIAN_3x3>(int w, int h) { return w*h < 1280 * 720; }
}
static bool openvx_medianFilter(InputArray _src, OutputArray _dst, int ksize)
{
    if (_src.type() != CV_8UC1 || _dst.type() != CV_8U
#ifndef VX_VERSION_1_1
        || ksize != 3
#endif
        )
        return false;

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    if (
#ifdef VX_VERSION_1_1
         ksize != 3 ? ovx::skipSmallImages<VX_KERNEL_NON_LINEAR_FILTER>(src.cols, src.rows) :
#endif
         ovx::skipSmallImages<VX_KERNEL_MEDIAN_3x3>(src.cols, src.rows)
       )
        return false;

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();
#ifdef VX_VERSION_1_1
        if ((vx_size)ksize > ctx.nonlinearMaxDimension())
            return false;
#endif

        Mat a;
        if (dst.data != src.data)
            a = src;
        else
            src.copyTo(a);

        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(a.cols, a.rows, 1, (vx_int32)(a.step)), a.data),
            ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                ivx::Image::createAddressing(dst.cols, dst.rows, 1, (vx_int32)(dst.step)), dst.data);

        //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
        //since OpenVX standard says nothing about thread-safety for now
        ivx::border_t prevBorder = ctx.immediateBorder();
        ctx.setImmediateBorder(VX_BORDER_REPLICATE);
#ifdef VX_VERSION_1_1
        if (ksize == 3)
#endif
        {
            ivx::IVX_CHECK_STATUS(vxuMedian3x3(ctx, ia, ib));
        }
#ifdef VX_VERSION_1_1
        else
        {
            ivx::Matrix mtx;
            if(ksize == 5)
                mtx = ivx::Matrix::createFromPattern(ctx, VX_PATTERN_BOX, ksize, ksize);
            else
            {
                vx_size supportedSize;
                ivx::IVX_CHECK_STATUS(vxQueryContext(ctx, VX_CONTEXT_NONLINEAR_MAX_DIMENSION, &supportedSize, sizeof(supportedSize)));
                if ((vx_size)ksize > supportedSize)
                {
                    ctx.setImmediateBorder(prevBorder);
                    return false;
                }
                Mat mask(ksize, ksize, CV_8UC1, Scalar(255));
                mtx = ivx::Matrix::create(ctx, VX_TYPE_UINT8, ksize, ksize);
                mtx.copyFrom(mask);
            }
            ivx::IVX_CHECK_STATUS(vxuNonLinearFilter(ctx, VX_NONLINEAR_FILTER_MEDIAN, ia, mtx, ib));
        }
#endif
        ctx.setImmediateBorder(prevBorder);
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
static bool ipp_medianFilter(Mat &src0, Mat &dst, int ksize)
{
    CV_INSTRUMENT_REGION_IPP();

#if IPP_VERSION_X100 < 201801
    // Degradations for big kernel
    if(ksize > 7)
        return false;
#endif

    {
        int         bufSize;
        IppiSize    dstRoiSize = ippiSize(dst.cols, dst.rows), maskSize = ippiSize(ksize, ksize);
        IppDataType ippType = ippiGetDataType(src0.type());
        int         channels = src0.channels();
        IppAutoBuffer<Ipp8u> buffer;

        if(src0.isSubmatrix())
            return false;

        Mat src;
        if(dst.data != src0.data)
            src = src0;
        else
            src0.copyTo(src);

        if(ippiFilterMedianBorderGetBufferSize(dstRoiSize, maskSize, ippType, channels, &bufSize) < 0)
            return false;

        buffer.allocate(bufSize);

        switch(ippType)
        {
        case ipp8u:
            if(channels == 1)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C1R, src.ptr<Ipp8u>(), (int)src.step, dst.ptr<Ipp8u>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else if(channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C3R, src.ptr<Ipp8u>(), (int)src.step, dst.ptr<Ipp8u>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else if(channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_8u_C4R, src.ptr<Ipp8u>(), (int)src.step, dst.ptr<Ipp8u>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else
                return false;
        case ipp16u:
            if(channels == 1)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C1R, src.ptr<Ipp16u>(), (int)src.step, dst.ptr<Ipp16u>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else if(channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C3R, src.ptr<Ipp16u>(), (int)src.step, dst.ptr<Ipp16u>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else if(channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16u_C4R, src.ptr<Ipp16u>(), (int)src.step, dst.ptr<Ipp16u>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else
                return false;
        case ipp16s:
            if(channels == 1)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C1R, src.ptr<Ipp16s>(), (int)src.step, dst.ptr<Ipp16s>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else if(channels == 3)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C3R, src.ptr<Ipp16s>(), (int)src.step, dst.ptr<Ipp16s>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else if(channels == 4)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_16s_C4R, src.ptr<Ipp16s>(), (int)src.step, dst.ptr<Ipp16s>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else
                return false;
        case ipp32f:
            if(channels == 1)
                return CV_INSTRUMENT_FUN_IPP(ippiFilterMedianBorder_32f_C1R, src.ptr<Ipp32f>(), (int)src.step, dst.ptr<Ipp32f>(), (int)dst.step, dstRoiSize, maskSize, ippBorderRepl, 0, buffer) >= 0;
            else
                return false;
        default:
            return false;
        }
    }
}
#endif

void medianBlur( InputArray _src0, OutputArray _dst, int ksize )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( (ksize % 2 == 1) && (_src0.dims() <= 2 ));

    if( ksize <= 1 || _src0.empty() )
    {
        _src0.copyTo(_dst);
        return;
    }

    CV_OCL_RUN(_dst.isUMat(),
               ocl_medianFilter(_src0,_dst, ksize))

    Mat src0 = _src0.getMat();
    _dst.create( src0.size(), src0.type() );
    Mat dst = _dst.getMat();

    CALL_HAL(medianBlur, cv_hal_medianBlur, src0.data, src0.step, dst.data, dst.step, src0.cols, src0.rows, src0.depth(),
             src0.channels(), ksize);

    CV_OVX_RUN(true,
               openvx_medianFilter(_src0, _dst, ksize))

    CV_IPP_RUN_FAST(ipp_medianFilter(src0, dst, ksize));

#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && tegra::medianBlur(src0, dst, ksize))
        return;
#endif

    CV_CPU_DISPATCH(medianBlur, (src0, dst, ksize),
        CV_CPU_DISPATCH_MODES_ALL);
}

}  // namespace

/* End of file. */
