/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "opencv2/core/openvx/ovx_defs.hpp"
#include "filter.hpp"

/****************************************************************************************\
                             Sobel & Scharr Derivative Filters
\****************************************************************************************/

namespace cv
{

static void getScharrKernels( OutputArray _kx, OutputArray _ky,
                              int dx, int dy, bool normalize, int ktype )
{
    const int ksize = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    _kx.create(ksize, 1, ktype, -1, true);
    _ky.create(ksize, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy == 1 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int kerI[3];

        if( order == 0 )
            kerI[0] = 3, kerI[1] = 10, kerI[2] = 3;
        else if( order == 1 )
            kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize || order == 1 ? 1. : 1./32;
        temp.convertTo(*kernel, ktype, scale);
    }
}


static void getSobelKernels( OutputArray _kx, OutputArray _ky,
                             int dx, int dy, int _ksize, bool normalize, int ktype )
{
    int i, j, ksizeX = _ksize, ksizeY = _ksize;
    if( ksizeX == 1 && dx > 0 )
        ksizeX = 3;
    if( ksizeY == 1 && dy > 0 )
        ksizeY = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    _kx.create(ksizeX, 1, ktype, -1, true);
    _ky.create(ksizeY, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    if( _ksize % 2 == 0 || _ksize > 31 )
        CV_Error( CV_StsOutOfRange, "The kernel size must be odd and not larger than 31" );
    std::vector<int> kerI(std::max(ksizeX, ksizeY) + 1);

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy > 0 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int ksize = k == 0 ? ksizeX : ksizeY;

        CV_Assert( ksize > order );

        if( ksize == 1 )
            kerI[0] = 1;
        else if( ksize == 3 )
        {
            if( order == 0 )
                kerI[0] = 1, kerI[1] = 2, kerI[2] = 1;
            else if( order == 1 )
                kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;
            else
                kerI[0] = 1, kerI[1] = -2, kerI[2] = 1;
        }
        else
        {
            int oldval, newval;
            kerI[0] = 1;
            for( i = 0; i < ksize; i++ )
                kerI[i+1] = 0;

            for( i = 0; i < ksize - order - 1; i++ )
            {
                oldval = kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j]+kerI[j-1];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }

            for( i = 0; i < order; i++ )
            {
                oldval = -kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j-1] - kerI[j];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }
        }

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize ? 1. : 1./(1 << (ksize-order-1));
        temp.convertTo(*kernel, ktype, scale);
    }
}

}

void cv::getDerivKernels( OutputArray kx, OutputArray ky, int dx, int dy,
                          int ksize, bool normalize, int ktype )
{
    if( ksize <= 0 )
        getScharrKernels( kx, ky, dx, dy, normalize, ktype );
    else
        getSobelKernels( kx, ky, dx, dy, ksize, normalize, ktype );
}


cv::Ptr<cv::FilterEngine> cv::createDerivFilter(int srcType, int dstType,
                                                int dx, int dy, int ksize, int borderType )
{
    Mat kx, ky;
    getDerivKernels( kx, ky, dx, dy, ksize, false, CV_32F );
    return createSeparableLinearFilter(srcType, dstType,
        kx, ky, Point(-1,-1), 0, borderType );
}

#ifdef HAVE_OPENVX
namespace cv
{
    namespace ovx {
        template <> inline bool skipSmallImages<VX_KERNEL_SOBEL_3x3>(int w, int h) { return w*h < 320 * 240; }
    }
    static bool openvx_sobel(InputArray _src, OutputArray _dst,
                             int dx, int dy, int ksize,
                             double scale, double delta, int borderType)
    {
        if (_src.type() != CV_8UC1 || _dst.type() != CV_16SC1 ||
            ksize != 3 || scale != 1.0 || delta != 0.0 ||
            (dx | dy) != 1 || (dx + dy) != 1 ||
            _src.cols() < ksize || _src.rows() < ksize ||
            ovx::skipSmallImages<VX_KERNEL_SOBEL_3x3>(_src.cols(), _src.rows())
            )
            return false;

        Mat src = _src.getMat();
        Mat dst = _dst.getMat();

        if ((borderType & BORDER_ISOLATED) == 0 && src.isSubmatrix())
            return false; //Process isolated borders only
        vx_enum border;
        switch (borderType & ~BORDER_ISOLATED)
        {
        case BORDER_CONSTANT:
            border = VX_BORDER_CONSTANT;
            break;
        case BORDER_REPLICATE:
//            border = VX_BORDER_REPLICATE;
//            break;
        default:
            return false;
        }

        try
        {
            ivx::Context ctx = ovx::getOpenVXContext();
            //if ((vx_size)ksize > ctx.convolutionMaxDimension())
            //    return false;

            Mat a;
            if (dst.data != src.data)
                a = src;
            else
                src.copyTo(a);

            ivx::Image
                ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                    ivx::Image::createAddressing(a.cols, a.rows, 1, (vx_int32)(a.step)), a.data),
                ib = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_S16,
                    ivx::Image::createAddressing(dst.cols, dst.rows, 2, (vx_int32)(dst.step)), dst.data);

            //ATTENTION: VX_CONTEXT_IMMEDIATE_BORDER attribute change could lead to strange issues in multi-threaded environments
            //since OpenVX standart says nothing about thread-safety for now
            ivx::border_t prevBorder = ctx.immediateBorder();
            ctx.setImmediateBorder(border, (vx_uint8)(0));
            if(dx)
                ivx::IVX_CHECK_STATUS(vxuSobel3x3(ctx, ia, ib, NULL));
            else
                ivx::IVX_CHECK_STATUS(vxuSobel3x3(ctx, ia, NULL, ib));
            ctx.setImmediateBorder(prevBorder);
        }
        catch (ivx::RuntimeError & e)
        {
            VX_DbgThrow(e.what());
        }
        catch (ivx::WrapperError & e)
        {
            VX_DbgThrow(e.what());
        }

        return true;
    }
}
#endif

#ifdef HAVE_IPP
namespace cv
{

static bool ipp_Deriv(InputArray _src, OutputArray _dst, int dx, int dy, int ksize, double scale, double delta, int borderType)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    ::ipp::IwiSize size(_src.size().width, _src.size().height);
    IppDataType   srcType   = ippiGetDataType(_src.depth());
    IppDataType   dstType   = ippiGetDataType(_dst.depth());
    int           channels  = _src.channels();
    bool          useScale  = false;
    bool          useScharr = false;

    if(channels != _dst.channels() || channels > 1)
        return false;

    if(fabs(delta) > FLT_EPSILON || fabs(scale-1) > FLT_EPSILON)
        useScale = true;

    if(ksize <= 0)
    {
        ksize     = 3;
        useScharr = true;
    }

    IppiMaskSize maskSize = ippiGetMaskSize(ksize, ksize);
    if(maskSize < 0)
        return false;

#if IPP_VERSION_X100 <= 201703
    // Bug with mirror wrap
    if(borderType == BORDER_REFLECT_101 && (ksize/2+1 > size.width || ksize/2+1 > size.height))
        return false;
#endif

    IwiDerivativeType derivType = ippiGetDerivType(dx, dy, (useScharr)?false:true);
    if(derivType < 0)
        return false;

    // Acquire data and begin processing
    try
    {
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        ::ipp::IwiImage iwSrc      = ippiGetImage(src);
        ::ipp::IwiImage iwDst      = ippiGetImage(dst);
        ::ipp::IwiImage iwSrcProc  = iwSrc;
        ::ipp::IwiImage iwDstProc  = iwDst;
        ::ipp::IwiBorderSize  borderSize(maskSize);
        ::ipp::IwiBorderType  ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        if(srcType == ipp8u && dstType == ipp8u)
        {
            iwDstProc.Alloc(iwDst.m_size, ipp16s, channels);
            useScale = true;
        }
        else if(srcType == ipp8u && dstType == ipp32f)
        {
            iwSrc -= borderSize;
            iwSrcProc.Alloc(iwSrc.m_size, ipp32f, channels);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwSrc, iwSrcProc, 1, 0, ::ipp::IwiScaleParams(ippAlgHintFast));
            iwSrcProc += borderSize;
        }

        if(useScharr)
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterScharr, iwSrcProc, iwDstProc, derivType, maskSize, ::ipp::IwDefault(), ippBorder);
        else
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterSobel, iwSrcProc, iwDstProc, derivType, maskSize, ::ipp::IwDefault(), ippBorder);

        if(useScale)
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwDstProc, iwDst, scale, delta, ::ipp::IwiScaleParams(ippAlgHintFast));
    }
    catch (::ipp::IwException)
    {
        return false;
    }

    return true;
#else
    CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(dx); CV_UNUSED(dy); CV_UNUSED(ksize); CV_UNUSED(scale); CV_UNUSED(delta); CV_UNUSED(borderType);
    return false;
#endif
}
}
#endif

#ifdef HAVE_OPENCL
namespace cv
{
static bool ocl_sepFilter3x3_8UC1(InputArray _src, OutputArray _dst, int ddepth,
                                  InputArray _kernelX, InputArray _kernelY, double delta, int borderType)
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if ( !(dev.isIntel() && (type == CV_8UC1) && (ddepth == CV_8U) &&
         (_src.offset() == 0) && (_src.step() % 4 == 0) &&
         (_src.cols() % 16 == 0) && (_src.rows() % 2 == 0)) )
        return false;

    Mat kernelX = _kernelX.getMat().reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    Mat kernelY = _kernelY.getMat().reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;

    if (ddepth < 0)
        ddepth = sdepth;

    Size size = _src.size();
    size_t globalsize[2] = { 0, 0 };
    size_t localsize[2] = { 0, 0 };

    globalsize[0] = size.width / 16;
    globalsize[1] = size.height / 2;

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };
    char build_opts[1024];
    sprintf(build_opts, "-D %s %s%s", borderMap[borderType],
            ocl::kernelToStr(kernelX, CV_32F, "KERNEL_MATRIX_X").c_str(),
            ocl::kernelToStr(kernelY, CV_32F, "KERNEL_MATRIX_Y").c_str());

    ocl::Kernel kernel("sepFilter3x3_8UC1_cols16_rows2", cv::ocl::imgproc::sepFilter3x3_oclsrc, build_opts);
    if (kernel.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, CV_MAKETYPE(ddepth, cn));
    if (!(_dst.offset() == 0 && _dst.step() % 4 == 0))
        return false;
    UMat dst = _dst.getUMat();

    int idxArg = kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = kernel.set(idxArg, (int)src.step);
    idxArg = kernel.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
    idxArg = kernel.set(idxArg, (int)dst.step);
    idxArg = kernel.set(idxArg, (int)dst.rows);
    idxArg = kernel.set(idxArg, (int)dst.cols);
    idxArg = kernel.set(idxArg, static_cast<float>(delta));

    return kernel.run(2, globalsize, (localsize[0] == 0) ? NULL : localsize, false);
}
}
#endif

void cv::Sobel( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
                int ksize, double scale, double delta, int borderType )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    int dtype = CV_MAKE_TYPE(ddepth, cn);
    _dst.create( _src.size(), dtype );

    int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

    Mat kx, ky;
    getDerivKernels( kx, ky, dx, dy, ksize, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differentiating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }

    CV_OCL_RUN(ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 && ksize == 3 &&
               (size_t)_src.rows() > ky.total() && (size_t)_src.cols() > kx.total(),
               ocl_sepFilter3x3_8UC1(_src, _dst, ddepth, kx, ky, delta, borderType));

    CV_OCL_RUN(ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 && (size_t)_src.rows() > kx.total() && (size_t)_src.cols() > kx.total(),
               ocl_sepFilter2D(_src, _dst, ddepth, kx, ky, Point(-1, -1), 0, borderType))

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType & BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

    CALL_HAL(sobel, cv_hal_sobel, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, ddepth, cn,
             ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, dx, dy, ksize, scale, delta, borderType&~BORDER_ISOLATED);

    CV_OVX_RUN(true,
               openvx_sobel(src, dst, dx, dy, ksize, scale, delta, borderType))

    CV_IPP_RUN_FAST(ipp_Deriv(src, dst, dx, dy, ksize, scale, delta, borderType));

    sepFilter2D(src, dst, ddepth, kx, ky, Point(-1, -1), delta, borderType );
}


void cv::Scharr( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
                 double scale, double delta, int borderType )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    int dtype = CV_MAKETYPE(ddepth, cn);
    _dst.create( _src.size(), dtype );

    int ktype = std::max(CV_32F, std::max(ddepth, sdepth));

    Mat kx, ky;
    getScharrKernels( kx, ky, dx, dy, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }

    CV_OCL_RUN(ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 &&
               (size_t)_src.rows() > ky.total() && (size_t)_src.cols() > kx.total(),
               ocl_sepFilter3x3_8UC1(_src, _dst, ddepth, kx, ky, delta, borderType));

    CV_OCL_RUN(ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 &&
               (size_t)_src.rows() > kx.total() && (size_t)_src.cols() > kx.total(),
               ocl_sepFilter2D(_src, _dst, ddepth, kx, ky, Point(-1, -1), 0, borderType))

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType & BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

    CALL_HAL(scharr, cv_hal_scharr, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, ddepth, cn,
             ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, dx, dy, scale, delta, borderType&~BORDER_ISOLATED);

    CV_IPP_RUN_FAST(ipp_Deriv(src, dst, dx, dy, 0, scale, delta, borderType));

    sepFilter2D( src, dst, ddepth, kx, ky, Point(-1, -1), delta, borderType );
}

#ifdef HAVE_OPENCL

namespace cv {

#define LAPLACIAN_LOCAL_MEM(tileX, tileY, ksize, elsize) (((tileX) + 2 * (int)((ksize) / 2)) * (3 * (tileY) + 2 * (int)((ksize) / 2)) * elsize)

static bool ocl_Laplacian5(InputArray _src, OutputArray _dst,
                           const Mat & kd, const Mat & ks, double scale, double delta,
                           int borderType, int depth, int ddepth)
{
    const size_t tileSizeX = 16;
    const size_t tileSizeYmin = 8;

    const ocl::Device dev = ocl::Device::getDefault();

    int stype = _src.type();
    int sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype), esz = CV_ELEM_SIZE(stype);

    bool doubleSupport = dev.doubleFPConfig() > 0;
    if (!doubleSupport && (sdepth == CV_64F || ddepth == CV_64F))
        return false;

    Mat kernelX = kd.reshape(1, 1);
    if (kernelX.cols % 2 != 1)
        return false;
    Mat kernelY = ks.reshape(1, 1);
    if (kernelY.cols % 2 != 1)
        return false;
    CV_Assert(kernelX.cols == kernelY.cols);

    size_t wgs = dev.maxWorkGroupSize();
    size_t lmsz = dev.localMemSize();
    size_t src_step = _src.step(), src_offset = _src.offset();
    const size_t tileSizeYmax = wgs / tileSizeX;

    // workaround for Nvidia: 3 channel vector type takes 4*elem_size in local memory
    int loc_mem_cn = dev.vendorID() == ocl::Device::VENDOR_NVIDIA && cn == 3 ? 4 : cn;

    if (((src_offset % src_step) % esz == 0) &&
        (
         (borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE) ||
         ((borderType == BORDER_REFLECT || borderType == BORDER_WRAP || borderType == BORDER_REFLECT_101) &&
          (_src.cols() >= (int) (kernelX.cols + tileSizeX) && _src.rows() >= (int) (kernelY.cols + tileSizeYmax)))
        ) &&
        (tileSizeX * tileSizeYmin <= wgs) &&
        (LAPLACIAN_LOCAL_MEM(tileSizeX, tileSizeYmin, kernelX.cols, loc_mem_cn * 4) <= lmsz)
       )
    {
        Size size = _src.size(), wholeSize;
        Point origin;
        int dtype = CV_MAKE_TYPE(ddepth, cn);
        int wdepth = CV_32F;

        size_t tileSizeY = tileSizeYmax;
        while ((tileSizeX * tileSizeY > wgs) || (LAPLACIAN_LOCAL_MEM(tileSizeX, tileSizeY, kernelX.cols, loc_mem_cn * 4) > lmsz))
        {
            tileSizeY /= 2;
        }
        size_t lt2[2] = { tileSizeX, tileSizeY};
        size_t gt2[2] = { lt2[0] * (1 + (size.width - 1) / lt2[0]), lt2[1] };

        char cvt[2][40];
        const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", "BORDER_WRAP",
                                           "BORDER_REFLECT_101" };

        String opts = cv::format("-D BLK_X=%d -D BLK_Y=%d -D RADIUS=%d%s%s"
                                 " -D convertToWT=%s -D convertToDT=%s"
                                 " -D %s -D srcT1=%s -D dstT1=%s -D WT1=%s"
                                 " -D srcT=%s -D dstT=%s -D WT=%s"
                                 " -D CN=%d ",
                                 (int)lt2[0], (int)lt2[1], kernelX.cols / 2,
                                 ocl::kernelToStr(kernelX, wdepth, "KERNEL_MATRIX_X").c_str(),
                                 ocl::kernelToStr(kernelY, wdepth, "KERNEL_MATRIX_Y").c_str(),
                                 ocl::convertTypeStr(sdepth, wdepth, cn, cvt[0]),
                                 ocl::convertTypeStr(wdepth, ddepth, cn, cvt[1]),
                                 borderMap[borderType],
                                 ocl::typeToStr(sdepth), ocl::typeToStr(ddepth), ocl::typeToStr(wdepth),
                                 ocl::typeToStr(CV_MAKETYPE(sdepth, cn)),
                                 ocl::typeToStr(CV_MAKETYPE(ddepth, cn)),
                                 ocl::typeToStr(CV_MAKETYPE(wdepth, cn)),
                                 cn);

        ocl::Kernel k("laplacian", ocl::imgproc::laplacian5_oclsrc, opts);
        if (k.empty())
            return false;
        UMat src = _src.getUMat();
        _dst.create(size, dtype);
        UMat dst = _dst.getUMat();

        int src_offset_x = static_cast<int>((src_offset % src_step) / esz);
        int src_offset_y = static_cast<int>(src_offset / src_step);

        src.locateROI(wholeSize, origin);

        k.args(ocl::KernelArg::PtrReadOnly(src), (int)src_step, src_offset_x, src_offset_y,
               wholeSize.height, wholeSize.width, ocl::KernelArg::WriteOnly(dst),
               static_cast<float>(scale), static_cast<float>(delta));

        return k.run(2, gt2, lt2, false);
    }
    int iscale = cvRound(scale), idelta = cvRound(delta);
    bool floatCoeff = std::fabs(delta - idelta) > DBL_EPSILON || std::fabs(scale - iscale) > DBL_EPSILON;
    int wdepth = std::max(depth, floatCoeff ? CV_32F : CV_32S), kercn = 1;

    if (!doubleSupport && wdepth == CV_64F)
        return false;

    char cvt[2][40];
    ocl::Kernel k("sumConvert", ocl::imgproc::laplacian5_oclsrc,
                  format("-D ONLY_SUM_CONVERT "
                         "-D srcT=%s -D WT=%s -D dstT=%s -D coeffT=%s -D wdepth=%d "
                         "-D convertToWT=%s -D convertToDT=%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(wdepth, kercn)),
                         ocl::typeToStr(CV_MAKE_TYPE(ddepth, kercn)),
                         ocl::typeToStr(wdepth), wdepth,
                         ocl::convertTypeStr(depth, wdepth, kercn, cvt[0]),
                         ocl::convertTypeStr(wdepth, ddepth, kercn, cvt[1]),
                         doubleSupport ? " -D DOUBLE_SUPPORT" : ""));
    if (k.empty())
        return false;

    UMat d2x, d2y;
    sepFilter2D(_src, d2x, depth, kd, ks, Point(-1, -1), 0, borderType);
    sepFilter2D(_src, d2y, depth, ks, kd, Point(-1, -1), 0, borderType);

    UMat dst = _dst.getUMat();

    ocl::KernelArg d2xarg = ocl::KernelArg::ReadOnlyNoSize(d2x),
            d2yarg = ocl::KernelArg::ReadOnlyNoSize(d2y),
            dstarg = ocl::KernelArg::WriteOnly(dst, cn, kercn);

    if (wdepth >= CV_32F)
        k.args(d2xarg, d2yarg, dstarg, (float)scale, (float)delta);
    else
        k.args(d2xarg, d2yarg, dstarg, iscale, idelta);

    size_t globalsize[] = { (size_t)dst.cols * cn / kercn, (size_t)dst.rows };
    return k.run(2, globalsize, NULL, false);
}

static bool ocl_Laplacian3_8UC1(InputArray _src, OutputArray _dst, int ddepth,
                                InputArray _kernel, double delta, int borderType)
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if ( !(dev.isIntel() && (type == CV_8UC1) && (ddepth == CV_8U) &&
         (borderType != BORDER_WRAP) &&
         (_src.offset() == 0) && (_src.step() % 4 == 0) &&
         (_src.cols() % 16 == 0) && (_src.rows() % 2 == 0)) )
        return false;

    Mat kernel = _kernel.getMat().reshape(1, 1);

    if (ddepth < 0)
        ddepth = sdepth;

    Size size = _src.size();
    size_t globalsize[2] = { 0, 0 };
    size_t localsize[2] = { 0, 0 };

    globalsize[0] = size.width / 16;
    globalsize[1] = size.height / 2;

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };
    char build_opts[1024];
    sprintf(build_opts, "-D %s %s", borderMap[borderType],
            ocl::kernelToStr(kernel, CV_32F, "KERNEL_MATRIX").c_str());

    ocl::Kernel k("laplacian3_8UC1_cols16_rows2", cv::ocl::imgproc::laplacian3_oclsrc, build_opts);
    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, CV_MAKETYPE(ddepth, cn));
    if (!(_dst.offset() == 0 && _dst.step() % 4 == 0))
        return false;
    UMat dst = _dst.getUMat();

    int idxArg = k.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = k.set(idxArg, (int)src.step);
    idxArg = k.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
    idxArg = k.set(idxArg, (int)dst.step);
    idxArg = k.set(idxArg, (int)dst.rows);
    idxArg = k.set(idxArg, (int)dst.cols);
    idxArg = k.set(idxArg, static_cast<float>(delta));

    return k.run(2, globalsize, (localsize[0] == 0) ? NULL : localsize, false);
}

}
#endif

#if defined(HAVE_IPP)
namespace cv
{

static bool ipp_Laplacian(InputArray _src, OutputArray _dst, int ksize, double scale, double delta, int borderType)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

    ::ipp::IwiSize size(_src.size().width, _src.size().height);
    IppDataType   srcType   = ippiGetDataType(_src.depth());
    IppDataType   dstType   = ippiGetDataType(_dst.depth());
    int           channels  = _src.channels();
    bool          useScale  = false;

    if(channels != _dst.channels() || channels > 1)
        return false;

    if(fabs(delta) > FLT_EPSILON || fabs(scale-1) > FLT_EPSILON)
        useScale = true;

    IppiMaskSize maskSize = ippiGetMaskSize(ksize, ksize);
    if(maskSize < 0)
        return false;

    // Acquire data and begin processing
    try
    {
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        ::ipp::IwiImage iwSrc      = ippiGetImage(src);
        ::ipp::IwiImage iwDst      = ippiGetImage(dst);
        ::ipp::IwiImage iwSrcProc  = iwSrc;
        ::ipp::IwiImage iwDstProc  = iwDst;
        ::ipp::IwiBorderSize  borderSize(maskSize);
        ::ipp::IwiBorderType  ippBorder(ippiGetBorder(iwSrc, borderType, borderSize));
        if(!ippBorder)
            return false;

        if(srcType == ipp8u && dstType == ipp8u)
        {
            iwDstProc.Alloc(iwDst.m_size, ipp16s, channels);
            useScale = true;
        }
        else if(srcType == ipp8u && dstType == ipp32f)
        {
            iwSrc -= borderSize;
            iwSrcProc.Alloc(iwSrc.m_size, ipp32f, channels);
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwSrc, iwSrcProc, 1, 0);
            iwSrcProc += borderSize;
        }

        CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterLaplacian, iwSrcProc, iwDstProc, maskSize, ::ipp::IwDefault(), ippBorder);

        if(useScale)
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiScale, iwDstProc, iwDst, scale, delta);

    }
    catch (::ipp::IwException ex)
    {
        return false;
    }

    return true;
#else
    CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(ksize); CV_UNUSED(scale); CV_UNUSED(delta); CV_UNUSED(borderType);
    return false;
#endif
}
}
#endif


void cv::Laplacian( InputArray _src, OutputArray _dst, int ddepth, int ksize,
                    double scale, double delta, int borderType )
{
    CV_INSTRUMENT_REGION()

    int stype = _src.type(), sdepth = CV_MAT_DEPTH(stype), cn = CV_MAT_CN(stype);
    if (ddepth < 0)
        ddepth = sdepth;
    _dst.create( _src.size(), CV_MAKETYPE(ddepth, cn) );

    if( ksize == 1 || ksize == 3 )
    {
        float K[2][9] =
        {
            { 0, 1, 0, 1, -4, 1, 0, 1, 0 },
            { 2, 0, 2, 0, -8, 0, 2, 0, 2 }
        };

        Mat kernel(3, 3, CV_32F, K[ksize == 3]);
        if( scale != 1 )
            kernel *= scale;

        CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
                   ocl_Laplacian3_8UC1(_src, _dst, ddepth, kernel, delta, borderType));
    }

    CV_IPP_RUN(!(cv::ocl::isOpenCLActivated() && _dst.isUMat()), ipp_Laplacian(_src, _dst, ksize, scale, delta, borderType));


#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::useTegra() && scale == 1.0 && delta == 0)
    {
        Mat src = _src.getMat(), dst = _dst.getMat();
        if (ksize == 1 && tegra::laplace1(src, dst, borderType))
            return;
        if (ksize == 3 && tegra::laplace3(src, dst, borderType))
            return;
        if (ksize == 5 && tegra::laplace5(src, dst, borderType))
            return;
    }
#endif

    if( ksize == 1 || ksize == 3 )
    {
        float K[2][9] =
        {
            { 0, 1, 0, 1, -4, 1, 0, 1, 0 },
            { 2, 0, 2, 0, -8, 0, 2, 0, 2 }
        };
        Mat kernel(3, 3, CV_32F, K[ksize == 3]);
        if( scale != 1 )
            kernel *= scale;

        filter2D( _src, _dst, ddepth, kernel, Point(-1, -1), delta, borderType );
    }
    else
    {
        int ktype = std::max(CV_32F, std::max(ddepth, sdepth));
        int wdepth = sdepth == CV_8U && ksize <= 5 ? CV_16S : sdepth <= CV_32F ? CV_32F : CV_64F;
        int wtype = CV_MAKETYPE(wdepth, cn);
        Mat kd, ks;
        getSobelKernels( kd, ks, 2, 0, ksize, false, ktype );

        CV_OCL_RUN(_dst.isUMat(),
                   ocl_Laplacian5(_src, _dst, kd, ks, scale,
                                  delta, borderType, wdepth, ddepth))

        Mat src = _src.getMat(), dst = _dst.getMat();
        Point ofs;
        Size wsz(src.cols, src.rows);
        if(!(borderType&BORDER_ISOLATED))
            src.locateROI( wsz, ofs );
        borderType = (borderType&~BORDER_ISOLATED);

        const size_t STRIPE_SIZE = 1 << 14;
        Ptr<FilterEngine> fx = createSeparableLinearFilter(stype,
            wtype, kd, ks, Point(-1,-1), 0, borderType, borderType, Scalar() );
        Ptr<FilterEngine> fy = createSeparableLinearFilter(stype,
            wtype, ks, kd, Point(-1,-1), 0, borderType, borderType, Scalar() );

        int y = fx->start(src, wsz, ofs), dsty = 0, dy = 0;
        fy->start(src, wsz, ofs);
        const uchar* sptr = src.ptr() + src.step[0] * y;

        int dy0 = std::min(std::max((int)(STRIPE_SIZE/(CV_ELEM_SIZE(stype)*src.cols)), 1), src.rows);
        Mat d2x( dy0 + kd.rows - 1, src.cols, wtype );
        Mat d2y( dy0 + kd.rows - 1, src.cols, wtype );

        for( ; dsty < src.rows; sptr += dy0*src.step, dsty += dy )
        {
            fx->proceed( sptr, (int)src.step, dy0, d2x.ptr(), (int)d2x.step );
            dy = fy->proceed( sptr, (int)src.step, dy0, d2y.ptr(), (int)d2y.step );
            if( dy > 0 )
            {
                Mat dstripe = dst.rowRange(dsty, dsty + dy);
                d2x.rows = d2y.rows = dy; // modify the headers, which should work
                d2x += d2y;
                d2x.convertTo( dstripe, ddepth, scale, delta );
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void
cvSobel( const void* srcarr, void* dstarr, int dx, int dy, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::Sobel( src, dst, dst.depth(), dx, dy, aperture_size, 1, 0, cv::BORDER_REPLICATE );
    if( CV_IS_IMAGE(srcarr) && ((IplImage*)srcarr)->origin && dy % 2 != 0 )
        dst *= -1;
}


CV_IMPL void
cvLaplace( const void* srcarr, void* dstarr, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::Laplacian( src, dst, dst.depth(), aperture_size, 1, 0, cv::BORDER_REPLICATE );
}

/* End of file. */
