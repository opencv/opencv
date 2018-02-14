// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "color.hpp"

namespace cv
{

//TODO: split ocl_cvtColor to different files

#ifdef HAVE_OPENCL

static bool ocl_cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    bool ok = false;
    UMat src = _src.getUMat(), dst;
    Size sz = src.size(), dstSz = sz;
    int scn = src.channels(), depth = src.depth(), bidx, uidx, yidx;
    int dims = 2;
    ocl::Kernel k;

    if (depth != CV_8U && depth != CV_16U && depth != CV_32F)
        return false;

    ocl::Device dev = ocl::Device::getDefault();
    int pxPerWIy = dev.isIntel() && (dev.type() & ocl::Device::TYPE_GPU) ? 4 : 1;
    int pxPerWIx = 1;

    size_t globalsize[] = { (size_t)src.cols, ((size_t)src.rows + pxPerWIy - 1) / pxPerWIy };
    cv::String opts = format("-D depth=%d -D scn=%d -D PIX_PER_WI_Y=%d ",
                             depth, scn, pxPerWIy);

    switch (code)
    {
    case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
    case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
    {
        dcn = code == COLOR_BGR2BGRA || code == COLOR_RGB2BGRA || code == COLOR_BGRA2RGBA ? 4 : 3;
        bool reverse = !(code == COLOR_BGR2BGRA || code == COLOR_BGRA2BGR);
        return oclCvtColorBGR2BGR(_src, _dst, dcn, reverse);
    }
    case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
    case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
    {
        dcn = code == COLOR_BGR5652BGRA || code == COLOR_BGR5552BGRA || code == COLOR_BGR5652RGBA || code == COLOR_BGR5552RGBA ? 4 : 3;
        bidx = code == COLOR_BGR5652BGR || code == COLOR_BGR5552BGR ||
            code == COLOR_BGR5652BGRA || code == COLOR_BGR5552BGRA ? 0 : 2;
        int greenbits = code == COLOR_BGR5652BGR || code == COLOR_BGR5652RGB ||
            code == COLOR_BGR5652BGRA || code == COLOR_BGR5652RGBA ? 6 : 5;
        return oclCvtColor5x52BGR(_src, _dst, dcn, bidx, greenbits);
    }
    case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
    case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
    {
        bidx = code == COLOR_BGR2BGR565 || code == COLOR_BGR2BGR555 ||
            code == COLOR_BGRA2BGR565 || code == COLOR_BGRA2BGR555 ? 0 : 2;
        int greenbits = code == COLOR_BGR2BGR565 || code == COLOR_RGB2BGR565 ||
            code == COLOR_BGRA2BGR565 || code == COLOR_RGBA2BGR565 ? 6 : 5;

        return oclCvtColorBGR25x5(_src, _dst, bidx, greenbits );
    }
    case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
    {
        int greenbits = code == COLOR_BGR5652GRAY ? 6 : 5;
        return oclCvtColor5x52Gray(_src, _dst, greenbits);
    }
    case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
    {
        int greenbits = code == COLOR_GRAY2BGR565 ? 6 : 5;
        return oclCvtColorGray25x5(_src, _dst, greenbits);
    }
    case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY:
    case COLOR_RGB2GRAY: case COLOR_RGBA2GRAY:
        bidx = code == COLOR_BGR2GRAY || code == COLOR_BGRA2GRAY ? 0 : 2;
        return oclCvtColorBGR2Gray(_src, _dst, bidx);

    case COLOR_GRAY2BGR:
    case COLOR_GRAY2BGRA:
        dcn = code == COLOR_GRAY2BGRA ? 4 : 3;
        //TODO: why no bidx?
        return oclCvtColorGray2BGR(_src, _dst, dcn);

    case COLOR_BGR2YUV:
    case COLOR_RGB2YUV:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_RGB2YUV ? 2 : 0;
        dcn = 3;
        k.create("RGB2YUV", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=3 -D bidx=%d", bidx));
        break;
    }
    case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
    {
        if(dcn <= 0) dcn = 3;
        CV_Assert(dcn == 3 || dcn == 4);
        bidx = code == COLOR_YUV2RGB ? 2 : 0;
        k.create("YUV2RGB", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d", dcn, bidx));
        break;
    }
    case COLOR_YUV2RGB_NV12: case COLOR_YUV2BGR_NV12: case COLOR_YUV2RGB_NV21: case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2RGBA_NV12: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV21:
    {
        CV_Assert( scn == 1 );
        CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
        dcn  = code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2RGBA_NV12 ||
               code == COLOR_YUV2BGRA_NV21 || code == COLOR_YUV2RGBA_NV21 ? 4 : 3;
        bidx = code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2BGR_NV12 ||
               code == COLOR_YUV2BGRA_NV21 || code == COLOR_YUV2BGR_NV21 ? 0 : 2;
        uidx = code == COLOR_YUV2RGBA_NV21 || code == COLOR_YUV2RGB_NV21 ||
               code == COLOR_YUV2BGRA_NV21 || code == COLOR_YUV2BGR_NV21 ? 1 : 0;

        dstSz = Size(sz.width, sz.height * 2 / 3);
        globalsize[0] = dstSz.width / 2; globalsize[1] = (dstSz.height/2 + pxPerWIy - 1) / pxPerWIy;
        k.create("YUV2RGB_NVx", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d -D uidx=%d", dcn, bidx, uidx));
        break;
    }
    case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
    {
        CV_Assert( scn == 1 );
        CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
        dcn  = code == COLOR_YUV2BGRA_YV12 || code == COLOR_YUV2RGBA_YV12 ||
               code == COLOR_YUV2BGRA_IYUV || code == COLOR_YUV2RGBA_IYUV ? 4 : 3;
        bidx = code == COLOR_YUV2BGRA_YV12 || code == COLOR_YUV2BGR_YV12 ||
               code == COLOR_YUV2BGRA_IYUV || code == COLOR_YUV2BGR_IYUV ? 0 : 2;
        uidx = code == COLOR_YUV2BGRA_YV12 || code == COLOR_YUV2BGR_YV12 ||
               code == COLOR_YUV2RGBA_YV12 || code == COLOR_YUV2RGB_YV12 ? 1 : 0;

        dstSz = Size(sz.width, sz.height * 2 / 3);
        globalsize[0] = dstSz.width / 2; globalsize[1] = (dstSz.height/2 + pxPerWIy - 1) / pxPerWIy;
        k.create("YUV2RGB_YV12_IYUV", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d -D uidx=%d%s", dcn, bidx, uidx,
                 src.isContinuous() ? " -D SRC_CONT" : ""));
        break;
    }
    case COLOR_YUV2GRAY_420:
    {
        if (dcn <= 0) dcn = 1;

        CV_Assert( dcn == 1 );
        CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

        dstSz = Size(sz.width, sz.height * 2 / 3);
        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();

        src.rowRange(0, dstSz.height).copyTo(dst);
        return true;
    }
    case COLOR_RGB2YUV_YV12: case COLOR_BGR2YUV_YV12: case COLOR_RGBA2YUV_YV12: case COLOR_BGRA2YUV_YV12:
    case COLOR_RGB2YUV_IYUV: case COLOR_BGR2YUV_IYUV: case COLOR_RGBA2YUV_IYUV: case COLOR_BGRA2YUV_IYUV:
    {
        if (dcn <= 0) dcn = 1;
        bidx = code == COLOR_BGRA2YUV_YV12 || code == COLOR_BGR2YUV_YV12 ||
               code == COLOR_BGRA2YUV_IYUV || code == COLOR_BGR2YUV_IYUV ? 0 : 2;
        uidx = code == COLOR_RGBA2YUV_YV12 || code == COLOR_RGB2YUV_YV12 ||
               code == COLOR_BGRA2YUV_YV12 || code == COLOR_BGR2YUV_YV12 ? 1 : 0;

        CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
        CV_Assert( dcn == 1 );
        CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0 );

        dstSz = Size(sz.width, sz.height / 2 * 3);
        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();

        if (dev.isIntel() && src.cols % 4 == 0 && src.step % 4 == 0 && src.offset % 4 == 0 &&
            dst.step % 4 == 0 && dst.offset % 4 == 0)
        {
            pxPerWIx = 2;
        }
        globalsize[0] = dstSz.width / (2 * pxPerWIx); globalsize[1] = (dstSz.height/3 + pxPerWIy - 1) / pxPerWIy;

        k.create("RGB2YUV_YV12_IYUV", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d -D uidx=%d -D PIX_PER_WI_X=%d", dcn, bidx, uidx, pxPerWIx));
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst));
        return k.run(2, globalsize, NULL, false);
    }
    case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2: case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
    {
        if (dcn <= 0)
            dcn = (code==COLOR_YUV2RGBA_UYVY || code==COLOR_YUV2BGRA_UYVY || code==COLOR_YUV2RGBA_YUY2 ||
                   code==COLOR_YUV2BGRA_YUY2 || code==COLOR_YUV2RGBA_YVYU || code==COLOR_YUV2BGRA_YVYU) ? 4 : 3;

        bidx = (code==COLOR_YUV2BGR_UYVY || code==COLOR_YUV2BGRA_UYVY || code==COLOR_YUV2BGRA_YUY2 ||
                code==COLOR_YUV2BGR_YUY2 || code==COLOR_YUV2BGRA_YVYU || code==COLOR_YUV2BGR_YVYU) ? 0 : 2;
        yidx = (code==COLOR_YUV2RGB_UYVY || code==COLOR_YUV2RGBA_UYVY || code==COLOR_YUV2BGR_UYVY || code==COLOR_YUV2BGRA_UYVY) ? 1 : 0;
        uidx = (code==COLOR_YUV2RGB_YVYU || code==COLOR_YUV2RGBA_YVYU ||
                code==COLOR_YUV2BGR_YVYU || code==COLOR_YUV2BGRA_YVYU) ? 2 : 0;
        uidx = 1 - yidx + uidx;

        CV_Assert( dcn == 3 || dcn == 4 );
        CV_Assert( scn == 2 && depth == CV_8U );

        k.create("YUV2RGB_422", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d -D uidx=%d -D yidx=%d%s", dcn, bidx, uidx, yidx,
                                src.offset % 4 == 0 && src.step % 4 == 0 ? " -D USE_OPTIMIZED_LOAD" : ""));
        break;
    }
    case COLOR_BGR2YCrCb:
    case COLOR_RGB2YCrCb:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2YCrCb ? 0 : 2;
        dcn = 3;
        k.create("RGB2YCrCb", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=3 -D bidx=%d", bidx));
        break;
    }
    case COLOR_YCrCb2BGR:
    case COLOR_YCrCb2RGB:
    {
        if( dcn <= 0 )
            dcn = 3;
        CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
        bidx = code == COLOR_YCrCb2BGR ? 0 : 2;
        k.create("YCrCb2RGB", ocl::imgproc::color_yuv_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d", dcn, bidx));
        break;
    }
    case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
    {
        bidx = code == COLOR_BGR2XYZ ? 0 : 2;

        return oclCvtColorBGR2XYZ(_src, _dst, bidx);
    }
    case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
    {
        bidx = code == COLOR_XYZ2BGR ? 0 : 2;

        return oclCvtColorXYZ2BGR(_src, _dst, dcn, bidx);
    }
    case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
    {
        bidx = code == COLOR_BGR2HSV || code == COLOR_BGR2HSV_FULL ? 0 : 2;
        bool full = !(code == COLOR_BGR2HSV || code == COLOR_RGB2HSV);
        return oclCvtColorBGR2HSV(_src, _dst, bidx, full);
    }
    case COLOR_BGR2HLS: case COLOR_RGB2HLS: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
    {
        bidx = code == COLOR_BGR2HLS || code == COLOR_BGR2HLS_FULL ? 0 : 2;
        bool full = !(code == COLOR_BGR2HLS || code == COLOR_RGB2HLS);
        return oclCvtColorBGR2HLS(_src, _dst, bidx, full);
    }
    case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
    {
        if (dcn <= 0)
            dcn = 3;
        bidx = code == COLOR_HSV2BGR || code == COLOR_HSV2BGR_FULL ? 0 : 2;
        bool full = !(code == COLOR_HSV2BGR || code == COLOR_HSV2RGB);
        return oclCvtColorHSV2BGR(_src, _dst, dcn, bidx, full);
    }
    case COLOR_HLS2BGR: case COLOR_HLS2RGB: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
    {
        if (dcn <= 0)
            dcn = 3;
        bidx = code == COLOR_HLS2BGR || code == COLOR_HLS2BGR_FULL ? 0 : 2;
        bool full = !(code == COLOR_HLS2BGR || code == COLOR_HLS2RGB);
        return oclCvtColorHLS2BGR(_src, _dst, dcn, bidx, full);
    }
    case COLOR_RGBA2mRGBA:
        return oclCvtColorRGBA2mRGBA(_src, _dst);

    case COLOR_mRGBA2RGBA:
        return oclCvtColormRGBA2RGBA(_src, _dst);

    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
    {
        bidx = code == COLOR_BGR2Lab || code == COLOR_LBGR2Lab ? 0 : 2;
        bool srgb = code == COLOR_BGR2Lab || code == COLOR_RGB2Lab ;

        return oclCvtColorBGR2Lab(_src, _dst, bidx, srgb);
    }
    case COLOR_BGR2Luv: case COLOR_RGB2Luv: case COLOR_LBGR2Luv: case COLOR_LRGB2Luv:
    {
        bidx = code == COLOR_BGR2Luv || code == COLOR_LBGR2Luv ? 0 : 2;
        bool srgb = code == COLOR_RGB2Luv || code == COLOR_BGR2Luv;

        return oclCvtColorBGR2Luv(_src, _dst, bidx, srgb);
    }
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
    {
        bidx = code == COLOR_Lab2BGR || code == COLOR_Lab2LBGR ? 0 : 2;
        bool srgb = code == COLOR_Lab2BGR || code == COLOR_Lab2RGB;

        return oclCvtColorLab2BGR(_src, _dst, dcn, bidx, srgb);
    }
    case COLOR_Luv2BGR: case COLOR_Luv2RGB: case COLOR_Luv2LBGR: case COLOR_Luv2LRGB:
    {
        bidx = code == COLOR_Luv2BGR || code == COLOR_Luv2LBGR ? 0 : 2;
        bool srgb = code == COLOR_Luv2BGR || code == COLOR_Luv2RGB;

        return oclCvtColorLuv2BGR(_src, _dst, dcn, bidx, srgb);
    }
    default:
        break;
    }

    if( !k.empty() )
    {
        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst));
        ok = k.run(dims, globalsize, NULL, false);
    }
    return ok;
}

#endif


// helper function for dual-plane modes
void cv::cvtColorTwoPlane( InputArray _ysrc, InputArray _uvsrc, OutputArray _dst, int code )
{
    // only YUV420 is currently supported
    switch (code) {
        case COLOR_YUV2BGR_NV21:  case COLOR_YUV2RGB_NV21:  case COLOR_YUV2BGR_NV12:  case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
            return;
    }

    int stype = _ysrc.type();
    int depth = CV_MAT_DEPTH(stype), uidx, dcn;

    Mat ysrc, uvsrc, dst;
    ysrc = _ysrc.getMat();
    uvsrc = _uvsrc.getMat();
    Size ysz = _ysrc.size();
    Size uvs = _uvsrc.size();

    // http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
    // http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples
    dcn = (code==COLOR_YUV420sp2BGRA || code==COLOR_YUV420sp2RGBA || code==COLOR_YUV2BGRA_NV12 || code==COLOR_YUV2RGBA_NV12) ? 4 : 3;
    uidx = (code==COLOR_YUV2BGR_NV21 || code==COLOR_YUV2BGRA_NV21 || code==COLOR_YUV2RGB_NV21 || code==COLOR_YUV2RGBA_NV21) ? 1 : 0;
    CV_Assert( dcn == 3 || dcn == 4 );
    CV_Assert( ysz.width == uvs.width * 2 );
    CV_Assert( ysz.width % 2 == 0 && depth == CV_8U );
    CV_Assert( ysz.height == uvs.height * 2 );
    _dst.create( ysz, CV_MAKETYPE(depth, dcn));
    dst = _dst.getMat();
    hal::cvtTwoPlaneYUVtoBGR(ysrc.data, uvsrc.data, ysrc.step, dst.data, dst.step, dst.cols, dst.rows, dcn, swapBlue(code), uidx);
}


//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  //
//////////////////////////////////////////////////////////////////////////////////////////

void cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN( _src.dims() <= 2 && _dst.isUMat() && !(CV_MAT_DEPTH(_src.type()) == CV_8U && (code == COLOR_Luv2BGR || code == COLOR_Luv2RGB)),
                ocl_cvtColor(_src, _dst, code, dcn) )

    if(dcn <= 0)
        dcn = dstChannels(code);

    switch( code )
    {
        case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
        case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
            cvtColorBGR2BGR(_src, _dst, dcn, swapBlue(code));
            break;

        case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
        case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
            cvtColorBGR25x5(_src, _dst, swapBlue(code), greenBits(code));
            break;

        case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
        case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
            cvtColor5x52BGR(_src, _dst, dcn, swapBlue(code), greenBits(code));
            break;

        case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY: case COLOR_RGB2GRAY: case COLOR_RGBA2GRAY:
            cvtColorBGR2Gray(_src, _dst, swapBlue(code));
            break;

        case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
            cvtColor5x52Gray(_src, _dst, greenBits(code));
            break;

        case COLOR_GRAY2BGR: case COLOR_GRAY2BGRA:
            cvtColorGray2BGR(_src, _dst, dcn);
            break;

        case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
            cvtColorGray25x5(_src, _dst, greenBits(code));
            break;

        case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
        case COLOR_BGR2YUV: case COLOR_RGB2YUV:
            cvtColorBGR2YUV(_src, _dst, swapBlue(code), code == COLOR_BGR2YCrCb || code == COLOR_RGB2YCrCb);
            break;

        case COLOR_YCrCb2BGR: case COLOR_YCrCb2RGB:
        case COLOR_YUV2BGR: case COLOR_YUV2RGB:
            cvtColorYUV2BGR(_src, _dst, dcn, swapBlue(code), code == COLOR_YCrCb2BGR || code == COLOR_YCrCb2RGB);
            break;

        case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
            cvtColorBGR2XYZ(_src, _dst, swapBlue(code));
            break;

        case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
            cvtColorXYZ2BGR(_src, _dst, dcn, swapBlue(code));
            break;

        case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
            cvtColorBGR2HSV(_src, _dst, swapBlue(code), isFullRange(code));
            break;

        case COLOR_BGR2HLS: case COLOR_RGB2HLS: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
            cvtColorBGR2HLS(_src, _dst, swapBlue(code), isFullRange(code));
            break;

        case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
            cvtColorHSV2BGR(_src, _dst, dcn, swapBlue(code), isFullRange(code));
            break;

        case COLOR_HLS2BGR: case COLOR_HLS2RGB: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
            cvtColorHLS2BGR(_src, _dst, dcn, swapBlue(code), isFullRange(code));
            break;

        case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
            cvtColorBGR2Lab(_src, _dst, swapBlue(code), issRGB(code));
            break;
        case COLOR_BGR2Luv: case COLOR_RGB2Luv: case COLOR_LBGR2Luv: case COLOR_LRGB2Luv:
            cvtColorBGR2Luv(_src, _dst, swapBlue(code), issRGB(code));
            break;

        case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
            cvtColorLab2BGR(_src, _dst, dcn, swapBlue(code), issRGB(code));
            break;
        case COLOR_Luv2BGR: case COLOR_Luv2RGB: case COLOR_Luv2LBGR: case COLOR_Luv2LRGB:
            cvtColorLuv2BGR(_src, _dst, dcn, swapBlue(code), issRGB(code));
            break;

        case COLOR_BayerBG2GRAY: case COLOR_BayerGB2GRAY: case COLOR_BayerRG2GRAY: case COLOR_BayerGR2GRAY:
        case COLOR_BayerBG2BGR: case COLOR_BayerGB2BGR: case COLOR_BayerRG2BGR: case COLOR_BayerGR2BGR:
        case COLOR_BayerBG2BGR_VNG: case COLOR_BayerGB2BGR_VNG: case COLOR_BayerRG2BGR_VNG: case COLOR_BayerGR2BGR_VNG:
        case COLOR_BayerBG2BGR_EA: case COLOR_BayerGB2BGR_EA: case COLOR_BayerRG2BGR_EA: case COLOR_BayerGR2BGR_EA:
        case COLOR_BayerBG2BGRA: case COLOR_BayerGB2BGRA: case COLOR_BayerRG2BGRA: case COLOR_BayerGR2BGRA:
            {
                Mat src;
                if (_src.getObj() == _dst.getObj()) // inplace processing (#6653)
                    _src.copyTo(src);
                else
                    src = _src.getMat();
                demosaicing(src, _dst, code, dcn);
                break;
            }

        case COLOR_YUV2BGR_NV21:  case COLOR_YUV2RGB_NV21:  case COLOR_YUV2BGR_NV12:  case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
            // http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
            // http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples
            cvtColorTwoPlaneYUV2BGR(_src, _dst, dcn, swapBlue(code), uIndex(code));
            break;

        case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
        case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
            //http://www.fourcc.org/yuv.php#YV12 == yuv420p -> It comprises an NxM Y plane followed by (N/2)x(M/2) V and U planes.
            //http://www.fourcc.org/yuv.php#IYUV == I420 -> It comprises an NxN Y plane followed by (N/2)x(N/2) U and V planes
            cvtColorThreePlaneYUV2BGR(_src, _dst, dcn, swapBlue(code), uIndex(code));
            break;

        case COLOR_YUV2GRAY_420:
            cvtColorYUV2Gray_420(_src, _dst);
            break;

        case COLOR_RGB2YUV_YV12: case COLOR_BGR2YUV_YV12: case COLOR_RGBA2YUV_YV12: case COLOR_BGRA2YUV_YV12:
        case COLOR_RGB2YUV_IYUV: case COLOR_BGR2YUV_IYUV: case COLOR_RGBA2YUV_IYUV: case COLOR_BGRA2YUV_IYUV:
            cvtColorBGR2ThreePlaneYUV(_src, _dst, swapBlue(code), uIndex(code));
            break;

        case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
        case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2: case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
        case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
            //http://www.fourcc.org/yuv.php#UYVY
            //http://www.fourcc.org/yuv.php#YUY2
            //http://www.fourcc.org/yuv.php#YVYU
            {
                int ycn  = (code==COLOR_YUV2RGB_UYVY || code==COLOR_YUV2BGR_UYVY || code==COLOR_YUV2RGBA_UYVY || code==COLOR_YUV2BGRA_UYVY) ? 1 : 0;
                cvtColorOnePlaneYUV2BGR(_src, _dst, dcn, swapBlue(code), uIndex(code), ycn);
                break;
            }

        case COLOR_YUV2GRAY_UYVY: case COLOR_YUV2GRAY_YUY2:
            cvtColorYUV2Gray_ch(_src, _dst, code == COLOR_YUV2GRAY_UYVY ? 1 : 0);
            break;

        case COLOR_RGBA2mRGBA:
            cvtColorRGBA2mRGBA(_src, _dst);
            break;

        case COLOR_mRGBA2RGBA:
            cvtColormRGBA2RGBA(_src, _dst);
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
    }
}
} //namespace cv


CV_IMPL void
cvCvtColor( const CvArr* srcarr, CvArr* dstarr, int code )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0;
    CV_Assert( src.depth() == dst.depth() );

    cv::cvtColor(src, dst, code, dst.channels());
    CV_Assert( dst.data == dst0.data );
}
