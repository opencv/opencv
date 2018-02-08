// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "color.hpp"

namespace cv
{

//
// Helper functions
//

namespace {
using namespace cv;

inline bool isHSV(int code)
{
    switch(code)
    {
    case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
    case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
        return true;
    default:
        return false;
    }
}

inline bool isLab(int code)
{
    switch (code)
    {
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
        return true;
    default:
        return false;
    }
}

inline bool issRGB(int code)
{
    switch (code)
    {
    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_BGR2Luv: case COLOR_RGB2Luv:
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Luv2BGR: case COLOR_Luv2RGB:
        return true;
    default:
        return false;
    }
}

inline bool swapBlue(int code)
{
    switch (code)
    {
    case COLOR_BGR2BGRA: case COLOR_BGRA2BGR:
    case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555:
    case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA:
    case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY:
    case COLOR_BGR2YCrCb: case COLOR_BGR2YUV:
    case COLOR_YCrCb2BGR: case COLOR_YUV2BGR:
    case COLOR_BGR2XYZ: case COLOR_XYZ2BGR:
    case COLOR_BGR2HSV: case COLOR_BGR2HLS: case COLOR_BGR2HSV_FULL: case COLOR_BGR2HLS_FULL:
    case COLOR_YUV2BGR_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2BGR_IYUV: case COLOR_YUV2BGRA_IYUV:
    case COLOR_YUV2BGR_NV21: case COLOR_YUV2BGRA_NV21: case COLOR_YUV2BGR_NV12: case COLOR_YUV2BGRA_NV12:
    case COLOR_Lab2BGR: case COLOR_Luv2BGR: case COLOR_Lab2LBGR: case COLOR_Luv2LBGR:
    case COLOR_BGR2Lab: case COLOR_BGR2Luv: case COLOR_LBGR2Lab: case COLOR_LBGR2Luv:
    case COLOR_HSV2BGR: case COLOR_HLS2BGR: case COLOR_HSV2BGR_FULL: case COLOR_HLS2BGR_FULL:
    case COLOR_YUV2BGR_UYVY: case COLOR_YUV2BGRA_UYVY: case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2BGRA_YUY2:  case COLOR_YUV2BGR_YVYU: case COLOR_YUV2BGRA_YVYU:
    case COLOR_BGR2YUV_IYUV: case COLOR_BGRA2YUV_IYUV: case COLOR_BGR2YUV_YV12: case COLOR_BGRA2YUV_YV12:
        return false;
    default:
        return true;
    }
}

inline bool isFullRange(int code)
{
    switch (code)
    {
    case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
    case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
        return true;
    default:
        return false;
    }
}

} // namespace::


//TODO: split ocl_cvtColor to different files

#ifdef HAVE_OPENCL

static bool ocl_cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    bool ok = false;
    UMat src = _src.getUMat(), dst;
    Size sz = src.size(), dstSz = sz;
    int scn = src.channels(), depth = src.depth(), bidx, uidx, yidx;
    int dims = 2, stripeSize = 1;
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
        CV_Assert(scn == 3 || scn == 4);
        dcn = code == COLOR_BGR2BGRA || code == COLOR_RGB2BGRA || code == COLOR_BGRA2RGBA ? 4 : 3;
        bool reverse = !(code == COLOR_BGR2BGRA || code == COLOR_BGRA2BGR);
        k.create("RGB", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=%d -D bidx=0 -D %s", dcn,
                        reverse ? "REVERSE" : "ORDER"));
        break;
    }
    case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
    case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
    {
        dcn = code == COLOR_BGR5652BGRA || code == COLOR_BGR5552BGRA || code == COLOR_BGR5652RGBA || code == COLOR_BGR5552RGBA ? 4 : 3;
        CV_Assert((dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U);
        bidx = code == COLOR_BGR5652BGR || code == COLOR_BGR5552BGR ||
            code == COLOR_BGR5652BGRA || code == COLOR_BGR5552BGRA ? 0 : 2;
        int greenbits = code == COLOR_BGR5652BGR || code == COLOR_BGR5652RGB ||
            code == COLOR_BGR5652BGRA || code == COLOR_BGR5652RGBA ? 6 : 5;
        k.create("RGB5x52RGB", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d -D greenbits=%d", dcn, bidx, greenbits));
        break;
    }
    case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
    case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
    {
        CV_Assert((scn == 3 || scn == 4) && depth == CV_8U );
        bidx = code == COLOR_BGR2BGR565 || code == COLOR_BGR2BGR555 ||
            code == COLOR_BGRA2BGR565 || code == COLOR_BGRA2BGR555 ? 0 : 2;
        int greenbits = code == COLOR_BGR2BGR565 || code == COLOR_RGB2BGR565 ||
            code == COLOR_BGRA2BGR565 || code == COLOR_RGBA2BGR565 ? 6 : 5;
        dcn = 2;
        k.create("RGB2RGB5x5", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=2 -D bidx=%d -D greenbits=%d", bidx, greenbits));
        break;
    }
    case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
    {
        CV_Assert(scn == 2 && depth == CV_8U);
        dcn = 1;
        int greenbits = code == COLOR_BGR5652GRAY ? 6 : 5;
        k.create("BGR5x52Gray", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=1 -D bidx=0 -D greenbits=%d", greenbits));
        break;
    }
    case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
    {
        CV_Assert(scn == 1 && depth == CV_8U);
        dcn = 2;
        int greenbits = code == COLOR_GRAY2BGR565 ? 6 : 5;
        k.create("Gray2BGR5x5", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=2 -D bidx=0 -D greenbits=%d", greenbits));
        break;
    }
    case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY:
    case COLOR_RGB2GRAY: case COLOR_RGBA2GRAY:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2GRAY || code == COLOR_BGRA2GRAY ? 0 : 2;
        dcn = 1;
        k.create("RGB2Gray", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=1 -D bidx=%d -D STRIPE_SIZE=%d",
                               bidx, stripeSize));
        globalsize[0] = (src.cols + stripeSize-1)/stripeSize;
        break;
    }
    case COLOR_GRAY2BGR:
    case COLOR_GRAY2BGRA:
    {
        CV_Assert(scn == 1);
        dcn = code == COLOR_GRAY2BGRA ? 4 : 3;
        k.create("Gray2RGB", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D bidx=0 -D dcn=%d", dcn));
        break;
    }
    case COLOR_BGR2YUV:
    case COLOR_RGB2YUV:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_RGB2YUV ? 2 : 0;
        dcn = 3;
        k.create("RGB2YUV", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=3 -D bidx=%d", bidx));
        break;
    }
    case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
    {
        if(dcn <= 0) dcn = 3;
        CV_Assert(dcn == 3 || dcn == 4);
        bidx = code == COLOR_YUV2RGB ? 2 : 0;
        k.create("YUV2RGB", ocl::imgproc::cvtcolor_oclsrc,
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
        k.create("YUV2RGB_NVx", ocl::imgproc::cvtcolor_oclsrc,
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
        k.create("YUV2RGB_YV12_IYUV", ocl::imgproc::cvtcolor_oclsrc,
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

        k.create("RGB2YUV_YV12_IYUV", ocl::imgproc::cvtcolor_oclsrc,
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

        k.create("YUV2RGB_422", ocl::imgproc::cvtcolor_oclsrc,
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
        k.create("RGB2YCrCb", ocl::imgproc::cvtcolor_oclsrc,
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
        k.create("YCrCb2RGB", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d", dcn, bidx));
        break;
    }
    case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2XYZ ? 0 : 2;

        UMat c;
        if (depth == CV_32F)
        {
            float coeffs[] =
            {
                0.412453f, 0.357580f, 0.180423f,
                0.212671f, 0.715160f, 0.072169f,
                0.019334f, 0.119193f, 0.950227f
            };
            if (bidx == 0)
            {
                std::swap(coeffs[0], coeffs[2]);
                std::swap(coeffs[3], coeffs[5]);
                std::swap(coeffs[6], coeffs[8]);
            }
            Mat(1, 9, CV_32FC1, &coeffs[0]).copyTo(c);
        }
        else
        {
            int coeffs[] =
            {
                1689,    1465,    739,
                871,     2929,    296,
                79,      488,     3892
            };
            if (bidx == 0)
            {
                std::swap(coeffs[0], coeffs[2]);
                std::swap(coeffs[3], coeffs[5]);
                std::swap(coeffs[6], coeffs[8]);
            }
            Mat(1, 9, CV_32SC1, &coeffs[0]).copyTo(c);
        }

        _dst.create(dstSz, CV_MAKETYPE(depth, 3));
        dst = _dst.getUMat();

        k.create("RGB2XYZ", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=3 -D bidx=%d", bidx));
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst), ocl::KernelArg::PtrReadOnly(c));
        return k.run(2, globalsize, 0, false);
    }
    case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
    {
        if (dcn <= 0)
            dcn = 3;
        CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
        bidx = code == COLOR_XYZ2BGR ? 0 : 2;

        UMat c;
        if (depth == CV_32F)
        {
            float coeffs[] =
            {
                3.240479f, -1.53715f, -0.498535f,
                -0.969256f, 1.875991f, 0.041556f,
                0.055648f, -0.204043f, 1.057311f
            };
            if (bidx == 0)
            {
                std::swap(coeffs[0], coeffs[6]);
                std::swap(coeffs[1], coeffs[7]);
                std::swap(coeffs[2], coeffs[8]);
            }
            Mat(1, 9, CV_32FC1, &coeffs[0]).copyTo(c);
        }
        else
        {
            int coeffs[] =
            {
                13273,  -6296,  -2042,
                -3970,   7684,    170,
                  228,   -836,   4331
            };
            if (bidx == 0)
            {
                std::swap(coeffs[0], coeffs[6]);
                std::swap(coeffs[1], coeffs[7]);
                std::swap(coeffs[2], coeffs[8]);
            }
            Mat(1, 9, CV_32SC1, &coeffs[0]).copyTo(c);
        }

        _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        dst = _dst.getUMat();

        k.create("XYZ2RGB", ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d", dcn, bidx));
        if (k.empty())
            return false;
        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst), ocl::KernelArg::PtrReadOnly(c));
        return k.run(2, globalsize, 0, false);
    }
    case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
    case COLOR_BGR2HLS: case COLOR_RGB2HLS: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
    {
        CV_Assert((scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F));
        bidx = code == COLOR_BGR2HSV || code == COLOR_BGR2HLS ||
            code == COLOR_BGR2HSV_FULL || code == COLOR_BGR2HLS_FULL ? 0 : 2;
        int hrange = depth == CV_32F ? 360 : code == COLOR_BGR2HSV || code == COLOR_RGB2HSV ||
            code == COLOR_BGR2HLS || code == COLOR_RGB2HLS ? 180 : 256;
        bool is_hsv = code == COLOR_BGR2HSV || code == COLOR_RGB2HSV || code == COLOR_BGR2HSV_FULL || code == COLOR_RGB2HSV_FULL;
        String kernelName = String("RGB2") + (is_hsv ? "HSV" : "HLS");
        dcn = 3;

        if (is_hsv && depth == CV_8U)
        {
            static UMat sdiv_data;
            static UMat hdiv_data180;
            static UMat hdiv_data256;
            static int sdiv_table[256];
            static int hdiv_table180[256];
            static int hdiv_table256[256];
            static volatile bool initialized180 = false, initialized256 = false;
            volatile bool & initialized = hrange == 180 ? initialized180 : initialized256;

            if (!initialized)
            {
                int * const hdiv_table = hrange == 180 ? hdiv_table180 : hdiv_table256, hsv_shift = 12;
                UMat & hdiv_data = hrange == 180 ? hdiv_data180 : hdiv_data256;

                sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;

                int v = 255 << hsv_shift;
                if (!initialized180 && !initialized256)
                {
                    for(int i = 1; i < 256; i++ )
                        sdiv_table[i] = saturate_cast<int>(v/(1.*i));
                    Mat(1, 256, CV_32SC1, sdiv_table).copyTo(sdiv_data);
                }

                v = hrange << hsv_shift;
                for (int i = 1; i < 256; i++ )
                    hdiv_table[i] = saturate_cast<int>(v/(6.*i));

                Mat(1, 256, CV_32SC1, hdiv_table).copyTo(hdiv_data);
                initialized = true;
            }

            _dst.create(dstSz, CV_8UC3);
            dst = _dst.getUMat();

            k.create("RGB2HSV", ocl::imgproc::cvtcolor_oclsrc,
                     opts + format("-D hrange=%d -D bidx=%d -D dcn=3",
                                   hrange, bidx));
            if (k.empty())
                return false;

            k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst),
                   ocl::KernelArg::PtrReadOnly(sdiv_data), hrange == 256 ? ocl::KernelArg::PtrReadOnly(hdiv_data256) :
                                                                       ocl::KernelArg::PtrReadOnly(hdiv_data180));

            return k.run(2, globalsize, NULL, false);
        }
        else
            k.create(kernelName.c_str(), ocl::imgproc::cvtcolor_oclsrc,
                     opts + format("-D hscale=%ff -D bidx=%d -D dcn=3",
                                   hrange*(1.f/360.f), bidx));
        break;
    }
    case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
    case COLOR_HLS2BGR: case COLOR_HLS2RGB: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
    {
        if (dcn <= 0)
            dcn = 3;
        CV_Assert(scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F));
        bidx = code == COLOR_HSV2BGR || code == COLOR_HLS2BGR ||
            code == COLOR_HSV2BGR_FULL || code == COLOR_HLS2BGR_FULL ? 0 : 2;
        int hrange = depth == CV_32F ? 360 : code == COLOR_HSV2BGR || code == COLOR_HSV2RGB ||
            code == COLOR_HLS2BGR || code == COLOR_HLS2RGB ? 180 : 255;
        bool is_hsv = code == COLOR_HSV2BGR || code == COLOR_HSV2RGB ||
                code == COLOR_HSV2BGR_FULL || code == COLOR_HSV2RGB_FULL;

        String kernelName = String(is_hsv ? "HSV" : "HLS") + "2RGB";
        k.create(kernelName.c_str(), ocl::imgproc::cvtcolor_oclsrc,
                 opts + format("-D dcn=%d -D bidx=%d -D hrange=%d -D hscale=%ff",
                               dcn, bidx, hrange, 6.f/hrange));
        break;
    }
    case COLOR_RGBA2mRGBA: case COLOR_mRGBA2RGBA:
    {
        CV_Assert(scn == 4 && depth == CV_8U);
        dcn = 4;

        k.create(code == COLOR_RGBA2mRGBA ? "RGBA2mRGBA" : "mRGBA2RGBA", ocl::imgproc::cvtcolor_oclsrc,
                 opts + "-D dcn=4 -D bidx=3");
        break;
    }
    case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
    case COLOR_BGR2Luv: case COLOR_RGB2Luv: case COLOR_LBGR2Luv: case COLOR_LRGB2Luv:
    {
        return oclCvtColorBGR2Lxx(scn, dcn, depth, bidx, code, k,
                                  opts, src, dst, _dst, sz, dstSz,
                                  globalsize);
    }
    case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
    case COLOR_Luv2BGR: case COLOR_Luv2RGB: case COLOR_Luv2LBGR: case COLOR_Luv2LRGB:
    {
        return oclCvtColorLxx2BGR(scn, dcn, depth, bidx, code, k,
                                  opts, src, dst, _dst, sz, dstSz,
                                  globalsize);
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

    int stype = _src.type();
    int scn = CV_MAT_CN(stype), depth = CV_MAT_DEPTH(stype), uidx, gbits, ycn;

    CV_OCL_RUN( _src.dims() <= 2 && _dst.isUMat() && !(depth == CV_8U && (code == COLOR_Luv2BGR || code == COLOR_Luv2RGB)),
                ocl_cvtColor(_src, _dst, code, dcn) )

    Mat src, dst;
    if (_src.getObj() == _dst.getObj()) // inplace processing (#6653)
        _src.copyTo(src);
    else
        src = _src.getMat();
    Size sz = src.size();
    CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_32F );

    switch( code )
    {
        case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
        case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
            CV_Assert( scn == 3 || scn == 4 );
            dcn = code == COLOR_BGR2BGRA || code == COLOR_RGB2BGRA || code == COLOR_BGRA2RGBA ? 4 : 3;
            _dst.create( sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtBGRtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, dcn, swapBlue(code));
            break;

        case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
        case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            gbits = code == COLOR_BGR2BGR565 || code == COLOR_RGB2BGR565 ||
                    code == COLOR_BGRA2BGR565 || code == COLOR_RGBA2BGR565 ? 6 : 5;
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();
            hal::cvtBGRtoBGR5x5(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                scn, swapBlue(code), gbits);
            break;

        case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
        case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
            if(dcn <= 0) dcn = (code==COLOR_BGR5652BGRA || code==COLOR_BGR5552BGRA || code==COLOR_BGR5652RGBA || code==COLOR_BGR5552RGBA) ? 4 : 3;
            CV_Assert( (dcn == 3 || dcn == 4) && scn == 2 && depth == CV_8U );
            gbits = code == COLOR_BGR5652BGR || code == COLOR_BGR5652RGB ||
                    code == COLOR_BGR5652BGRA || code == COLOR_BGR5652RGBA ? 6 : 5;
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtBGR5x5toBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                dcn, swapBlue(code), gbits);
            break;

        case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY: case COLOR_RGB2GRAY: case COLOR_RGBA2GRAY:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 1));
            dst = _dst.getMat();
            hal::cvtBGRtoGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                              depth, scn, swapBlue(code));
            break;

        case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
            CV_Assert( scn == 2 && depth == CV_8U );
            gbits = code == COLOR_BGR5652GRAY ? 6 : 5;
            _dst.create(sz, CV_8UC1);
            dst = _dst.getMat();
            hal::cvtBGR5x5toGray(src.data, src.step, dst.data, dst.step, src.cols, src.rows, gbits);
            break;

        case COLOR_GRAY2BGR: case COLOR_GRAY2BGRA:
            if( dcn <= 0 ) dcn = (code==COLOR_GRAY2BGRA) ? 4 : 3;
            CV_Assert( scn == 1 && (dcn == 3 || dcn == 4));
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtGraytoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn);
            break;

        case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
            CV_Assert( scn == 1 && depth == CV_8U );
            gbits = code == COLOR_GRAY2BGR565 ? 6 : 5;
            _dst.create(sz, CV_8UC2);
            dst = _dst.getMat();
            hal::cvtGraytoBGR5x5(src.data, src.step, dst.data, dst.step, src.cols, src.rows, gbits);
            break;

        case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
        case COLOR_BGR2YUV: case COLOR_RGB2YUV:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoYUV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, swapBlue(code), code == COLOR_BGR2YCrCb || code == COLOR_RGB2YCrCb);
            break;

        case COLOR_YCrCb2BGR: case COLOR_YCrCb2RGB:
        case COLOR_YUV2BGR: case COLOR_YUV2RGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, dcn, swapBlue(code), code == COLOR_YCrCb2BGR || code == COLOR_YCrCb2RGB);
            break;

        case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
            CV_Assert( scn == 3 || scn == 4 );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoXYZ(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, scn, swapBlue(code));
            break;

        case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtXYZtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows, depth, dcn, swapBlue(code));
            break;

        case COLOR_BGR2HSV: case COLOR_RGB2HSV: case COLOR_BGR2HSV_FULL: case COLOR_RGB2HSV_FULL:
        case COLOR_BGR2HLS: case COLOR_RGB2HLS: case COLOR_BGR2HLS_FULL: case COLOR_RGB2HLS_FULL:
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoHSV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, swapBlue(code), isFullRange(code), isHSV(code));
            break;

        case COLOR_HSV2BGR: case COLOR_HSV2RGB: case COLOR_HSV2BGR_FULL: case COLOR_HSV2RGB_FULL:
        case COLOR_HLS2BGR: case COLOR_HLS2RGB: case COLOR_HLS2BGR_FULL: case COLOR_HLS2RGB_FULL:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtHSVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, dcn, swapBlue(code), isFullRange(code), isHSV(code));
            break;

        case COLOR_BGR2Lab: case COLOR_RGB2Lab: case COLOR_LBGR2Lab: case COLOR_LRGB2Lab:
        case COLOR_BGR2Luv: case COLOR_RGB2Luv: case COLOR_LBGR2Luv: case COLOR_LRGB2Luv:
            CV_Assert( (scn == 3 || scn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, 3));
            dst = _dst.getMat();
            hal::cvtBGRtoLab(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, scn, swapBlue(code), isLab(code), issRGB(code));
            break;

        case COLOR_Lab2BGR: case COLOR_Lab2RGB: case COLOR_Lab2LBGR: case COLOR_Lab2LRGB:
        case COLOR_Luv2BGR: case COLOR_Luv2RGB: case COLOR_Luv2LBGR: case COLOR_Luv2LRGB:
            if( dcn <= 0 ) dcn = 3;
            CV_Assert( scn == 3 && (dcn == 3 || dcn == 4) && (depth == CV_8U || depth == CV_32F) );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtLabtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                             depth, dcn, swapBlue(code), isLab(code), issRGB(code));
            break;

        case COLOR_BayerBG2GRAY: case COLOR_BayerGB2GRAY: case COLOR_BayerRG2GRAY: case COLOR_BayerGR2GRAY:
        case COLOR_BayerBG2BGR: case COLOR_BayerGB2BGR: case COLOR_BayerRG2BGR: case COLOR_BayerGR2BGR:
        case COLOR_BayerBG2BGR_VNG: case COLOR_BayerGB2BGR_VNG: case COLOR_BayerRG2BGR_VNG: case COLOR_BayerGR2BGR_VNG:
        case COLOR_BayerBG2BGR_EA: case COLOR_BayerGB2BGR_EA: case COLOR_BayerRG2BGR_EA: case COLOR_BayerGR2BGR_EA:
        case COLOR_BayerBG2BGRA: case COLOR_BayerGB2BGRA: case COLOR_BayerRG2BGRA: case COLOR_BayerGR2BGRA:
            demosaicing(src, _dst, code, dcn);
            break;

        case COLOR_YUV2BGR_NV21:  case COLOR_YUV2RGB_NV21:  case COLOR_YUV2BGR_NV12:  case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
            // http://www.fourcc.org/yuv.php#NV21 == yuv420sp -> a plane of 8 bit Y samples followed by an interleaved V/U plane containing 8 bit 2x2 subsampled chroma samples
            // http://www.fourcc.org/yuv.php#NV12 -> a plane of 8 bit Y samples followed by an interleaved U/V plane containing 8 bit 2x2 subsampled colour difference samples
            if (dcn <= 0) dcn = (code==COLOR_YUV420sp2BGRA || code==COLOR_YUV420sp2RGBA || code==COLOR_YUV2BGRA_NV12 || code==COLOR_YUV2RGBA_NV12) ? 4 : 3;
            uidx = (code==COLOR_YUV2BGR_NV21 || code==COLOR_YUV2BGRA_NV21 || code==COLOR_YUV2RGB_NV21 || code==COLOR_YUV2RGBA_NV21) ? 1 : 0;
            CV_Assert( dcn == 3 || dcn == 4 );
            CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
            _dst.create(Size(sz.width, sz.height * 2 / 3), CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtTwoPlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, dst.cols, dst.rows,
                                     dcn, swapBlue(code), uidx);
            break;
        case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
        case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
            //http://www.fourcc.org/yuv.php#YV12 == yuv420p -> It comprises an NxM Y plane followed by (N/2)x(M/2) V and U planes.
            //http://www.fourcc.org/yuv.php#IYUV == I420 -> It comprises an NxN Y plane followed by (N/2)x(N/2) U and V planes
            if (dcn <= 0) dcn = (code==COLOR_YUV2BGRA_YV12 || code==COLOR_YUV2RGBA_YV12 || code==COLOR_YUV2RGBA_IYUV || code==COLOR_YUV2BGRA_IYUV) ? 4 : 3;
            uidx  = (code==COLOR_YUV2BGR_YV12 || code==COLOR_YUV2RGB_YV12 || code==COLOR_YUV2BGRA_YV12 || code==COLOR_YUV2RGBA_YV12) ? 1 : 0;
            CV_Assert( dcn == 3 || dcn == 4 );
            CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
            _dst.create(Size(sz.width, sz.height * 2 / 3), CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtThreePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, dst.cols, dst.rows,
                                       dcn, swapBlue(code), uidx);
            break;

        case COLOR_YUV2GRAY_420:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );

                Size dstSz(sz.width, sz.height * 2 / 3);
                _dst.create(dstSz, CV_MAKETYPE(depth, dcn));
                dst = _dst.getMat();
#ifdef HAVE_IPP
#if IPP_VERSION_X100 >= 201700
                if (CV_INSTRUMENT_FUN_IPP(ippiCopy_8u_C1R_L, src.data, (IppSizeL)src.step, dst.data, (IppSizeL)dst.step,
                                                   ippiSizeL(dstSz.width, dstSz.height)) >= 0)
                    break;
#endif
#endif
                src(Range(0, dstSz.height), Range::all()).copyTo(dst);
            }
            break;
        case COLOR_RGB2YUV_YV12: case COLOR_BGR2YUV_YV12: case COLOR_RGBA2YUV_YV12: case COLOR_BGRA2YUV_YV12:
        case COLOR_RGB2YUV_IYUV: case COLOR_BGR2YUV_IYUV: case COLOR_RGBA2YUV_IYUV: case COLOR_BGRA2YUV_IYUV:
            if (dcn <= 0) dcn = 1;
            uidx = (code == COLOR_BGR2YUV_IYUV || code == COLOR_BGRA2YUV_IYUV || code == COLOR_RGB2YUV_IYUV || code == COLOR_RGBA2YUV_IYUV) ? 1 : 2;
            CV_Assert( (scn == 3 || scn == 4) && depth == CV_8U );
            CV_Assert( dcn == 1 );
            CV_Assert( sz.width % 2 == 0 && sz.height % 2 == 0 );
            _dst.create(Size(sz.width, sz.height / 2 * 3), CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtBGRtoThreePlaneYUV(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                       scn, swapBlue(code), uidx);
            break;
        case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
        case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2: case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
        case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
            //http://www.fourcc.org/yuv.php#UYVY
            //http://www.fourcc.org/yuv.php#YUY2
            //http://www.fourcc.org/yuv.php#YVYU
            if (dcn <= 0) dcn = (code==COLOR_YUV2RGBA_UYVY || code==COLOR_YUV2BGRA_UYVY || code==COLOR_YUV2RGBA_YUY2 || code==COLOR_YUV2BGRA_YUY2 || code==COLOR_YUV2RGBA_YVYU || code==COLOR_YUV2BGRA_YVYU) ? 4 : 3;
            ycn  = (code==COLOR_YUV2RGB_UYVY || code==COLOR_YUV2BGR_UYVY || code==COLOR_YUV2RGBA_UYVY || code==COLOR_YUV2BGRA_UYVY) ? 1 : 0;
            uidx = (code==COLOR_YUV2RGB_YVYU || code==COLOR_YUV2BGR_YVYU || code==COLOR_YUV2RGBA_YVYU || code==COLOR_YUV2BGRA_YVYU) ? 1 : 0;
            CV_Assert( dcn == 3 || dcn == 4 );
            CV_Assert( scn == 2 && depth == CV_8U );
            _dst.create(sz, CV_8UC(dcn));
            dst = _dst.getMat();
            hal::cvtOnePlaneYUVtoBGR(src.data, src.step, dst.data, dst.step, src.cols, src.rows,
                                     dcn, swapBlue(code), uidx, ycn);
            break;
        case COLOR_YUV2GRAY_UYVY: case COLOR_YUV2GRAY_YUY2:
            {
                if (dcn <= 0) dcn = 1;

                CV_Assert( dcn == 1 );
                CV_Assert( scn == 2 && depth == CV_8U );

                src.release(); // T-API datarace fixup
                extractChannel(_src, _dst, code == COLOR_YUV2GRAY_UYVY ? 1 : 0);
            }
            break;
        case COLOR_RGBA2mRGBA:
            if (dcn <= 0) dcn = 4;
            CV_Assert( scn == 4 && dcn == 4 && depth == CV_8U );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtRGBAtoMultipliedRGBA(src.data, src.step, dst.data, dst.step, src.cols, src.rows);
            break;
        case COLOR_mRGBA2RGBA:
            if (dcn <= 0) dcn = 4;
            CV_Assert( scn == 4 && dcn == 4 && depth == CV_8U );
            _dst.create(sz, CV_MAKETYPE(depth, dcn));
            dst = _dst.getMat();
            hal::cvtMultipliedRGBAtoRGBA(src.data, src.step, dst.data, dst.step, src.cols, src.rows);
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
