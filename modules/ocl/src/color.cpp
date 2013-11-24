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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Wang Weiyan, wangweiyanster@gmail.com
//    Peng Xiao, pengxiao@multicorewareinc.com
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
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

static void fromRGB_caller(const oclMat &src, oclMat &dst, int bidx, const std::string & kernelName,
                           const std::string & additionalOptions = std::string(),
                           const oclMat & data1 = oclMat(), const oclMat & data2 = oclMat())
{
    int src_offset = src.offset / src.elemSize1(), src_step = src.step1();
    int dst_offset = dst.offset / dst.elemSize1(), dst_step = dst.step1();

    String build_options = format("-D DEPTH_%d", src.depth());
    if (!additionalOptions.empty())
        build_options = build_options + additionalOptions;

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&bidx));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_offset ));

    if (!data1.empty())
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&data1.data ));
    if (!data2.empty())
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&data2.data ));

    size_t gt[3] = { dst.cols, dst.rows, 1 }, lt[3] = { 16, 16, 1 };
    openCLExecuteKernel(src.clCxt, &cvt_color, kernelName.c_str(), gt, lt, args, -1, -1, build_options.c_str());
}

static void toRGB_caller(const oclMat &src, oclMat &dst, int bidx, const std::string & kernelName,
                         const std::string & additionalOptions = std::string(), const oclMat & data = oclMat())
{
    String build_options = format("-D DEPTH_%d -D dcn=%d", src.depth(), dst.channels());
    if (!additionalOptions.empty())
        build_options = build_options + additionalOptions;

    int src_offset = src.offset / src.elemSize1(), src_step = src.step1();
    int dst_offset = dst.offset / dst.elemSize1(), dst_step = dst.step1();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&bidx));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_offset ));

    if (!data.empty())
        args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&data.data ));

    size_t gt[3] = { dst.cols, dst.rows, 1 }, lt[3] = { 16, 16, 1 };
    openCLExecuteKernel(src.clCxt, &cvt_color, kernelName.c_str(), gt, lt, args, -1, -1, build_options.c_str());
}

static void RGB_caller(const oclMat &src, oclMat &dst, bool reverse)
{
    String build_options = format("-D DEPTH_%d -D dcn=%d -D scn=%d -D %s", src.depth(),
                                  dst.channels(), src.channels(), reverse ? "REVERSE" : "ORDER");
    int src_offset = src.offset / src.elemSize1(), src_step = src.step1();
    int dst_offset = dst.offset / dst.elemSize1(), dst_step = dst.step1();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_step));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_offset ));

    size_t gt[3] = { dst.cols, dst.rows, 1 }, lt[3] = { 16, 16, 1 };
    openCLExecuteKernel(src.clCxt, &cvt_color, "RGB", gt, lt, args, -1, -1, build_options.c_str());
}

static void fromRGB5x5_caller(const oclMat &src, oclMat &dst, int bidx, int greenbits, const std::string & kernelName)
{
    String build_options = format("-D DEPTH_%d -D greenbits=%d -D dcn=%d",
                                  src.depth(), greenbits, dst.channels());
    int src_offset = src.offset >> 1, src_step = src.step >> 1;
    int dst_offset = dst.offset / dst.elemSize1(), dst_step = dst.step / dst.elemSize1();

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&bidx));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_offset ));

    size_t gt[3] = { dst.cols, dst.rows, 1 }, lt[3] = { 16, 16, 1 };
    openCLExecuteKernel(src.clCxt, &cvt_color, kernelName.c_str(), gt, lt, args, -1, -1, build_options.c_str());
}

static void toRGB5x5_caller(const oclMat &src, oclMat &dst, int bidx, int greenbits, const std::string & kernelName)
{
    String build_options = format("-D DEPTH_%d -D greenbits=%d -D scn=%d",
                                  src.depth(), greenbits, src.channels());
    int src_offset = (int)src.offset, src_step = (int)src.step;
    int dst_offset = dst.offset >> 1, dst_step = dst.step >> 1;

    std::vector<std::pair<size_t , const void *> > args;
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.cols));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst.rows));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_step));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&bidx));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&src.data));
    args.push_back( std::make_pair( sizeof(cl_mem) , (void *)&dst.data));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&src_offset ));
    args.push_back( std::make_pair( sizeof(cl_int) , (void *)&dst_offset ));

    size_t gt[3] = { dst.cols, dst.rows, 1 }, lt[3] = { 16, 16, 1 };
    openCLExecuteKernel(src.clCxt, &cvt_color, kernelName.c_str(), gt, lt, args, -1, -1, build_options.c_str());
}

static void cvtColor_caller(const oclMat &src, oclMat &dst, int code, int dcn)
{
    Size sz = src.size();
    int scn = src.channels(), depth = src.depth(), bidx;

    CV_Assert(depth == CV_8U || depth == CV_16U || depth == CV_32F);

    switch (code)
    {
    case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
    case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
    {
        CV_Assert(scn == 3 || scn == 4);
        dcn = code == COLOR_BGR2BGRA || code == COLOR_RGB2BGRA || code == COLOR_BGRA2RGBA ? 4 : 3;
        bool reverse = !(code == COLOR_BGR2BGRA || code == COLOR_BGRA2BGR);
        dst.create(sz, CV_MAKE_TYPE(depth, dcn));
        RGB_caller(src, dst, reverse);
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
        dst.create(sz, CV_8UC2);
        toRGB5x5_caller(src, dst, bidx, greenbits, "RGB2RGB5x5");
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
        dst.create(sz, CV_MAKETYPE(depth, dcn));
        fromRGB5x5_caller(src, dst, bidx, greenbits, "RGB5x52RGB");
        break;
    }
    case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
    {
        CV_Assert(scn == 2 && depth == CV_8U);
        dst.create(sz, CV_8UC1);
        int greenbits = code == COLOR_BGR5652GRAY ? 6 : 5;
        fromRGB5x5_caller(src, dst, -1, greenbits, "BGR5x52Gray");
        break;
    }
    case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
    {
        CV_Assert(scn == 1 && depth == CV_8U);
        dst.create(sz, CV_8UC2);
        int greenbits = code == COLOR_GRAY2BGR565 ? 6 : 5;
        toRGB5x5_caller(src, dst, -1, greenbits, "Gray2BGR5x5");
        break;
    }
    case COLOR_RGB2GRAY: case COLOR_BGR2GRAY: case COLOR_RGBA2GRAY: case COLOR_BGRA2GRAY:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2GRAY || code == COLOR_BGRA2GRAY ? 0 : 2;
        dst.create(sz, CV_MAKETYPE(depth, 1));
        fromRGB_caller(src, dst, bidx, "RGB2Gray");
        break;
    }
    case COLOR_GRAY2BGR: case COLOR_GRAY2BGRA:
    {
        CV_Assert(scn == 1);
        dcn  = code == COLOR_GRAY2BGRA ? 4 : 3;
        dst.create(sz, CV_MAKETYPE(depth, dcn));
        toRGB_caller(src, dst, 0, "Gray2RGB");
        break;
    }
    case COLOR_BGR2YUV: case COLOR_RGB2YUV:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2YUV ? 0 : 2;
        dst.create(sz, CV_MAKETYPE(depth, 3));
        fromRGB_caller(src, dst, bidx, "RGB2YUV");
        break;
    }
    case COLOR_YUV2BGR: case COLOR_YUV2RGB:
    {
        if( dcn <= 0 )
            dcn = 3;
        CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
        bidx = code == COLOR_YUV2BGR ? 0 : 2;
        dst.create(sz, CV_MAKETYPE(depth, dcn));
        toRGB_caller(src, dst, bidx, "YUV2RGB");
        break;
    }
    case COLOR_YUV2RGB_NV12: case COLOR_YUV2BGR_NV12:
    case COLOR_YUV2RGBA_NV12: case COLOR_YUV2BGRA_NV12:
    {
        CV_Assert(scn == 1);
        CV_Assert( sz.width % 2 == 0 && sz.height % 3 == 0 && depth == CV_8U );
        dcn = code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2RGBA_NV12 ? 4 : 3;
        bidx = code == COLOR_YUV2BGRA_NV12 || code == COLOR_YUV2BGR_NV12 ? 0 : 2;

        Size dstSz(sz.width, sz.height * 2 / 3);
        dst.create(dstSz, CV_MAKETYPE(depth, dcn));
        toRGB_caller(src, dst, bidx, "YUV2RGBA_NV12");
        break;
    }
    case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2YCrCb ? 0 : 2;
        dst.create(sz, CV_MAKETYPE(depth, 3));
        fromRGB_caller(src, dst, bidx, "RGB2YCrCb");
        break;
    }
    case COLOR_YCrCb2BGR: case COLOR_YCrCb2RGB:
    {
        if( dcn <= 0 )
            dcn = 3;
        CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
        bidx = code == COLOR_YCrCb2BGR ? 0 : 2;
        dst.create(sz, CV_MAKETYPE(depth, dcn));
        toRGB_caller(src, dst, bidx, "YCrCb2RGB");
        break;
    }
    case COLOR_BGR2XYZ: case COLOR_RGB2XYZ:
    {
        CV_Assert(scn == 3 || scn == 4);
        bidx = code == COLOR_BGR2XYZ ? 0 : 2;
        dst.create(sz, CV_MAKE_TYPE(depth, 3));

        Mat c;
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
        oclMat oclCoeffs(c);

        fromRGB_caller(src, dst, bidx, "RGB2XYZ", "", oclCoeffs);
        break;
    }
    case COLOR_XYZ2BGR: case COLOR_XYZ2RGB:
    {
        if (dcn <= 0)
            dcn = 3;
        CV_Assert(scn == 3 && (dcn == 3 || dcn == 4));
        bidx = code == COLOR_XYZ2BGR ? 0 : 2;
        dst.create(sz, CV_MAKE_TYPE(depth, dcn));

        Mat c;
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
        oclMat oclCoeffs(c);

        toRGB_caller(src, dst, bidx, "XYZ2RGB", "", oclCoeffs);
        break;
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
        dst.create(sz, CV_MAKETYPE(depth, 3));
        std::string kernelName = std::string("RGB2") + (is_hsv ? "HSV" : "HLS");

        if (is_hsv && depth == CV_8U)
        {
            static oclMat sdiv_data;
            static oclMat hdiv_data180;
            static oclMat hdiv_data256;
            static int sdiv_table[256];
            static int hdiv_table180[256];
            static int hdiv_table256[256];
            static volatile bool initialized180 = false, initialized256 = false;
            volatile bool & initialized = hrange == 180 ? initialized180 : initialized256;

            if (!initialized)
            {
                int * const hdiv_table = hrange == 180 ? hdiv_table180 : hdiv_table256, hsv_shift = 12;
                oclMat & hdiv_data = hrange == 180 ? hdiv_data180 : hdiv_data256;

                sdiv_table[0] = hdiv_table180[0] = hdiv_table256[0] = 0;

                int v = 255 << hsv_shift;
                if (!initialized180 && !initialized256)
                {
                    for(int i = 1; i < 256; i++ )
                        sdiv_table[i] = saturate_cast<int>(v/(1.*i));
                    sdiv_data.upload(Mat(1, 256, CV_32SC1, sdiv_table));
                }

                v = hrange << hsv_shift;
                for (int i = 1; i < 256; i++ )
                    hdiv_table[i] = saturate_cast<int>(v/(6.*i));

                hdiv_data.upload(Mat(1, 256, CV_32SC1, hdiv_table));
                initialized = true;
            }

            fromRGB_caller(src, dst, bidx, kernelName, format(" -D hrange=%d", hrange), sdiv_data, hrange == 256 ? hdiv_data256 : hdiv_data180);
            return;
        }

        fromRGB_caller(src, dst, bidx, kernelName, format(" -D hscale=%f", hrange*(1.f/360.f)));
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

        dst.create(sz, CV_MAKETYPE(depth, dcn));

        std::string kernelName = std::string(is_hsv ? "HSV" : "HLS") + "2RGB";
        toRGB_caller(src, dst, bidx, kernelName, format(" -D hrange=%d -D hscale=%f", hrange, 6.f/hrange));
        break;
    }
    case COLOR_RGBA2mRGBA: case COLOR_mRGBA2RGBA:
        {
            CV_Assert(scn == 4 && depth == CV_8U);
            dst.create(sz, CV_MAKETYPE(depth, 4));
            std::string kernelName = code == COLOR_RGBA2mRGBA ? "RGBA2mRGBA" : "mRGBA2RGBA";

            fromRGB_caller(src, dst, 0, kernelName);
            break;
        }
    default:
        CV_Error(Error::StsBadFlag, "Unknown/unsupported color conversion code" );
    }
}

void cv::ocl::cvtColor(const oclMat &src, oclMat &dst, int code, int dcn)
{
    cvtColor_caller(src, dst, code, dcn);
}
