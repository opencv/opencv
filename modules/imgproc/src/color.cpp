// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "color.hpp"

namespace cv
{

#ifdef HAVE_OPENCL

static bool ocl_cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    int bidx = swapBlue(code) ? 2 : 0;

    switch (code)
    {
    case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
    case COLOR_RGBA2BGR: case COLOR_RGB2BGR: case COLOR_BGRA2RGBA:
    {
        bool reverse = !(code == COLOR_BGR2BGRA || code == COLOR_BGRA2BGR);
        return oclCvtColorBGR2BGR(_src, _dst, dcn, reverse);
    }
    case COLOR_BGR5652BGR: case COLOR_BGR5552BGR: case COLOR_BGR5652RGB: case COLOR_BGR5552RGB:
    case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
        return oclCvtColor5x52BGR(_src, _dst, dcn, bidx, greenBits(code));

    case COLOR_BGR2BGR565: case COLOR_BGR2BGR555: case COLOR_RGB2BGR565: case COLOR_RGB2BGR555:
    case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
        return oclCvtColorBGR25x5(_src, _dst, bidx, greenBits(code) );

    case COLOR_BGR5652GRAY: case COLOR_BGR5552GRAY:
        return oclCvtColor5x52Gray(_src, _dst, greenBits(code));

    case COLOR_GRAY2BGR565: case COLOR_GRAY2BGR555:
        return oclCvtColorGray25x5(_src, _dst, greenBits(code));

    case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY:
    case COLOR_RGB2GRAY: case COLOR_RGBA2GRAY:
        return oclCvtColorBGR2Gray(_src, _dst, bidx);

    case COLOR_GRAY2BGR:
    case COLOR_GRAY2BGRA:
        return oclCvtColorGray2BGR(_src, _dst, dcn);

    case COLOR_BGR2YUV:
    case COLOR_RGB2YUV:
        return oclCvtColorBGR2YUV(_src, _dst, bidx);

    case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
        return oclCvtColorYUV2BGR(_src, _dst, dcn, bidx);

    case COLOR_YUV2RGB_NV12: case COLOR_YUV2BGR_NV12: case COLOR_YUV2RGB_NV21: case COLOR_YUV2BGR_NV21:
    case COLOR_YUV2RGBA_NV12: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV21:
    {
        int uidx = code == COLOR_YUV2RGBA_NV21 || code == COLOR_YUV2RGB_NV21 ||
                   code == COLOR_YUV2BGRA_NV21 || code == COLOR_YUV2BGR_NV21 ? 1 : 0;
        return oclCvtColorTwoPlaneYUV2BGR(_src, _dst, dcn, bidx, uidx);
    }
    case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12: case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV: case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
    {
        int uidx = code == COLOR_YUV2BGRA_YV12 || code == COLOR_YUV2BGR_YV12 ||
                   code == COLOR_YUV2RGBA_YV12 || code == COLOR_YUV2RGB_YV12 ? 1 : 0;
        return oclCvtColorThreePlaneYUV2BGR(_src, _dst, dcn, bidx, uidx);
    }
    case COLOR_YUV2GRAY_420:
    {
        return oclCvtColorYUV2Gray_420(_src, _dst);
    }
    case COLOR_RGB2YUV_YV12: case COLOR_BGR2YUV_YV12: case COLOR_RGBA2YUV_YV12: case COLOR_BGRA2YUV_YV12:
    case COLOR_RGB2YUV_IYUV: case COLOR_BGR2YUV_IYUV: case COLOR_RGBA2YUV_IYUV: case COLOR_BGRA2YUV_IYUV:
    {
        int uidx = code == COLOR_RGBA2YUV_YV12 || code == COLOR_RGB2YUV_YV12 ||
                   code == COLOR_BGRA2YUV_YV12 || code == COLOR_BGR2YUV_YV12 ? 1 : 0;
        return oclCvtColorBGR2ThreePlaneYUV(_src, _dst, bidx, uidx );
    }
    case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY: case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2: case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
    case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2: case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
    {
        int yidx = (code==COLOR_YUV2RGB_UYVY || code==COLOR_YUV2RGBA_UYVY ||
                    code==COLOR_YUV2BGR_UYVY || code==COLOR_YUV2BGRA_UYVY) ? 1 : 0;
        int uidx = (code==COLOR_YUV2RGB_YVYU || code==COLOR_YUV2RGBA_YVYU ||
                    code==COLOR_YUV2BGR_YVYU || code==COLOR_YUV2BGRA_YVYU) ? 2 : 0;
        uidx = 1 - yidx + uidx;

        return oclCvtColorOnePlaneYUV2BGR(_src, _dst, dcn, bidx, uidx, yidx);
    }
    case COLOR_BGR2YCrCb:
    case COLOR_RGB2YCrCb:
        return oclCvtColorBGR2YCrCb(_src, _dst, bidx);

    case COLOR_YCrCb2BGR:
    case COLOR_YCrCb2RGB:
        return oclCvtcolorYCrCb2BGR(_src, _dst, dcn, bidx);

    case COLOR_BGR2XYZ:
    case COLOR_RGB2XYZ:
        return oclCvtColorBGR2XYZ(_src, _dst, bidx);

    case COLOR_XYZ2BGR:
    case COLOR_XYZ2RGB:
        return oclCvtColorXYZ2BGR(_src, _dst, dcn, bidx);

    case COLOR_BGR2HSV: case COLOR_BGR2HSV_FULL:
    case COLOR_RGB2HSV: case COLOR_RGB2HSV_FULL:
        return oclCvtColorBGR2HSV(_src, _dst, bidx, isFullRangeHSV(code));

    case COLOR_BGR2HLS: case COLOR_BGR2HLS_FULL:
    case COLOR_RGB2HLS: case COLOR_RGB2HLS_FULL:
        return oclCvtColorBGR2HLS(_src, _dst, bidx, isFullRangeHSV(code));

    case COLOR_HSV2BGR: case COLOR_HSV2BGR_FULL:
    case COLOR_HSV2RGB: case COLOR_HSV2RGB_FULL:
        return oclCvtColorHSV2BGR(_src, _dst, dcn, bidx, isFullRangeHSV(code));

    case COLOR_HLS2BGR: case COLOR_HLS2BGR_FULL:
    case COLOR_HLS2RGB: case COLOR_HLS2RGB_FULL:
        return oclCvtColorHLS2BGR(_src, _dst, dcn, bidx, isFullRangeHSV(code));

    case COLOR_RGBA2mRGBA:
        return oclCvtColorRGBA2mRGBA(_src, _dst);

    case COLOR_mRGBA2RGBA:
        return oclCvtColormRGBA2RGBA(_src, _dst);

    case COLOR_BGR2Lab: case COLOR_LBGR2Lab:
    case COLOR_RGB2Lab: case COLOR_LRGB2Lab:
        return oclCvtColorBGR2Lab(_src, _dst, bidx, is_sRGB(code));

    case COLOR_BGR2Luv: case COLOR_LBGR2Luv:
    case COLOR_RGB2Luv: case COLOR_LRGB2Luv:
        return oclCvtColorBGR2Luv(_src, _dst, bidx, is_sRGB(code));

    case COLOR_Lab2BGR: case COLOR_Lab2LBGR:
    case COLOR_Lab2RGB: case COLOR_Lab2LRGB:
        return oclCvtColorLab2BGR(_src, _dst, dcn, bidx, is_sRGB(code));

    case COLOR_Luv2BGR: case COLOR_Luv2LBGR:
    case COLOR_Luv2RGB: case COLOR_Luv2LRGB:
        return oclCvtColorLuv2BGR(_src, _dst, dcn, bidx, is_sRGB(code));

    default:
        return false;
    }
}

#endif


// helper function for dual-plane modes

void cvtColorTwoPlane( InputArray _ysrc, InputArray _uvsrc, OutputArray _dst, int code )
{
    // only YUV420 is currently supported
    switch (code)
    {
        case COLOR_YUV2BGR_NV21:  case COLOR_YUV2RGB_NV21:  case COLOR_YUV2BGR_NV12:  case COLOR_YUV2RGB_NV12:
        case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21: case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
            break;
        default:
            CV_Error( CV_StsBadFlag, "Unknown/unsupported color conversion code" );
            return;
    }

    cvtColorTwoPlaneYUV2BGRpair(_ysrc, _uvsrc, _dst, dstChannels(code), swapBlue(code), uIndex(code));
}


//////////////////////////////////////////////////////////////////////////////////////////
//                                   The main function                                  //
//////////////////////////////////////////////////////////////////////////////////////////

void cvtColor( InputArray _src, OutputArray _dst, int code, int dcn )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_src.empty());

    if(dcn <= 0)
            dcn = dstChannels(code);

    CV_OCL_RUN( _src.dims() <= 2 && _dst.isUMat() &&
                !(CV_MAT_DEPTH(_src.type()) == CV_8U && (code == COLOR_Luv2BGR || code == COLOR_Luv2RGB)),
                ocl_cvtColor(_src, _dst, code, dcn) )

    switch( code )
    {
        case COLOR_BGR2BGRA: case COLOR_RGB2BGRA: case COLOR_BGRA2BGR:
        case COLOR_RGBA2BGR: case COLOR_RGB2BGR:  case COLOR_BGRA2RGBA:
            cvtColorBGR2BGR(_src, _dst, dcn, swapBlue(code));
            break;

        case COLOR_BGR2BGR565:  case COLOR_BGR2BGR555: case COLOR_BGRA2BGR565: case COLOR_BGRA2BGR555:
        case COLOR_RGB2BGR565:  case COLOR_RGB2BGR555: case COLOR_RGBA2BGR565: case COLOR_RGBA2BGR555:
            cvtColorBGR25x5(_src, _dst, swapBlue(code), greenBits(code));
            break;

        case COLOR_BGR5652BGR:  case COLOR_BGR5552BGR: case COLOR_BGR5652BGRA: case COLOR_BGR5552BGRA:
        case COLOR_BGR5652RGB:  case COLOR_BGR5552RGB: case COLOR_BGR5652RGBA: case COLOR_BGR5552RGBA:
            cvtColor5x52BGR(_src, _dst, dcn, swapBlue(code), greenBits(code));
            break;

        case COLOR_BGR2GRAY: case COLOR_BGRA2GRAY:
        case COLOR_RGB2GRAY: case COLOR_RGBA2GRAY:
            cvtColorBGR2Gray(_src, _dst, swapBlue(code));
            break;

        case COLOR_BGR5652GRAY:
        case COLOR_BGR5552GRAY:
            cvtColor5x52Gray(_src, _dst, greenBits(code));
            break;

        case COLOR_GRAY2BGR:
        case COLOR_GRAY2BGRA:
            cvtColorGray2BGR(_src, _dst, dcn);
            break;

        case COLOR_GRAY2BGR565:
        case COLOR_GRAY2BGR555:
            cvtColorGray25x5(_src, _dst, greenBits(code));
            break;

        case COLOR_BGR2YCrCb: case COLOR_RGB2YCrCb:
        case COLOR_BGR2YUV:   case COLOR_RGB2YUV:
            cvtColorBGR2YUV(_src, _dst, swapBlue(code), code == COLOR_BGR2YCrCb || code == COLOR_RGB2YCrCb);
            break;

        case COLOR_YCrCb2BGR: case COLOR_YCrCb2RGB:
        case COLOR_YUV2BGR:   case COLOR_YUV2RGB:
            cvtColorYUV2BGR(_src, _dst, dcn, swapBlue(code), code == COLOR_YCrCb2BGR || code == COLOR_YCrCb2RGB);
            break;

        case COLOR_BGR2XYZ:
        case COLOR_RGB2XYZ:
            cvtColorBGR2XYZ(_src, _dst, swapBlue(code));
            break;

        case COLOR_XYZ2BGR:
        case COLOR_XYZ2RGB:
            cvtColorXYZ2BGR(_src, _dst, dcn, swapBlue(code));
            break;

        case COLOR_BGR2HSV: case COLOR_BGR2HSV_FULL:
        case COLOR_RGB2HSV: case COLOR_RGB2HSV_FULL:
            cvtColorBGR2HSV(_src, _dst, swapBlue(code), isFullRangeHSV(code));
            break;

        case COLOR_BGR2HLS: case COLOR_BGR2HLS_FULL:
        case COLOR_RGB2HLS: case COLOR_RGB2HLS_FULL:
            cvtColorBGR2HLS(_src, _dst, swapBlue(code), isFullRangeHSV(code));
            break;

        case COLOR_HSV2BGR: case COLOR_HSV2BGR_FULL:
        case COLOR_HSV2RGB: case COLOR_HSV2RGB_FULL:
            cvtColorHSV2BGR(_src, _dst, dcn, swapBlue(code), isFullRangeHSV(code));
            break;

        case COLOR_HLS2BGR: case COLOR_HLS2BGR_FULL:
        case COLOR_HLS2RGB: case COLOR_HLS2RGB_FULL:
            cvtColorHLS2BGR(_src, _dst, dcn, swapBlue(code), isFullRangeHSV(code));
            break;

        case COLOR_BGR2Lab: case COLOR_LBGR2Lab:
        case COLOR_RGB2Lab: case COLOR_LRGB2Lab:
            cvtColorBGR2Lab(_src, _dst, swapBlue(code), is_sRGB(code));
            break;

        case COLOR_BGR2Luv: case COLOR_LBGR2Luv:
        case COLOR_RGB2Luv: case COLOR_LRGB2Luv:
            cvtColorBGR2Luv(_src, _dst, swapBlue(code), is_sRGB(code));
            break;

        case COLOR_Lab2BGR: case COLOR_Lab2LBGR:
        case COLOR_Lab2RGB: case COLOR_Lab2LRGB:
            cvtColorLab2BGR(_src, _dst, dcn, swapBlue(code), is_sRGB(code));
            break;

        case COLOR_Luv2BGR: case COLOR_Luv2LBGR:
        case COLOR_Luv2RGB: case COLOR_Luv2LRGB:
            cvtColorLuv2BGR(_src, _dst, dcn, swapBlue(code), is_sRGB(code));
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
                int ycn  = (code==COLOR_YUV2RGB_UYVY || code==COLOR_YUV2BGR_UYVY ||
                            code==COLOR_YUV2RGBA_UYVY || code==COLOR_YUV2BGRA_UYVY) ? 1 : 0;
                cvtColorOnePlaneYUV2BGR(_src, _dst, dcn, swapBlue(code), uIndex(code), ycn);
                break;
            }

        case COLOR_YUV2GRAY_UYVY:
        case COLOR_YUV2GRAY_YUY2:
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
