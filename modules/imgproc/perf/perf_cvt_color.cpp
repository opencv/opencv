// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

//extra color conversions supported implicitly
enum
{
    CX_BGRA2HLS      = COLOR_COLORCVT_MAX + COLOR_BGR2HLS,
    CX_BGRA2HLS_FULL = COLOR_COLORCVT_MAX + COLOR_BGR2HLS_FULL,
    CX_BGRA2HSV      = COLOR_COLORCVT_MAX + COLOR_BGR2HSV,
    CX_BGRA2HSV_FULL = COLOR_COLORCVT_MAX + COLOR_BGR2HSV_FULL,
    CX_BGRA2Lab      = COLOR_COLORCVT_MAX + COLOR_BGR2Lab,
    CX_BGRA2Luv      = COLOR_COLORCVT_MAX + COLOR_BGR2Luv,
    CX_BGRA2XYZ      = COLOR_COLORCVT_MAX + COLOR_BGR2XYZ,
    CX_BGRA2YCrCb    = COLOR_COLORCVT_MAX + COLOR_BGR2YCrCb,
    CX_BGRA2YUV      = COLOR_COLORCVT_MAX + COLOR_BGR2YUV,
    CX_HLS2BGRA      = COLOR_COLORCVT_MAX + COLOR_HLS2BGR,
    CX_HLS2BGRA_FULL = COLOR_COLORCVT_MAX + COLOR_HLS2BGR_FULL,
    CX_HLS2RGBA      = COLOR_COLORCVT_MAX + COLOR_HLS2RGB,
    CX_HLS2RGBA_FULL = COLOR_COLORCVT_MAX + COLOR_HLS2RGB_FULL,
    CX_HSV2BGRA      = COLOR_COLORCVT_MAX + COLOR_HSV2BGR,
    CX_HSV2BGRA_FULL = COLOR_COLORCVT_MAX + COLOR_HSV2BGR_FULL,
    CX_HSV2RGBA      = COLOR_COLORCVT_MAX + COLOR_HSV2RGB,
    CX_HSV2RGBA_FULL = COLOR_COLORCVT_MAX + COLOR_HSV2RGB_FULL,
    CX_Lab2BGRA      = COLOR_COLORCVT_MAX + COLOR_Lab2BGR,
    CX_Lab2LBGRA     = COLOR_COLORCVT_MAX + COLOR_Lab2LBGR,
    CX_Lab2LRGBA     = COLOR_COLORCVT_MAX + COLOR_Lab2LRGB,
    CX_Lab2RGBA      = COLOR_COLORCVT_MAX + COLOR_Lab2RGB,
    CX_LBGRA2Lab     = COLOR_COLORCVT_MAX + COLOR_LBGR2Lab,
    CX_LBGRA2Luv     = COLOR_COLORCVT_MAX + COLOR_LBGR2Luv,
    CX_LRGBA2Lab     = COLOR_COLORCVT_MAX + COLOR_LRGB2Lab,
    CX_LRGBA2Luv     = COLOR_COLORCVT_MAX + COLOR_LRGB2Luv,
    CX_Luv2BGRA      = COLOR_COLORCVT_MAX + COLOR_Luv2BGR,
    CX_Luv2LBGRA     = COLOR_COLORCVT_MAX + COLOR_Luv2LBGR,
    CX_Luv2LRGBA     = COLOR_COLORCVT_MAX + COLOR_Luv2LRGB,
    CX_Luv2RGBA      = COLOR_COLORCVT_MAX + COLOR_Luv2RGB,
    CX_RGBA2HLS      = COLOR_COLORCVT_MAX + COLOR_RGB2HLS,
    CX_RGBA2HLS_FULL = COLOR_COLORCVT_MAX + COLOR_RGB2HLS_FULL,
    CX_RGBA2HSV      = COLOR_COLORCVT_MAX + COLOR_RGB2HSV,
    CX_RGBA2HSV_FULL = COLOR_COLORCVT_MAX + COLOR_RGB2HSV_FULL,
    CX_RGBA2Lab      = COLOR_COLORCVT_MAX + COLOR_RGB2Lab,
    CX_RGBA2Luv      = COLOR_COLORCVT_MAX + COLOR_RGB2Luv,
    CX_RGBA2XYZ      = COLOR_COLORCVT_MAX + COLOR_RGB2XYZ,
    CX_RGBA2YCrCb    = COLOR_COLORCVT_MAX + COLOR_RGB2YCrCb,
    CX_RGBA2YUV      = COLOR_COLORCVT_MAX + COLOR_RGB2YUV,
    CX_XYZ2BGRA      = COLOR_COLORCVT_MAX + COLOR_XYZ2BGR,
    CX_XYZ2RGBA      = COLOR_COLORCVT_MAX + COLOR_XYZ2RGB,
    CX_YCrCb2BGRA    = COLOR_COLORCVT_MAX + COLOR_YCrCb2BGR,
    CX_YCrCb2RGBA    = COLOR_COLORCVT_MAX + COLOR_YCrCb2RGB,
    CX_YUV2BGRA      = COLOR_COLORCVT_MAX + COLOR_YUV2BGR,
    CX_YUV2RGBA      = COLOR_COLORCVT_MAX + COLOR_YUV2RGB
};

CV_ENUM(CvtMode,
    COLOR_BGR2BGR555, COLOR_BGR2BGR565, COLOR_BGR2BGRA, COLOR_BGR2GRAY,
    COLOR_BGR2HLS, COLOR_BGR2HLS_FULL, COLOR_BGR2HSV, COLOR_BGR2HSV_FULL,
    COLOR_BGR2Lab, COLOR_BGR2Luv, COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_BGR2XYZ,
    COLOR_BGR2YCrCb, COLOR_BGR2YUV, COLOR_BGR5552BGR, COLOR_BGR5552BGRA,

    COLOR_BGR5552GRAY, COLOR_BGR5552RGB, COLOR_BGR5552RGBA, COLOR_BGR5652BGR,
    COLOR_BGR5652BGRA, COLOR_BGR5652GRAY, COLOR_BGR5652RGB, COLOR_BGR5652RGBA,

    COLOR_BGRA2BGR, COLOR_BGRA2BGR555, COLOR_BGRA2BGR565, COLOR_BGRA2GRAY, COLOR_BGRA2RGBA,
    CX_BGRA2HLS, CX_BGRA2HLS_FULL, CX_BGRA2HSV, CX_BGRA2HSV_FULL,
    CX_BGRA2Lab, CX_BGRA2Luv, CX_BGRA2XYZ,
    CX_BGRA2YCrCb, CX_BGRA2YUV,

    COLOR_GRAY2BGR, COLOR_GRAY2BGR555, COLOR_GRAY2BGR565, COLOR_GRAY2BGRA,

    COLOR_HLS2BGR, COLOR_HLS2BGR_FULL, COLOR_HLS2RGB, COLOR_HLS2RGB_FULL,
    CX_HLS2BGRA, CX_HLS2BGRA_FULL, CX_HLS2RGBA, CX_HLS2RGBA_FULL,

    COLOR_HSV2BGR, COLOR_HSV2BGR_FULL, COLOR_HSV2RGB, COLOR_HSV2RGB_FULL,
    CX_HSV2BGRA, CX_HSV2BGRA_FULL, CX_HSV2RGBA,    CX_HSV2RGBA_FULL,

    COLOR_Lab2BGR, COLOR_Lab2LBGR, COLOR_Lab2LRGB, COLOR_Lab2RGB,
    CX_Lab2BGRA, CX_Lab2LBGRA, CX_Lab2LRGBA, CX_Lab2RGBA,

    COLOR_LBGR2Lab, COLOR_LBGR2Luv, COLOR_LRGB2Lab, COLOR_LRGB2Luv,
    CX_LBGRA2Lab, CX_LBGRA2Luv, CX_LRGBA2Lab, CX_LRGBA2Luv,

    COLOR_Luv2BGR, COLOR_Luv2LBGR, COLOR_Luv2LRGB, COLOR_Luv2RGB,
    CX_Luv2BGRA, CX_Luv2LBGRA, CX_Luv2LRGBA, CX_Luv2RGBA,

    COLOR_RGB2BGR555, COLOR_RGB2BGR565, COLOR_RGB2GRAY,
    COLOR_RGB2HLS, COLOR_RGB2HLS_FULL, COLOR_RGB2HSV, COLOR_RGB2HSV_FULL,
    COLOR_RGB2Lab, COLOR_RGB2Luv, COLOR_RGB2XYZ, COLOR_RGB2YCrCb, COLOR_RGB2YUV,

    COLOR_RGBA2BGR, COLOR_RGBA2BGR555, COLOR_RGBA2BGR565, COLOR_RGBA2GRAY,
    CX_RGBA2HLS, CX_RGBA2HLS_FULL, CX_RGBA2HSV, CX_RGBA2HSV_FULL,
    CX_RGBA2Lab, CX_RGBA2Luv, CX_RGBA2XYZ,
    CX_RGBA2YCrCb, CX_RGBA2YUV,

    COLOR_XYZ2BGR, COLOR_XYZ2RGB, CX_XYZ2BGRA, CX_XYZ2RGBA,

    COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, CX_YCrCb2BGRA, CX_YCrCb2RGBA,
    COLOR_YUV2BGR, COLOR_YUV2RGB, CX_YUV2BGRA, CX_YUV2RGBA
    )

CV_ENUM(CvtMode16U,
    COLOR_BGR2BGRA, COLOR_BGR2GRAY,
    COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_BGR2XYZ,
    COLOR_BGR2YCrCb, COLOR_BGR2YUV,

    COLOR_BGRA2BGR, COLOR_BGRA2GRAY, COLOR_BGRA2RGBA,
    CX_BGRA2XYZ,
    CX_BGRA2YCrCb, CX_BGRA2YUV,

    COLOR_GRAY2BGR, COLOR_GRAY2BGRA,

    COLOR_RGB2GRAY,
    COLOR_RGB2XYZ, COLOR_RGB2YCrCb, COLOR_RGB2YUV,

    COLOR_RGBA2BGR, COLOR_RGBA2GRAY,
    CX_RGBA2XYZ,
    CX_RGBA2YCrCb, CX_RGBA2YUV,

    COLOR_XYZ2BGR, COLOR_XYZ2RGB, CX_XYZ2BGRA, CX_XYZ2RGBA,

    COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, CX_YCrCb2BGRA, CX_YCrCb2RGBA,
    COLOR_YUV2BGR, COLOR_YUV2RGB, CX_YUV2BGRA, CX_YUV2RGBA
    )

CV_ENUM(CvtMode32F,
    COLOR_BGR2BGRA, COLOR_BGR2GRAY,
    COLOR_BGR2HLS, COLOR_BGR2HLS_FULL, COLOR_BGR2HSV, COLOR_BGR2HSV_FULL,
    COLOR_BGR2Lab, COLOR_BGR2Luv, COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_BGR2XYZ,
    COLOR_BGR2YCrCb, COLOR_BGR2YUV,

    COLOR_BGRA2BGR, COLOR_BGRA2GRAY, COLOR_BGRA2RGBA,
    CX_BGRA2HLS, CX_BGRA2HLS_FULL, CX_BGRA2HSV, CX_BGRA2HSV_FULL,
    CX_BGRA2Lab, CX_BGRA2Luv, CX_BGRA2XYZ,
    CX_BGRA2YCrCb, CX_BGRA2YUV,

    COLOR_GRAY2BGR, COLOR_GRAY2BGRA,

    COLOR_HLS2BGR, COLOR_HLS2BGR_FULL, COLOR_HLS2RGB, COLOR_HLS2RGB_FULL,
    CX_HLS2BGRA, CX_HLS2BGRA_FULL, CX_HLS2RGBA, CX_HLS2RGBA_FULL,

    COLOR_HSV2BGR, COLOR_HSV2BGR_FULL, COLOR_HSV2RGB, COLOR_HSV2RGB_FULL,
    CX_HSV2BGRA, CX_HSV2BGRA_FULL, CX_HSV2RGBA,    CX_HSV2RGBA_FULL,

    COLOR_Lab2BGR, COLOR_Lab2LBGR, COLOR_Lab2LRGB, COLOR_Lab2RGB,
    CX_Lab2BGRA, CX_Lab2LBGRA, CX_Lab2LRGBA, CX_Lab2RGBA,

    COLOR_LBGR2Lab, COLOR_LBGR2Luv, COLOR_LRGB2Lab, COLOR_LRGB2Luv,
    CX_LBGRA2Lab, CX_LBGRA2Luv, CX_LRGBA2Lab, CX_LRGBA2Luv,

    COLOR_Luv2BGR, COLOR_Luv2LBGR, COLOR_Luv2LRGB, COLOR_Luv2RGB,
    CX_Luv2BGRA, CX_Luv2LBGRA, CX_Luv2LRGBA, CX_Luv2RGBA,

    COLOR_RGB2GRAY,
    COLOR_RGB2HLS, COLOR_RGB2HLS_FULL, COLOR_RGB2HSV, COLOR_RGB2HSV_FULL,
    COLOR_RGB2Lab, COLOR_RGB2Luv, COLOR_RGB2XYZ, COLOR_RGB2YCrCb, COLOR_RGB2YUV,

    COLOR_RGBA2BGR, COLOR_RGBA2GRAY,
    CX_RGBA2HLS, CX_RGBA2HLS_FULL, CX_RGBA2HSV, CX_RGBA2HSV_FULL,
    CX_RGBA2Lab, CX_RGBA2Luv, CX_RGBA2XYZ,
    CX_RGBA2YCrCb, CX_RGBA2YUV,

    COLOR_XYZ2BGR, COLOR_XYZ2RGB, CX_XYZ2BGRA, CX_XYZ2RGBA,

    COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, CX_YCrCb2BGRA, CX_YCrCb2RGBA,
    COLOR_YUV2BGR, COLOR_YUV2RGB, CX_YUV2BGRA, CX_YUV2RGBA
    )

CV_ENUM(CvtModeBayer,
    COLOR_BayerBG2BGR, COLOR_BayerBG2BGRA, COLOR_BayerBG2BGR_VNG, COLOR_BayerBG2GRAY,
    COLOR_BayerGB2BGR, COLOR_BayerGB2BGRA, COLOR_BayerGB2BGR_VNG, COLOR_BayerGB2GRAY,
    COLOR_BayerGR2BGR, COLOR_BayerGR2BGRA, COLOR_BayerGR2BGR_VNG, COLOR_BayerGR2GRAY,
    COLOR_BayerRG2BGR, COLOR_BayerRG2BGRA, COLOR_BayerRG2BGR_VNG, COLOR_BayerRG2GRAY
    )


CV_ENUM(CvtMode2, COLOR_YUV2BGR_NV12, COLOR_YUV2BGRA_NV12, COLOR_YUV2RGB_NV12, COLOR_YUV2RGBA_NV12, COLOR_YUV2BGR_NV21, COLOR_YUV2BGRA_NV21, COLOR_YUV2RGB_NV21, COLOR_YUV2RGBA_NV21,
                  COLOR_YUV2BGR_YV12, COLOR_YUV2BGRA_YV12, COLOR_YUV2RGB_YV12, COLOR_YUV2RGBA_YV12, COLOR_YUV2BGR_IYUV, COLOR_YUV2BGRA_IYUV, COLOR_YUV2RGB_IYUV, COLOR_YUV2RGBA_IYUV,
                  COLOR_YUV2GRAY_420, COLOR_YUV2RGB_UYVY, COLOR_YUV2BGR_UYVY, COLOR_YUV2RGBA_UYVY, COLOR_YUV2BGRA_UYVY, COLOR_YUV2RGB_YUY2, COLOR_YUV2BGR_YUY2, COLOR_YUV2RGB_YVYU,
                  COLOR_YUV2BGR_YVYU, COLOR_YUV2RGBA_YUY2, COLOR_YUV2BGRA_YUY2, COLOR_YUV2RGBA_YVYU, COLOR_YUV2BGRA_YVYU,
                  COLOR_RGB2YUV_UYVY, COLOR_BGR2YUV_UYVY, COLOR_RGBA2YUV_UYVY, COLOR_BGRA2YUV_UYVY, COLOR_RGB2YUV_YUY2, COLOR_BGR2YUV_YUY2, COLOR_RGB2YUV_YVYU,
                  COLOR_BGR2YUV_YVYU, COLOR_RGBA2YUV_YUY2, COLOR_BGRA2YUV_YUY2, COLOR_RGBA2YUV_YVYU, COLOR_BGRA2YUV_YVYU)

CV_ENUM(CvtMode3, COLOR_RGB2YUV_IYUV, COLOR_BGR2YUV_IYUV, COLOR_RGBA2YUV_IYUV, COLOR_BGRA2YUV_IYUV,
                  COLOR_RGB2YUV_YV12, COLOR_BGR2YUV_YV12, COLOR_RGBA2YUV_YV12, COLOR_BGRA2YUV_YV12)

struct ChPair
{
    ChPair(int _scn, int _dcn): scn(_scn), dcn(_dcn) {}
    int scn, dcn;
};

static ChPair getConversionInfo(int cvtMode)
{
    switch(cvtMode)
    {
    case COLOR_BayerBG2GRAY: case COLOR_BayerGB2GRAY:
    case COLOR_BayerGR2GRAY: case COLOR_BayerRG2GRAY:
    case COLOR_YUV2GRAY_420:
        return ChPair(1,1);
    case COLOR_GRAY2BGR555: case COLOR_GRAY2BGR565:
        return ChPair(1,2);
    case COLOR_BayerBG2BGR: case COLOR_BayerBG2BGR_VNG:
    case COLOR_BayerGB2BGR: case COLOR_BayerGB2BGR_VNG:
    case COLOR_BayerGR2BGR: case COLOR_BayerGR2BGR_VNG:
    case COLOR_BayerRG2BGR: case COLOR_BayerRG2BGR_VNG:
    case COLOR_GRAY2BGR:
    case COLOR_YUV2BGR_NV12: case COLOR_YUV2RGB_NV12:
    case COLOR_YUV2BGR_NV21: case COLOR_YUV2RGB_NV21:
    case COLOR_YUV2BGR_YV12: case COLOR_YUV2RGB_YV12:
    case COLOR_YUV2BGR_IYUV: case COLOR_YUV2RGB_IYUV:
        return ChPair(1,3);
    case COLOR_GRAY2BGRA:
    case COLOR_YUV2BGRA_NV12: case COLOR_YUV2RGBA_NV12:
    case COLOR_YUV2BGRA_NV21: case COLOR_YUV2RGBA_NV21:
    case COLOR_YUV2BGRA_YV12: case COLOR_YUV2RGBA_YV12:
    case COLOR_YUV2BGRA_IYUV: case COLOR_YUV2RGBA_IYUV:
    case COLOR_BayerBG2BGRA: case COLOR_BayerGB2BGRA:
    case COLOR_BayerGR2BGRA: case COLOR_BayerRG2BGRA:
        return ChPair(1,4);
    case COLOR_BGR5552GRAY: case COLOR_BGR5652GRAY:
        return ChPair(2,1);
    case COLOR_BGR5552BGR: case COLOR_BGR5552RGB:
    case COLOR_BGR5652BGR: case COLOR_BGR5652RGB:
    case COLOR_YUV2RGB_UYVY: case COLOR_YUV2BGR_UYVY:
    case COLOR_YUV2RGB_YUY2: case COLOR_YUV2BGR_YUY2:
    case COLOR_YUV2RGB_YVYU: case COLOR_YUV2BGR_YVYU:
        return ChPair(2,3);
    case COLOR_RGB2YUV_UYVY: case COLOR_BGR2YUV_UYVY:
    case COLOR_RGB2YUV_YUY2: case COLOR_BGR2YUV_YUY2:
    case COLOR_RGB2YUV_YVYU: case COLOR_BGR2YUV_YVYU:
        return ChPair(3,2);
    case COLOR_BGR5552BGRA: case COLOR_BGR5552RGBA:
    case COLOR_BGR5652BGRA: case COLOR_BGR5652RGBA:
    case COLOR_YUV2RGBA_UYVY: case COLOR_YUV2BGRA_UYVY:
    case COLOR_YUV2RGBA_YUY2: case COLOR_YUV2BGRA_YUY2:
    case COLOR_YUV2RGBA_YVYU: case COLOR_YUV2BGRA_YVYU:
        return ChPair(2,4);
    case COLOR_RGBA2YUV_UYVY: case COLOR_BGRA2YUV_UYVY:
    case COLOR_RGBA2YUV_YUY2: case COLOR_BGRA2YUV_YUY2:
    case COLOR_RGBA2YUV_YVYU: case COLOR_BGRA2YUV_YVYU:
        return ChPair(4,2);
    case COLOR_BGR2GRAY: case COLOR_RGB2GRAY:
    case COLOR_RGB2YUV_IYUV: case COLOR_RGB2YUV_YV12:
    case COLOR_BGR2YUV_IYUV: case COLOR_BGR2YUV_YV12:
        return ChPair(3,1);
    case COLOR_BGR2BGR555: case COLOR_BGR2BGR565:
    case COLOR_RGB2BGR555: case COLOR_RGB2BGR565:
        return ChPair(3,2);
    case COLOR_BGR2HLS: case COLOR_BGR2HLS_FULL:
    case COLOR_BGR2HSV: case COLOR_BGR2HSV_FULL:
    case COLOR_BGR2Lab: case COLOR_BGR2Luv:
    case COLOR_BGR2RGB: case COLOR_BGR2XYZ:
    case COLOR_BGR2YCrCb: case COLOR_BGR2YUV:
    case COLOR_HLS2BGR: case COLOR_HLS2BGR_FULL:
    case COLOR_HLS2RGB: case COLOR_HLS2RGB_FULL:
    case COLOR_HSV2BGR: case COLOR_HSV2BGR_FULL:
    case COLOR_HSV2RGB: case COLOR_HSV2RGB_FULL:
    case COLOR_Lab2BGR: case COLOR_Lab2LBGR:
    case COLOR_Lab2LRGB: case COLOR_Lab2RGB:
    case COLOR_LBGR2Lab: case COLOR_LBGR2Luv:
    case COLOR_LRGB2Lab: case COLOR_LRGB2Luv:
    case COLOR_Luv2BGR: case COLOR_Luv2LBGR:
    case COLOR_Luv2LRGB: case COLOR_Luv2RGB:
    case COLOR_RGB2HLS: case COLOR_RGB2HLS_FULL:
    case COLOR_RGB2HSV: case COLOR_RGB2HSV_FULL:
    case COLOR_RGB2Lab: case COLOR_RGB2Luv:
    case COLOR_RGB2XYZ: case COLOR_RGB2YCrCb:
    case COLOR_RGB2YUV: case COLOR_XYZ2BGR:
    case COLOR_XYZ2RGB: case COLOR_YCrCb2BGR:
    case COLOR_YCrCb2RGB: case COLOR_YUV2BGR:
    case COLOR_YUV2RGB:
        return ChPair(3,3);
    case COLOR_BGR2BGRA: case COLOR_BGR2RGBA:
    case CX_HLS2BGRA: case CX_HLS2BGRA_FULL:
    case CX_HLS2RGBA: case CX_HLS2RGBA_FULL:
    case CX_HSV2BGRA: case CX_HSV2BGRA_FULL:
    case CX_HSV2RGBA: case CX_HSV2RGBA_FULL:
    case CX_Lab2BGRA: case CX_Lab2LBGRA:
    case CX_Lab2LRGBA: case CX_Lab2RGBA:
    case CX_Luv2BGRA: case CX_Luv2LBGRA:
    case CX_Luv2LRGBA: case CX_Luv2RGBA:
    case CX_XYZ2BGRA: case CX_XYZ2RGBA:
    case CX_YCrCb2BGRA: case CX_YCrCb2RGBA:
    case CX_YUV2BGRA: case CX_YUV2RGBA:
        return ChPair(3,4);
    case COLOR_BGRA2GRAY: case COLOR_RGBA2GRAY:
    case COLOR_RGBA2YUV_IYUV: case COLOR_RGBA2YUV_YV12:
    case COLOR_BGRA2YUV_IYUV: case COLOR_BGRA2YUV_YV12:
        return ChPair(4,1);
    case COLOR_BGRA2BGR555: case COLOR_BGRA2BGR565:
    case COLOR_RGBA2BGR555: case COLOR_RGBA2BGR565:
        return ChPair(4,2);
    case COLOR_BGRA2BGR: case CX_BGRA2HLS:
    case CX_BGRA2HLS_FULL: case CX_BGRA2HSV:
    case CX_BGRA2HSV_FULL: case CX_BGRA2Lab:
    case CX_BGRA2Luv: case CX_BGRA2XYZ:
    case CX_BGRA2YCrCb: case CX_BGRA2YUV:
    case CX_LBGRA2Lab: case CX_LBGRA2Luv:
    case CX_LRGBA2Lab: case CX_LRGBA2Luv:
    case COLOR_RGBA2BGR: case CX_RGBA2HLS:
    case CX_RGBA2HLS_FULL: case CX_RGBA2HSV:
    case CX_RGBA2HSV_FULL: case CX_RGBA2Lab:
    case CX_RGBA2Luv: case CX_RGBA2XYZ:
    case CX_RGBA2YCrCb: case CX_RGBA2YUV:
        return ChPair(4,3);
    case COLOR_BGRA2RGBA:
        return ChPair(4,4);
    default:
        ADD_FAILURE() << "Unknown conversion type";
        break;
    };
    return ChPair(0,0);
}

typedef tuple<Size, CvtMode> Size_CvtMode_t;
typedef perf::TestBaseWithParam<Size_CvtMode_t> Size_CvtMode;

PERF_TEST_P(Size_CvtMode, cvtColor8u,
            testing::Combine(
                testing::Values(::perf::szODD, ::perf::szVGA, ::perf::sz1080p),
                CvtMode::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int _mode = get<1>(GetParam()), mode = _mode;
    ChPair ch = getConversionInfo(mode);
    mode %= COLOR_COLORCVT_MAX;

    Mat src(sz, CV_8UC(ch.scn));
    Mat dst(sz, CV_8UC(ch.dcn));

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    int runs = sz.width <= 320 ? 100 : 5;
    TEST_CYCLE_MULTIRUN(runs) cvtColor(src, dst, mode, ch.dcn);

#if defined(__APPLE__) && defined(HAVE_IPP)
    SANITY_CHECK(dst, _mode == CX_BGRA2HLS_FULL ? 2 : 1);
#elif defined(_MSC_VER) && _MSC_VER >= 1900 /* MSVC 14 */
    if (_mode == CX_Luv2BGRA)
        SANITY_CHECK_NOTHING();
    else
        SANITY_CHECK(dst, 1);
#else
    SANITY_CHECK(dst, 1);
#endif
}


typedef tuple<Size, CvtMode16U> Size_CvtMode16U_t;
typedef perf::TestBaseWithParam<Size_CvtMode16U_t> Size_CvtMode16U;

PERF_TEST_P(Size_CvtMode16U, DISABLED_cvtColor_16u,
            testing::Combine(
                testing::Values(::perf::szODD, ::perf::szVGA, ::perf::sz1080p),
                CvtMode16U::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int _mode = get<1>(GetParam()), mode = _mode;
    ChPair ch = getConversionInfo(mode);
    mode %= COLOR_COLORCVT_MAX;
    Mat src(sz, CV_16UC(ch.scn));
    Mat dst(sz, CV_16UC(ch.scn));

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    int runs = sz.width <= 320 ? 100 : 5;
    TEST_CYCLE_MULTIRUN(runs) cvtColor(src, dst, mode, ch.dcn);

    SANITY_CHECK(dst, 1);
}


typedef tuple<Size, CvtMode32F> Size_CvtMode32F_t;
typedef perf::TestBaseWithParam<Size_CvtMode32F_t> Size_CvtMode32F;

PERF_TEST_P(Size_CvtMode32F, DISABLED_cvtColor_32f,
            testing::Combine(
                testing::Values(::perf::szODD, ::perf::szVGA, ::perf::sz1080p),
                CvtMode32F::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int _mode = get<1>(GetParam()), mode = _mode;
    ChPair ch = getConversionInfo(mode);
    mode %= COLOR_COLORCVT_MAX;
    Mat src(sz, CV_32FC(ch.scn));
    Mat dst(sz, CV_32FC(ch.scn));

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    int runs = sz.width <= 320 ? 100 : 5;
    TEST_CYCLE_MULTIRUN(runs) cvtColor(src, dst, mode, ch.dcn);

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, CvtModeBayer> Size_CvtMode_Bayer_t;
typedef perf::TestBaseWithParam<Size_CvtMode_Bayer_t> Size_CvtMode_Bayer;

PERF_TEST_P(Size_CvtMode_Bayer, cvtColorBayer8u,
            testing::Combine(
                testing::Values(::perf::szODD, ::perf::szVGA),
                CvtModeBayer::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    ChPair ch = getConversionInfo(mode);
    mode %= COLOR_COLORCVT_MAX;

    Mat src(sz, CV_8UC(ch.scn));
    Mat dst(sz, CV_8UC(ch.dcn));

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() cvtColor(src, dst, mode, ch.dcn);

    SANITY_CHECK_NOTHING();
}

typedef tuple<Size, CvtMode2> Size_CvtMode2_t;
typedef perf::TestBaseWithParam<Size_CvtMode2_t> Size_CvtMode2;

PERF_TEST_P(Size_CvtMode2, cvtColorYUV420,
            testing::Combine(
                testing::Values(szVGA, sz1080p, Size(130, 60)),
                CvtMode2::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    ChPair ch = getConversionInfo(mode);

    Mat src(sz.height + sz.height / 2, sz.width, CV_8UC(ch.scn));
    Mat dst(sz, CV_8UC(ch.dcn));

    declare.in(src, WARMUP_RNG).out(dst);

    int runs = (sz.width <= 640) ? 8 : 1;
    TEST_CYCLE_MULTIRUN(runs) cvtColor(src, dst, mode, ch.dcn);

    SANITY_CHECK(dst, 1);
}

typedef tuple<Size, CvtMode3> Size_CvtMode3_t;
typedef perf::TestBaseWithParam<Size_CvtMode3_t> Size_CvtMode3;

PERF_TEST_P(Size_CvtMode3, cvtColorRGB2YUV420p,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p, Size(130, 60)),
                CvtMode3::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    ChPair ch = getConversionInfo(mode);

    Mat src(sz, CV_8UC(ch.scn));
    Mat dst(sz.height + sz.height / 2, sz.width, CV_8UC(ch.dcn));

    declare.time(100);
    declare.in(src, WARMUP_RNG).out(dst);

    int runs = (sz.width <= 640) ? 10 : 1;
    TEST_CYCLE_MULTIRUN(runs) cvtColor(src, dst, mode, ch.dcn);

    SANITY_CHECK(dst, 1);
}

CV_ENUM(EdgeAwareBayerMode, COLOR_BayerBG2BGR_EA, COLOR_BayerGB2BGR_EA, COLOR_BayerRG2BGR_EA, COLOR_BayerGR2BGR_EA)

typedef tuple<Size, EdgeAwareBayerMode> EdgeAwareParams;
typedef perf::TestBaseWithParam<EdgeAwareParams> EdgeAwareDemosaicingTest;

PERF_TEST_P(EdgeAwareDemosaicingTest, demosaicingEA,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p, Size(130, 60)),
                EdgeAwareBayerMode::all()
                )
            )
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());

    Mat src(sz, CV_8UC1);
    Mat dst(sz, CV_8UC3);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() cvtColor(src, dst, mode, 3);

    SANITY_CHECK(dst, 1);
}

} // namespace
