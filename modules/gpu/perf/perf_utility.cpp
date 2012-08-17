#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

void fillRandom(Mat& m, double a, double b)
{
    RNG rng(123456789);
    rng.fill(m, RNG::UNIFORM, Scalar::all(a), Scalar::all(b));
}

Mat readImage(const string& fileName, int flags)
{
    return imread(perf::TestBase::getDataPath(fileName), flags);
}

void PrintTo(const CvtColorInfo& info, ostream* os)
{
    static const char* str[] =
    {
        "BGR2BGRA",
        "BGRA2BGR",
        "BGR2RGBA",
        "RGBA2BGR",
        "BGR2RGB",
        "BGRA2RGBA",

        "BGR2GRAY",
        "RGB2GRAY",
        "GRAY2BGR",
        "GRAY2BGRA",
        "BGRA2GRAY",
        "RGBA2GRAY",

        "BGR2BGR565",
        "RGB2BGR565",
        "BGR5652BGR",
        "BGR5652RGB",
        "BGRA2BGR565",
        "RGBA2BGR565",
        "BGR5652BGRA",
        "BGR5652RGBA",

        "GRAY2BGR565",
        "BGR5652GRAY",

        "BGR2BGR555",
        "RGB2BGR555",
        "BGR5552BGR",
        "BGR5552RGB",
        "BGRA2BGR555",
        "RGBA2BGR555",
        "BGR5552BGRA",
        "BGR5552RGBA",

        "GRAY2BGR555",
        "BGR5552GRAY",

        "BGR2XYZ",
        "RGB2XYZ",
        "XYZ2BGR",
        "XYZ2RGB",

        "BGR2YCrCb",
        "RGB2YCrCb",
        "YCrCb2BGR",
        "YCrCb2RGB",

        "BGR2HSV",
        "RGB2HSV",

        "",
        "",

        "BGR2Lab",
        "RGB2Lab",

        "BayerBG2BGR",
        "BayerGB2BGR",
        "BayerRG2BGR",
        "BayerGR2BGR",

        "BGR2Luv",
        "RGB2Luv",

        "BGR2HLS",
        "RGB2HLS",

        "HSV2BGR",
        "HSV2RGB",

        "Lab2BGR",
        "Lab2RGB",
        "Luv2BGR",
        "Luv2RGB",

        "HLS2BGR",
        "HLS2RGB",

        "BayerBG2BGR_VNG",
        "BayerGB2BGR_VNG",
        "BayerRG2BGR_VNG",
        "BayerGR2BGR_VNG",

        "BGR2HSV_FULL",
        "RGB2HSV_FULL",
        "BGR2HLS_FULL",
        "RGB2HLS_FULL",

        "HSV2BGR_FULL",
        "HSV2RGB_FULL",
        "HLS2BGR_FULL",
        "HLS2RGB_FULL",

        "LBGR2Lab",
        "LRGB2Lab",
        "LBGR2Luv",
        "LRGB2Luv",

        "Lab2LBGR",
        "Lab2LRGB",
        "Luv2LBGR",
        "Luv2LRGB",

        "BGR2YUV",
        "RGB2YUV",
        "YUV2BGR",
        "YUV2RGB",

        "BayerBG2GRAY",
        "BayerGB2GRAY",
        "BayerRG2GRAY",
        "BayerGR2GRAY",

        //YUV 4:2:0 formats family
        "YUV2RGB_NV12",
        "YUV2BGR_NV12",
        "YUV2RGB_NV21",
        "YUV2BGR_NV21",

        "YUV2RGBA_NV12",
        "YUV2BGRA_NV12",
        "YUV2RGBA_NV21",
        "YUV2BGRA_NV21",

        "YUV2RGB_YV12",
        "YUV2BGR_YV12",
        "YUV2RGB_IYUV",
        "YUV2BGR_IYUV",

        "YUV2RGBA_YV12",
        "YUV2BGRA_YV12",
        "YUV2RGBA_IYUV",
        "YUV2BGRA_IYUV",

        "YUV2GRAY_420",

        //YUV 4:2:2 formats family
        "YUV2RGB_UYVY",
        "YUV2BGR_UYVY",
        "YUV2RGB_VYUY",
        "YUV2BGR_VYUY",

        "YUV2RGBA_UYVY",
        "YUV2BGRA_UYVY",
        "YUV2RGBA_VYUY",
        "YUV2BGRA_VYUY",

        "YUV2RGB_YUY2",
        "YUV2BGR_YUY2",
        "YUV2RGB_YVYU",
        "YUV2BGR_YVYU",

        "YUV2RGBA_YUY2",
        "YUV2BGRA_YUY2",
        "YUV2RGBA_YVYU",
        "YUV2BGRA_YVYU",

        "YUV2GRAY_UYVY",
        "YUV2GRAY_YUY2",

        // alpha premultiplication
        "RGBA2mRGBA",
        "mRGBA2RGBA",

        "COLORCVT_MAX"
    };

    *os << str[info.code];
}
