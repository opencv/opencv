// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4)

typedef TestBaseWithParam< tuple<Size, MatType, MatType, InterType> > TestRemap;

PERF_TEST_P( TestRemap, Remap,
             Combine(
                Values( szVGA, sz1080p ),
                Values( CV_16UC1, CV_16SC1, CV_32FC1 ),
                Values( CV_16SC2, CV_32FC1, CV_32FC2 ),
                InterType::all()
             )
)
{
    Size sz;
    int src_type, map1_type, inter_type;

    sz         = get<0>(GetParam());
    src_type   = get<1>(GetParam());
    map1_type  = get<2>(GetParam());
    inter_type = get<3>(GetParam());

    Mat src(sz, src_type), dst(sz, src_type), map1(sz, map1_type), map2;
    if (map1_type == CV_32FC1)
        map2.create(sz, CV_32FC1);
    else if (inter_type != INTER_NEAREST && map1_type == CV_16SC2)
    {
        map2.create(sz, CV_16UC1);
        map2 = Scalar::all(0);
    }

    RNG rng;
    rng.fill(src, RNG::UNIFORM, 0, 256);

    for (int j = 0; j < map1.rows; ++j)
        for (int i = 0; i < map1.cols; ++i)
            switch (map1_type)
            {
                case CV_32FC1:
                    map1.at<float>(j, i) = static_cast<float>(src.cols - i - 1);
                    map2.at<float>(j, i) = static_cast<float>(j);
                    break;
                case CV_32FC2:
                    map1.at<Vec2f>(j, i)[0] = static_cast<float>(src.cols - i - 1);
                    map1.at<Vec2f>(j, i)[1] = static_cast<float>(j);
                    break;
                case CV_16SC2:
                    map1.at<Vec2s>(j, i)[0] = static_cast<short>(src.cols - i - 1);
                    map1.at<Vec2s>(j, i)[1] = static_cast<short>(j);
                    break;
                default:
                    CV_Assert(0);
            }


    declare.in(src, WARMUP_RNG).out(dst).time(20);

    int runs = (sz.width <= 640) ? 3 : 1;
    TEST_CYCLE_MULTIRUN(runs) remap(src, dst, map1, map2, inter_type);

    if (inter_type == INTER_CUBIC)
    {
        SANITY_CHECK_NOTHING();
    }
    else
    {
        SANITY_CHECK(dst);
    }
}

// INTER_CUBIC is not supported for half depths
CV_ENUM(HalfRemapInter, INTER_LINEAR, INTER_LANCZOS4)

typedef TestBaseWithParam< tuple<Size, MatType, HalfRemapInter> > TestRemapHalf;

// mirrored map with a fractional offset, so every pixel is interpolated;
// values in [0, 1) keep fp16 away from its overflow range
static void halfRemapData(Size sz, int type, Mat& src, Mat& mapx, Mat& mapy)
{
    Mat src32(sz, CV_MAKETYPE(CV_32F, CV_MAT_CN(type)));
    randu(src32, Scalar::all(0), Scalar::all(1));
    src32.convertTo(src, type);

    mapx.create(sz, CV_32FC1);
    mapy.create(sz, CV_32FC1);
    for (int j = 0; j < sz.height; ++j)
        for (int i = 0; i < sz.width; ++i)
        {
            mapx.at<float>(j, i) = static_cast<float>(sz.width - i - 1) - 0.375f;
            mapy.at<float>(j, i) = static_cast<float>(j) + 0.375f;
        }
}

// the fp32 rows are the reference for the half rows
PERF_TEST_P( TestRemapHalf, RemapHalf,
             Combine(
                Values( TYPICAL_MAT_SIZES ),
                Values( CV_16FC1, CV_16FC3, CV_16BFC1, CV_16BFC3, CV_32FC1, CV_32FC3 ),
                HalfRemapInter::all()
             )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int inter = get<2>(GetParam());

    Mat src, mapx, mapy, dst(sz, type);
    halfRemapData(sz, type, src, mapx, mapy);

    declare.in(src).out(dst);

    TEST_CYCLE() remap(src, dst, mapx, mapy, inter);

    SANITY_CHECK_NOTHING();
}

// convert to fp32, remap, convert back: the route available without native
// half support, so the native half path has to beat it
PERF_TEST_P( TestRemapHalf, RemapHalfViaFp32,
             Combine(
                Values( TYPICAL_MAT_SIZES ),
                Values( CV_16FC1, CV_16FC3, CV_16BFC1, CV_16BFC3 ),
                HalfRemapInter::all()
             )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int inter = get<2>(GetParam());

    Mat src, mapx, mapy, dst(sz, type), src32, dst32;
    halfRemapData(sz, type, src, mapx, mapy);

    declare.in(src).out(dst);

    TEST_CYCLE()
    {
        src.convertTo(src32, CV_32F);
        remap(src32, dst32, mapx, mapy, inter);
        dst32.convertTo(dst, CV_MAT_DEPTH(type));
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
