// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include "perf_precomp.hpp"

namespace opencv_test {

CV_ENUM(PatchDepth, CV_8U, CV_32F)

typedef tuple<Size, MatType, PatchDepth> GetRectSubPixParams;
typedef perf::TestBaseWithParam<GetRectSubPixParams> GetRectSubPix_Test;

// Chessboard inner corners (9x6) of the cv/stereo/case1 boards, detected once with
// findChessboardCorners and hard-coded so this perf test needs no calib dependency.
static const Point2f corners_left01[] = {
    Point2f(244.4541f, 94.3314f), Point2f(274.6218f, 92.2413f), Point2f(305.4939f, 90.4029f),
    Point2f(338.3641f, 88.8363f), Point2f(371.5922f, 87.9836f), Point2f(406.8435f, 86.9164f),
    Point2f(441.6335f, 86.3721f), Point2f(477.6230f, 86.3797f), Point2f(513.9866f, 86.7991f),
    Point2f(244.9995f, 126.5384f), Point2f(274.9684f, 124.7523f), Point2f(306.0444f, 124.0323f),
    Point2f(338.7149f, 123.2651f), Point2f(372.0939f, 122.1290f), Point2f(406.6485f, 122.1684f),
    Point2f(442.1944f, 122.2465f), Point2f(477.8674f, 122.1053f), Point2f(514.0326f, 122.9214f),
    Point2f(245.4698f, 158.3554f), Point2f(275.3708f, 158.3770f), Point2f(306.4206f, 157.6200f),
    Point2f(338.7076f, 157.4221f), Point2f(372.3548f, 157.3757f), Point2f(406.5436f, 157.4915f),
    Point2f(442.1193f, 157.6772f), Point2f(477.6427f, 158.4171f), Point2f(513.5308f, 159.2809f),
    Point2f(246.3835f, 190.2633f), Point2f(275.7964f, 190.6393f), Point2f(307.2744f, 191.0970f),
    Point2f(339.1837f, 191.5760f), Point2f(372.3740f, 191.8712f), Point2f(406.9517f, 192.4470f),
    Point2f(441.5515f, 193.6797f), Point2f(477.3305f, 194.3325f), Point2f(513.2557f, 195.5269f),
    Point2f(247.4330f, 222.4675f), Point2f(276.8233f, 223.4360f), Point2f(307.4332f, 224.2435f),
    Point2f(339.5576f, 225.2065f), Point2f(372.9427f, 226.4245f), Point2f(406.4261f, 227.5323f),
    Point2f(441.4084f, 228.5345f), Point2f(476.7176f, 230.1001f), Point2f(511.5294f, 231.5994f),
    Point2f(248.7999f, 253.5990f), Point2f(277.5086f, 255.1773f), Point2f(308.5765f, 256.4403f),
    Point2f(340.3875f, 258.2953f), Point2f(372.7524f, 259.7994f), Point2f(406.1046f, 261.6587f),
    Point2f(440.4359f, 263.3060f), Point2f(475.4150f, 264.7134f), Point2f(510.1775f, 266.1992f),
};

static const Point2f corners_left02[] = {
    Point2f(256.5867f, 357.3861f), Point2f(255.3215f, 334.4808f), Point2f(254.7554f, 309.0343f),
    Point2f(253.2735f, 280.1345f), Point2f(252.5384f, 248.0337f), Point2f(251.7535f, 212.7699f),
    Point2f(251.2085f, 172.4518f), Point2f(251.3873f, 128.3270f), Point2f(251.4556f, 78.3560f),
    Point2f(291.6361f, 365.9622f), Point2f(292.5572f, 343.3497f), Point2f(293.5726f, 318.2415f),
    Point2f(295.2780f, 289.9100f), Point2f(296.7264f, 257.9767f), Point2f(298.7410f, 222.1348f),
    Point2f(301.2631f, 182.4788f), Point2f(303.7642f, 136.7058f), Point2f(307.1471f, 86.7391f),
    Point2f(327.6240f, 374.3524f), Point2f(330.8632f, 352.3100f), Point2f(334.4490f, 327.3103f),
    Point2f(337.8723f, 299.4167f), Point2f(342.3027f, 267.7063f), Point2f(347.3625f, 232.3861f),
    Point2f(352.3509f, 192.1027f), Point2f(358.5134f, 147.0249f), Point2f(365.3164f, 97.3291f),
    Point2f(364.5553f, 381.6275f), Point2f(369.4413f, 360.3667f), Point2f(375.1765f, 335.9256f),
    Point2f(381.4051f, 308.1661f), Point2f(388.5566f, 277.6599f), Point2f(396.0829f, 242.0081f),
    Point2f(404.5149f, 202.7227f), Point2f(413.7823f, 158.0101f), Point2f(424.3409f, 107.9548f),
    Point2f(401.3313f, 389.5627f), Point2f(408.5090f, 368.3929f), Point2f(416.5022f, 344.4312f),
    Point2f(425.1089f, 317.5573f), Point2f(434.4975f, 286.7095f), Point2f(445.0846f, 252.4719f),
    Point2f(456.7321f, 213.5597f), Point2f(469.1978f, 169.3279f), Point2f(483.2666f, 120.4614f),
    Point2f(437.8366f, 396.5025f), Point2f(446.5643f, 375.8865f), Point2f(456.9603f, 352.2726f),
    Point2f(467.8020f, 326.0813f), Point2f(480.4469f, 295.9886f), Point2f(493.1928f, 262.2359f),
    Point2f(508.0341f, 223.9814f), Point2f(523.4081f, 181.1591f), Point2f(540.2076f, 133.2355f),
};

static const Point2f corners_left03[] = {
    Point2f(277.4323f, 72.5290f), Point2f(313.4367f, 81.5630f), Point2f(352.9303f, 91.0506f),
    Point2f(393.5079f, 101.8541f), Point2f(434.6314f, 113.5358f), Point2f(476.8674f, 126.6859f),
    Point2f(519.5198f, 139.4719f), Point2f(562.0754f, 154.0302f), Point2f(603.5757f, 168.5026f),
    Point2f(259.6532f, 105.6526f), Point2f(297.3900f, 115.0693f), Point2f(337.2212f, 126.4162f),
    Point2f(378.3126f, 137.9841f), Point2f(421.3062f, 150.6102f), Point2f(464.5258f, 164.5774f),
    Point2f(508.5670f, 178.7557f), Point2f(552.4766f, 193.4379f), Point2f(595.2234f, 208.4552f),
    Point2f(242.7253f, 141.3230f), Point2f(279.8400f, 152.2348f), Point2f(320.5120f, 163.9313f),
    Point2f(362.5777f, 177.0284f), Point2f(406.0779f, 190.5102f), Point2f(450.5823f, 205.4244f),
    Point2f(495.9811f, 220.1399f), Point2f(541.2142f, 235.4248f), Point2f(585.0867f, 250.9424f),
    Point2f(223.8041f, 178.0292f), Point2f(262.7206f, 190.8443f), Point2f(303.1327f, 204.2207f),
    Point2f(345.8119f, 218.4479f), Point2f(389.5984f, 233.0941f), Point2f(435.4800f, 248.5219f),
    Point2f(481.7506f, 264.2025f), Point2f(528.0433f, 280.0055f), Point2f(573.3781f, 295.7106f),
    Point2f(205.9287f, 217.1566f), Point2f(244.4685f, 230.9739f), Point2f(285.0102f, 245.5548f),
    Point2f(328.2179f, 261.5405f), Point2f(373.1269f, 277.2205f), Point2f(419.5606f, 293.7301f),
    Point2f(466.3259f, 310.0434f), Point2f(513.5237f, 326.6496f), Point2f(559.9616f, 342.5871f),
    Point2f(187.4462f, 257.4862f), Point2f(225.3331f, 272.8211f), Point2f(266.7754f, 289.3292f),
    Point2f(309.9885f, 305.9286f), Point2f(354.7533f, 323.2353f), Point2f(402.4073f, 340.2387f),
    Point2f(449.5339f, 357.9061f), Point2f(497.6018f, 374.2831f), Point2f(544.2325f, 390.1797f),
};

struct Board { const char* path; const Point2f* corners; int ncorners; };
static const Board g_boards[] = {
    { "cv/stereo/case1/left01.png", corners_left01, (int)(sizeof(corners_left01) / sizeof(Point2f)) },
    { "cv/stereo/case1/left02.png", corners_left02, (int)(sizeof(corners_left02) / sizeof(Point2f)) },
    { "cv/stereo/case1/left03.png", corners_left03, (int)(sizeof(corners_left03) / sizeof(Point2f)) },
};

PERF_TEST_P(GetRectSubPix_Test, getRectSubPix,
            testing::Combine(
                testing::Values(Size(16, 16), Size(32, 32), Size(64, 64), Size(128, 128)),
                testing::Values(CV_8UC1, CV_32FC1),
                PatchDepth::all()
                )
            )
{
    Size patchSize = get<0>(GetParam());
    int srcType    = get<1>(GetParam());
    int patchDepth = get<2>(GetParam());

    // getRectSubPix supports only 8U->8U, 8U->32F and 32F->32F
    if (CV_MAT_DEPTH(srcType) == CV_32F && patchDepth == CV_8U)
        throw SkipTestException("getRectSubPix does not support 32F source with 8U patch");

    const size_t nboards = sizeof(g_boards) / sizeof(g_boards[0]);
    std::vector<Mat> srcs;
    for (size_t b = 0; b < nboards; b++)
    {
        Mat image = imread(getDataPath(g_boards[b].path), IMREAD_GRAYSCALE);
        ASSERT_FALSE(image.empty());
        if (CV_MAT_DEPTH(srcType) == CV_32F)
        {
            Mat f;
            image.convertTo(f, CV_32F);
            srcs.push_back(f);
        }
        else
            srcs.push_back(image);
    }

    int patchType = CV_MAKETYPE(patchDepth, 1);
    Mat patch(patchSize, patchType);

    declare.in(srcs[0]);

    TEST_CYCLE()
    {
        for (size_t b = 0; b < nboards; b++)
            for (int i = 0; i < g_boards[b].ncorners; i++)
                getRectSubPix(srcs[b], patchSize, g_boards[b].corners[i], patch, patchType);
    }
// Copyright (C) 2026, Intel Corporation, all rights reserved.
#include "perf_precomp.hpp"

namespace opencv_test
{
using namespace perf;

// src/dst depths swept independently (matches IPP getRectSubPix coverage).
#define GETRECTSUBPIX_TYPES CV_8UC1, CV_32FC1

typedef tuple<Size, MatType, MatType> Size_SrcType_DstType_t;
typedef perf::TestBaseWithParam<Size_SrcType_DstType_t> Size_SrcType_DstType;

PERF_TEST_P(Size_SrcType_DstType, getRectSubPix,
            testing::Combine(
                testing::Values(szVGA, sz720p, sz1080p),
                testing::Values(GETRECTSUBPIX_TYPES),
                testing::Values(GETRECTSUBPIX_TYPES)
                ))
{
    Size sz      = get<0>(GetParam());
    int  srcType = get<1>(GetParam());
    int  dstType = get<2>(GetParam());

    // getRectSubPix does not support a lower-depth output than the input
    if (CV_MAT_DEPTH(dstType) < CV_MAT_DEPTH(srcType))
        throw ::perf::TestBase::PerfSkipTestException();

    Size    rectSize(std::min(800, sz.width), std::min(600, sz.height));
    Point2f rectCenter(100.33f, 100.77f);

    Mat src(sz, srcType);
    Mat dst(rectSize, dstType);

    declare.in(src, WARMUP_RNG).out(dst);

    TEST_CYCLE() cv::getRectSubPix(src, rectSize, rectCenter, dst, dstType);

    SANITY_CHECK_NOTHING();
}

} // namespace
