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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Size, int, int> > Video_Acc_Cn4;

TEST_P(Video_Acc_Cn4, accuracy)
{
    const Size size = get<0>(GetParam());
    const int pattern = get<1>(GetParam());
    const int srcType = get<2>(GetParam());

    RNG& rng = theRNG();

    Mat src(size, srcType);
    Mat dst(size, CV_32FC4);
    Mat mask(size, CV_8UC1);

    if (srcType == CV_8UC4)
        rng.fill(src, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    else
        rng.fill(src, RNG::UNIFORM, Scalar::all(-10.0), Scalar::all(10.0));

    rng.fill(dst, RNG::UNIFORM, Scalar::all(-1000.0), Scalar::all(1000.0));

    for (int y = 0; y < mask.rows; ++y)
    {
        uchar* row = mask.ptr<uchar>(y);

        for (int x = 0; x < mask.cols; ++x)
        {
            switch (pattern)
            {
            case 0:
                row[x] = 0;
                break;
            case 1:
                row[x] = 255;
                break;
            case 2:
                row[x] = ((x + y) % 2) ? 255 : 0;
                break;
            case 3:
                row[x] = ((x * 13 + y * 7) % 5) ? 255 : 0;
                break;
            default:
                row[x] = ((x * 17 + y * 11) % 3) ? 255 : 0;
                break;
            }
        }
    }

    Mat dstRef = dst.clone();

    if (srcType == CV_32FC4)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            const Vec4f* srcRow = src.ptr<Vec4f>(y);
            Vec4f* dstRefRow = dstRef.ptr<Vec4f>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                if (maskRow[x])
                {
                    for (int c = 0; c < 4; ++c)
                        dstRefRow[x][c] += srcRow[x][c];
                }
            }
        }
    }
    else
    {
        CV_Assert(srcType == CV_8UC4);

        for (int y = 0; y < src.rows; ++y)
        {
            const Vec4b* srcRow = src.ptr<Vec4b>(y);
            Vec4f* dstRefRow = dstRef.ptr<Vec4f>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);

            for (int x = 0; x < src.cols; ++x)
            {
                if (maskRow[x])
                {
                    for (int c = 0; c < 4; ++c)
                        dstRefRow[x][c] += static_cast<float>(srcRow[x][c]);
                }
            }
        }
    }

    cv::accumulate(src, dst, mask);

    const double err = cv::norm(dst, dstRef, NORM_INF);

    EXPECT_EQ(0.0, err)
        << "size=" << size
        << ", pattern=" << pattern
        << ", srcType=" << srcType;
}

INSTANTIATE_TEST_CASE_P(Accumulate,
    Video_Acc_Cn4,
    testing::Combine(
        testing::Values(Size(1, 1),
                        Size(3, 5),
                        Size(17, 7),
                        Size(37, 19),
                        Size(128, 16),
                        Size(641, 37)),
        testing::Values(0, 1, 2, 3, 4),
        testing::Values(CV_32FC4, CV_8UC4)));

}} // namespace
