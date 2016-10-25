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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "test_precomp.hpp"
#include "opencv2/highgui.hpp"

#include <vector>
#include <string>
using namespace std;
using namespace cv;

#undef RENDER_MSERS
#define RENDER_MSERS 0

#if defined RENDER_MSERS && RENDER_MSERS
static void renderMSERs(const Mat& gray, Mat& img, const vector<vector<Point> >& msers)
{
    cvtColor(gray, img, COLOR_GRAY2BGR);
    RNG rng((uint64)1749583);
    for( int i = 0; i < (int)msers.size(); i++ )
    {
        uchar b = rng.uniform(0, 256);
        uchar g = rng.uniform(0, 256);
        uchar r = rng.uniform(0, 256);
        Vec3b color(b, g, r);

        const Point* pt = &msers[i][0];
        size_t j, n = msers[i].size();
        for( j = 0; j < n; j++ )
            img.at<Vec3b>(pt[j]) = color;
    }
}
#endif

TEST(Features2d_MSER, cases)
{
    uchar buf[] =
    {
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,
         255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,
         255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,
         255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
         255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
    };
    Mat big_image = imread(cvtest::TS::ptr()->get_data_path() + "mser/puzzle.png", 0);
    Mat small_image(14, 26, CV_8U, buf);
    static const int thresharr[] = { 0, 70, 120, 180, 255 };

    const int kDelta = 5;
    Ptr<MSER> mserExtractor = MSER::create( kDelta );
    vector<vector<Point> > msers;
    vector<Rect> boxes;

    RNG rng((uint64)123456);

    for( int i = 0; i < 100; i++ )
    {
        bool use_big_image = rng.uniform(0, 7) != 0;
        bool invert = rng.uniform(0, 2) != 0;
        bool binarize = use_big_image ? rng.uniform(0, 5) != 0 : false;
        bool blur = rng.uniform(0, 2) != 0;
        int thresh = thresharr[rng.uniform(0, 5)];

        /*if( i == 0 )
        {
            use_big_image = true;
            invert = binarize = blur = false;
        }*/

        const Mat& src0 = use_big_image ? big_image : small_image;
        Mat src = src0.clone();

        int kMinArea = use_big_image ? 256 : 10;
        int kMaxArea = (int)src.total()/4;

        mserExtractor->setMinArea(kMinArea);
        mserExtractor->setMaxArea(kMaxArea);

        if( invert )
            bitwise_not(src, src);
        if( binarize )
            threshold(src, src, thresh, 255, THRESH_BINARY);
        if( blur )
            GaussianBlur(src, src, Size(5, 5), 1.5, 1.5);

        int minRegs = use_big_image ? 7 : 2;
        int maxRegs = use_big_image ? 1000 : 20;
        if( binarize && (thresh == 0 || thresh == 255) )
            minRegs = maxRegs = 0;

        mserExtractor->detectRegions( src, msers, boxes );
        int nmsers = (int)msers.size();
        ASSERT_EQ(nmsers, (int)boxes.size());

        if( maxRegs < nmsers || minRegs > nmsers )
        {
            printf("%d. minArea=%d, maxArea=%d, nmsers=%d, minRegs=%d, maxRegs=%d, "
                   "image=%s, invert=%d, binarize=%d, thresh=%d, blur=%d\n",
                   i, kMinArea, kMaxArea, nmsers, minRegs, maxRegs, use_big_image ? "big" : "small",
                   (int)invert, (int)binarize, thresh, (int)blur);
    #if defined RENDER_MSERS && RENDER_MSERS
            Mat image;
            imshow("source", src);
            renderMSERs(src, image, msers);
            imshow("result", image);
            waitKey();
    #endif
        }

        ASSERT_LE(minRegs, nmsers);
        ASSERT_GE(maxRegs, nmsers);
    }
}
