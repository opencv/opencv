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
// Copyright (C) 2015, Smart Engines Ltd, all rights reserved.
// Copyright (C) 2015, Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute), all rights reserved.
// Copyright (C) 2015, Dmitry Nikolaev, Simon Karpenko, Michail Aliev, Elena "Erik Yorsh" Kuznetsova, all rights reserved.
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

#include "perf_precomp.hpp"
#include "fast_hough_transform.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<string, MatDepth> Image_dstDepth_t;
typedef perf::TestBaseWithParam<Image_dstDepth_t> Image_dstDepth;

#define MAT_DEPHTS CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F

PERF_TEST_P(Image_dstDepth, FastHoughTransform,
            testing::Combine(
                testing::Values("cv/shared/pic5.png", "stitching/a1.png"),
                testing::Values(MAT_DEPHTS)
                )
            )
{
    string   filename = getDataPath(get<0>(GetParam()));
    MatDepth depth    = get<1>(GetParam());

    Mat src = imread(filename, IMREAD_GRAYSCALE);
    if (src.empty())
        FAIL() << "Unable to load source image" << filename;

    Mat fht(2 * (src.cols + src.rows) - 3,
            src.cols + src.rows,
            CV_MAKETYPE(depth, src.channels()));;
    declare.in(src).out(fht);
    TEST_CYCLE() FastHoughTransform(src,
                                    fht,
                                    depth);

    SANITY_CHECK(fht);
}

#undef MAT_DEPHTS