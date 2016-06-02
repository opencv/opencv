// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;
using std::tr1::make_tuple;
using std::tr1::get;

typedef std::tr1::tuple<string, Point, int, int, int, int> Size_Source_Fl_t;
typedef perf::TestBaseWithParam<Size_Source_Fl_t> Size_Source_Fl;

PERF_TEST_P(Size_Source_Fl, floodFill1, Combine(
    testing::Values("cv/shared/fruits.png", "cv/optflow/RubberWhale1.png"), //images
            testing::Values(Point(120, 82), Point(200, 140)), //seed points
            testing::Values(4,8), //connectivity
            testing::Values((int)IMREAD_COLOR, (int)IMREAD_GRAYSCALE), //color image, or not
            testing::Values(0, 1, 2), //use fixed(1), gradient (2) or simple(0) mode
            testing::Values((int)CV_8U, (int)CV_32F, (int)CV_32S) //image depth
            ))
{
    //test given image(s)
    string filename = getDataPath(get<0>(GetParam()));
    Point pseed;
    pseed = get<1>(GetParam());

    int connectivity = get<2>(GetParam());
    int colorType = get<3>(GetParam());
    int modeType = get<4>(GetParam());
    int imdepth = get<5>(GetParam());

    Mat image0 = imread(filename, colorType);

    Scalar newval, loVal, upVal;
    if (modeType == 0)
    {
        loVal = Scalar(0, 0, 0);
        upVal = Scalar(0, 0, 0);
    }
    else
    {
        loVal = Scalar(4, 4, 4);
        upVal = Scalar(20, 20, 20);
    }
    int newMaskVal = 255;  //base mask for floodfill type
    int flags = connectivity + (newMaskVal << 8) + (modeType == 1 ? FLOODFILL_FIXED_RANGE : 0);

    int b = 152;//(unsigned)theRNG() & 255;
    int g = 136;//(unsigned)theRNG() & 255;
    int r = 53;//(unsigned)theRNG() & 255;
    newval = (colorType == IMREAD_COLOR) ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);

    Rect outputRect = Rect();
    Mat source = Mat();

    for (;  next(); )
    {
        image0.convertTo(source, imdepth);
        startTimer();
        cv::floodFill(source, pseed, newval, &outputRect, loVal, upVal, flags);
        stopTimer();
    }
    EXPECT_EQ(image0.cols, source.cols);
    EXPECT_EQ(image0.rows, source.rows);
    SANITY_CHECK_NOTHING();
}
