/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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

namespace opencv_test { namespace {

//#define DRAW_TEST_IMAGE

class CV_DrawingTest : public cvtest::BaseTest
{
public:
    CV_DrawingTest(){}
protected:
    void run( int );
    virtual void draw( Mat& img ) = 0;
    virtual int checkLineIterator( Mat& img) = 0;
    virtual int checkLineVirtualIterator() = 0;
};

void CV_DrawingTest::run( int )
{
    Mat testImg, valImg;
    const string fname = "../highgui/drawing/image.png";
    string path = ts->get_data_path(), filename;
    filename = path + fname;

    draw( testImg );

    valImg = imread( filename );
    if( valImg.empty() )
    {
        //imwrite( filename, testImg );
        ts->printf( ts->LOG, "test image can not be read");
#if defined(HAVE_PNG) || defined(HAVE_SPNG)
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
#else
        ts->printf( ts->LOG, "PNG image support is not available");
        ts->set_failed_test_info(cvtest::TS::OK);
#endif
        return;
    }
    else
    {
        // image should match exactly
        float err = (float)cvtest::norm( testImg, valImg, NORM_L1 );
        float Eps = 1;
        if( err > Eps)
        {
            ts->printf( ts->LOG, "NORM_L1 between testImg and valImg is equal %f (larger than %f)\n", err, Eps );
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }
        else
        {
            ts->set_failed_test_info(checkLineIterator( testImg ));
        }
    }
    ts->set_failed_test_info(checkLineVirtualIterator());
    ts->set_failed_test_info(cvtest::TS::OK);
}

class CV_DrawingTest_CPP : public CV_DrawingTest
{
public:
    CV_DrawingTest_CPP() {}
protected:
    virtual void draw( Mat& img );
    virtual int checkLineIterator( Mat& img);
    virtual int checkLineVirtualIterator();
};

void CV_DrawingTest_CPP::draw( Mat& img )
{
    Size imgSize( 600, 400 );
    img.create( imgSize, CV_8UC3 );

    vector<Point> polyline(4);
    polyline[0] = Point(0, 0);
    polyline[1] = Point(imgSize.width, 0);
    polyline[2] = Point(imgSize.width, imgSize.height);
    polyline[3] = Point(0, imgSize.height);
    const Point* pts = &polyline[0];
    int n = (int)polyline.size();
    fillPoly( img, &pts, &n, 1, Scalar::all(255) );

    Point p1(1,1), p2(3,3);
    if( clipLine(Rect(0,0,imgSize.width,imgSize.height), p1, p2) && clipLine(imgSize, p1, p2) )
        circle( img, Point(300,100), 40, Scalar(0,0,255), 3 ); // draw

    p2 = Point(3,imgSize.height+1000);
    if( clipLine(Rect(0,0,imgSize.width,imgSize.height), p1, p2) && clipLine(imgSize, p1, p2) )
        circle( img, Point(500,300), 50, Scalar(255, 0, 0), 5, 8, 1 ); // draw

    p1 = Point(imgSize.width,1), p2 = Point(imgSize.width,3);
    if( clipLine(Rect(0,0,imgSize.width,imgSize.height), p1, p2) && clipLine(imgSize, p1, p2) )
        circle( img, Point(390,100), 10, Scalar(0,0,255), 3 ); // not draw

    p1 = Point(imgSize.width-1,1), p2 = Point(imgSize.width,3);
    if( clipLine(Rect(0,0,imgSize.width,imgSize.height), p1, p2) && clipLine(imgSize, p1, p2) )
        ellipse( img, Point(390,100), Size(20,30), 60, 0, 220.0, Scalar(0,200,0), 4 ); //draw

    ellipse( img, RotatedRect(Point(100,200),Size(200,100),160), Scalar(200,200,255), 5 );

    polyline.clear();
    ellipse2Poly( Point(430,180), Size(100,150), 30, 0, 150, 20, polyline );
    pts = &polyline[0];
    n = (int)polyline.size();
    polylines( img, &pts, &n, 1, false, Scalar(0,0,150), 4, cv::LINE_AA );
    n = 0;
    for( vector<Point>::const_iterator it = polyline.begin(); n < (int)polyline.size()-1; ++it, n++ )
    {
        line( img, *it, *(it+1), Scalar(50,250,100));
    }

    polyline.clear();
    ellipse2Poly( Point(500,300), Size(50,80), 0, 0, 180, 10, polyline );
    pts = &polyline[0];
    n = (int)polyline.size();
    polylines( img, &pts, &n, 1, true, Scalar(100,200,100), 20 );
    fillConvexPoly( img, pts, n, Scalar(0, 80, 0) );

    polyline.resize(8);
    // external rectengular
    polyline[0] = Point(0, 0);
    polyline[1] = Point(80, 0);
    polyline[2] = Point(80, 80);
    polyline[3] = Point(0, 80);
    // internal rectangular
    polyline[4] = Point(20, 20);
    polyline[5] = Point(60, 20);
    polyline[6] = Point(60, 60);
    polyline[7] = Point(20, 60);
    const Point* ppts[] = {&polyline[0], &polyline[0]+4};
    int pn[] = {4, 4};
    fillPoly( img, ppts, pn, 2, Scalar(100, 100, 0), 8, 0, Point(500, 20) );

    rectangle( img, Point(0, 300), Point(50, 398), Scalar(0,0,255) );

    string text1 = "OpenCV";
    int baseline = 0, thickness = 3, fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    float fontScale = 2;
    Size textSize = getTextSize( text1, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    Point textOrg((img.cols - textSize.width)/2, (img.rows + textSize.height)/2);
    rectangle(img, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0,0,255));
    line(img, textOrg + Point(0, thickness), textOrg + Point(textSize.width, thickness), Scalar(0, 0, 255));
    putText(img, text1, textOrg, fontFace, fontScale, Scalar(150,0,150), thickness, 8);

    string text2 = "abcdefghijklmnopqrstuvwxyz1234567890";
    Scalar color(200,0,0);
    fontScale = 0.5, thickness = 1;
    int dist = 5;

    textSize = getTextSize( text2, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    textOrg = Point(5,5)+Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);

    fontScale = 1;
    textSize = getTextSize( text2, FONT_HERSHEY_PLAIN, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_PLAIN, fontScale, color, thickness, cv::LINE_AA);

    fontScale = 0.5;
    textSize = getTextSize( text2, FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_DUPLEX, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_TRIPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_TRIPLEX, fontScale, color, thickness, cv::LINE_AA);

    fontScale = 1;
    textSize = getTextSize( text2, FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness, &baseline);
    textOrg += Point(0,180) + Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, color, thickness, cv::LINE_AA);

    dist = 15, fontScale = 0.5;
    textSize = getTextSize( text2, FONT_ITALIC, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_ITALIC, fontScale, color, thickness, cv::LINE_AA);
}

int CV_DrawingTest_CPP::checkLineIterator( Mat& img )
{
    LineIterator it( img, Point(0,300), Point(1000, 300) );
    for(int i = 0; i < it.count; ++it, i++ )
    {
        Vec3b v = (Vec3b)(*(*it)) - img.at<Vec3b>(300,i);
        float err = (float)cvtest::norm( v, NORM_L2 );
        if( err != 0 )
        {
            ts->printf( ts->LOG, "LineIterator works incorrect" );
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
    }
    ts->set_failed_test_info(cvtest::TS::OK);
    return 0;
}

int CV_DrawingTest_CPP::checkLineVirtualIterator(  )
{
    RNG randomGenerator(1);
    for (size_t test = 0; test < 10000; ++test)
    {
        int width = randomGenerator.uniform(0, 512+1);
        int height = randomGenerator.uniform(0, 512+1);
        int x1 = randomGenerator.uniform(-512, 1024+1);
        int y1 = randomGenerator.uniform(-512, 1024+1);
        int x2 = randomGenerator.uniform(-512, 1024+1);
        int y2 = randomGenerator.uniform(-512, 1024+1);
        int x3 = randomGenerator.uniform(-512, 1024+1);
        int y3 = randomGenerator.uniform(-512, 1024+1);
        int channels = randomGenerator.uniform(1, 3+1);
        Mat m(cv::Size(width, height), CV_MAKETYPE(8U, channels));
        Point p1(x1, y1);
        Point p2(x2, y2);
        Point offset(x3, y3);
        LineIterator it( m, p1, p2 );
        LineIterator vit(Rect(offset.x, offset.y, width, height), p1 + offset, p2 + offset);
        if (it.count != vit.count)
        {
           ts->printf( ts->LOG, "virtual LineIterator works incorrectly" );
           ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
           break;
        }
        else
        {
            for(int i = 0; i < it.count; ++it, ++vit, i++ )
            {
                Point pIt = it.pos();
                Point pVit = vit.pos() - offset;
                if (pIt != pVit)
                {
                    ts->printf( ts->LOG, "virtual LineIterator works incorrectly" );
                    ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
                    break;
                }
            }
        }
    }
    ts->set_failed_test_info(cvtest::TS::OK);
    return 0;
}

class CV_DrawingTest_Far : public CV_DrawingTest_CPP
{
public:
    CV_DrawingTest_Far() {}
protected:
    virtual void draw(Mat& img);
};

void CV_DrawingTest_Far::draw(Mat& img)
{
    Size imgSize(32768 + 600, 400);
    img.create(imgSize, CV_8UC3);

    vector<Point> polyline(4);
    polyline[0] = Point(32768 + 0, 0);
    polyline[1] = Point(imgSize.width, 0);
    polyline[2] = Point(imgSize.width, imgSize.height);
    polyline[3] = Point(32768 + 0, imgSize.height);
    const Point* pts = &polyline[0];
    int n = (int)polyline.size();
    fillPoly(img, &pts, &n, 1, Scalar::all(255));

    Point p1(32768 + 1, 1), p2(32768 + 3, 3);
    if (clipLine(Rect(32768 + 0, 0, imgSize.width, imgSize.height), p1, p2) && clipLine(imgSize, p1, p2))
        circle(img, Point(32768 + 300, 100), 40, Scalar(0, 0, 255), 3); // draw

    p2 = Point(32768 + 3, imgSize.height + 1000);
    if (clipLine(Rect(32768 + 0, 0, imgSize.width, imgSize.height), p1, p2) && clipLine(imgSize, p1, p2))
        circle(img, Point(65536 + 500, 300), 50, Scalar(255, 0, 0), 5, 8, 1); // draw

    p1 = Point(imgSize.width, 1), p2 = Point(imgSize.width, 3);
    if (clipLine(Rect(32768 + 0, 0, imgSize.width, imgSize.height), p1, p2) && clipLine(imgSize, p1, p2))
        circle(img, Point(32768 + 390, 100), 10, Scalar(0, 0, 255), 3); // not draw

    p1 = Point(imgSize.width - 1, 1), p2 = Point(imgSize.width, 3);
    if (clipLine(Rect(32768 + 0, 0, imgSize.width, imgSize.height), p1, p2) && clipLine(imgSize, p1, p2))
        ellipse(img, Point(32768 + 390, 100), Size(20, 30), 60, 0, 220.0, Scalar(0, 200, 0), 4); //draw

    ellipse(img, RotatedRect(Point(32768 + 100, 200), Size(200, 100), 160), Scalar(200, 200, 255), 5);

    polyline.clear();
    ellipse2Poly(Point(32768 + 430, 180), Size(100, 150), 30, 0, 150, 20, polyline);
    pts = &polyline[0];
    n = (int)polyline.size();
    polylines(img, &pts, &n, 1, false, Scalar(0, 0, 150), 4, cv::LINE_AA);
    n = 0;
    for (vector<Point>::const_iterator it = polyline.begin(); n < (int)polyline.size() - 1; ++it, n++)
    {
        line(img, *it, *(it + 1), Scalar(50, 250, 100));
    }

    polyline.clear();
    ellipse2Poly(Point(32768 + 500, 300), Size(50, 80), 0, 0, 180, 10, polyline);
    pts = &polyline[0];
    n = (int)polyline.size();
    polylines(img, &pts, &n, 1, true, Scalar(100, 200, 100), 20);
    fillConvexPoly(img, pts, n, Scalar(0, 80, 0));

    polyline.resize(8);
    // external rectengular
    polyline[0] = Point(32768 + 0, 0);
    polyline[1] = Point(32768 + 80, 0);
    polyline[2] = Point(32768 + 80, 80);
    polyline[3] = Point(32768 + 0, 80);
    // internal rectangular
    polyline[4] = Point(32768 + 20, 20);
    polyline[5] = Point(32768 + 60, 20);
    polyline[6] = Point(32768 + 60, 60);
    polyline[7] = Point(32768 + 20, 60);
    const Point* ppts[] = { &polyline[0], &polyline[0] + 4 };
    int pn[] = { 4, 4 };
    fillPoly(img, ppts, pn, 2, Scalar(100, 100, 0), 8, 0, Point(500, 20));

    rectangle(img, Point(32768 + 0, 300), Point(32768 + 50, 398), Scalar(0, 0, 255));

    string text1 = "OpenCV";
    int baseline = 0, thickness = 3, fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    float fontScale = 2;
    Size textSize = getTextSize(text1, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    Point textOrg((32768 + img.cols - textSize.width) / 2, (img.rows + textSize.height) / 2);
    rectangle(img, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0, 0, 255));
    line(img, textOrg + Point(0, thickness), textOrg + Point(textSize.width, thickness), Scalar(0, 0, 255));
    putText(img, text1, textOrg, fontFace, fontScale, Scalar(150, 0, 150), thickness, 8);

    string text2 = "abcdefghijklmnopqrstuvwxyz1234567890";
    Scalar color(200, 0, 0);
    fontScale = 0.5, thickness = 1;
    int dist = 5;

    textSize = getTextSize(text2, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    textOrg = Point(32768 + 5, 5) + Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);

    fontScale = 1;
    textSize = getTextSize(text2, FONT_HERSHEY_PLAIN, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_PLAIN, fontScale, color, thickness, cv::LINE_AA);

    fontScale = 0.5;
    textSize = getTextSize(text2, FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_DUPLEX, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_TRIPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_TRIPLEX, fontScale, color, thickness, cv::LINE_AA);

    fontScale = 1;
    textSize = getTextSize(text2, FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness, &baseline);
    textOrg += Point(0, 180) + Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, color, thickness, cv::LINE_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, color, thickness, cv::LINE_AA);

    dist = 15, fontScale = 0.5;
    textSize = getTextSize(text2, FONT_ITALIC, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_ITALIC, fontScale, color, thickness, cv::LINE_AA);

    img = img(Rect(32768, 0, 600, 400)).clone();
}

TEST(Drawing,    cpp_regression) { CV_DrawingTest_CPP test; test.safe_run(); }
TEST(Drawing,    far_regression) { CV_DrawingTest_Far test; test.safe_run(); }

class CV_FillConvexPolyTest : public cvtest::BaseTest
{
public:
    CV_FillConvexPolyTest() {}
    ~CV_FillConvexPolyTest() {}
protected:
    void run(int)
    {
        vector<Point> line1;
        vector<Point> line2;

        line1.push_back(Point(1, 1));
        line1.push_back(Point(5, 1));
        line1.push_back(Point(5, 8));
        line1.push_back(Point(1, 8));

        line2.push_back(Point(2, 2));
        line2.push_back(Point(10, 2));
        line2.push_back(Point(10, 16));
        line2.push_back(Point(2, 16));

        Mat gray0(10,10,CV_8U, Scalar(0));
        fillConvexPoly(gray0, line1, Scalar(255), 8, 0);
        int nz1 = countNonZero(gray0);

        fillConvexPoly(gray0, line2, Scalar(0), 8, 1);
        int nz2 = countNonZero(gray0)/255;

        CV_Assert( nz1 == 40 && nz2 == 0 );
    }
};

TEST(Drawing, fillconvexpoly_clipping) { CV_FillConvexPolyTest test; test.safe_run(); }

class CV_DrawingTest_UTF8 : public cvtest::BaseTest
{
public:
    CV_DrawingTest_UTF8() {}
    ~CV_DrawingTest_UTF8() {}
protected:
    void run(int)
    {
        vector<string> lines;
        lines.push_back("abcdefghijklmnopqrstuvwxyz1234567890");
        // cyrillic letters small
        lines.push_back("\xD0\xB0\xD0\xB1\xD0\xB2\xD0\xB3\xD0\xB4\xD0\xB5\xD1\x91\xD0\xB6\xD0\xB7"
                        "\xD0\xB8\xD0\xB9\xD0\xBA\xD0\xBB\xD0\xBC\xD0\xBD\xD0\xBE\xD0\xBF\xD1\x80"
                        "\xD1\x81\xD1\x82\xD1\x83\xD1\x84\xD1\x85\xD1\x86\xD1\x87\xD1\x88\xD1\x89"
                        "\xD1\x8A\xD1\x8B\xD1\x8C\xD1\x8D\xD1\x8E\xD1\x8F");
        // cyrillic letters capital
        lines.push_back("\xD0\x90\xD0\x91\xD0\x92\xD0\x93\xD0\x94\xD0\x95\xD0\x81\xD0\x96\xD0\x97"
                        "\xD0\x98\xD0\x99\xD0\x9A\xD0\x9B\xD0\x9C\xD0\x9D\xD0\x9E\xD0\x9F\xD0\xA0"
                        "\xD0\xA1\xD0\xA2\xD0\xA3\xD0\xA4\xD0\xA5\xD0\xA6\xD0\xA7\xD0\xA8\xD0\xA9"
                        "\xD0\xAA\xD0\xAB\xD0\xAC\xD0\xAD\xD0\xAE\xD0\xAF");
        // bounds
        lines.push_back("-\xD0\x80-\xD0\x8E-\xD0\x8F-");
        lines.push_back("-\xD1\x90-\xD1\x91-\xD1\xBF-");
        // bad utf8
        lines.push_back("-\x81-\x82-\x83-");
        lines.push_back("--\xF0--");
        lines.push_back("-\xF0");

        vector<int> fonts;
        fonts.push_back(FONT_HERSHEY_SIMPLEX);
        fonts.push_back(FONT_HERSHEY_PLAIN);
        fonts.push_back(FONT_HERSHEY_DUPLEX);
        fonts.push_back(FONT_HERSHEY_COMPLEX);
        fonts.push_back(FONT_HERSHEY_TRIPLEX);
        fonts.push_back(FONT_HERSHEY_COMPLEX_SMALL);
        fonts.push_back(FONT_HERSHEY_SCRIPT_SIMPLEX);
        fonts.push_back(FONT_HERSHEY_SCRIPT_COMPLEX);

        vector<Mat> results;
        Size bigSize(0, 0);
        for (vector<int>::const_iterator font = fonts.begin(); font != fonts.end(); ++font)
        {
            for (int italic = 0; italic <= FONT_ITALIC; italic += FONT_ITALIC)
            {
                for (vector<string>::const_iterator line = lines.begin(); line != lines.end(); ++line)
                {
                    const float fontScale = 1;
                    const int thickness = 1;
                    const Scalar color(20,20,20);
                    int baseline = 0;

                    Size textSize = getTextSize(*line, *font | italic, fontScale, thickness, &baseline);
                    Point textOrg(0, textSize.height + 2);
                    Mat img(textSize + Size(0, baseline), CV_8UC3, Scalar(255, 255, 255));
                    putText(img, *line, textOrg, *font | italic, fontScale, color, thickness, cv::LINE_AA);

                    results.push_back(img);
                    bigSize.width = max(bigSize.width, img.size().width);
                    bigSize.height += img.size().height + 1;
                }
            }
        }

        int shift = 0;
        Mat result(bigSize, CV_8UC3, Scalar(100, 100, 100));
        for (vector<Mat>::const_iterator img = results.begin(); img != results.end(); ++img)
        {
            Rect roi(Point(0, shift), img->size());
            Mat sub(result, roi);
            img->copyTo(sub);
            shift += img->size().height + 1;
        }
        if (cvtest::debugLevel > 0)
            imwrite("all_fonts.png", result);
    }
};

TEST(Drawing, utf8_support) { CV_DrawingTest_UTF8 test; test.safe_run(); }


TEST(Drawing, _914)
{
    const int rows = 256;
    const int cols = 256;

    Mat img(rows, cols, CV_8UC1, Scalar(255));

    line(img, Point(0, 10), Point(255, 10), Scalar(0), 2, 4);
    line(img, Point(-5, 20), Point(260, 20), Scalar(0), 2, 4);
    line(img, Point(10, 0), Point(10, 255), Scalar(0), 2, 4);

    double x0 = 0.0/pow(2.0, -2.0);
    double x1 = 255.0/pow(2.0, -2.0);
    double y = 30.5/pow(2.0, -2.0);

    line(img, Point(int(x0), int(y)), Point(int(x1), int(y)), Scalar(0), 2, 4, 2);

    int pixelsDrawn = rows*cols - countNonZero(img);
    ASSERT_EQ( (3*rows + cols)*3 - 3*9, pixelsDrawn);
}

TEST(Drawing, polylines_empty)
{
    Mat img(100, 100, CV_8UC1, Scalar(0));
    vector<Point> pts; // empty
    polylines(img, pts, false, Scalar(255));
    int cnt = countNonZero(img);
    ASSERT_EQ(cnt, 0);
}

TEST(Drawing, polylines)
{
    Mat img(100, 100, CV_8UC1, Scalar(0));
    vector<Point> pts;
    pts.push_back(Point(0, 0));
    pts.push_back(Point(20, 0));
    polylines(img, pts, false, Scalar(255));
    int cnt = countNonZero(img);
    ASSERT_EQ(cnt, 21);
}

TEST(Drawing, longline)
{
    Mat mat = Mat::zeros(256, 256, CV_8UC1);

    line(mat, cv::Point(34, 204), cv::Point(46400, 47400), cv::Scalar(255), 3);
    EXPECT_EQ(310, cv::countNonZero(mat));

    Point pt[6];
    pt[0].x = 32;
    pt[0].y = 204;
    pt[1].x = 34;
    pt[1].y = 202;
    pt[2].x = 87;
    pt[2].y = 255;
    pt[3].x = 82;
    pt[3].y = 255;
    pt[4].x = 37;
    pt[4].y = 210;
    pt[5].x = 37;
    pt[5].y = 209;
    fillConvexPoly(mat, pt, 6, cv::Scalar(0));

    EXPECT_EQ(0, cv::countNonZero(mat));
}


TEST(Drawing, putText_no_garbage)
{
    Size sz(640, 480);
    Mat mat = Mat::zeros(sz, CV_8UC1);

    mat = Scalar::all(0);
    putText(mat, "029", Point(10, 350), 0, 10, Scalar(128), 15);

    EXPECT_EQ(0, cv::countNonZero(mat(Rect(0, 0,           10, sz.height))));
    EXPECT_EQ(0, cv::countNonZero(mat(Rect(sz.width-10, 0, 10, sz.height))));
    EXPECT_EQ(0, cv::countNonZero(mat(Rect(205, 0,         10, sz.height))));
    EXPECT_EQ(0, cv::countNonZero(mat(Rect(405, 0,         10, sz.height))));
}


TEST(Drawing, line)
{
    Mat mat = Mat::zeros(Size(100,100), CV_8UC1);

    ASSERT_THROW(line(mat, Point(1,1),Point(99,99),Scalar(255),0), cv::Exception);
}

TEST(Drawing, regression_16308)
{
    Mat_<uchar> img(Size(100, 100), (uchar)0);
    circle(img, Point(50, 50), 50, 255, 1, LINE_AA);
    EXPECT_NE(0, (int)img.at<uchar>(0, 50));
    EXPECT_NE(0, (int)img.at<uchar>(50, 0));
    EXPECT_NE(0, (int)img.at<uchar>(50, 99));
    EXPECT_NE(0, (int)img.at<uchar>(99, 50));
}

TEST(Drawing, fillpoly_circle)
{
    Mat img_c(640, 480, CV_8UC3, Scalar::all(0));
    Mat img_fp = img_c.clone(), img_fcp = img_c.clone(), img_fp3 = img_c.clone();

    Point center1(img_c.cols/2, img_c.rows/2);
    Point center2(img_c.cols/10, img_c.rows*3/4);
    Point center3 = Point(img_c.cols, img_c.rows) - center2;
    int radius = img_c.rows/4;
    int radius_small = img_c.cols/15;
    Scalar color(0, 0, 255);

    circle(img_c, center1, radius, color, -1);

    // check that circle, fillConvexPoly and fillPoly
    // give almost the same result then asked to draw a single circle
    vector<Point> vtx;
    ellipse2Poly(center1, Size(radius, radius), 0, 0, 360, 1, vtx);
    fillConvexPoly(img_fcp, vtx, color);
    fillPoly(img_fp, vtx, color);
    double diff_fp = cv::norm(img_c, img_fp, NORM_L1)/(255*radius*2*CV_PI);
    double diff_fcp = cv::norm(img_c, img_fcp, NORM_L1)/(255*radius*2*CV_PI);
    EXPECT_LT(diff_fp, 1.);
    EXPECT_LT(diff_fcp, 1.);

    // check that fillPoly can draw 3 disjoint circles at once
    circle(img_c, center2, radius_small, color, -1);
    circle(img_c, center3, radius_small, color, -1);

    vector<vector<Point> > vtx3(3);
    vtx3[0] = vtx;
    ellipse2Poly(center2, Size(radius_small, radius_small), 0, 0, 360, 1, vtx3[1]);
    ellipse2Poly(center3, Size(radius_small, radius_small), 0, 0, 360, 1, vtx3[2]);
    fillPoly(img_fp3, vtx3, color);
    double diff_fp3 = cv::norm(img_c, img_fp3, NORM_L1)/(255*(radius+radius_small*2)*2*CV_PI);
    EXPECT_LT(diff_fp3, 1.);
}

TEST(Drawing, fillpoly_contours)
{
    const int imgSize = 50;
    const int type = CV_8UC1;
    const int shift = 0;
    const Scalar cl = Scalar::all(255);
    const cv::LineTypes lineType = LINE_8;

    // check that contours of fillPoly and polylines match
    {
        cv::Mat img(imgSize, imgSize, type);
        img = 0;
        std::vector<std::vector<cv::Point>> polygonPoints{
            { {44, 27}, {7, 37}, {7, 19}, {38, 19} }
        };
        cv::fillPoly(img, polygonPoints, cl, lineType, shift);
        cv::polylines(img, polygonPoints, true, 0, 1, lineType, shift);

        {
            cv::Mat labelImage(img.size(), CV_32S);
            int labels = cv::connectedComponents(img, labelImage, 4);
            EXPECT_EQ(2, labels) << "filling went over the border";
        }
    }

    // check that line generated with fillPoly and polylines match
    {
        cv::Mat img1(imgSize, imgSize, type), img2(imgSize, imgSize, type);
        img1 = 0;
        img2 = 0;
        std::vector<std::vector<cv::Point>> polygonPoints{
            { {44, 27}, {38, 19} }
        };
        cv::fillPoly(img1, polygonPoints, cl, lineType, shift);
        cv::polylines(img2, polygonPoints, true, cl, 1, lineType, shift);
        EXPECT_MAT_N_DIFF(img1, img2, 0);
    }
}

TEST(Drawing, fillpoly_match_lines)
{
    const int imgSize = 49;
    const int type = CV_8UC1;
    const int shift = 0;
    const Scalar cl = Scalar::all(255);
    const cv::LineTypes lineType = LINE_8;
    cv::Mat img1(imgSize, imgSize, type), img2(imgSize, imgSize, type);
    for (int x1 = 0; x1 < imgSize; x1 += imgSize / 2)
    {
        for (int y1 = 0; y1 < imgSize; y1 += imgSize / 2)
        {
            for (int x2 = 0; x2 < imgSize; x2++)
            {
                for (int y2 = 0; y2 < imgSize; y2++)
                {
                    img1 = 0;
                    img2 = 0;
                    std::vector<std::vector<cv::Point>> polygonPoints{
                        { {x1, y1}, {x2, y2} }
                    };
                    cv::fillPoly(img1, polygonPoints, cl, lineType, shift);
                    cv::polylines(img2, polygonPoints, true, cl, 1, lineType, shift);
                    EXPECT_MAT_N_DIFF(img1, img2, 0);
                }
            }
        }
    }
}

TEST(Drawing, fillpoly_fully)
{
    unsigned imageWidth = 256;
    unsigned imageHeight = 256;
    int type = CV_8UC1;
    int shift = 0;
    Point offset(0, 0);
    cv::LineTypes lineType = LINE_4;

    int imageSizeOffset = 15;

    cv::Mat img(imageHeight, imageWidth, type);
    img = 0;

    std::vector<cv::Point> polygonPoints;
    polygonPoints.push_back(cv::Point(100, -50));
    polygonPoints.push_back(cv::Point(imageSizeOffset, imageHeight - imageSizeOffset));
    polygonPoints.push_back(cv::Point(imageSizeOffset, imageSizeOffset));

    // convert data
    std::vector<const cv::Point*> polygonPointPointers(polygonPoints.size());
    for (size_t i = 0; i < polygonPoints.size(); i++)
    {
        polygonPointPointers[i] = &polygonPoints[i];
    }

    const cv::Point** data = &polygonPointPointers.front();
    int size = (int)polygonPoints.size();
    const int* npts = &size;
    int ncontours = 1;

    // generate image
    cv::fillPoly(img, data, npts, ncontours, 255, lineType, shift, offset);

    // check for artifacts
    {
        cv::Mat binary = img < 128;
        cv::Mat labelImage(binary.size(), CV_32S);
        cv::Mat labelCentroids;
        int labels = cv::connectedComponents(binary, labelImage, 4);
        EXPECT_EQ(2, labels) << "artifacts occured";
    }

    // check if filling went over border
    {
        int xy_shift = 16, delta = offset.y + ((1 << shift) >> 1);
        int xy_one = 1 << xy_shift;

        Point pt0(polygonPoints[polygonPoints.size() - 1]), pt1;
        for (size_t i = 0; i < polygonPoints.size(); i++, pt0 = pt1)
        {
            pt1 = polygonPoints[i];

            // offset/shift treated like in fillPoly
            Point t0(pt0), t1(pt1);

            t0.x = (t0.x + offset.x) << (xy_shift - shift);
            t0.y = (t0.y + delta) >> shift;

            t1.x = (t1.x + offset.x) << (xy_shift - shift);
            t1.y = (t1.y + delta) >> shift;

            if (lineType < cv::LINE_AA)
            {
                t0.x = (t0.x + (xy_one >> 1)) >> xy_shift;
                t1.x = (t1.x + (xy_one >> 1)) >> xy_shift;

                // LINE_4 to use the same type of line which is used in fillPoly
                line(img, t0, t1, 0, 1, LINE_4, 0);
            }
            else
            {
                t0.x >>= (xy_shift);
                t1.x >>= (xy_shift);
                line(img, t0, t1, 0, 1, lineType, 0);
            }

        }
        cv::Mat binary = img < 254;
        cv::Mat labelImage(binary.size(), CV_32S);
        int labels = cv::connectedComponents(binary, labelImage, 4);
        EXPECT_EQ(2, labels) << "filling went over the border";
    }
}

PARAM_TEST_CASE(FillPolyFully, unsigned, unsigned, int, int, Point, cv::LineTypes)
{
    unsigned imageWidth;
    unsigned imageHeight;
    int type;
    int shift;
    Point offset;
    cv::LineTypes lineType;

    virtual void SetUp()
    {
        imageWidth = GET_PARAM(0);
        imageHeight = GET_PARAM(1);
        type = GET_PARAM(2);
        shift = GET_PARAM(3);
        offset = GET_PARAM(4);
        lineType = GET_PARAM(5);
    }

    void draw_polygon(cv::Mat& img, const std::vector<cv::Point>& polygonPoints)
    {
        // convert data
        std::vector<const cv::Point*> polygonPointPointers(polygonPoints.size());
        for (size_t i = 0; i < polygonPoints.size(); i++)
        {
            polygonPointPointers[i] = &polygonPoints[i];
        }

        const cv::Point** data = &polygonPointPointers.front();
        int size = (int)polygonPoints.size();
        const int* npts = &size;
        int ncontours = 1;

        // generate image
        cv::fillPoly(img, data, npts, ncontours, 255, lineType, shift, offset);
    }

    void check_artifacts(cv::Mat& img)
    {
        // check for artifacts
        cv::Mat binary = img < 128;
        cv::Mat labelImage(binary.size(), CV_32S);
        cv::Mat labelCentroids;
        int labels = cv::connectedComponents(binary, labelImage, 4);
        EXPECT_EQ(2, labels) << "artifacts occured";
    }

    void check_filling_over_border(cv::Mat& img, const std::vector<cv::Point>& polygonPoints)
    {
        int xy_shift = 16, delta = offset.y + ((1 << shift) >> 1);
        int xy_one = 1 << xy_shift;

        Point pt0(polygonPoints[polygonPoints.size() - 1]), pt1;
        for (size_t i = 0; i < polygonPoints.size(); i++, pt0 = pt1)
        {
            pt1 = polygonPoints[i];

            // offset/shift treated like in fillPoly
            Point t0(pt0), t1(pt1);

            t0.x = (t0.x + offset.x) << (xy_shift - shift);
            t0.y = (t0.y + delta) >> shift;

            t1.x = (t1.x + offset.x) << (xy_shift - shift);
            t1.y = (t1.y + delta) >> shift;

            if (lineType < cv::LINE_AA)
            {
                t0.x = (t0.x + (xy_one >> 1)) >> xy_shift;
                t1.x = (t1.x + (xy_one >> 1)) >> xy_shift;

                // LINE_4 to use the same type of line which is used in fillPoly
                line(img, t0, t1, 0, 1, LINE_4, 0);
            }
            else
            {
                t0.x >>= (xy_shift);
                t1.x >>= (xy_shift);
                line(img, t0, t1, 0, 1, lineType, 0);
            }

        }
        cv::Mat binary = img < 254;
        cv::Mat labelImage(binary.size(), CV_32S);
        int labels = cv::connectedComponents(binary, labelImage, 4);
        EXPECT_EQ(2, labels) << "filling went over the border";
    }

    void run_test(const std::vector<cv::Point>& polygonPoints)
    {
        cv::Mat img(imageHeight, imageWidth, type);
        img = 0;

        draw_polygon(img, polygonPoints);
        check_artifacts(img);
        check_filling_over_border(img, polygonPoints);
    }
};

TEST_P(FillPolyFully, DISABLED_fillpoly_fully)
{
    int imageSizeOffset = 15;

    // testing for polygon with straight edge at left/right side
    int positions1[2] = { imageSizeOffset, (int)imageWidth - imageSizeOffset };
    for (size_t i = 0; i < 2; i++)
    {
        for (int y = imageHeight + 50; y > -50; y -= 1)
        {
            // define polygon
            std::vector<cv::Point> polygonPoints;
            polygonPoints.push_back(cv::Point(100, imageHeight - y));
            polygonPoints.push_back(cv::Point(positions1[i], positions1[1]));
            polygonPoints.push_back(cv::Point(positions1[i], positions1[0]));

            run_test(polygonPoints);
        }
    }

    // testing for polygon with straight edge at top/bottom side
    int positions2[2] = { imageSizeOffset, (int)imageHeight - imageSizeOffset };
    for (size_t i = 0; i < 2; i++)
    {
        for (int x = imageWidth + 50; x > -50; x -= 1)
        {
            // define polygon
            std::vector<cv::Point> polygonPoints;
            polygonPoints.push_back(cv::Point(imageWidth - x, 100));
            polygonPoints.push_back(cv::Point(positions2[1], positions2[i]));
            polygonPoints.push_back(cv::Point(positions2[0], positions2[i]));

            run_test(polygonPoints);
        }
    }
}

INSTANTIATE_TEST_CASE_P(
    FillPolyTest, FillPolyFully,
    testing::Combine(
        testing::Values(256),
        testing::Values(256),
        testing::Values(CV_8UC1),
        testing::Values(0, 1, 2),
        testing::Values(cv::Point(0, 0), cv::Point(10, 10)),
        testing::Values(LINE_4, LINE_8, LINE_AA)
    )
);

TEST(Drawing, circle_overflow)
{
    applyTestTag(CV_TEST_TAG_VERYLONG);
    cv::Mat1b matrix = cv::Mat1b::zeros(600, 600);
    cv::Scalar kBlue = cv::Scalar(0, 0, 255);
    cv::circle(matrix, cv::Point(275, -2147483318), 2147483647, kBlue, 1, 8, 0);
}

TEST(Drawing, circle_memory_access)
{
    cv::Mat1b matrix = cv::Mat1b::zeros(10, 10);
    cv::Scalar kBlue = cv::Scalar(0, 0, 255);
    cv::circle(matrix, cv::Point(-1, -1), 0, kBlue, 2, 8, 16);
}

inline static Mat mosaic2x2(Mat &img)
{
    const Size sz = img.size();
    Mat res(sz * 2, img.type(), Scalar::all(0));
    img.copyTo(res(Rect(Point(0, 0), sz)));
    img.copyTo(res(Rect(Point(0, sz.height), sz)));
    img.copyTo(res(Rect(Point(sz.width, 0), sz)));
    img.copyTo(res(Rect(Point(sz.width, sz.height), sz)));
    return res;
}

TEST(Drawing, contours_filled)
{
    const Scalar white(255);
    const Scalar black(0);
    const Size sz(100, 100);

    Mat img(sz, CV_8UC1, black);
    rectangle(img, Point(20, 20), Point(80, 80), white, -1);
    rectangle(img, Point(30, 30), Point(70, 70), black, -1);
    rectangle(img, Point(40, 40), Point(60, 60), white, -1);
    img = mosaic2x2(img);

    Mat img1(sz, CV_8UC1, black);
    rectangle(img1, Point(20, 20), Point(80, 80), white, -1);
    img1 = mosaic2x2(img1);

    Mat img2(sz, CV_8UC1, black);
    rectangle(img2, Point(20, 20), Point(80, 80), white, -1);
    rectangle(img2, Point(30, 30), Point(70, 70), black, -1);
    img2 = mosaic2x2(img2);

    Mat img3(sz, CV_8UC1, black);
    rectangle(img3, Point(40, 40), Point(60, 60), white, -1);
    img3 = mosaic2x2(img3);

    // inverted contours - corners and left edge adjusted
    Mat imgi(sz, CV_8UC1, black);
    rectangle(imgi, Point(29, 29), Point(71, 71), white, -1);
    rectangle(imgi, Point(41, 41), Point(59, 59), black, -1);
    imgi.at<uchar>(Point(29, 29)) = 0;
    imgi.at<uchar>(Point(29, 71)) = 0;
    imgi = mosaic2x2(imgi);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    ASSERT_EQ(12u, contours.size());

    // NOTE:
    // assuming contour tree has following structure (idx = 0, 1, ...):
    //   idx (top level)
    //     - idx + 1
    //         - idx + 2
    //   idx + 3 (top level)
    //     - idx + 4
    //         - idx + 5
    //   ...
    const vector<int> top_contours {0, 3, 6, 9};
    {
        // all contours
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        drawContours(res, contours, -1, white, -1, cv::LINE_8, hierarchy);
        EXPECT_LT(cvtest::norm(img, res, NORM_INF), 1);
    }
    {
        // all contours
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        drawContours(res, contours, -1, white, -1, cv::LINE_8, hierarchy, 3);
        EXPECT_LT(cvtest::norm(img, res, NORM_INF), 1);
    }
    {
        // all contours
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        drawContours(res, contours, -1, white, -1, cv::LINE_8, hierarchy, 0);
        EXPECT_LT(cvtest::norm(img, res, NORM_INF), 1);
    }
    {
        // all external contours one by one
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        for (int idx : top_contours)
            drawContours(res, contours, idx, white, -1, cv::LINE_8, hierarchy, 0);
        EXPECT_LT(cvtest::norm(img1, res, NORM_INF), 1);
    }
    {
        // all external contours + 1-level deep hole (one by one)
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        for (int idx : top_contours)
            drawContours(res, contours, idx, white, -1, cv::LINE_8, hierarchy, 1);
        EXPECT_LT(cvtest::norm(img2, res, NORM_INF), 1);
    }
    {
        // 2-level deep contours
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        for (int idx : top_contours)
            drawContours(res, contours, idx + 2, white, -1, cv::LINE_8, hierarchy);
        EXPECT_LT(cvtest::norm(img3, res, NORM_INF), 1);
    }
    {
        // holes become inverted here, LINE_8 -> LINE_4
        Mat res(img.size(), CV_8UC1, Scalar::all(0));
        for (int idx : top_contours)
            drawContours(res, contours, idx + 1, white, -1, cv::LINE_4, hierarchy);
        EXPECT_LT(cvtest::norm(imgi, res, NORM_INF), 1);
    }
}

}} // namespace
