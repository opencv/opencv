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
#ifdef HAVE_PNG
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
    ts->set_failed_test_info(cvtest::TS::OK);
}

class CV_DrawingTest_CPP : public CV_DrawingTest
{
public:
    CV_DrawingTest_CPP() {}
protected:
    virtual void draw( Mat& img );
    virtual int checkLineIterator( Mat& img);
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
        circle( img, Point(500,300), 50, cvColorToScalar(255,CV_8UC3), 5, 8, 1 ); // draw

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
    polylines( img, &pts, &n, 1, false, Scalar(0,0,150), 4, CV_AA );
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
    putText(img, text2, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, CV_AA);

    fontScale = 1;
    textSize = getTextSize( text2, FONT_HERSHEY_PLAIN, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_PLAIN, fontScale, color, thickness, CV_AA);

    fontScale = 0.5;
    textSize = getTextSize( text2, FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_DUPLEX, fontScale, color, thickness, CV_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX, fontScale, color, thickness, CV_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_TRIPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_TRIPLEX, fontScale, color, thickness, CV_AA);

    fontScale = 1;
    textSize = getTextSize( text2, FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness, &baseline);
    textOrg += Point(0,180) + Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, CV_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, color, thickness, CV_AA);

    textSize = getTextSize( text2, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, color, thickness, CV_AA);

    dist = 15, fontScale = 0.5;
    textSize = getTextSize( text2, FONT_ITALIC, fontScale, thickness, &baseline);
    textOrg += Point(0,textSize.height+dist);
    putText(img, text2, textOrg, FONT_ITALIC, fontScale, color, thickness, CV_AA);
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
        circle(img, Point(65536 + 500, 300), 50, cvColorToScalar(255, CV_8UC3), 5, 8, 1); // draw

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
    polylines(img, &pts, &n, 1, false, Scalar(0, 0, 150), 4, CV_AA);
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
    putText(img, text2, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, CV_AA);

    fontScale = 1;
    textSize = getTextSize(text2, FONT_HERSHEY_PLAIN, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_PLAIN, fontScale, color, thickness, CV_AA);

    fontScale = 0.5;
    textSize = getTextSize(text2, FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_DUPLEX, fontScale, color, thickness, CV_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX, fontScale, color, thickness, CV_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_TRIPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_TRIPLEX, fontScale, color, thickness, CV_AA);

    fontScale = 1;
    textSize = getTextSize(text2, FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness, &baseline);
    textOrg += Point(0, 180) + Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_COMPLEX_SMALL, fontScale, color, thickness, CV_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale, color, thickness, CV_AA);

    textSize = getTextSize(text2, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_HERSHEY_SCRIPT_COMPLEX, fontScale, color, thickness, CV_AA);

    dist = 15, fontScale = 0.5;
    textSize = getTextSize(text2, FONT_ITALIC, fontScale, thickness, &baseline);
    textOrg += Point(0, textSize.height + dist);
    putText(img, text2, textOrg, FONT_ITALIC, fontScale, color, thickness, CV_AA);

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
                    putText(img, *line, textOrg, *font | italic, fontScale, color, thickness, CV_AA);

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
        //imwrite("/tmp/all_fonts.png", result);
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

}} // namespace
