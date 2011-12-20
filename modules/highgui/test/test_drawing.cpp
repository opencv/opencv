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
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

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
    const string name = "drawing/image.jpg";
    string path = ts->get_data_path(), filename;
    filename = path + name;

    draw( testImg );

    valImg = imread( filename );
    if( valImg.empty() )
    {
        imwrite( filename, testImg );
        //ts->printf( ts->LOG, "test image can not be read");
        //ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
    }
    else
    {
        float err = (float)norm( testImg, valImg, CV_RELATIVE_L1 );
        float Eps = 0.9f;
        if( err > Eps)
        {
            ts->printf( ts->LOG, "CV_RELATIVE_L1 between testImg and valImg is equal %f (larger than %f)\n", err, Eps );
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
        float err = (float)norm( v );
        if( err != 0 )
        {
            ts->printf( ts->LOG, "LineIterator works incorrect" );
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
    }
    ts->set_failed_test_info(cvtest::TS::OK);
    return 0;
}

class CV_DrawingTest_C : public CV_DrawingTest
{
public:
    CV_DrawingTest_C() {}
protected:
    virtual void draw( Mat& img );
    virtual int checkLineIterator( Mat& img);
};

void CV_DrawingTest_C::draw( Mat& _img )
{
    CvSize imgSize = cvSize(600, 400);
    _img.create( imgSize, CV_8UC3 );
    CvMat img = _img;

    vector<CvPoint> polyline(4);
    polyline[0] = cvPoint(0, 0);
    polyline[1] = cvPoint(imgSize.width, 0);
    polyline[2] = cvPoint(imgSize.width, imgSize.height);
    polyline[3] = cvPoint(0, imgSize.height);
    CvPoint* pts = &polyline[0];
    int n = (int)polyline.size();
    cvFillPoly( &img, &pts, &n, 1, cvScalar(255,255,255) );

    CvPoint p1 = cvPoint(1,1), p2 = cvPoint(3,3);
    if( cvClipLine(imgSize, &p1, &p2) )
        cvCircle( &img, cvPoint(300,100), 40, cvScalar(0,0,255), 3 ); // draw

    p1 = cvPoint(1,1), p2 = cvPoint(3,imgSize.height+1000);
    if( cvClipLine(imgSize, &p1, &p2) )
        cvCircle( &img, cvPoint(500,300), 50, cvScalar(255,0,0), 5, 8, 1 ); // draw

    p1 = cvPoint(imgSize.width,1), p2 = cvPoint(imgSize.width,3);
    if( cvClipLine(imgSize, &p1, &p2) )
        cvCircle( &img, cvPoint(390,100), 10, cvScalar(0,0,255), 3 ); // not draw

    p1 = Point(imgSize.width-1,1), p2 = Point(imgSize.width,3);
    if( cvClipLine(imgSize, &p1, &p2) )
        cvEllipse( &img, cvPoint(390,100), cvSize(20,30), 60, 0, 220.0, cvScalar(0,200,0), 4 ); //draw

    CvBox2D box;
    box.center.x = 100;
    box.center.y = 200;
    box.size.width = 200;
    box.size.height = 100;
    box.angle = 160;
    cvEllipseBox( &img, box, Scalar(200,200,255), 5 );

    polyline.resize(9);
    pts = &polyline[0];
    n = (int)polyline.size();
    assert( cvEllipse2Poly( cvPoint(430,180), cvSize(100,150), 30, 0, 150, &polyline[0], 20 ) == n );
    cvPolyLine( &img, &pts, &n, 1, false, cvScalar(0,0,150), 4, CV_AA );
    n = 0;
    for( vector<CvPoint>::const_iterator it = polyline.begin(); n < (int)polyline.size()-1; ++it, n++ )
    {
        cvLine( &img, *it, *(it+1), cvScalar(50,250,100) );
    }

    polyline.resize(19);
    pts = &polyline[0];
    n = (int)polyline.size();
    assert( cvEllipse2Poly( cvPoint(500,300), cvSize(50,80), 0, 0, 180, &polyline[0], 10 ) == n );
    cvPolyLine( &img, &pts, &n, 1, true, Scalar(100,200,100), 20 );
    cvFillConvexPoly( &img, pts, n, cvScalar(0, 80, 0) );

    polyline.resize(8);
    // external rectengular
    polyline[0] = cvPoint(500, 20);
    polyline[1] = cvPoint(580, 20);
    polyline[2] = cvPoint(580, 100);
    polyline[3] = cvPoint(500, 100);
    // internal rectangular
    polyline[4] = cvPoint(520, 40);
    polyline[5] = cvPoint(560, 40);
    polyline[6] = cvPoint(560, 80);
    polyline[7] = cvPoint(520, 80);
    CvPoint* ppts[] = {&polyline[0], &polyline[0]+4};
    int pn[] = {4, 4};
    cvFillPoly( &img, ppts, pn, 2, cvScalar(100, 100, 0), 8, 0 );

    cvRectangle( &img, cvPoint(0, 300), cvPoint(50, 398), cvScalar(0,0,255) );

    string text1 = "OpenCV";
    CvFont font;
    cvInitFont( &font, FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2, 0, 3 );
    int baseline = 0;
    CvSize textSize;
    cvGetTextSize( text1.c_str(), &font, &textSize, &baseline );
    baseline += font.thickness;
    CvPoint textOrg = cvPoint((imgSize.width - textSize.width)/2, (imgSize.height + textSize.height)/2);
    cvRectangle( &img, cvPoint( textOrg.x, textOrg.y + baseline),
                 cvPoint(textOrg.x + textSize.width, textOrg.y - textSize.height), cvScalar(0,0,255));
    cvLine( &img, cvPoint(textOrg.x, textOrg.y + font.thickness),
            cvPoint(textOrg.x + textSize.width, textOrg.y + font.thickness), cvScalar(0, 0, 255));
    cvPutText( &img, text1.c_str(), textOrg, &font, cvScalar(150,0,150) );

    int dist = 5;
    string text2 = "abcdefghijklmnopqrstuvwxyz1234567890";
    CvScalar color = cvScalar(200,0,0);
    cvInitFont( &font, FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(5, 5+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_PLAIN, 1, 1, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_DUPLEX, 0.5, 0.5, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_TRIPLEX, 0.5, 0.5, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist + 180);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    cvInitFont( &font, FONT_HERSHEY_SCRIPT_COMPLEX, 1, 1, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );

    dist = 15;
    cvInitFont( &font, FONT_ITALIC, 0.5, 0.5, 0, 1, CV_AA );
    cvGetTextSize( text2.c_str(), &font, &textSize, &baseline );
    textOrg = cvPoint(textOrg.x,textOrg.y+textSize.height+dist);
    cvPutText(&img, text2.c_str(), textOrg, &font, color );
}

int CV_DrawingTest_C::checkLineIterator( Mat& _img )
{
    CvLineIterator it;
    CvMat img = _img;
    int count = cvInitLineIterator( &img, cvPoint(0,300), cvPoint(1000, 300), &it );
    for(int i = 0; i < count; i++ )
    {
        Vec3b v = (Vec3b)(*(it.ptr)) - _img.at<Vec3b>(300,i);
        float err = (float)norm( v );
        if( err != 0 )
        {
            ts->printf( ts->LOG, "CvLineIterator works incorrect" );
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_OUTPUT);
        }
        CV_NEXT_LINE_POINT(it);
    }
    ts->set_failed_test_info(cvtest::TS::OK);
    return 0;
}

TEST(Highgui_Drawing_CPP,    regression) { CV_DrawingTest_CPP test; test.safe_run(); }
TEST(Highgui_Drawing_C,      regression) { CV_DrawingTest_C   test; test.safe_run(); }
