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

#include "cvtest.h"
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <iostream>
#include "cvaux.h"

using namespace cv;
using namespace std;

//#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
#define MARKERS

#ifdef MARKERS
	#define marker(x) cout << (x)  << endl
#else
	#define marker(x) 
#endif


class CV_HighGuiOnlyGuiTest : public CvTest
{
public:
    CV_HighGuiOnlyGuiTest();
    ~CV_HighGuiOnlyGuiTest();    
protected:    
    void run(int);				
};

CV_HighGuiOnlyGuiTest::CV_HighGuiOnlyGuiTest(): CvTest( "z-highgui-gui-only", "?" )
{
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}
CV_HighGuiOnlyGuiTest::~CV_HighGuiOnlyGuiTest() {}

void Foo(int /*k*/, void* /*z*/) {}

void CV_HighGuiOnlyGuiTest::run( int /*start_from */)
{	   
    cout << "GUI 1" << endl;
	namedWindow("Win");
    cout << "GUI 2" << endl;
	Mat m(30, 30, CV_8U);	
	m = Scalar(128);	
    cout << "GUI 3" << endl;
	imshow("Win", m);	
    cout << "GUI 4" << endl;
	int value = 50;
    cout << "GUI 5" << endl;
	createTrackbar( "trackbar", "Win", &value, 100, Foo, &value);	
    cout << "GUI 6" << endl;
	getTrackbarPos( "trackbar", "Win" );	
    cout << "GUI 7" << endl;
	waitKey(500);		
    cout << "GUI 8" << endl;
	cvDestroyAllWindows();
    cout << "GUI 9" << endl;
	
    ts->set_failed_test_info(CvTS::OK);
}

CV_HighGuiOnlyGuiTest highGuiOnlyGui_test;


