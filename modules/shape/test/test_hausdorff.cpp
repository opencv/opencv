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
#include <stdlib.h>

using namespace cv;
using namespace std;

class CV_HaussTest : public cvtest::BaseTest
{
public:
    CV_HaussTest();
    ~CV_HaussTest();
protected:
    void run(int);
private:
    void testShort();
    void testLong();
};

CV_HaussTest::CV_HaussTest()
{
}
CV_HaussTest::~CV_HaussTest()
{
}

void CV_HaussTest::testShort()
{
    vector<Point2f> cont1, cont2;
    cont1.push_back(Point2f(1,1));
    cont1.push_back(Point2f(1,2));
    cont1.push_back(Point2f(1,3));
    cont1.push_back(Point2f(1,4));
    cont1.push_back(Point2f(1,5));
    cont1.push_back(Point2f(5,1));
    cont1.push_back(Point2f(5,2));
    cont1.push_back(Point2f(5,3));
    cont1.push_back(Point2f(5,4));
    cont1.push_back(Point2f(5,5));

    cont2.push_back(Point2f(5,5));
    cont2.push_back(Point2f(1,2));
    cont2.push_back(Point2f(1.1,3));
    cont2.push_back(Point2f(1,4.1));
    cont2.push_back(Point2f(1,5));
    cont2.push_back(Point2f(5.1,1));
    cont2.push_back(Point2f(5,2.1));
    cont2.push_back(Point2f(5,3));
    cont2.push_back(Point2f(5.1,4));
    cont2.push_back(Point2f(1.1,1.1));

    std::cout<<"TEST SHORT"<<std::endl;
    std::cout<<"HAUSS Distance (K=0.6): "<<hausdorff(cont1, cont2, DIST_L2, 0.6)<<std::endl;
    std::cout<<"HAUSS Distance (K=1.0): "<<hausdorff(cont1, cont2)<<std::endl;
}

void CV_HaussTest::testLong()
{
    vector<Point2f> cont1, cont2;
    cont1.push_back(Point2f(1,1));
    cont1.push_back(Point2f(1,2));
    cont1.push_back(Point2f(1,3));
    cont1.push_back(Point2f(1,4));
    cont1.push_back(Point2f(1,5));
    cont1.push_back(Point2f(5,1));
    cont1.push_back(Point2f(5,2));
    cont1.push_back(Point2f(5,3));
    cont1.push_back(Point2f(5,4));
    cont1.push_back(Point2f(5,5));

    cont2.push_back(Point2f(15,5));
    cont2.push_back(Point2f(1,12));
    cont2.push_back(Point2f(11.1,3));
    cont2.push_back(Point2f(1,4.1));
    cont2.push_back(Point2f(1,5));
    cont2.push_back(Point2f(5.1,1));
    cont2.push_back(Point2f(5,2.1));
    cont2.push_back(Point2f(5,3));
    cont2.push_back(Point2f(5.1,4));
    cont2.push_back(Point2f(1.1,1.1));

    std::cout<<"TEST LONG"<<std::endl;
    std::cout<<"HAUSS Distance (K=0.6): "<<hausdorff(cont1, cont2, DIST_L2, 0.6)<<std::endl;
    std::cout<<"HAUSS Distance (K=1.0): "<<hausdorff(cont1, cont2)<<std::endl;
}

void CV_HaussTest::run(int /* */)
{
    testShort();
    testLong();
	ts->set_failed_test_info(cvtest::TS::OK);	
}

TEST(Hauss, regression) { CV_HaussTest test; test.safe_run(); }
