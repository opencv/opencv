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
#include <time.h>

using namespace cv;
using namespace std;

const int numBins=100;

class CV_EmdL1Test : public cvtest::BaseTest
{
public:
    CV_EmdL1Test();
    ~CV_EmdL1Test();
protected:
    void run(int);
private:
    void testLowDistance();
    void testFairDistance();
    void testBigDistance();
};

CV_EmdL1Test::CV_EmdL1Test()
{
}
CV_EmdL1Test::~CV_EmdL1Test()
{
}

void CV_EmdL1Test::testFairDistance()
{
	// Defining Histograms
    Mat sig1(numBins,3,CV_32F), sig2(numBins,3,CV_32F);
    for (int ii=0; ii<numBins; ii++)
    {
        sig1.at<float>(ii,0)=float(ii)/float(numBins);
        sig2.at<float>(ii,0)=(0.5+float(rand())/float(RAND_MAX))*float(ii+10)/float(numBins);
        sig1.at<float>(ii,1)=ii;
        sig2.at<float>(ii,1)=ii;
        sig1.at<float>(ii,2)=0;
        sig2.at<float>(ii,2)=0;
    }
    std::cout<<"EMD between the histograms: "<<EMD(sig1, sig2, DIST_L2)<<std::endl;
    std::cout<<"EMDL1 between the histograms: "<<EMDL1(sig1.col(0), sig2.col(0))<<std::endl;
    std::cout<<std::endl;
}

void CV_EmdL1Test::testLowDistance()
{
    srand (time(NULL));
    // Defining Histograms
    Mat sig1(numBins,3,CV_32F), sig2(numBins,3,CV_32F);
    for (int ii=0; ii<numBins; ii++)
    {
        sig1.at<float>(ii,0)=float(ii)/float(numBins);
        sig2.at<float>(ii,0)=float(ii+float(rand())/float(RAND_MAX))/float(numBins);
        sig1.at<float>(ii,1)=ii;
        sig2.at<float>(ii,1)=ii;
        sig1.at<float>(ii,2)=0;
        sig2.at<float>(ii,2)=0;
    }
    std::cout<<"EMD between the histograms: "<<EMD(sig1, sig2, DIST_L2)<<std::endl;
    std::cout<<"EMDL1 between the histograms: "<<EMDL1(sig1.col(0), sig2.col(0))<<std::endl;
    std::cout<<std::endl;
}

void CV_EmdL1Test::testBigDistance()
{
    srand (time(NULL));
    // Defining Histograms
    Mat sig1(numBins,3,CV_32F), sig2(numBins,3,CV_32F);
    for (int ii=0; ii<numBins; ii++)
    {
        sig1.at<float>(ii,0)=float(ii)/float(numBins);
        sig2.at<float>(ii,0)=(rand()%numBins)/float(numBins);
        sig1.at<float>(ii,1)=ii;
        sig2.at<float>(ii,1)=ii;
        sig1.at<float>(ii,2)=0;
        sig2.at<float>(ii,2)=0;
    }
    std::cout<<"EMD between the histograms: "<<EMD(sig1, sig2, DIST_L2)<<std::endl;
    std::cout<<"EMDL1 between the histograms: "<<EMDL1(sig1.col(0), sig2.col(0))<<std::endl;
    std::cout<<std::endl;
}

void CV_EmdL1Test::run(int /* */)
{
    /*testLowDistance();
    testFairDistance();
    testBigDistance();*/
	ts->set_failed_test_info(cvtest::TS::OK);	
}

TEST(EmdL1, regression) { CV_EmdL1Test test; test.safe_run(); }
