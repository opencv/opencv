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
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

#define GET_RES 0
class CV_CalonderTest : public CvTest
{
public:
    CV_CalonderTest() : CvTest("CalonderDescriptorExtractor", "CalonderDescriptorExtractor::compute") {}
protected:
    void run(int);
};

void writeMatInBin( const Mat& mat, const string& filename )
{
    ofstream os( filename.c_str() );
    int type = mat.type();
    os.write( (char*)&mat.rows, sizeof(int) );
    os.write( (char*)&mat.cols, sizeof(int) );
    os.write( (char*)&type, sizeof(int) );
    os.write( (char*)&mat.step, sizeof(int) );
    os.write( (char*)mat.data, mat.step*mat.rows );
}

Mat readMatFromBin( const string& filename )
{
    ifstream is( filename.c_str() );
    int rows, cols, type, step;
    is.read( (char*)&rows, sizeof(int) );
    is.read( (char*)&cols, sizeof(int) );
    is.read( (char*)&type, sizeof(int) );
    is.read( (char*)&step, sizeof(int) );

    uchar* data = (uchar*)cvAlloc(step*rows);
    is.read( (char*)data, step*rows );
    return Mat( rows, cols, type, data );
}

void CV_CalonderTest::run(int)
{
    string dir = string(ts->get_data_path()) + "/calonder";
    Mat img = imread(dir +"/boat.png",0);
    if( img.empty() )
    {
        ts->printf(CvTS::LOG, "Test image can not be read\n");
        ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
        return;
    }

    vector<KeyPoint> keypoints;
#if GET_RES
    FastFeatureDetector fd;
    fd.detect(img, keypoints);

    FileStorage fs( dir + "/keypoints.xml", FileStorage::WRITE );
    if( fs.isOpened() )
        write( fs, "keypoints", keypoints );
    else
    {
        ts->printf(CvTS::LOG, "File for writting keypoints can not be opened\n");
        ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
        return;
    }

#else
    FileStorage fs( dir + "/keypoints.xml", FileStorage::READ);
    if( fs.isOpened() )
        read( fs.getFirstTopLevelNode(), keypoints );
    else
    {
        ts->printf(CvTS::LOG, "File for reading keypoints can not be opened\n");
        ts->set_failed_test_info( CvTS::FAIL_INVALID_TEST_DATA );
        return;
    }
#endif //GET_RES

    CalonderDescriptorExtractor<float> fde(dir + "/classifier.rtc");

    Mat fdescriptors;
    double t = getTickCount();
    fde.compute(img, keypoints, fdescriptors);
    t = getTickCount() - t;
    ts->printf(CvTS::LOG, "\nAverage time of computiting float descriptor = %g ms\n", t/((double)cvGetTickFrequency()*1000.)/fdescriptors.rows );

#if GET_RES
    assert(fdescriptors.type() == CV_32FC1);
    writeMatInBin( fdescriptors, "" );
#else
    Mat ros_fdescriptors = readMatFromBin( dir + "/ros_float_desc" );
    double fnorm = norm(fdescriptors, ros_fdescriptors, NORM_INF );
    ts->printf(CvTS::LOG, "nofm (inf) BTW valid and calculated float descriptors = %f\n", fnorm );
    if( fnorm > FLT_EPSILON )
        ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
#endif // GET_RES

    CalonderDescriptorExtractor<uchar> cde(dir + "/classifier.rtc");
    Mat cdescriptors;
    t = getTickCount();
    cde.compute(img, keypoints, cdescriptors);
    t = getTickCount() - t;
    ts->printf(CvTS::LOG, "Average time of computiting uchar descriptor = %g ms\n", t/((double)cvGetTickFrequency()*1000.)/cdescriptors.rows );

#if GET_RES
    assert(cdescriptors.type() == CV_8UC1);
    writeMatInBin( fdescriptors, "" );
#else
    Mat ros_cdescriptors = readMatFromBin( dir + "/ros_uchar_desc" );
    double cnorm = norm(cdescriptors, ros_cdescriptors, NORM_INF );
    ts->printf(CvTS::LOG, "nofm (inf) BTW valid and calculated uchar descriptors = %f\n", cnorm );
    if( cnorm > FLT_EPSILON )
        ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
#endif // GET_RES
}

#if CV_SSE2
CV_CalonderTest calonderTest;
#endif // CV_SSE2
