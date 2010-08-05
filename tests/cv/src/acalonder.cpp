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

using namespace cv;
using namespace std;

#define WRITE_KEYPOINTS     0
#define WRITE_DESCRIPTORS   0

class CV_CalonderTest : public CvTest
{
public:
    CV_CalonderTest() : CvTest("calonder-descriptor-extractor", "CalonderDescriptorExtractor::compute") {}
protected:
    void run(int);
};

void writeMatInBin( const Mat& mat, const string& filename )
{
    FILE* f = fopen( filename.c_str(), "wb");
    int type = mat.type();
    fwrite( (void*)&mat.rows, sizeof(int), 1, f );
    fwrite( (void*)&mat.cols, sizeof(int), 1, f );
    fwrite( (void*)&type, sizeof(int), 1, f );
    fwrite( (void*)&mat.step, sizeof(int), 1, f );
    fwrite( (void*)mat.data, 1, mat.step*mat.rows, f );
    fclose(f);
}

Mat readMatFromBin( const string& filename )
{
    FILE* f = fopen( filename.c_str(), "rb" );
    int rows, cols, type, step;
    fread( (void*)&rows, sizeof(int), 1, f );
    fread( (void*)&cols, sizeof(int), 1, f );
    fread( (void*)&type, sizeof(int), 1, f );
    fread( (void*)&step, sizeof(int), 1, f );

    uchar* data = (uchar*)cvAlloc(step*rows);
    fread( (void*)data, 1, step*rows, f );
    fclose(f);

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
#if WRITE_KEYPOINTS
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
#endif //WRITE_KEYPOINTS

    CalonderDescriptorExtractor<float> fde(dir + "/classifier.rtc");

    Mat fdescriptors;
    double t = (double)getTickCount();
    fde.compute(img, keypoints, fdescriptors);
    t = getTickCount() - t;
    ts->printf(CvTS::LOG, "\nAverage time of computiting float descriptor = %g ms\n", t/((double)cvGetTickFrequency()*1000.)/fdescriptors.rows );

#if WRITE_DESCRIPTORS
    assert(fdescriptors.type() == CV_32FC1);
    writeMatInBin( fdescriptors, dir + "/ros_float_desc" );
#else
    Mat ros_fdescriptors = readMatFromBin( dir + "/ros_float_desc" );
    double fnorm = norm(fdescriptors, ros_fdescriptors, NORM_INF );
    ts->printf(CvTS::LOG, "nofm (inf) BTW valid and calculated float descriptors = %f\n", fnorm );
    if( fnorm > FLT_EPSILON )
        ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
#endif // WRITE_DESCRIPTORS

    CalonderDescriptorExtractor<uchar> cde(dir + "/classifier.rtc");
    Mat cdescriptors;
    t = (double)getTickCount();
    cde.compute(img, keypoints, cdescriptors);
    t = getTickCount() - t;
    ts->printf(CvTS::LOG, "Average time of computiting uchar descriptor = %g ms\n", t/((double)cvGetTickFrequency()*1000.)/cdescriptors.rows );

#if WRITE_DESCRIPTORS
    assert(cdescriptors.type() == CV_8UC1);
    writeMatInBin( cdescriptors, dir + "/ros_uchar_desc" );
#else
    Mat ros_cdescriptors = readMatFromBin( dir + "/ros_uchar_desc" );
    double cnorm = norm(cdescriptors, ros_cdescriptors, NORM_INF );
    ts->printf(CvTS::LOG, "nofm (inf) BTW valid and calculated uchar descriptors = %f\n", cnorm );
    if( cnorm > FLT_EPSILON + 1 ) // + 1 because of quantization float to uchar
        ts->set_failed_test_info( CvTS::FAIL_BAD_ACCURACY );
#endif // WRITE_DESCRIPTORS
}

#if CV_SSE2
CV_CalonderTest calonderTest;
#endif // CV_SSE2
