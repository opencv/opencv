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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Matthias Bady aegirxx ==> gmail.com
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

using namespace std;
using namespace cv;
using namespace ocl;

#ifdef HAVE_OPENCL

namespace
{
IMPLEMENT_PARAM_CLASS( BRIEF_Bytes, int )
}

PARAM_TEST_CASE( BRIEF, BRIEF_Bytes )
{
    int bytes;

    virtual void SetUp( )
    {
        bytes = GET_PARAM( 0 );
    }
};

OCL_TEST_P( BRIEF, Accuracy )
{
    Mat img = readImage( "gpu/opticalflow/rubberwhale1.png", IMREAD_GRAYSCALE );
    ASSERT_TRUE( !img.empty( ) ) << "no input image";

    FastFeatureDetector fast( 20 );
    std::vector<KeyPoint> keypoints;
    fast.detect( img, keypoints, Mat( ) );

    Mat descriptorsGold;
    BriefDescriptorExtractor brief( bytes );
    brief.compute( img, keypoints, descriptorsGold );

    Mat kpMat( 2, int( keypoints.size() ), CV_32FC1 );
    for ( int i = 0, size = (int)keypoints.size( ); i < size; ++i )
    {
        kpMat.col( i ).row( 0 ) = int( keypoints[i].pt.x );
        kpMat.col( i ).row( 1 ) = int( keypoints[i].pt.y );
    }
    oclMat imgOcl( img ), keypointsOcl( kpMat ), descriptorsOcl, maskOcl;

    BRIEF_OCL briefOcl( bytes );
    briefOcl.compute( imgOcl, keypointsOcl, maskOcl, descriptorsOcl );
    Mat mask, descriptors;
    maskOcl.download( mask );
    descriptorsOcl.download( descriptors );

    const int numDesc = cv::countNonZero( mask );
    if ( numDesc != descriptors.cols )
    {
        int idx = 0;
        Mat tmp( numDesc, bytes, CV_8UC1 );
        for ( int i = 0; i < descriptors.rows; ++i )
        {
            if ( mask.at<uchar>(i) )
            {
                descriptors.row( i ).copyTo( tmp.row( idx++ ) );
            }
        }
        descriptors = tmp;
    }
    ASSERT_TRUE( descriptors.size( ) == descriptorsGold.size( ) ) << "Different number of descriptors";
    ASSERT_TRUE( 0 == norm( descriptors, descriptorsGold, NORM_HAMMING ) ) << "Descriptors different";
}

INSTANTIATE_TEST_CASE_P( OCL_Features2D, BRIEF, testing::Values( 16, 32, 64 ) );
#endif
