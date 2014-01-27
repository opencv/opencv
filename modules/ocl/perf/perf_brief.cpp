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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Matthias Bady, aegirxx ==> gmail.com
//
//M*/

#include "perf_precomp.hpp"

using namespace std;
using namespace cv;
using namespace ocl;
using namespace perf;

///////////// BRIEF ////////////////////////
typedef TestBaseWithParam<std::tr1::tuple<std::string, int, size_t> > OCL_BRIEF;

PERF_TEST_P( OCL_BRIEF, extract, testing::Combine(
                                                   testing::Values( string( "gpu/opticalflow/rubberwhale1.png" ),
                                                                    string( "gpu/stereobm/aloe-L.png" )
                                                                    ), testing::Values( 16, 32, 64 ), testing::Values( 250, 500, 1000, 2500, 3000 ) ) )
{
    const std::string filename = std::tr1::get<0>(GetParam( ));
    const int bytes = std::tr1::get<1>(GetParam( ));
    const size_t numKp = std::tr1::get<2>(GetParam( ));

    Mat img = imread( getDataPath( filename ), IMREAD_GRAYSCALE );
    ASSERT_TRUE( !img.empty( ) ) << "no input image";

    int threshold = 15;
    std::vector<KeyPoint> keypoints;
    while (threshold > 0 && keypoints.size( ) < numKp)
    {
        FastFeatureDetector fast( threshold );
        fast.detect( img, keypoints, Mat( ) );
        threshold -= 5;
        KeyPointsFilter::runByImageBorder( keypoints, img.size( ), BRIEF_OCL::getBorderSize( ) );
    }
    ASSERT_TRUE( keypoints.size( ) >= numKp ) << "not enough keypoints";
    keypoints.resize( numKp );

    if ( RUN_OCL_IMPL )
    {
        Mat kpMat( 2, int( keypoints.size() ), CV_32FC1 );
        for ( size_t i = 0; i < keypoints.size( ); ++i )
        {
            kpMat.col( int( i ) ).row( 0 ) = keypoints[i].pt.x;
            kpMat.col( int( i ) ).row( 1 ) = keypoints[i].pt.y;
        }
        BRIEF_OCL brief( bytes );
        oclMat imgCL( img ), keypointsCL(kpMat), mask;
        while (next( ))
        {
            startTimer( );
            oclMat descriptorsCL;
            brief.compute( imgCL, keypointsCL, mask, descriptorsCL );
            cv::ocl::finish( );
            stopTimer( );
        }
        SANITY_CHECK_NOTHING( )
    }
    else if ( RUN_PLAIN_IMPL )
    {
        BriefDescriptorExtractor brief( bytes );

        while (next( ))
        {
            startTimer( );
            Mat descriptors;
            brief.compute( img, keypoints, descriptors );
            stopTimer( );
        }
        SANITY_CHECK_NOTHING( )
    }
    else
        OCL_PERF_ELSE;
}