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

typedef TestBaseWithParam<std::tr1::tuple<std::string, int> > OCL_BRIEF;

#define BRIEF_IMAGES \
    "cv/detectors_descriptors_evaluation/images_datasets/leuven/img1.png",\
    "stitching/a3.png"

PERF_TEST_P( OCL_BRIEF, extract, testing::Combine( testing::Values( BRIEF_IMAGES ), testing::Values( 16, 32, 64 ) ) )
{
    const int threshold = 20;
    const std::string filename = std::tr1::get<0>(GetParam( ));
    const int bytes = std::tr1::get<1>(GetParam( ));
    const Mat img = imread( getDataPath( filename ), IMREAD_GRAYSCALE );
    ASSERT_FALSE( img.empty( ) );

    if ( RUN_OCL_IMPL )
    {
        oclMat d_img( img );
        oclMat d_keypoints;
        FAST_OCL fast( threshold );
        fast( d_img, oclMat( ), d_keypoints );

        BRIEF_OCL brief( bytes );

        OCL_TEST_CYCLE( )
        {
            oclMat d_descriptors;
            brief.compute( d_img, d_keypoints, d_descriptors );
        }

        std::vector<KeyPoint> ocl_keypoints;
        fast.downloadKeypoints( d_keypoints, ocl_keypoints );
        SANITY_CHECK_KEYPOINTS( ocl_keypoints );
    }
    else if ( RUN_PLAIN_IMPL )
    {
        std::vector<KeyPoint> keypoints;
        FAST( img, keypoints, threshold );

        BriefDescriptorExtractor brief( bytes );

        TEST_CYCLE( )
        {
            Mat descriptors;
            brief.compute( img, keypoints, descriptors );
        }

        SANITY_CHECK_KEYPOINTS( keypoints );
    }
    else
        OCL_PERF_ELSE;
}