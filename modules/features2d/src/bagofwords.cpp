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

#include "precomp.hpp"

using namespace std;

namespace cv
{

BOWKMeansTrainer::BOWKMeansTrainer( int _clusterCount, const TermCriteria& _termcrit,
                                    int _attempts, int _flags ) :
    clusterCount(_clusterCount), termcrit(_termcrit), attempts(_attempts), flags(_flags)
{}

void BOWKMeansTrainer::cluster( const Mat& descriptors, Mat& vocabulary )
{
    Mat labels;
    kmeans( descriptors, clusterCount, labels, termcrit, attempts, flags, &vocabulary );
}


BOWImgDescriptorExtractor::BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& _dextractor,
                                                      const Ptr<DescriptorMatcher>& _dmatcher ) :
    dextractor(_dextractor), dmatcher(_dmatcher)
{}

void BOWImgDescriptorExtractor::set( const Mat& _vocabulary )
{
    dmatcher->clear();
    vocabulary = _vocabulary;
    dmatcher->add( vocabulary );
}

void BOWImgDescriptorExtractor::compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& imgDescriptor,
                                         vector<vector<int> >& pointIdxsInClusters )
{
    int clusterCount = descriptorSize(); // = vocabulary.rows

    // Compute descriptors for the image.
    Mat descriptors;
    dextractor->compute( image, keypoints, descriptors );

    // Match keypoint descriptors to cluster center (to vocabulary)
    vector<DMatch> matches;
    dmatcher->match( descriptors, matches );

    // Compute image descriptor
    pointIdxsInClusters = vector<vector<int> >(clusterCount);
    imgDescriptor = Mat( 1, clusterCount, descriptorType(), Scalar::all(0.0) );
    float *dptr = (float*)imgDescriptor.data;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        int queryIdx = matches[i].indexQuery;
        int trainIdx = matches[i].indexTrain; // cluster index
        CV_Assert( queryIdx == (int)i );

        dptr[trainIdx] = dptr[trainIdx] + 1.f;
        pointIdxsInClusters[trainIdx].push_back( queryIdx );
    }

    // Normalize image descriptor.
    imgDescriptor /= descriptors.rows;
}

}
