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

namespace cv
{

BOWTrainer::BOWTrainer() : size(0)
{}

BOWTrainer::~BOWTrainer()
{}

void BOWTrainer::add( const Mat& _descriptors )
{
    CV_Assert( !_descriptors.empty() );
    if( !descriptors.empty() )
    {
        CV_Assert( descriptors[0].cols == _descriptors.cols );
        CV_Assert( descriptors[0].type() == _descriptors.type() );
        size += _descriptors.rows;
    }
    else
    {
        size = _descriptors.rows;
    }

    descriptors.push_back(_descriptors);
}

const std::vector<Mat>& BOWTrainer::getDescriptors() const
{
    return descriptors;
}

int BOWTrainer::descriptorsCount() const
{
    return descriptors.empty() ? 0 : size;
}

void BOWTrainer::clear()
{
    descriptors.clear();
}

BOWKMeansTrainer::BOWKMeansTrainer( int _clusterCount, const TermCriteria& _termcrit,
                                    int _attempts, int _flags ) :
    clusterCount(_clusterCount), termcrit(_termcrit), attempts(_attempts), flags(_flags)
{}

Mat BOWKMeansTrainer::cluster() const
{
    CV_INSTRUMENT_REGION();

    CV_Assert( !descriptors.empty() );

    Mat mergedDescriptors( descriptorsCount(), descriptors[0].cols, descriptors[0].type() );
    for( size_t i = 0, start = 0; i < descriptors.size(); i++ )
    {
        Mat submut = mergedDescriptors.rowRange((int)start, (int)(start + descriptors[i].rows));
        descriptors[i].copyTo(submut);
        start += descriptors[i].rows;
    }
    return cluster( mergedDescriptors );
}

BOWKMeansTrainer::~BOWKMeansTrainer()
{}

Mat BOWKMeansTrainer::cluster( const Mat& _descriptors ) const
{
    CV_INSTRUMENT_REGION();

    Mat labels, vocabulary;
    kmeans( _descriptors, clusterCount, labels, termcrit, attempts, flags, vocabulary );
    return vocabulary;
}


BOWImgDescriptorExtractor::BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& _dextractor,
                                                      const Ptr<DescriptorMatcher>& _dmatcher ) :
    dextractor(_dextractor), dmatcher(_dmatcher)
{}

BOWImgDescriptorExtractor::BOWImgDescriptorExtractor( const Ptr<DescriptorMatcher>& _dmatcher ) :
    dmatcher(_dmatcher)
{}

BOWImgDescriptorExtractor::~BOWImgDescriptorExtractor()
{}

void BOWImgDescriptorExtractor::setVocabulary( const Mat& _vocabulary )
{
    dmatcher->clear();
    vocabulary = _vocabulary;
    dmatcher->add( std::vector<Mat>(1, vocabulary) );
}

const Mat& BOWImgDescriptorExtractor::getVocabulary() const
{
    return vocabulary;
}

void BOWImgDescriptorExtractor::compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray imgDescriptor,
                                         std::vector<std::vector<int> >* pointIdxsOfClusters, Mat* descriptors )
{
    CV_INSTRUMENT_REGION();

    imgDescriptor.release();

    if( keypoints.empty() )
        return;

    // Compute descriptors for the image.
    Mat _descriptors;
    dextractor->compute( image, keypoints, _descriptors );

    compute( _descriptors, imgDescriptor, pointIdxsOfClusters );

    // Add the descriptors of image keypoints
    if (descriptors) {
        *descriptors = _descriptors.clone();
    }
}

int BOWImgDescriptorExtractor::descriptorSize() const
{
    return vocabulary.empty() ? 0 : vocabulary.rows;
}

int BOWImgDescriptorExtractor::descriptorType() const
{
    return CV_32FC1;
}

void BOWImgDescriptorExtractor::compute( InputArray keypointDescriptors, OutputArray _imgDescriptor, std::vector<std::vector<int> >* pointIdxsOfClusters )
{
    CV_INSTRUMENT_REGION();

    CV_Assert( !vocabulary.empty() );
    CV_Assert(!keypointDescriptors.empty());

    int clusterCount = descriptorSize(); // = vocabulary.rows

    // Match keypoint descriptors to cluster center (to vocabulary)
    std::vector<DMatch> matches;
    dmatcher->match( keypointDescriptors, matches );

    // Compute image descriptor
    if( pointIdxsOfClusters )
    {
        pointIdxsOfClusters->clear();
        pointIdxsOfClusters->resize(clusterCount);
    }

    _imgDescriptor.create(1, clusterCount, descriptorType());
    _imgDescriptor.setTo(Scalar::all(0));

    Mat imgDescriptor = _imgDescriptor.getMat();

    float *dptr = imgDescriptor.ptr<float>();
    for( size_t i = 0; i < matches.size(); i++ )
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );

        dptr[trainIdx] = dptr[trainIdx] + 1.f;
        if( pointIdxsOfClusters )
            (*pointIdxsOfClusters)[trainIdx].push_back( queryIdx );
    }

    // Normalize image descriptor.
    imgDescriptor /= keypointDescriptors.size().height;
}

}
