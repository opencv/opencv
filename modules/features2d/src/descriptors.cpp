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
#include <limits>

namespace cv
{

/****************************************************************************************\
*                                 DescriptorExtractor                                    *
\****************************************************************************************/
/*
 *   DescriptorExtractor
 */
DescriptorExtractor::~DescriptorExtractor()
{}

void DescriptorExtractor::compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const
{
    if( image.empty() || keypoints.empty() )
    {
        descriptors.release();
        return;
    }

    KeyPointsFilter::runByImageBorder( keypoints, image.size(), 0 );
    KeyPointsFilter::runByKeypointSize( keypoints, std::numeric_limits<float>::epsilon() );

    computeImpl( image, keypoints, descriptors );
}

void DescriptorExtractor::compute( InputArrayOfArrays _imageCollection, std::vector<std::vector<KeyPoint> >& pointCollection, OutputArrayOfArrays _descCollection ) const
{
    std::vector<Mat> imageCollection, descCollection;
    _imageCollection.getMatVector(imageCollection);
    _descCollection.getMatVector(descCollection);
    CV_Assert( imageCollection.size() == pointCollection.size() );
    descCollection.resize( imageCollection.size() );
    for( size_t i = 0; i < imageCollection.size(); i++ )
        compute( imageCollection[i], pointCollection[i], descCollection[i] );
}

/*void DescriptorExtractor::read( const FileNode& )
{}

void DescriptorExtractor::write( FileStorage& ) const
{}*/

bool DescriptorExtractor::empty() const
{
    return false;
}

void DescriptorExtractor::removeBorderKeypoints( std::vector<KeyPoint>& keypoints,
                                                 Size imageSize, int borderSize )
{
    KeyPointsFilter::runByImageBorder( keypoints, imageSize, borderSize );
}

Ptr<DescriptorExtractor> DescriptorExtractor::create(const String& descriptorExtractorType)
{
    return Algorithm::create<DescriptorExtractor>("Feature2D." + descriptorExtractorType);
}


CV_WRAP void Feature2D::compute( InputArray image, CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints, OutputArray descriptors ) const
{
   DescriptorExtractor::compute(image, keypoints, descriptors);
}

}
