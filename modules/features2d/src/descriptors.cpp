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

/****************************************************************************************\
*                                 DescriptorExtractor                                    *
\****************************************************************************************/
/*
 *   DescriptorExtractor
 */
struct RoiPredicate
{
    RoiPredicate(float _minX, float _minY, float _maxX, float _maxY)
        : minX(_minX), minY(_minY), maxX(_maxX), maxY(_maxY)
    {}

    bool operator()( const KeyPoint& keyPt) const
    {
        Point2f pt = keyPt.pt;
        return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
    }

    float minX, minY, maxX, maxY;
};

DescriptorExtractor::~DescriptorExtractor()
{}

void DescriptorExtractor::compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const
{
	if( image.empty() || keypoints.empty() )
		return;

	// Check keypoints are in image. Do filter bad points here?
    //for( size_t i = 0; i < keypoints.size(); i++ )
    //  CV_Assert( Rect(0,0, image.cols, image.rows).contains(keypoints[i].pt) );

	computeImpl( image, keypoints, descriptors );
}

void DescriptorExtractor::compute( const vector<Mat>& imageCollection, vector<vector<KeyPoint> >& pointCollection, vector<Mat>& descCollection ) const
{
    CV_Assert( imageCollection.size() == pointCollection.size() );
    descCollection.resize( imageCollection.size() );
    for( size_t i = 0; i < imageCollection.size(); i++ )
        compute( imageCollection[i], pointCollection[i], descCollection[i] );
}

void DescriptorExtractor::read( const FileNode& )
{}

void DescriptorExtractor::write( FileStorage& ) const
{}

void DescriptorExtractor::removeBorderKeypoints( vector<KeyPoint>& keypoints,
                                                 Size imageSize, int borderSize )
{
    if( borderSize > 0)
    {
        keypoints.erase( remove_if(keypoints.begin(), keypoints.end(),
                                   RoiPredicate((float)borderSize, (float)borderSize,
                                                (float)(imageSize.width - borderSize),
                                                (float)(imageSize.height - borderSize))),
                         keypoints.end() );
    }
}

/****************************************************************************************\
*                                SiftDescriptorExtractor                                 *
\****************************************************************************************/
SiftDescriptorExtractor::SiftDescriptorExtractor(const SIFT::DescriptorParams& descriptorParams,
                                                 const SIFT::CommonParams& commonParams)
    : sift( descriptorParams.magnification, descriptorParams.isNormalize, descriptorParams.recalculateAngles,
            commonParams.nOctaves, commonParams.nOctaveLayers, commonParams.firstOctave, commonParams.angleMode )
{}

SiftDescriptorExtractor::SiftDescriptorExtractor( double magnification, bool isNormalize, bool recalculateAngles,
                                                  int nOctaves, int nOctaveLayers, int firstOctave, int angleMode )
    : sift( magnification, isNormalize, recalculateAngles, nOctaves, nOctaveLayers, firstOctave, angleMode )
{}

void SiftDescriptorExtractor::computeImpl( const Mat& image,
										   vector<KeyPoint>& keypoints,
										   Mat& descriptors) const
{
    bool useProvidedKeypoints = true;
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    sift(grayImage, Mat(), keypoints, descriptors, useProvidedKeypoints);
}

void SiftDescriptorExtractor::read (const FileNode &fn)
{
    double magnification = fn["magnification"];
    bool isNormalize = (int)fn["isNormalize"] != 0;
    bool recalculateAngles = (int)fn["recalculateAngles"] != 0;
    int nOctaves = fn["nOctaves"];
    int nOctaveLayers = fn["nOctaveLayers"];
    int firstOctave = fn["firstOctave"];
    int angleMode = fn["angleMode"];

    sift = SIFT( magnification, isNormalize, recalculateAngles, nOctaves, nOctaveLayers, firstOctave, angleMode );
}

void SiftDescriptorExtractor::write (FileStorage &fs) const
{
//    fs << "algorithm" << getAlgorithmName ();

    SIFT::CommonParams commParams = sift.getCommonParams ();
    SIFT::DescriptorParams descriptorParams = sift.getDescriptorParams ();
    fs << "magnification" << descriptorParams.magnification;
    fs << "isNormalize" << descriptorParams.isNormalize;
    fs << "recalculateAngles" << descriptorParams.recalculateAngles;
    fs << "nOctaves" << commParams.nOctaves;
    fs << "nOctaveLayers" << commParams.nOctaveLayers;
    fs << "firstOctave" << commParams.firstOctave;
    fs << "angleMode" << commParams.angleMode;
}

int SiftDescriptorExtractor::descriptorSize() const
{
    return sift.descriptorSize();
}

int SiftDescriptorExtractor::descriptorType() const
{
    return CV_32FC1;
}

/****************************************************************************************\
*                                SurfDescriptorExtractor                                 *
\****************************************************************************************/
SurfDescriptorExtractor::SurfDescriptorExtractor( int nOctaves,
                                                  int nOctaveLayers, bool extended )
    : surf( 0.0, nOctaves, nOctaveLayers, extended )
{}

void SurfDescriptorExtractor::computeImpl( const Mat& image,
                                           vector<KeyPoint>& keypoints,
                                           Mat& descriptors) const
{
    // Compute descriptors for given keypoints
    vector<float> _descriptors;
    Mat mask;
    bool useProvidedKeypoints = true;
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    surf(grayImage, mask, keypoints, _descriptors, useProvidedKeypoints);

    descriptors.create((int)keypoints.size(), (int)surf.descriptorSize(), CV_32FC1);
    assert( (int)_descriptors.size() == descriptors.rows * descriptors.cols );
    std::copy(_descriptors.begin(), _descriptors.end(), descriptors.begin<float>());
}

void SurfDescriptorExtractor::read( const FileNode &fn )
{
    int nOctaves = fn["nOctaves"];
    int nOctaveLayers = fn["nOctaveLayers"];
    bool extended = (int)fn["extended"] != 0;

    surf = SURF( 0.0, nOctaves, nOctaveLayers, extended );
}

void SurfDescriptorExtractor::write( FileStorage &fs ) const
{
//    fs << "algorithm" << getAlgorithmName ();

    fs << "nOctaves" << surf.nOctaves;
    fs << "nOctaveLayers" << surf.nOctaveLayers;
    fs << "extended" << surf.extended;
}

int SurfDescriptorExtractor::descriptorSize() const
{
    return surf.descriptorSize();
}

int SurfDescriptorExtractor::descriptorType() const
{
    return CV_32FC1;
}

/****************************************************************************************\
*                             OpponentColorDescriptorExtractor                           *
\****************************************************************************************/
OpponentColorDescriptorExtractor::OpponentColorDescriptorExtractor( const Ptr<DescriptorExtractor>& _descriptorExtractor ) :
        descriptorExtractor(_descriptorExtractor)
{}

void convertBGRImageToOpponentColorSpace( const Mat& bgrImage, vector<Mat>& opponentChannels )
{
    if( bgrImage.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "input image must be an BGR image of type CV_8UC3" );

    // Split image into RGB to allow conversion to Opponent Color Space.
    vector<Mat> bgrChannels(3);
    split( bgrImage, bgrChannels );

    // Prepare opponent color space storage matrices.
    opponentChannels.resize( 3 );
    opponentChannels[0] = cv::Mat(bgrImage.size(), CV_8UC1); // R-G RED-GREEN
    opponentChannels[1] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G-2B YELLOW-BLUE
    opponentChannels[2] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G+B

    // Calculate the channels of the opponent color space
    {
        // (R - G) / sqrt(2)
        MatConstIterator_<char> rIt = bgrChannels[2].begin<char>();
        MatConstIterator_<char> gIt = bgrChannels[1].begin<char>();
        MatIterator_<char> dstIt = opponentChannels[0].begin<char>();
        float factor = 1.f / sqrt(2.f);
        for( ; dstIt != opponentChannels[0].end<char>(); ++rIt, ++gIt, ++dstIt )
        {
            int value = static_cast<int>( static_cast<float>(static_cast<int>(*gIt)-static_cast<int>(*rIt)) * factor );
            if( value < 0 ) value = 0;
            if( value > 255 ) value = 255;
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
    {
        // (R + G - 2B)/sqrt(6)
        MatConstIterator_<char> rIt = bgrChannels[2].begin<char>();
        MatConstIterator_<char> gIt = bgrChannels[1].begin<char>();
        MatConstIterator_<char> bIt = bgrChannels[0].begin<char>();
        MatIterator_<char> dstIt = opponentChannels[1].begin<char>();
        float factor = 1.f / sqrt(6.f);
        for( ; dstIt != opponentChannels[1].end<char>(); ++rIt, ++gIt, ++bIt, ++dstIt )
        {
            int value = static_cast<int>( static_cast<float>(static_cast<int>(*rIt) + static_cast<int>(*gIt) - 2*static_cast<int>(*bIt)) *
                                          factor );
            if( value < 0 ) value = 0;
            if( value > 255 ) value = 255;
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
    {
        // (R + G + B)/sqrt(3)
        MatConstIterator_<char> rIt = bgrChannels[2].begin<char>();
        MatConstIterator_<char> gIt = bgrChannels[1].begin<char>();
        MatConstIterator_<char> bIt = bgrChannels[0].begin<char>();
        MatIterator_<char> dstIt = opponentChannels[2].begin<char>();
        float factor = 1.f / sqrt(3.f);
        for( ; dstIt != opponentChannels[2].end<char>(); ++rIt, ++gIt, ++bIt, ++dstIt )
        {
            int value = static_cast<int>( static_cast<float>(static_cast<int>(*rIt) + static_cast<int>(*gIt) + static_cast<int>(*bIt)) *
                                          factor );
            if( value < 0 ) value = 0;
            if( value > 255 ) value = 255;
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
}

void OpponentColorDescriptorExtractor::computeImpl( const Mat& bgrImage, vector<KeyPoint>& keypoints, Mat& descriptors ) const
{
    vector<Mat> opponentChannels;
    convertBGRImageToOpponentColorSpace( bgrImage, opponentChannels );

    // Compute descriptors three times, once for each Opponent channel
    // and concatenate into a single color surf descriptor
    int descriptorSize = descriptorExtractor->descriptorSize();
    descriptors.create( static_cast<int>(keypoints.size()), 3*descriptorSize, CV_32FC1 );
    for( int i = 0; i < 3/*channel count*/; i++ )
    {
        CV_Assert( opponentChannels[i].type() == CV_8UC1 );
        Mat opponentDescriptors = descriptors.colRange( i*descriptorSize, (i+1)*descriptorSize );
        descriptorExtractor->compute( opponentChannels[i], keypoints, opponentDescriptors );
    }
}

void OpponentColorDescriptorExtractor::read( const FileNode& fn )
{
    descriptorExtractor->read(fn);
}

void OpponentColorDescriptorExtractor::write( FileStorage& fs ) const
{
    descriptorExtractor->write(fs);
}

int OpponentColorDescriptorExtractor::descriptorSize() const
{
    return 3*descriptorExtractor->descriptorSize();
}

int OpponentColorDescriptorExtractor::descriptorType() const
{
    return descriptorExtractor->descriptorType();
}
/****************************************************************************************\
*                   Factory function for descriptor extractor creating                   *
\****************************************************************************************/

Ptr<DescriptorExtractor> createDescriptorExtractor(const string& descriptorExtractorType)
{
  DescriptorExtractor* de = 0;
  if (!descriptorExtractorType.compare("SIFT"))
  {
    de = new SiftDescriptorExtractor();
  }
  else if (!descriptorExtractorType.compare("SURF"))
  {
    de = new SurfDescriptorExtractor();
  }
  else if (!descriptorExtractorType.compare("OpponentSIFT"))
  {
    de = new OpponentColorDescriptorExtractor(new SiftDescriptorExtractor);
  }
  else if (!descriptorExtractorType.compare("OpponentSURF"))
  {
    de = new OpponentColorDescriptorExtractor(new SurfDescriptorExtractor);
  }
  else if (!descriptorExtractorType.compare("BRIEF"))
  {
    de = new BriefDescriptorExtractor(32);
  }
  return de;
}

}
