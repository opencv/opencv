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
DescriptorExtractor::~DescriptorExtractor()
{}

void DescriptorExtractor::compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const
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

bool DescriptorExtractor::empty() const
{
    return false;
}

void DescriptorExtractor::removeBorderKeypoints( vector<KeyPoint>& keypoints,
                                                 Size imageSize, int borderSize )
{
    KeyPointsFilter::runByImageBorder( keypoints, imageSize, borderSize );
}

Ptr<DescriptorExtractor> DescriptorExtractor::create(const string& descriptorExtractorType)
{
    DescriptorExtractor* de = 0;

    size_t pos = 0;
    if (!descriptorExtractorType.compare("SIFT"))
    {
        de = new SiftDescriptorExtractor();
    }
    else if (!descriptorExtractorType.compare("SURF"))
    {
        de = new SurfDescriptorExtractor();
    }
    else if (!descriptorExtractorType.compare("ORB"))
    {
        de = new OrbDescriptorExtractor();
    }
    else if (!descriptorExtractorType.compare("BRIEF"))
    {
        de = new BriefDescriptorExtractor();
    }
    else if ( (pos=descriptorExtractorType.find("Opponent")) == 0)
    {
        pos += string("Opponent").size();
        de = new OpponentColorDescriptorExtractor( DescriptorExtractor::create(descriptorExtractorType.substr(pos)) );
    }
    return de;
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
                                                  int nOctaveLayers, bool extended, bool upright )
    : surf( 0.0, nOctaves, nOctaveLayers, extended, upright )
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
    bool upright = (int)fn["upright"] != 0;

    surf = SURF( 0.0, nOctaves, nOctaveLayers, extended, upright );
}

void SurfDescriptorExtractor::write( FileStorage &fs ) const
{
//    fs << "algorithm" << getAlgorithmName ();

    fs << "nOctaves" << surf.nOctaves;
    fs << "nOctaveLayers" << surf.nOctaveLayers;
    fs << "extended" << surf.extended;
    fs << "upright" << surf.upright;
}

int SurfDescriptorExtractor::descriptorSize() const
{
    return surf.descriptorSize();
}

int SurfDescriptorExtractor::descriptorType() const
{
    return CV_32FC1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/** Default constructor */
OrbDescriptorExtractor::OrbDescriptorExtractor(ORB::CommonParams params)
{
  orb_ = ORB(0, params);
}
void OrbDescriptorExtractor::computeImpl(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                                         cv::Mat& descriptors) const
{
  cv::Mat empty_mask;
  orb_(image, empty_mask, keypoints, descriptors, true);
}
void OrbDescriptorExtractor::read(const cv::FileNode& fn)
{
  orb_.read(fn);
}
void OrbDescriptorExtractor::write(cv::FileStorage& fs) const
{
  orb_.write(fs);
}
int OrbDescriptorExtractor::descriptorSize() const
{
  return orb_.descriptorSize();
}
int OrbDescriptorExtractor::descriptorType() const
{
  return CV_8UC1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/****************************************************************************************\
*                             OpponentColorDescriptorExtractor                           *
\****************************************************************************************/
OpponentColorDescriptorExtractor::OpponentColorDescriptorExtractor( const Ptr<DescriptorExtractor>& _descriptorExtractor ) :
        descriptorExtractor(_descriptorExtractor)
{
    CV_Assert( !descriptorExtractor.empty() );
}

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
        MatConstIterator_<signed char> rIt = bgrChannels[2].begin<signed char>();
        MatConstIterator_<signed char> gIt = bgrChannels[1].begin<signed char>();
        MatIterator_<unsigned char> dstIt = opponentChannels[0].begin<unsigned char>();
        float factor = 1.f / sqrt(2.f);
        for( ; dstIt != opponentChannels[0].end<unsigned char>(); ++rIt, ++gIt, ++dstIt )
        {
            int value = static_cast<int>( static_cast<float>(static_cast<int>(*gIt)-static_cast<int>(*rIt)) * factor );
            if( value < 0 ) value = 0;
            if( value > 255 ) value = 255;
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
    {
        // (R + G - 2B)/sqrt(6)
        MatConstIterator_<signed char> rIt = bgrChannels[2].begin<signed char>();
        MatConstIterator_<signed char> gIt = bgrChannels[1].begin<signed char>();
        MatConstIterator_<signed char> bIt = bgrChannels[0].begin<signed char>();
        MatIterator_<unsigned char> dstIt = opponentChannels[1].begin<unsigned char>();
        float factor = 1.f / sqrt(6.f);
        for( ; dstIt != opponentChannels[1].end<unsigned char>(); ++rIt, ++gIt, ++bIt, ++dstIt )
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
        MatConstIterator_<signed char> rIt = bgrChannels[2].begin<signed char>();
        MatConstIterator_<signed char> gIt = bgrChannels[1].begin<signed char>();
        MatConstIterator_<signed char> bIt = bgrChannels[0].begin<signed char>();
        MatIterator_<unsigned char> dstIt = opponentChannels[2].begin<unsigned char>();
        float factor = 1.f / sqrt(3.f);
        for( ; dstIt != opponentChannels[2].end<unsigned char>(); ++rIt, ++gIt, ++bIt, ++dstIt )
        {
            int value = static_cast<int>( static_cast<float>(static_cast<int>(*rIt) + static_cast<int>(*gIt) + static_cast<int>(*bIt)) *
                                          factor );
            if( value < 0 ) value = 0;
            if( value > 255 ) value = 255;
            (*dstIt) = static_cast<unsigned char>(value);
        }
    }
}

struct KP_LessThan
{
    KP_LessThan(const vector<KeyPoint>& _kp) : kp(&_kp) {}
    bool operator()(int i, int j) const
    {
        return (*kp)[i].class_id < (*kp)[j].class_id;
    }
    const vector<KeyPoint>* kp;
};

void OpponentColorDescriptorExtractor::computeImpl( const Mat& bgrImage, vector<KeyPoint>& keypoints, Mat& descriptors ) const
{
    vector<Mat> opponentChannels;
    convertBGRImageToOpponentColorSpace( bgrImage, opponentChannels );

    const int N = 3; // channels count
    vector<KeyPoint> channelKeypoints[N];
    Mat channelDescriptors[N];
    vector<int> idxs[N];

    // Compute descriptors three times, once for each Opponent channel to concatenate into a single color descriptor
    int maxKeypointsCount = 0;
    for( int ci = 0; ci < N; ci++ )
    {
        channelKeypoints[ci].insert( channelKeypoints[ci].begin(), keypoints.begin(), keypoints.end() );
        // Use class_id member to get indices into initial keypoints vector
        for( size_t ki = 0; ki < channelKeypoints[ci].size(); ki++ )
            channelKeypoints[ci][ki].class_id = (int)ki;

        descriptorExtractor->compute( opponentChannels[ci], channelKeypoints[ci], channelDescriptors[ci] );
        idxs[ci].resize( channelKeypoints[ci].size() );
        for( size_t ki = 0; ki < channelKeypoints[ci].size(); ki++ )
        {
            idxs[ci][ki] = (int)ki;
        }
        std::sort( idxs[ci].begin(), idxs[ci].end(), KP_LessThan(channelKeypoints[ci]) );
        maxKeypointsCount = std::max( maxKeypointsCount, (int)channelKeypoints[ci].size());
    }

    vector<KeyPoint> outKeypoints;
    outKeypoints.reserve( keypoints.size() );

    int descriptorSize = descriptorExtractor->descriptorSize();
    Mat mergedDescriptors( maxKeypointsCount, 3*descriptorSize, descriptorExtractor->descriptorType() );
    int mergedCount = 0;
    // cp - current channel position
    size_t cp[] = {0, 0, 0}; 
    while( cp[0] < channelKeypoints[0].size() &&
           cp[1] < channelKeypoints[1].size() &&
           cp[2] < channelKeypoints[2].size() )
    {
        const int maxInitIdx = std::max( channelKeypoints[0][idxs[0][cp[0]]].class_id,
                                         std::max( channelKeypoints[1][idxs[1][cp[1]]].class_id,
                                                   channelKeypoints[2][idxs[2][cp[2]]].class_id ) );

        while( channelKeypoints[0][idxs[0][cp[0]]].class_id < maxInitIdx && cp[0] < channelKeypoints[0].size() ) { cp[0]++; }
        while( channelKeypoints[1][idxs[1][cp[1]]].class_id < maxInitIdx && cp[1] < channelKeypoints[1].size() ) { cp[1]++; }
        while( channelKeypoints[2][idxs[2][cp[2]]].class_id < maxInitIdx && cp[2] < channelKeypoints[2].size() ) { cp[2]++; }
        if( cp[0] >= channelKeypoints[0].size() || cp[1] >= channelKeypoints[1].size() || cp[2] >= channelKeypoints[2].size() )
            break;

        if( channelKeypoints[0][idxs[0][cp[0]]].class_id == maxInitIdx &&
            channelKeypoints[1][idxs[1][cp[1]]].class_id == maxInitIdx &&
            channelKeypoints[2][idxs[2][cp[2]]].class_id == maxInitIdx )
        {
            outKeypoints.push_back( keypoints[maxInitIdx] );
            // merge descriptors
            for( int ci = 0; ci < N; ci++ )
            {
                Mat dst = mergedDescriptors(Range(mergedCount, mergedCount+1), Range(ci*descriptorSize, (ci+1)*descriptorSize));
                channelDescriptors[ci].row( idxs[ci][cp[ci]] ).copyTo( dst );
                cp[ci]++;
            }
            mergedCount++;
        }
    }
    mergedDescriptors.rowRange(0, mergedCount).copyTo( descriptors );
    std::swap( outKeypoints, keypoints );
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

bool OpponentColorDescriptorExtractor::empty() const
{
    return descriptorExtractor.empty() || (DescriptorExtractor*)(descriptorExtractor)->empty();
}

}
