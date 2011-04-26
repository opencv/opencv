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
#include <iostream>
#include <fstream>

using namespace std;

namespace cv
{

/*
 * Functions
 */

static void readDirContent( const string& descrFilename, vector<string>& names )
{
    names.clear();

    ifstream file( descrFilename.c_str() );
    if ( !file.is_open() )
        return;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        if( str[0] == '#' ) continue; // comment
        names.push_back(str);
    }
    file.close();
}

static void computeGradients( const Mat& image, Mat& magnitudes, Mat& angles )
{
    Mat dx, dy;
    cv::Sobel( image, dx, CV_32F, 1, 0, 3 );
    cv::Sobel( image, dy, CV_32F, 0, 1, 3 );

    cv::cartToPolar( dx, dy, magnitudes, angles, true );
    CV_Assert( magnitudes.type() == CV_32FC1 );
    CV_Assert( angles.type() == CV_32FC1 );
}

static void computeWinData( const Mat& image, const Mat& mask, const Size& winSize,
                            Mat& winImage, Mat& winMask,
                            Mat& winMagnitudes, Mat& winAngles, int border=0 )
{
    CV_Assert( border >= 0 );

    Size extSize;
    extSize.width = winSize.width + 2*border;
    extSize.height = winSize.height + 2*border;

    if( mask.empty() )
    {
        image.copyTo( winImage );
        winMask.release();
    }
    else
    {
        vector<Point> points;
        points.reserve( image.rows * image.cols );
        for( int y = 0; y < mask.rows; y++ )
        {
            for( int x = 0; x < mask.cols; x++ )
            {
                if( mask.at<uchar>(y,x) )
                    points.push_back( cv::Point(x,y) );
            }
        }

        cv::Rect_<float> brect = cv::boundingRect( cv::Mat(points) );

        float ratio = std::min( (float)winSize.width/ brect.width, (float)winSize.height / brect.height );

        float rectWidth = winSize.width / ratio;
        float rectHeight = winSize.height / ratio;

        float scaledBorder = border / ratio;
        brect.x -= (rectWidth - brect.width) / 2.f + scaledBorder ;
        brect.y -= (rectHeight - brect.height) / 2.f + scaledBorder;
        brect.width = rectWidth + 2*scaledBorder;
        brect.height = rectHeight + 2*scaledBorder;

        // TODO the following cases:
        assert( Rect(0, 0, image.cols, image.rows ).contains( brect.tl() ) );
        assert( Rect(0, 0, image.cols, image.rows ).contains( brect.br() ) );

        Mat subImage( image, brect );
        Mat subMask( mask, brect );

        cv::resize( subImage, winImage, extSize );
        cv::resize( subMask, winMask, extSize );

        CV_Assert( winImage.size() == extSize );
        CV_Assert( winMask.size() == extSize );
    }

    computeGradients( winImage, winMagnitudes, winAngles );

    //    Mat bluredWinImage;
    //    cv::GaussianBlur( winImage, bluredWinImage, Size(), 0.5, 0.5 );
    //    computeGradients( bluredWinImage, magnitudes, angles );
}

inline int getBin( double angle )
{
    double angle1 = angle >= 180 ? angle - 180 : angle;
    int orientationBin = (int)(angle1 / DOTDetector::TrainParams::BIN_RANGE() );
    assert( orientationBin < 7 );

    return orientationBin;
}

static void copyTrainData( const Mat& magnitudesSrc, const Mat& anglesSrc, const Mat& mask,
                           Mat& magnitudesDst, Mat& anglesDst )
{
    magnitudesDst = Mat( magnitudesSrc.size(), magnitudesSrc.type(), Scalar::all(-1) );
    anglesDst = Mat( anglesSrc.size(), anglesSrc.type(), Scalar::all(-1) );

    magnitudesSrc.copyTo( magnitudesDst, mask );
    anglesSrc.copyTo( anglesDst, mask );
}

inline int countNonZeroBits( uchar val )
{
    uchar v = val;
    v = (v & 0x55) + ((v >> 1) & 0x55);
    v = (v & 0x33) + ((v >> 2) & 0x33);

    return (v & 0x0f) + ((v >> 4) & 0x0f);
}

inline void countNonZeroAndTexturelessBits( const Mat& mat, int& nonZeroBitsCount, int& texturelessBitsCount )
{
    CV_Assert( mat.type() == CV_8UC1 );

    nonZeroBitsCount = 0;
    texturelessBitsCount = 0;

    const uchar texturelessValue = 1 << DOTDetector::TrainParams::BIN_COUNT;
    for( int y = 0; y < mat.rows; y++ )
    {
        for( int x = 0; x < mat.cols; x++ )
        {
            int curCount = countNonZeroBits( mat.at<uchar>(y,x) );
            if( curCount )
            {
                nonZeroBitsCount += curCount;
                if( mat.at<uchar>(y,x) == texturelessValue )
                    texturelessBitsCount++;
            }
        }
    }
}

static void quantizeToTrain( const Mat& _magnitudesExt, const Mat& _anglesExt, const Mat& maskExt,
                             Mat& quantizedImage, const DOTDetector::TrainParams& params )
{
    CV_DbgAssert( params.winSize.height % params.regionSize == 0 );
    CV_DbgAssert( params.winSize.width % params.regionSize == 0 );
    CV_DbgAssert( params.regionSize % 2 == 1 );

    const int regionSize_2 = params.regionSize / 2;

    Mat magnitudesExt, anglesExt;
    copyTrainData( _magnitudesExt, _anglesExt, maskExt, magnitudesExt, anglesExt );

    const int verticalRegionCount = params.winSize.height / params.regionSize;
    const int horizontalRegionCount = params.winSize.width / params.regionSize;

    quantizedImage = Mat( verticalRegionCount, horizontalRegionCount, CV_8UC1, Scalar::all(0) );

    Rect curRect( regionSize_2, regionSize_2, params.regionSize, params.regionSize );

    for( int vRegIdx = 0; vRegIdx < verticalRegionCount; vRegIdx++ )
    {
        for( int hRegIdx = 0; hRegIdx < horizontalRegionCount; hRegIdx++ )
        {
            uchar curRectBits = 0;

            for( int yShift = -regionSize_2; yShift <= regionSize_2; yShift++ ) // TODO yShift += regionSize/2
            {
                Rect shiftedRect = curRect;

                shiftedRect.y = curRect.y + yShift;

                for( int xShift = -regionSize_2; xShift <= regionSize_2; xShift++ ) // TODO xShift += regionSize/2
                {
                    shiftedRect.x = curRect.x + xShift;

                    Mat subMagnitudes( magnitudesExt, shiftedRect ), subMagnitudesCopy;
                    subMagnitudes.copyTo( subMagnitudesCopy );
                    Mat subAngles( anglesExt, shiftedRect );

                    double maxMagnitude;
                    int strongestCount = 0;
                    for( ; strongestCount < params.maxStrongestCount; strongestCount++ )
                    {
                        Point maxLoc;
                        cv::minMaxLoc( subMagnitudesCopy, 0, &maxMagnitude, 0, &maxLoc );

                        if( maxMagnitude < params.minMagnitude )
                            break;

                        subMagnitudesCopy.at<float>( maxLoc ) = -1;

                        double angle = subAngles.at<float>( maxLoc );
                        int orientationBin = getBin( angle );

                        curRectBits |= 1 << orientationBin;
                    }
                    if( strongestCount == 0 && maxMagnitude > 0 )
                        curRectBits |= 1 << DOTDetector::TrainParams::BIN_COUNT;
                }
            }

            if( !( curRectBits == (1 << DOTDetector::TrainParams::BIN_COUNT) && cv::countNonZero(magnitudesExt(curRect) == -1) ) )
            {
                if( countNonZeroBits( curRectBits ) <= params.maxNonzeroBits )
                    quantizedImage.at<uchar>(vRegIdx, hRegIdx) = curRectBits;
            }

            curRect.x += params.regionSize;
        }
        curRect.x = regionSize_2;
        curRect.y += params.regionSize;
    }
}

static void quantizeToDetect( const Mat& _magnitudes, const Mat& angles, Mat& quantizedImage, const DOTDetector::TrainParams& params )
{
    Mat magnitudes; _magnitudes.copyTo( magnitudes );

    const int verticalRegionCount = magnitudes.rows / params.regionSize;
    const int horizontalRegionCount = magnitudes.cols / params.regionSize;

    quantizedImage = Mat( verticalRegionCount, horizontalRegionCount, CV_8UC1, Scalar::all(0) );

    Rect curRect(0, 0, params.regionSize, params.regionSize);
    const int maxStrongestCount = 1;
    for( int vRegIdx = 0; vRegIdx < verticalRegionCount; vRegIdx++ )
    {
        for( int hRegIdx = 0; hRegIdx < horizontalRegionCount; hRegIdx++ )
        {
            uchar curRectBits = 0;

            Mat subMagnitudes( magnitudes, curRect ), subMagnitudesCopy;
            subMagnitudes.copyTo( subMagnitudesCopy );
            Mat subAngles( angles, curRect );

            double maxMagnitude = -1;
            int strongestCount = 0;

            for( ; strongestCount < maxStrongestCount; strongestCount++ )
            {
                Point maxLoc;
                cv::minMaxLoc( subMagnitudesCopy, 0, &maxMagnitude, 0, &maxLoc );

                if( maxMagnitude < params.minMagnitude )
                    break;

                subMagnitudesCopy.at<float>( maxLoc ) = -1;

                double angle = subAngles.at<float>( maxLoc );
                int orientationBin = getBin( angle );

                curRectBits |= 1 << orientationBin;
            }
            if( strongestCount == 0 && maxMagnitude > 0 )
                curRectBits |= 1 << DOTDetector::TrainParams::BIN_COUNT;

            quantizedImage.at<uchar>(vRegIdx, hRegIdx) = curRectBits;
            curRect.x += params.regionSize;
        }
        curRect.x = 0;
        curRect.y += params.regionSize;
    }
}

inline void andQuantizedImages( const Mat& queryQuantizedImage, const Mat& trainQuantizedImage, float& ratio, float& texturelessRatio )
{
    int nonZeroCount = 0, texturelessCount = 0;
    countNonZeroAndTexturelessBits( trainQuantizedImage & queryQuantizedImage, nonZeroCount, texturelessCount );

    CV_Assert( nonZeroCount > 0 );
    int area = cv::countNonZero( trainQuantizedImage );

    ratio = (float)nonZeroCount / area;
    texturelessRatio = texturelessCount / nonZeroCount;
}

static void computeTrainUsedStrongestMask( const Mat& _magnitudesExt, const Mat& _anglesExt, const Mat& maskExt, const Mat& quantizedImage,
                                           Mat& winUsedStrongestMask, int regionSize, int minMagnitude )
{
    const int usedLabel = 255;
    const int regionSize_2 = regionSize / 2;

    Mat magnitudesExt, anglesExt;
    copyTrainData( _magnitudesExt, _anglesExt, maskExt, magnitudesExt, anglesExt );

    const int verticalRegionCount = quantizedImage.rows;
    const int horizontalRegionCount = quantizedImage.cols;

    Mat binsExt( anglesExt.size(), CV_32SC1, Scalar::all(-1) );
    for( int y = 0; y < binsExt.rows; y++ )
    {
        for( int x = 0; x < binsExt.cols; x++ )
        {
            if( magnitudesExt.at<float>(y,x) >= minMagnitude )
            {
                binsExt.at<int>(y,x) = getBin( anglesExt.at<float>(y,x) );
            }
        }
    }

    Rect curRect( 0, 0, regionSize + 2*regionSize_2, regionSize + 2*regionSize_2 );

    Mat colorsExt( anglesExt.size(), CV_8UC1, Scalar::all(0) );
    for( int vRegIdx = 0; vRegIdx < verticalRegionCount; vRegIdx++ )
    {
        for( int hRegIdx = 0; hRegIdx < horizontalRegionCount; hRegIdx++ )
        {
            Mat subColors = colorsExt( curRect );
            Mat subBins = binsExt( curRect );

            uchar bits = quantizedImage.at<uchar>(vRegIdx, hRegIdx);

            for( int binIdx = 0; binIdx < DOTDetector::TrainParams::BIN_COUNT; binIdx++ )
            {
                if( bits & (1 << binIdx) )
                {
                    float gray = usedLabel/* * weights[vRegIdx*horizontalRegionCount+hRegIdx][binIdx]*/;
                    subColors.setTo( Scalar((uchar)cvRound(gray)), subBins == binIdx );
                }
            }

            curRect.x += regionSize;
        }

        curRect.x = 0;
        curRect.y += regionSize;
    }

    Mat colors = colorsExt( Rect(regionSize_2, regionSize_2, binsExt.cols - 2*regionSize_2, binsExt.rows - 2*regionSize_2) );

    colors.convertTo( winUsedStrongestMask, CV_8UC1 );
}

/*
 * DOTDetector::Params
 */

DOTDetector::TrainParams::TrainParams() : winSize(Size(84,84)), regionSize(7),
                                          minMagnitude(60), maxStrongestCount(7), maxNonzeroBits(6),
                                          minRatio(0.85f) {}

DOTDetector::TrainParams::TrainParams( const Size& _winSize, int _regionSize, int _minMagnitude,
                                       int _maxStrongestCount, int _maxNonzeroBits,
                                       float _minRatio ) :
                                winSize(_winSize), regionSize(_regionSize), minMagnitude(_minMagnitude),
                                maxStrongestCount(_maxStrongestCount), maxNonzeroBits(_maxNonzeroBits),
                                minRatio(_minRatio)
{
    asserts();
}

void DOTDetector::TrainParams::asserts() const
{
    CV_Assert( winSize.width > 0 && winSize.height > 0 );
    CV_Assert( regionSize > 0 && regionSize % 2 == 1);

    CV_Assert( winSize.width % regionSize == 0 );
    CV_Assert( winSize.height % regionSize == 0 );

    CV_Assert( minMagnitude > 0 );

    CV_Assert( maxStrongestCount > 0 && maxStrongestCount <= BIN_COUNT );
    CV_Assert( maxNonzeroBits > 0 && maxNonzeroBits <= BIN_COUNT );

    CV_Assert( minRatio > 0.f && minRatio < 1.f );
}

void DOTDetector::TrainParams::read( FileNode& fn )
{
    winSize.width = fn["winSize.width"];
    winSize.height = fn["winSize.height"];
    regionSize = fn["regionSize"];

    minMagnitude = fn["minMagnitude"];
    maxStrongestCount = fn["maxStrongestCount"];
    maxNonzeroBits = fn["maxNonzeroBits"];

    minRatio = fn["minRatio"];

    asserts();
}

void DOTDetector::TrainParams::write( FileStorage& fs ) const
{
    CV_Assert( fs.isOpened() );

    fs << "winSize.width" << winSize.width;
    fs << "winSize.height" << winSize.height;
    fs << "regionSize" << regionSize;

    fs << "minMagnitude" << minMagnitude;
    fs << "maxStrongestCount" << maxStrongestCount;
    fs << "maxNonzeroBits" << maxNonzeroBits;

    fs << "minRatio" << minRatio;
}

DOTDetector::DetectParams::DetectParams() : minRatio(0.8f), minRegionSize(7), maxRegionSize(9), regionSizeStep(2),
                                            isGroup(true), groupThreshold(3), groupEps(0.2) {}

DOTDetector::DetectParams::DetectParams( float _minRatio, int _minRegionSize, int _maxRegionSize, int _regionSizeStep,
                                         bool _isGroup, int _groupThreshold, double _groupEps ) :
    minRatio(_minRatio), minRegionSize(_minRegionSize), maxRegionSize(_maxRegionSize), regionSizeStep(_regionSizeStep),
    isGroup(_isGroup), groupThreshold(_groupThreshold), groupEps(_groupEps)
{
    asserts();
}

void DOTDetector::DetectParams::asserts( float minTrainRatio ) const
{
    CV_Assert( minRatio > 0 && minRatio < 1 );
    CV_Assert( minRatio <= minTrainRatio );

    CV_Assert( minRegionSize > 0 && minRegionSize % 2 == 1 );
    CV_Assert( maxRegionSize > 0 && maxRegionSize % 2 == 1 );
    CV_Assert( minRegionSize <= maxRegionSize );

    CV_Assert( regionSizeStep % 2 == 0 );

    if( isGroup )
    {
        CV_Assert( groupThreshold > 0 );
        CV_Assert( groupEps > 0 && groupEps < 1 );
    }
}

/*
 * DOTDetector::DOTTemplate
 */

DOTDetector::DOTTemplate::DOTTemplate() : texturelessRatio(-1.f) {}

DOTDetector::DOTTemplate::DOTTemplate( const cv::Mat& _quantizedImage, int _classID, const cv::Mat& _maskedImage, const cv::Mat& _gradientMask ) :
        quantizedImage(_quantizedImage), texturelessRatio(computeTexturelessRatio(_quantizedImage))
{
    addClassID( _classID, _maskedImage, _gradientMask );
}

void DOTDetector::DOTTemplate::addClassID( int _classID, const cv::Mat& _maskedImage, const cv::Mat& _gradientMask )
{
    CV_Assert( _classID >= 0 );
    bool isFound = false;

    for( size_t i = 0; i < classIDs.size(); i++ )
    {
        if( classIDs[i] == _classID )
        {
            isFound = true;
            break;
        }
    }

    if( !isFound )
    {
        classIDs.push_back( _classID );
        if( !_maskedImage.empty() )
        {
            CV_Assert( !_gradientMask.empty() );
            maskedImages.push_back( _maskedImage );
            gradientMasks.push_back( _gradientMask );
        }
    }
}

float DOTDetector::DOTTemplate::computeTexturelessRatio( const cv::Mat& quantizedImage )
{
    const uchar TEXTURELESS_VAL = 1 << DOTDetector::TrainParams::BIN_COUNT;
    int texturelessCount = 0;
    for( int y = 0; y < quantizedImage.rows; y++ )
    {
        for( int x = 0; x < quantizedImage.cols; x++ )
        {
            if( quantizedImage.at<uchar>(y,x) & TEXTURELESS_VAL )
                texturelessCount++;
        }
    }
    return (float)texturelessCount/ (float)(quantizedImage.cols * quantizedImage.rows);
}

void DOTDetector::DOTTemplate::read( FileNode& fn )
{
    fn["template"] >> quantizedImage;
    fn["classIDs"] >> classIDs;
    texturelessRatio = fn["texturelessRatio"];
}

void DOTDetector::DOTTemplate::write( FileStorage& fs ) const
{
    fs << "template" << quantizedImage;
    fs << "classIDs" << classIDs;
    fs << "texturelessRatio" << texturelessRatio;
}

/*
 * DOTDetector
 */

DOTDetector::DOTDetector() : isAddImageAndGradientMask( false )
{
}

DOTDetector::DOTDetector( const std::string& filename ) : isAddImageAndGradientMask( false )
{
    load( filename );
}

DOTDetector::~DOTDetector()
{
    clear();
}

void DOTDetector::clear()
{
    classNames.clear();
    dotTemplates.clear();
}

void DOTDetector::read( FileNode& fn )
{
    clear();

    // read params
    FileNode fn_params = fn["train_params"];
    trainParams.read( fn_params );

    // read class names
    int classCount = fn["class_count"];
    FileNodeIterator fni = fn["class_names"].begin();
    for( int i = 0; i < classCount; i++ )
    {
        string name;
        fni >> name;
        classNames.push_back( name );
    }

    // read DOT templates
    int templatesCount = fn["templates_count"];
    fni = fn["templates"].begin();
    for( int i = 0; i < templatesCount; i++ )
    {
        dotTemplates.push_back( DOTTemplate() );
        FileNode cur_fn = *fni;
        dotTemplates.rbegin()->read( cur_fn );
    }
}

void DOTDetector::write( FileStorage& fs ) const
{
    // write params
    fs << "train_params" << "{";
    trainParams.write( fs );
    fs << "}"; //params

    // write class names
    fs << "class_count" << (int)classNames.size();
    fs << "class_names" << "[";
    for( size_t i = 0; i < classNames.size(); i++ )
    {
        fs << classNames[i];
    }
    fs << "]";

    // write dot templates
    fs << "templates_count" << (int)dotTemplates.size();
    fs << "templates" << "[";
    for( size_t i = 0; i < dotTemplates.size(); i++ )
    {
        dotTemplates[i].write( fs );
    }
    fs << "]";
}

void DOTDetector::load( const std::string& filename )
{
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
        FileNode fn = fs.getFirstTopLevelNode();
        read( fn );
    }
}

void DOTDetector::save( const std::string& filename ) const
{
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "dot_detector" << "{";
        write( fs );
        fs << "}";
    }
}

void DOTDetector::train( const string& _baseDirName, const TrainParams& _trainParams, bool _isAddImageAndGradientMask )
{
    clear();

    trainParams = _trainParams;
    trainParams.asserts();

    string baseDirName = _baseDirName + (*(_baseDirName.end()-1) == '/' ? "" : "/");
    const int regionSize_2 = trainParams.regionSize / 2;

    readDirContent( baseDirName+"objects.txt", classNames );

    for( size_t objIdx = 0; objIdx < classNames.size(); objIdx++ )
    {
        string curObjDirName = baseDirName + classNames[objIdx] + "/";

        cout << "===============" << classNames[objIdx] << "===============" << endl;
        vector<string> imagesFilenames;
        readDirContent( curObjDirName + "images.txt", imagesFilenames );

        if( imagesFilenames.empty() )
            continue;

        int countSamples = 0;
        for( size_t imgIdx = 0; imgIdx < imagesFilenames.size(); imgIdx++ )
        {
            cout << imagesFilenames[imgIdx] ;
            Mat image = cv::imread( curObjDirName + imagesFilenames[imgIdx], 0 );
            if( image.empty() )
                continue;

            Mat mask;
            {
                Mat _mask = cv::imread( curObjDirName + imagesFilenames[imgIdx] + ".mask.png", 0 );
                if( _mask.empty() )
                {
                    cout << " - FAIL" << endl;
                    continue;
                }
                mask = _mask;
            }
            cout << " - OK" << endl;

            countSamples++;

            Mat trainImageExt, trainMaskExt, trainQuantizedImage, detectQuantizedImage;
            Mat trainMagnitudesExt, trainAnglesExt;

            computeWinData( image, mask, trainParams.winSize,
                            trainImageExt, trainMaskExt,
                            trainMagnitudesExt, trainAnglesExt, regionSize_2 );

            quantizeToTrain( trainMagnitudesExt, trainAnglesExt, trainMaskExt, trainQuantizedImage, trainParams );

            quantizeToDetect( trainMagnitudesExt, trainAnglesExt, detectQuantizedImage, trainParams );

            vector<vector<Rect> > rects;
            vector<vector<float> > ratios;
            vector<vector<int> > trainTemplatesIdxs;
            detectQuantized( detectQuantizedImage, trainParams.minRatio, rects, ratios, trainTemplatesIdxs );

            Mat maskedTrainImage, trainGradientMask;
            if( isAddImageAndGradientMask )
            {
                trainImageExt.copyTo( maskedTrainImage, trainMaskExt);
                computeTrainUsedStrongestMask( trainMagnitudesExt, trainAnglesExt, trainMaskExt, trainQuantizedImage,
                                               trainGradientMask, trainParams.regionSize, trainParams.minMagnitude );
            }
            int classID = classNames.size()-1;
            bool isFound = false;
            for( size_t cIdx = 0; cIdx < trainTemplatesIdxs.size(); cIdx++ )
            {
                if( trainTemplatesIdxs[cIdx].size() )
                {
                    for( size_t i = 0; i < trainTemplatesIdxs[cIdx].size(); i++ )
                    {
                        int tIdx = trainTemplatesIdxs[cIdx][i];

                        dotTemplates[tIdx].addClassID( classID, maskedTrainImage, trainGradientMask );
                        isFound = true;
                    }
                }
            }
            if( !isFound )
            {
                dotTemplates.push_back( DOTTemplate(trainQuantizedImage, classID, maskedTrainImage, trainGradientMask) );
            }

            cout << "dot templates size = " << dotTemplates.size() << endl;
        }
    }
}

void DOTDetector::detectQuantized( const Mat& testQuantizedImage, float minRatio,
                                   vector<vector<Rect> >& rects, vector<vector<float> >& ratios, vector<vector<int> >& trainTemlateIdxs ) const
{
    if( dotTemplates.empty() )
        return;

    const int regionsPerRow = dotTemplates[0].quantizedImage.rows;
    const int regionsPerCol = dotTemplates[0].quantizedImage.cols;

    int classCount = classNames.size();

    rects.resize( classCount );
    ratios.resize( classCount );
    trainTemlateIdxs.resize( classCount );

    for( size_t tIdx = 0; tIdx < dotTemplates.size(); tIdx++ )
    {
        Rect r( 0, 0, regionsPerCol, regionsPerRow );
        for( r.y = 0; r.y <= testQuantizedImage.rows-r.height; r.y++ )
        {
            for( r.x = 0; r.x <= testQuantizedImage.cols-r.width; r.x++ )
            {
                float ratio, texturelessRatio;
                andQuantizedImages( testQuantizedImage(r), dotTemplates[tIdx].quantizedImage, ratio, texturelessRatio );
                if( ratio > minRatio && texturelessRatio < dotTemplates[tIdx].texturelessRatio )
                {
                    for( size_t cIdx = 0; cIdx < dotTemplates[tIdx].classIDs.size(); cIdx++ )
                    {
                        int classID =  dotTemplates[tIdx].classIDs[cIdx];
                        rects[classID].push_back( r );
                        ratios[classID].push_back( ratio );
                        trainTemlateIdxs[classID].push_back( tIdx );
                    }
                }
            }
        }
    }
}

void DOTDetector::detectMultiScale( const Mat& image, vector<vector<Rect> >& rects,
                                    const DetectParams& detectParams, vector<vector<float> >* ratios, vector<vector<int> >* trainTemplateIndices ) const
{
    detectParams.asserts( trainParams.minRatio );

    int classCount = classNames.size();
    rects.resize( classCount );
    if( ratios )
    {
        ratios->clear();
        if( !detectParams.isGroup )
            ratios->resize( classCount );
    }
    if( trainTemplateIndices )
    {
        trainTemplateIndices->clear();
        if( !detectParams.isGroup )
            trainTemplateIndices->resize( classCount );
    }

    Mat magnitudes, angles;
    computeGradients( image, magnitudes, angles );
    for( int regionSize = detectParams.minRegionSize; regionSize <= detectParams.maxRegionSize; regionSize += detectParams.regionSizeStep )
    {
        Mat quantizedImage;
        vector<vector<Rect> > curRects;
        vector<vector<float> > curRatios;
        vector<vector<int> > curTrainTemlateIdxs;
        quantizeToDetect( magnitudes, angles, quantizedImage, trainParams );
        detectQuantized( quantizedImage, detectParams.minRatio, curRects, curRatios, curTrainTemlateIdxs );

        for( int ci = 0; ci < classCount; ci++ )
        {
            for( size_t ri = 0; ri < curRects[ci].size(); ri++  )
            {
                Rect r = curRects[ci][ri];
                r.x *= regionSize;
                r.y *= regionSize;
                r.width *= regionSize;
                r.height *= regionSize;

                rects[ci].push_back( r );
                if( ratios && !detectParams.isGroup )
                    (*ratios)[ci].push_back( curRatios[ci][ri] );
                if( trainTemplateIndices && !detectParams.isGroup )
                    (*trainTemplateIndices)[ci].push_back( curTrainTemlateIdxs[ci][ri] );
            }
        }
    }

    if( detectParams.isGroup )
        groupRectanglesList( rects, detectParams.groupThreshold, detectParams.groupEps );
}


const vector<DOTDetector::DOTTemplate>& DOTDetector::getDOTTemplates() const
{
    return dotTemplates;
}

const vector<string>& DOTDetector::getClassNames() const
{
    return classNames;
}

void DOTDetector::groupRectanglesList( std::vector<std::vector<cv::Rect> >& rectList, int groupThreshold, double eps )
{
    for( size_t i = 0; i < rectList.size(); i++ )
        groupRectangles( rectList[i], groupThreshold, eps );
}

} // namespace cv

/* End of file. */
