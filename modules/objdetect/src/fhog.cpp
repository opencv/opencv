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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

#include "precomp.hpp"
#include "fhog.hpp"

/****************************************************************************************\
*                      Felzenszwalb's Histogram of Oriented Gradients                    *
\****************************************************************************************/

namespace cv {

FHOGDescriptor::FHOGDescriptor(uint cellSize, uint scale){

    _cellSize = cellSize;
    _scale = scale;

}

FHOGDescriptor::~FHOGDescriptor(){

    freeFeatureMapObject(&_map);

}

FHOGDescriptor* FHOGDescriptor::clone() const{

    return new FHOGDescriptor(*this);

}

Mat FHOGDescriptor::extractFeatures(Mat image){

    _tmplSz.width = _scale * image.cols;
    _tmplSz.height = _scale * image.rows;

    // Round to cell size and also make it even
    _tmplSz.width = ( ( (int)(_tmplSz.width / (2 * _cellSize)) ) * 2 * _cellSize ) + _cellSize*2;
    _tmplSz.height = ( ( (int)(_tmplSz.height / (2 * _cellSize)) ) * 2 * _cellSize ) + _cellSize*2;

    image.convertTo(image, CV_32F, 1 / 255.f);
    if (image.cols != _tmplSz.width || image.rows != _tmplSz.height) {
        resize(image, image, _tmplSz);
    }

    // Add extra cell filled with zeros around the image
    cv::Mat featurePaddingMat( _tmplSz.height+_cellSize*2, _tmplSz.width+_cellSize*2, CV_32FC3, cvScalar(0,0,0) );
    //image.copyTo(featurePaddingMat.rowRange(_cellSize, _cellSize+_tmplSz.height).colRange(_cellSize, _cellSize+_tmplSz.width));
    image.copyTo(featurePaddingMat);

    // HOG features
    getFeatureMaps(featurePaddingMat, _cellSize, &_map);
    normalizeAndTruncate(_map, 0.2f);
    PCAFeatureMaps(_map);
    _featuresMap = Mat(Size(_map->numFeatures*_map->sizeX*_map->sizeY,1), CV_32F, _map->map);  // Procedure do deal with cv::Mat multichannel bug
    _featuresMap = _featuresMap.clone();
    freeFeatureMapObject(&_map);
    return _featuresMap;

}

}