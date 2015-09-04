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

#ifndef FHOG_H
#define FHOG_H

#include "precomp.hpp"
#include "fhogtools.hpp"


/****************************************************************************************\
*                      Felzenszwalb's Histogram of Oriented Gradients                    *
\****************************************************************************************/


/*!
Felzenszwalb's Histogram of Oriented Gradients
FHOG provide an easy-to-use implementation of the Felzenszwalb HOG features extractor, as presented in @cite Felzenszwalb10 .

@code
// Cell size
int cellSize = 4;

// Scale applied
int scale = 1;

// Create object
HogFeature FHOG(cellSize, scale);

// Extract FHOG features
cv::Mat features = FHOG.getFeature(image);
@endcode
*/

namespace cv
{
namespace ml
{

class FHOG {

    public:

    /** @brief FHOG constructor

        @param cellSize cell size, in pixels
        @param scale scale applied to the image

    */
    FHOG(uint cellSize = 4, uint scale = 1);

    virtual ~FHOG();

    virtual FHOG* clone() const;

    /** @brief Get cell size value.

        @param image source image

    */
    virtual uint getCellSize(){ return _cellSize; };

    /** @brief Set cell size value.

        @returns cellSize cell size

    */
    virtual void setCellSize(uint cellSize){ _cellSize = cellSize; };

    /** @brief Get scale value.

        @param scale scale value

    */
    virtual uint getScale(){ return _scale; };

    /** @brief Set scale value.

        @returns scale scale value

    */
    virtual void setScale(uint scale){ _scale = scale; };

    /** @brief Performs features extraction

        @param image source image

    */
    virtual cv::Mat extractFeatures(cv::Mat image);

    private:
        uint _cellSize;
        uint _scale;
        cv::Size _tmplSz;
        cv::Mat _featuresMap;
        cv::Mat _featurePaddingMat;
        CvLSVMFeatureMapCaskade *_map;
};

}
}

#endif