/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//M*/

#ifndef __OPENCV_SHAPE_SHAPE_TRANSFORM_HPP__
#define __OPENCV_SHAPE_SHAPE_TRANSFORM_HPP__
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

/*!
 * The base class for ShapeTransformer.
 * This is just to define the common interface for
 * shape transformation techniques.
 */
class CV_EXPORTS_W ShapeTransformer : public Algorithm
{
public:
    /* Estimate, Apply Transformation and return Transforming cost*/
    CV_WRAP virtual void estimateTransformation(InputArray transformingShape, InputArray targetShape,
                                                 std::vector<DMatch>& matches) = 0;

    CV_WRAP virtual float applyTransformation(InputArray input, OutputArray output=noArray()) = 0;

    CV_WRAP virtual void warpImage(InputArray transformingImage, OutputArray output,
                                   int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT,
                                   const Scalar& borderValue=Scalar()) const = 0;
};

/***********************************************************************************/
/***********************************************************************************/
/*!
 * Thin Plate Spline Transformation
 * Implementation of the TPS transformation
 * according to "Principal Warps: Thin-Plate Splines and the
 * Decomposition of Deformations" by Juan Manuel Perez for the GSOC 2013
 */

class CV_EXPORTS_W ThinPlateSplineShapeTransformer : public ShapeTransformer
{
public:
    CV_WRAP virtual void setRegularizationParameter(double beta) = 0;
    CV_WRAP virtual double getRegularizationParameter() const = 0;
};

/* Complete constructor */
CV_EXPORTS_W Ptr<ThinPlateSplineShapeTransformer>
    createThinPlateSplineShapeTransformer(double regularizationParameter=0);

/***********************************************************************************/
/***********************************************************************************/
/*!
 * Affine Transformation as a derivated from ShapeTransformer
 */

class CV_EXPORTS_W AffineTransformer : public ShapeTransformer
{
public:
    CV_WRAP virtual void setFullAffine(bool fullAffine) = 0;
    CV_WRAP virtual bool getFullAffine() const = 0;
};

/* Complete constructor */
CV_EXPORTS_W Ptr<AffineTransformer> createAffineTransformer(bool fullAffine);

} // cv
#endif
