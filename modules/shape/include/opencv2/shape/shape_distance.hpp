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

#ifndef __OPENCV_SHAPE_SHAPE_DISTANCE_HPP__
#define __OPENCV_SHAPE_SHAPE_DISTANCE_HPP__
#include "opencv2/core.hpp"
#include "opencv2/shape/hist_cost.hpp"
#include "opencv2/shape/shape_transformer.hpp"

namespace cv
{

/*!
 * The base class for ShapeDistanceExtractor.
 * This is just to define the common interface for
 * shape comparisson techniques.
 */
class CV_EXPORTS_W ShapeDistanceExtractor : public Algorithm
{
public:
    CV_WRAP virtual float computeDistance(InputArray contour1, InputArray contour2) = 0;
};

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*!
 * Shape Context implementation.
 * The SCD class implements SCD algorithm proposed by Belongie et al.in
 * "Shape Matching and Object Recognition Using Shape Contexts".
 * Implemented by Juan M. Perez for the GSOC 2013.
 */
class CV_EXPORTS_W ShapeContextDistanceExtractor : public ShapeDistanceExtractor
{
public:
    CV_WRAP virtual void setAngularBins(int nAngularBins) = 0;
    CV_WRAP virtual int getAngularBins() const = 0;

    CV_WRAP virtual void setRadialBins(int nRadialBins) = 0;
    CV_WRAP virtual int getRadialBins() const = 0;

    CV_WRAP virtual void setInnerRadius(float innerRadius) = 0;
    CV_WRAP virtual float getInnerRadius() const = 0;

    CV_WRAP virtual void setOuterRadius(float outerRadius) = 0;
    CV_WRAP virtual float getOuterRadius() const = 0;

    CV_WRAP virtual void setRotationInvariant(bool rotationInvariant) = 0;
    CV_WRAP virtual bool getRotationInvariant() const = 0;

    CV_WRAP virtual void setShapeContextWeight(float shapeContextWeight) = 0;
    CV_WRAP virtual float getShapeContextWeight() const = 0;

    CV_WRAP virtual void setImageAppearanceWeight(float imageAppearanceWeight) = 0;
    CV_WRAP virtual float getImageAppearanceWeight() const = 0;

    CV_WRAP virtual void setBendingEnergyWeight(float bendingEnergyWeight) = 0;
    CV_WRAP virtual float getBendingEnergyWeight() const = 0;

    CV_WRAP virtual void setImages(InputArray image1, InputArray image2) = 0;
    CV_WRAP virtual void getImages(OutputArray image1, OutputArray image2) const = 0;

    CV_WRAP virtual void setIterations(int iterations) = 0;
    CV_WRAP virtual int getIterations() const = 0;

    CV_WRAP virtual void setCostExtractor(Ptr<HistogramCostExtractor> comparer) = 0;
    CV_WRAP virtual Ptr<HistogramCostExtractor> getCostExtractor() const = 0;

    CV_WRAP virtual void setStdDev(float sigma) = 0;
    CV_WRAP virtual float getStdDev() const = 0;

    CV_WRAP virtual void setTransformAlgorithm(Ptr<ShapeTransformer> transformer) = 0;
    CV_WRAP virtual Ptr<ShapeTransformer> getTransformAlgorithm() const = 0;
};

/* Complete constructor */
CV_EXPORTS_W Ptr<ShapeContextDistanceExtractor>
    createShapeContextDistanceExtractor(int nAngularBins=12, int nRadialBins=4,
                                        float innerRadius=0.2f, float outerRadius=2, int iterations=3,
                                        const Ptr<HistogramCostExtractor> &comparer = createChiHistogramCostExtractor(),
                                        const Ptr<ShapeTransformer> &transformer = createThinPlateSplineShapeTransformer());

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/*!
 * Hausdorff distace implementation based on
 */
class CV_EXPORTS_W HausdorffDistanceExtractor : public ShapeDistanceExtractor
{
public:
    CV_WRAP virtual void setDistanceFlag(int distanceFlag) = 0;
    CV_WRAP virtual int getDistanceFlag() const = 0;

    CV_WRAP virtual void setRankProportion(float rankProportion) = 0;
    CV_WRAP virtual float getRankProportion() const = 0;
};

/* Constructor */
CV_EXPORTS_W Ptr<HausdorffDistanceExtractor> createHausdorffDistanceExtractor(int distanceFlag=cv::NORM_L2, float rankProp=0.6f);

} // cv
#endif
