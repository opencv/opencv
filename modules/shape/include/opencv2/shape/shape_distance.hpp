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

#ifndef OPENCV_SHAPE_SHAPE_DISTANCE_HPP
#define OPENCV_SHAPE_SHAPE_DISTANCE_HPP
#include "opencv2/core.hpp"
#include "opencv2/shape/hist_cost.hpp"
#include "opencv2/shape/shape_transformer.hpp"

namespace cv
{

//! @addtogroup shape
//! @{

/** @brief Abstract base class for shape distance algorithms.
 */
class CV_EXPORTS_W ShapeDistanceExtractor : public Algorithm
{
public:
    /** @brief Compute the shape distance between two shapes defined by its contours.

    @param contour1 Contour defining first shape.
    @param contour2 Contour defining second shape.
     */
    CV_WRAP virtual float computeDistance(InputArray contour1, InputArray contour2) = 0;
};

/***********************************************************************************/
/***********************************************************************************/
/***********************************************************************************/
/** @brief Implementation of the Shape Context descriptor and matching algorithm

proposed by Belongie et al. in "Shape Matching and Object Recognition Using Shape Contexts" (PAMI
2002). This implementation is packaged in a generic scheme, in order to allow you the
implementation of the common variations of the original pipeline.
*/
class CV_EXPORTS_W ShapeContextDistanceExtractor : public ShapeDistanceExtractor
{
public:
    /** @brief Establish the number of angular bins for the Shape Context Descriptor used in the shape matching
    pipeline.

    @param nAngularBins The number of angular bins in the shape context descriptor.
     */
    CV_WRAP virtual void setAngularBins(int nAngularBins) = 0;
    CV_WRAP virtual int getAngularBins() const = 0;

    /** @brief Establish the number of radial bins for the Shape Context Descriptor used in the shape matching
    pipeline.

    @param nRadialBins The number of radial bins in the shape context descriptor.
     */
    CV_WRAP virtual void setRadialBins(int nRadialBins) = 0;
    CV_WRAP virtual int getRadialBins() const = 0;

    /** @brief Set the inner radius of the shape context descriptor.

    @param innerRadius The value of the inner radius.
     */
    CV_WRAP virtual void setInnerRadius(float innerRadius) = 0;
    CV_WRAP virtual float getInnerRadius() const = 0;

    /** @brief Set the outer radius of the shape context descriptor.

    @param outerRadius The value of the outer radius.
     */
    CV_WRAP virtual void setOuterRadius(float outerRadius) = 0;
    CV_WRAP virtual float getOuterRadius() const = 0;

    CV_WRAP virtual void setRotationInvariant(bool rotationInvariant) = 0;
    CV_WRAP virtual bool getRotationInvariant() const = 0;

    /** @brief Set the weight of the shape context distance in the final value of the shape distance. The shape
    context distance between two shapes is defined as the symmetric sum of shape context matching costs
    over best matching points. The final value of the shape distance is a user-defined linear
    combination of the shape context distance, an image appearance distance, and a bending energy.

    @param shapeContextWeight The weight of the shape context distance in the final distance value.
     */
    CV_WRAP virtual void setShapeContextWeight(float shapeContextWeight) = 0;
    CV_WRAP virtual float getShapeContextWeight() const = 0;

    /** @brief Set the weight of the Image Appearance cost in the final value of the shape distance. The image
    appearance cost is defined as the sum of squared brightness differences in Gaussian windows around
    corresponding image points. The final value of the shape distance is a user-defined linear
    combination of the shape context distance, an image appearance distance, and a bending energy. If
    this value is set to a number different from 0, is mandatory to set the images that correspond to
    each shape.

    @param imageAppearanceWeight The weight of the appearance cost in the final distance value.
     */
    CV_WRAP virtual void setImageAppearanceWeight(float imageAppearanceWeight) = 0;
    CV_WRAP virtual float getImageAppearanceWeight() const = 0;

    /** @brief Set the weight of the Bending Energy in the final value of the shape distance. The bending energy
    definition depends on what transformation is being used to align the shapes. The final value of the
    shape distance is a user-defined linear combination of the shape context distance, an image
    appearance distance, and a bending energy.

    @param bendingEnergyWeight The weight of the Bending Energy in the final distance value.
     */
    CV_WRAP virtual void setBendingEnergyWeight(float bendingEnergyWeight) = 0;
    CV_WRAP virtual float getBendingEnergyWeight() const = 0;

    /** @brief Set the images that correspond to each shape. This images are used in the calculation of the Image
    Appearance cost.

    @param image1 Image corresponding to the shape defined by contours1.
    @param image2 Image corresponding to the shape defined by contours2.
     */
    CV_WRAP virtual void setImages(InputArray image1, InputArray image2) = 0;
    CV_WRAP virtual void getImages(OutputArray image1, OutputArray image2) const = 0;

    CV_WRAP virtual void setIterations(int iterations) = 0;
    CV_WRAP virtual int getIterations() const = 0;

    /** @brief Set the algorithm used for building the shape context descriptor cost matrix.

    @param comparer Smart pointer to a HistogramCostExtractor, an algorithm that defines the cost
    matrix between descriptors.
     */
    CV_WRAP virtual void setCostExtractor(Ptr<HistogramCostExtractor> comparer) = 0;
    CV_WRAP virtual Ptr<HistogramCostExtractor> getCostExtractor() const = 0;

    /** @brief Set the value of the standard deviation for the Gaussian window for the image appearance cost.

    @param sigma Standard Deviation.
     */
    CV_WRAP virtual void setStdDev(float sigma) = 0;
    CV_WRAP virtual float getStdDev() const = 0;

    /** @brief Set the algorithm used for aligning the shapes.

    @param transformer Smart pointer to a ShapeTransformer, an algorithm that defines the aligning
    transformation.
     */
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
/** @brief A simple Hausdorff distance measure between shapes defined by contours

according to the paper "Comparing Images using the Hausdorff distance." by D.P. Huttenlocher, G.A.
Klanderman, and W.J. Rucklidge. (PAMI 1993). :
 */
class CV_EXPORTS_W HausdorffDistanceExtractor : public ShapeDistanceExtractor
{
public:
    /** @brief Set the norm used to compute the Hausdorff value between two shapes. It can be L1 or L2 norm.

    @param distanceFlag Flag indicating which norm is used to compute the Hausdorff distance
    (NORM_L1, NORM_L2).
     */
    CV_WRAP virtual void setDistanceFlag(int distanceFlag) = 0;
    CV_WRAP virtual int getDistanceFlag() const = 0;

    /** @brief This method sets the rank proportion (or fractional value) that establish the Kth ranked value of
    the partial Hausdorff distance. Experimentally had been shown that 0.6 is a good value to compare
    shapes.

    @param rankProportion fractional value (between 0 and 1).
     */
    CV_WRAP virtual void setRankProportion(float rankProportion) = 0;
    CV_WRAP virtual float getRankProportion() const = 0;
};

/* Constructor */
CV_EXPORTS_W Ptr<HausdorffDistanceExtractor> createHausdorffDistanceExtractor(int distanceFlag=cv::NORM_L2, float rankProp=0.6f);

//! @}

} // cv
#endif
