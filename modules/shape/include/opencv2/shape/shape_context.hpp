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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_SHAPE_SHAPE_CONTEXT_HPP__
#define __OPENCV_SHAPE_SHAPE_CONTEXT_HPP__

#include <vector>

namespace cv
{

/*!
 * SCD implementation.
 * The SCD class implements SCD algorithm by Belongie et al.
 * The SCMatcher implements the matching pipeline presented in the same 
 * paper.
 */

/****************************************************************************************\
*                                   Descriptor Class                                    *
\****************************************************************************************/

class CV_EXPORTS_W SCD 
{
public:
    //! the default constructor
    CV_WRAP SCD();
    //! the full constructor taking all the necessary parameters
    explicit CV_WRAP SCD(int nAngularBins=12, int nRadialBins = 5, 
                           double innerRadius=0.1, double outerRadius=1, bool rotationInvariant=false);

    //! returns the descriptor size in float's 
    CV_WRAP int descriptorSize() const;

    //! Compute keypoints descriptors. 
    CV_WRAP void extractSCD(InputArray contour, Mat& descriptors);

    //! Setters
    void setAngularBins(int);
    void setRadialBins(int);
    void setInnerRadius(double);
    void setOuterRadius(double);
    void setRotationInvariant(bool);

    //! Getters
    int getAngularBins(void);
    int getRadialBins(void);
    double getInnerRadius(void);
    double getOuterRadius(void);
    bool getRotationInvariant(void);
    float getMeanDistance(void);
    
private:
    CV_PROP_RW int nAngularBins;
    CV_PROP_RW int nRadialBins;
    CV_PROP_RW double innerRadius;
    CV_PROP_RW double outerRadius;
    CV_PROP_RW bool rotationInvariant;
    CV_PROP_RW float meanDistance;

protected:
    CV_WRAP void logarithmicSpaces(std::vector<double>& vecSpaces) const;
    CV_WRAP void angularSpaces(std::vector<double>& vecSpaces) const;                              
    CV_WRAP void buildNormalizedDistanceMatrix(InputArray contour, 
                              Mat& disMatrix);
    CV_WRAP void buildAngleMatrix(InputArray contour, 
                              Mat& angleMatrix) const;
    CV_WRAP double distance(Point2f p, Point2f q) const;
};

typedef SCD ShapeContextDescriptorExtractor;

/****************************************************************************************\
*                                     Matching Class                                     *
\****************************************************************************************/
struct CV_EXPORTS DistanceSCDFlags
{
    enum
    { 
        DEFAULT = 0, // CHI Squared Distance
        DIST_CHI = 0,
        DIST_EMD = 1, // Earth Mover's Distance (Not defined yet)
        DIST_L2 = 2 // L2 Distance (Not defined yet)
    };
};

class CV_EXPORTS_W SCDMatcher 
{
public:
    //! the full constructor
    CV_WRAP SCDMatcher(float outlierWeight=0.1, int flags=DistanceSCDFlags::DEFAULT);
    //! the matcher function using Hungarian method
    CV_WRAP void matchDescriptors(Mat& descriptors1,  Mat& descriptors2, std::vector<DMatch>& matches);
    //! *etters
    CV_WRAP float getMatchingCost(void);
private:
    CV_PROP_RW float outlierWeight;
    CV_PROP_RW int configFlags;
    CV_PROP_RW float minMatchCost;
protected:
    CV_WRAP void buildCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix, int flags) const;
    CV_WRAP void buildChiCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const;
    CV_WRAP void buildEMDCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const;
    CV_WRAP void buildL2CostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const;
    CV_WRAP void hungarian(Mat&, std::vector<DMatch>&);
};

typedef SCDMatcher ShapeContextDescriptorMatcher;

/****************************************************************************************\
*                              Transform Base Class                                     *
\****************************************************************************************/
class CV_EXPORTS_W Transform
{
public:
    //! destructors
    CV_WRAP virtual ~Transform(){}
    //! methods
    CV_WRAP virtual void applyTransformation(InputArray pts1, InputArray pts2,
                                     std::vector<DMatch>&, std::vector<Point2f> &outPts)=0;
    //! getters
    CV_WRAP virtual float getTranformCost(void) const=0;
protected:
    CV_PROP_RW float transformCost;
};

/****************************************************************************************\
*                                       TPS  Class                                      *
\****************************************************************************************/
class CV_EXPORTS_W ThinPlateSplineTransform : public Transform
{
public:
    //! *tructors
    CV_WRAP ThinPlateSplineTransform();
    CV_WRAP ThinPlateSplineTransform(double beta);

    //! getters-setters
    void setRegularizationParam(double beta);
    double getRegularizationParam(void);
    //! methods
    CV_WRAP void applyTransformation(InputArray pts1, InputArray pts2,
                             std::vector<DMatch>&, std::vector<Point2f>& outPts);
    //! getters
    CV_WRAP float getTranformCost(void) const;
private:
    double beta;
    double distance(Point2f, Point2f) const;
};

/****************************************************************************************\
*                                  Affine Class                                          *
\****************************************************************************************/
class CV_EXPORTS_W AffineTransform : public Transform
{
public:
    //! *tructors
    CV_WRAP AffineTransform();
    CV_WRAP AffineTransform(bool fullAffine);

    //! getters-setters
    CV_WRAP float getTranformCost(void) const;
    CV_WRAP void setFullAffine(bool fullAffine);
    CV_WRAP bool getFullAffine(void);
    //! methods
    CV_WRAP void applyTransformation(InputArray pts1, InputArray pts2,
                             std::vector<DMatch>&, std::vector<Point2f>& outPts);
private:
    bool fullAffine;
};

/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/
struct CV_EXPORTS DrawSCDFlags
{
    enum
    { 
        DEFAULT = 0, // Descriptor image is not normalized
        DRAW_NORM = 1, // Dscriptor is normalized
        DRAW_NORM_NEG = 2 //Descriptor normalized and color negative
    };
};

//! Draw a single point descriptor.
CV_EXPORTS_W void drawSCD(const Mat& descriptor, int angularBins, int radialBins, CV_OUT Mat& outImage, 
                           int index=0, float pixelsPerBin=5, int flags=DrawSCDFlags::DEFAULT);
    
} /* namespace cv */

#endif
