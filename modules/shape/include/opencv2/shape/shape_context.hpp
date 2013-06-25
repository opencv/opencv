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
                           double innerRadius=0.1, double outerRadius=1);

    //! returns the descriptor size in float's 
    CV_WRAP int descriptorSize() const;

    //! Compute keypoints descriptors. 
    CV_WRAP void extractSCD(InputArray contour, Mat& descriptors) const;

    //! Setters
    void setAngularBins(int);
    void setRadialBins(int);
    void setInnerRadius(double);
    void setOuterRadius(double);

    //! Getters
    int getAngularBins(void);
    int getRadialBins(void);
    double getInnerRadius(void);
    double getOuterRadius(void);
    
private:
    CV_PROP_RW int nAngularBins;
    CV_PROP_RW int nRadialBins;
    CV_PROP_RW double innerRadius;
    CV_PROP_RW double outerRadius;
protected:
    CV_WRAP void logarithmicSpaces(std::vector<double>& vecSpaces) const;
    CV_WRAP void angularSpaces(std::vector<double>& vecSpaces) const;                              
    CV_WRAP void buildNormalizedDistanceMatrix(InputArray contour, 
                              Mat& disMatrix) const;
    CV_WRAP void buildAngleMatrix(InputArray contour, 
                              Mat& angleMatrix) const;
    CV_WRAP double distance(Point p, Point q) const;
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
        DIST_EMD = 1, // Earth Mover's Distance
        DIST_EUCLIDEAN = 2 // Euclidean Distance
    };
};

class CV_EXPORTS_W SCDMatcher 
{
public:
    //! the default constructor
    CV_WRAP SCDMatcher();
    //! the full constructor taking all the necessary parameters
    // Define it here
private:
protected:
    CV_WRAP void buildCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix,
                                 int flags=DistanceSCDFlags::DEFAULT) const;
    CV_WRAP void buildChiCostMatrix(Mat& descriptors1,  Mat& descriptors2, Mat& costMatrix) const;
};

typedef SCDMatcher ShapeContextDescriptorMatcher;

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
