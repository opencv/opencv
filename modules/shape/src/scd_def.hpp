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

#include <stdlib.h>
#include <math.h>
#include <vector>

namespace cv
{
/*
 * ShapeContextDescriptor class
 */
class SCD
{
public:
    //! the full constructor taking all the necessary parameters
    explicit SCD(int _nAngularBins=12, int _nRadialBins=5,
                 double _innerRadius=0.1, double _outerRadius=1, bool _rotationInvariant=false)
    {
        setAngularBins(_nAngularBins);
        setRadialBins(_nRadialBins);
        setInnerRadius(_innerRadius);
        setOuterRadius(_outerRadius);
        setRotationInvariant(_rotationInvariant);
    }

    void extractSCD(cv::Mat& contour, cv::Mat& descriptors,
                    const std::vector<int>& queryInliers=std::vector<int>(),
                    const float _meanDistance=-1);

    int descriptorSize() {return nAngularBins*nRadialBins;}
    void setAngularBins(int angularBins) { nAngularBins=angularBins; }
    void setRadialBins(int radialBins) { nRadialBins=radialBins; }
    void setInnerRadius(double _innerRadius) { innerRadius=_innerRadius; }
    void setOuterRadius(double _outerRadius) { outerRadius=_outerRadius; }
    void setRotationInvariant(bool _rotationInvariant) { rotationInvariant=_rotationInvariant; }
    int getAngularBins() const { return nAngularBins; }
    int getRadialBins() const { return nRadialBins; }
    double getInnerRadius() const { return innerRadius; }
    double getOuterRadius() const { return outerRadius; }
    bool getRotationInvariant() const { return rotationInvariant; }
    float getMeanDistance() const { return meanDistance; }

private:
    int nAngularBins;
    int nRadialBins;
    double innerRadius;
    double outerRadius;
    bool rotationInvariant;
    float meanDistance;

protected:
    void logarithmicSpaces(std::vector<double>& vecSpaces) const;
    void angularSpaces(std::vector<double>& vecSpaces) const;

    void buildNormalizedDistanceMatrix(cv::Mat& contour,
                          cv::Mat& disMatrix, const std::vector<int> &queryInliers,
                          const float _meanDistance=-1);

    void buildAngleMatrix(cv::Mat& contour,
                              cv::Mat& angleMatrix) const;
};

/*
 * Matcher
 */
class SCDMatcher
{
public:
    // the full constructor
    SCDMatcher()
    {
    }

    // the matcher function using Hungarian method
    void matchDescriptors(cv::Mat& descriptors1,  cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, cv::Ptr<cv::HistogramCostExtractor>& comparer,
                                      std::vector<int>& inliers1, std::vector<int> &inliers2);

    // matching cost
    float getMatchingCost() const {return minMatchCost;}

private:
    float minMatchCost;
protected:
    void buildCostMatrix(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                                     cv::Mat& costMatrix, cv::Ptr<cv::HistogramCostExtractor>& comparer) const;
    void hungarian(cv::Mat& costMatrix, std::vector<cv::DMatch>& outMatches, std::vector<int> &inliers1,
                   std::vector<int> &inliers2, int sizeScd1=0, int sizeScd2=0);

};

}
