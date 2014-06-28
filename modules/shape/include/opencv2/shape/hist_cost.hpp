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

#ifndef __OPENCV_HIST_COST_HPP__
#define __OPENCV_HIST_COST_HPP__

#include "opencv2/imgproc.hpp"

namespace cv
{

/*!
 * The base class for HistogramCostExtractor.
 */
class CV_EXPORTS_W HistogramCostExtractor : public Algorithm
{
public:
    CV_WRAP virtual void buildCostMatrix(InputArray descriptors1, InputArray descriptors2, OutputArray costMatrix) = 0;

    CV_WRAP virtual void setNDummies(int nDummies) = 0;
    CV_WRAP virtual int getNDummies() const = 0;

    CV_WRAP virtual void setDefaultCost(float defaultCost) = 0;
    CV_WRAP virtual float getDefaultCost() const = 0;
};

/*!  */
class CV_EXPORTS_W NormHistogramCostExtractor : public HistogramCostExtractor
{
public:
    CV_WRAP virtual void setNormFlag(int flag) = 0;
    CV_WRAP virtual int getNormFlag() const = 0;
};

CV_EXPORTS_W Ptr<HistogramCostExtractor>
    createNormHistogramCostExtractor(int flag=DIST_L2, int nDummies=25, float defaultCost=0.2f);

/*!  */
class CV_EXPORTS_W EMDHistogramCostExtractor : public HistogramCostExtractor
{
public:
    CV_WRAP virtual void setNormFlag(int flag) = 0;
    CV_WRAP virtual int getNormFlag() const = 0;
};

CV_EXPORTS_W Ptr<HistogramCostExtractor>
    createEMDHistogramCostExtractor(int flag=DIST_L2, int nDummies=25, float defaultCost=0.2f);

/*!  */
class CV_EXPORTS_W ChiHistogramCostExtractor : public HistogramCostExtractor
{};

CV_EXPORTS_W Ptr<HistogramCostExtractor> createChiHistogramCostExtractor(int nDummies=25, float defaultCost=0.2f);

/*!  */
class CV_EXPORTS_W EMDL1HistogramCostExtractor : public HistogramCostExtractor
{};

CV_EXPORTS_W Ptr<HistogramCostExtractor>
    createEMDL1HistogramCostExtractor(int nDummies=25, float defaultCost=0.2f);

} // cv
#endif
