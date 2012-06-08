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

using namespace cv;

/////////////////////// AlgorithmInfo for various detector & descriptors ////////////////////////////

/* NOTE!!!
   All the AlgorithmInfo-related stuff should be in the same file as initModule_features2d().
   Otherwise, linker may throw away some seemingly unused stuff.
*/

CV_INIT_ALGORITHM(BriefDescriptorExtractor, "Feature2D.BRIEF",
                  obj.info()->addParam(obj, "bytes", obj.bytes_));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(FastFeatureDetector, "Feature2D.FAST",
                  obj.info()->addParam(obj, "threshold", obj.threshold);
                  obj.info()->addParam(obj, "nonmaxSuppression", obj.nonmaxSuppression));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(StarDetector, "Feature2D.STAR",
                  obj.info()->addParam(obj, "maxSize", obj.maxSize);
                  obj.info()->addParam(obj, "responseThreshold", obj.responseThreshold);
                  obj.info()->addParam(obj, "lineThresholdProjected", obj.lineThresholdProjected);
                  obj.info()->addParam(obj, "lineThresholdBinarized", obj.lineThresholdBinarized);
                  obj.info()->addParam(obj, "suppressNonmaxSize", obj.suppressNonmaxSize));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(MSER, "Feature2D.MSER",
                  obj.info()->addParam(obj, "delta", obj.delta);
                  obj.info()->addParam(obj, "minArea", obj.minArea);
                  obj.info()->addParam(obj, "maxArea", obj.maxArea);
                  obj.info()->addParam(obj, "maxVariation", obj.maxVariation);
                  obj.info()->addParam(obj, "minDiversity", obj.minDiversity);
                  obj.info()->addParam(obj, "maxEvolution", obj.maxEvolution);
                  obj.info()->addParam(obj, "areaThreshold", obj.areaThreshold);
                  obj.info()->addParam(obj, "minMargin", obj.minMargin);
                  obj.info()->addParam(obj, "edgeBlurSize", obj.edgeBlurSize));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(ORB, "Feature2D.ORB",
                  obj.info()->addParam(obj, "nFeatures", obj.nfeatures);
                  obj.info()->addParam(obj, "scaleFactor", obj.scaleFactor);
                  obj.info()->addParam(obj, "nLevels", obj.nlevels);
                  obj.info()->addParam(obj, "firstLevel", obj.firstLevel);
                  obj.info()->addParam(obj, "edgeThreshold", obj.edgeThreshold);
                  obj.info()->addParam(obj, "patchSize", obj.patchSize);
                  obj.info()->addParam(obj, "WTA_K", obj.WTA_K);
                  obj.info()->addParam(obj, "scoreType", obj.scoreType));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(GFTTDetector, "Feature2D.GFTT",
                  obj.info()->addParam(obj, "nfeatures", obj.nfeatures);
                  obj.info()->addParam(obj, "qualityLevel", obj.qualityLevel);
                  obj.info()->addParam(obj, "minDistance", obj.minDistance);
                  obj.info()->addParam(obj, "useHarrisDetector", obj.useHarrisDetector);
                  obj.info()->addParam(obj, "k", obj.k));

///////////////////////////////////////////////////////////////////////////////////////////////////////////

class CV_EXPORTS HarrisDetector : public GFTTDetector
{
public:
    HarrisDetector( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
                    int blockSize=3, bool useHarrisDetector=true, double k=0.04 )
    : GFTTDetector( maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k ) {}
    AlgorithmInfo* info() const;
};

CV_INIT_ALGORITHM(HarrisDetector, "Feature2D.HARRIS",
                  obj.info()->addParam(obj, "nfeatures", obj.nfeatures);
                  obj.info()->addParam(obj, "qualityLevel", obj.qualityLevel);
                  obj.info()->addParam(obj, "minDistance", obj.minDistance);
                  obj.info()->addParam(obj, "useHarrisDetector", obj.useHarrisDetector);
                  obj.info()->addParam(obj, "k", obj.k));

////////////////////////////////////////////////////////////////////////////////////////////////////////////

CV_INIT_ALGORITHM(DenseFeatureDetector, "Feature2D.Dense",
                  obj.info()->addParam(obj, "initFeatureScale", obj.initFeatureScale);
                  obj.info()->addParam(obj, "featureScaleLevels", obj.featureScaleLevels);
                  obj.info()->addParam(obj, "featureScaleMul", obj.featureScaleMul);
                  obj.info()->addParam(obj, "initXyStep", obj.initXyStep);
                  obj.info()->addParam(obj, "initImgBound", obj.initImgBound);
                  obj.info()->addParam(obj, "varyXyStepWithScale", obj.varyXyStepWithScale);
                  obj.info()->addParam(obj, "varyImgBoundWithScale", obj.varyImgBoundWithScale));

CV_INIT_ALGORITHM(GridAdaptedFeatureDetector, "Feature2D.Grid",
                  obj.info()->addParam(obj, "detector", (Ptr<Algorithm>&)obj.detector);
                  obj.info()->addParam(obj, "maxTotalKeypoints", obj.maxTotalKeypoints);
                  obj.info()->addParam(obj, "gridRows", obj.gridRows);
                  obj.info()->addParam(obj, "gridCols", obj.gridCols));

bool cv::initModule_features2d(void)
{
    bool all = true;
    all &= !BriefDescriptorExtractor_info_auto.name().empty();
    all &= !FastFeatureDetector_info_auto.name().empty();
    all &= !StarDetector_info_auto.name().empty();
    all &= !MSER_info_auto.name().empty();
    all &= !ORB_info_auto.name().empty();
    all &= !GFTTDetector_info_auto.name().empty();
    all &= !HarrisDetector_info_auto.name().empty();
    all &= !DenseFeatureDetector_info_auto.name().empty();
    all &= !GridAdaptedFeatureDetector_info_auto.name().empty();

    return all;
}
