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

namespace cv
{

/////////////////////// AlgorithmInfo for various detector & descriptors ////////////////////////////

/* NOTE!!!
   All the AlgorithmInfo-related stuff should be in the same file as initModule_features2d().
   Otherwise, linker may throw away some seemingly unused stuff.
*/
    
static Algorithm* createBRIEF() { return new BriefDescriptorExtractor; }
static AlgorithmInfo& brief_info()
{
    static AlgorithmInfo brief_info_var("Feature2D.BRIEF", createBRIEF);
    return brief_info_var;
}

static AlgorithmInfo& brief_info_auto = brief_info();

AlgorithmInfo* BriefDescriptorExtractor::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        BriefDescriptorExtractor brief;
        brief_info().addParam(brief, "bytes", brief.bytes_);
        
        initialized = true;
    }
    return &brief_info();
}
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
static Algorithm* createFAST() { return new FastFeatureDetector; }
static AlgorithmInfo& fast_info()
{
    static AlgorithmInfo fast_info_var("Feature2D.FAST", createFAST);
    return fast_info_var;
}

static AlgorithmInfo& fast_info_auto = fast_info();

AlgorithmInfo* FastFeatureDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        FastFeatureDetector obj;
        fast_info().addParam(obj, "threshold", obj.threshold);
        fast_info().addParam(obj, "nonmaxSuppression", obj.nonmaxSuppression);
        
        initialized = true;
    }
    return &fast_info();
}
    

///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
static Algorithm* createStarDetector() { return new StarDetector; }
static AlgorithmInfo& star_info()
{
    static AlgorithmInfo star_info_var("Feature2D.STAR", createStarDetector);
    return star_info_var;
}

static AlgorithmInfo& star_info_auto = star_info();

AlgorithmInfo* StarDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        StarDetector obj;
        star_info().addParam(obj, "maxSize", obj.maxSize);
        star_info().addParam(obj, "responseThreshold", obj.responseThreshold);
        star_info().addParam(obj, "lineThresholdProjected", obj.lineThresholdProjected);
        star_info().addParam(obj, "lineThresholdBinarized", obj.lineThresholdBinarized);
        star_info().addParam(obj, "suppressNonmaxSize", obj.suppressNonmaxSize);
        
        initialized = true;
    }
    return &star_info();
}    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
static Algorithm* createMSER() { return new MSER; }
static AlgorithmInfo& mser_info()
{
    static AlgorithmInfo mser_info_var("Feature2D.MSER", createMSER);
    return mser_info_var;
}

static AlgorithmInfo& mser_info_auto = mser_info();

AlgorithmInfo* MSER::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        MSER obj;
        mser_info().addParam(obj, "delta", obj.delta);
        mser_info().addParam(obj, "minArea", obj.minArea);
        mser_info().addParam(obj, "maxArea", obj.maxArea);
        mser_info().addParam(obj, "maxVariation", obj.maxVariation);
        mser_info().addParam(obj, "minDiversity", obj.minDiversity);
        mser_info().addParam(obj, "maxEvolution", obj.maxEvolution);
        mser_info().addParam(obj, "areaThreshold", obj.areaThreshold);
        mser_info().addParam(obj, "minMargin", obj.minMargin);
        mser_info().addParam(obj, "edgeBlurSize", obj.edgeBlurSize);
        
        initialized = true;
    }
    return &mser_info();
}
    
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////

static Algorithm* createORB() { return new ORB; }
static AlgorithmInfo& orb_info()
{
    static AlgorithmInfo orb_info_var("Feature2D.ORB", createORB);
    return orb_info_var;
}

static AlgorithmInfo& orb_info_auto = orb_info();

AlgorithmInfo* ORB::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        ORB obj;
        orb_info().addParam(obj, "nFeatures", obj.nfeatures);
        orb_info().addParam(obj, "scaleFactor", obj.scaleFactor);
        orb_info().addParam(obj, "nLevels", obj.nlevels);
        orb_info().addParam(obj, "firstLevel", obj.firstLevel);
        orb_info().addParam(obj, "edgeThreshold", obj.edgeThreshold);
        orb_info().addParam(obj, "patchSize", obj.patchSize);
        orb_info().addParam(obj, "WTA_K", obj.WTA_K);
        orb_info().addParam(obj, "scoreType", obj.scoreType);
        
        initialized = true;
    }
    return &orb_info();
}

static Algorithm* createGFTT() { return new GFTTDetector; }
static Algorithm* createHarris()
{
    GFTTDetector* d = new GFTTDetector;
    d->set("useHarris", true);
    return d;
}

static AlgorithmInfo gftt_info("Feature2D.GFTT", createGFTT);
static AlgorithmInfo harris_info("Feature2D.HARRIS", createHarris);
    
AlgorithmInfo* GFTTDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        GFTTDetector obj;
        gftt_info.addParam(obj, "nfeatures", obj.nfeatures);
        gftt_info.addParam(obj, "qualityLevel", obj.qualityLevel);
        gftt_info.addParam(obj, "minDistance", obj.minDistance);
        gftt_info.addParam(obj, "useHarrisDetector", obj.useHarrisDetector);
        gftt_info.addParam(obj, "k", obj.k);
        
        harris_info.addParam(obj, "nfeatures", obj.nfeatures);
        harris_info.addParam(obj, "qualityLevel", obj.qualityLevel);
        harris_info.addParam(obj, "minDistance", obj.minDistance);
        harris_info.addParam(obj, "useHarrisDetector", obj.useHarrisDetector);
        harris_info.addParam(obj, "k", obj.k);
        
        initialized = true;
    }
    return &gftt_info;
}

static Algorithm* createDense() { return new DenseFeatureDetector; }
static AlgorithmInfo dense_info("Feature2D.Dense", createDense);
    
AlgorithmInfo* DenseFeatureDetector::info() const
{
    static volatile bool initialized = false;
    if( !initialized )
    {
        DenseFeatureDetector obj;
        dense_info.addParam(obj, "initFeatureScale", obj.initFeatureScale);
        dense_info.addParam(obj, "featureScaleLevels", obj.featureScaleLevels);
        dense_info.addParam(obj, "featureScaleMul", obj.featureScaleMul);
        dense_info.addParam(obj, "initXyStep", obj.initXyStep);
        dense_info.addParam(obj, "initImgBound", obj.initImgBound);
        dense_info.addParam(obj, "varyXyStepWithScale", obj.varyXyStepWithScale);
        dense_info.addParam(obj, "varyImgBoundWithScale", obj.varyImgBoundWithScale);
        
        initialized = true;
    }
    return &dense_info;
}    

bool initModule_features2d(void)
{
    Ptr<Algorithm> brief = createBRIEF(), orb = createORB(),
        star = createStarDetector(), fastd = createFAST(), mser = createMSER(),
        dense = createDense(), gftt = createGFTT(), harris = createHarris();
        
    return brief->info() != 0 && orb->info() != 0 && star->info() != 0 &&
        fastd->info() != 0 && mser->info() != 0 && dense->info() != 0 &&
        gftt->info() != 0 && harris->info() != 0;
}

}

