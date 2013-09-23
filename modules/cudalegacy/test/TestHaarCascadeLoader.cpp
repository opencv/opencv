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

#include "test_precomp.hpp"


TestHaarCascadeLoader::TestHaarCascadeLoader(std::string testName_, std::string cascadeName_)
    :
    NCVTestProvider(testName_),
    cascadeName(cascadeName_)
{
}


bool TestHaarCascadeLoader::toString(std::ofstream &strOut)
{
    strOut << "cascadeName=" << cascadeName << std::endl;
    return true;
}


bool TestHaarCascadeLoader::init()
{
    return true;
}


bool TestHaarCascadeLoader::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    Ncv32u numStages, numNodes, numFeatures;
    Ncv32u numStages_2 = 0, numNodes_2 = 0, numFeatures_2 = 0;

    ncvStat = ncvHaarGetClassifierSize(this->cascadeName, numStages, numNodes, numFeatures);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);

    NCVVectorAlloc<HaarStage64> h_HaarStages(*this->allocatorCPU.get(), numStages);
    ncvAssertReturn(h_HaarStages.isMemAllocated(), false);
    NCVVectorAlloc<HaarClassifierNode128> h_HaarNodes(*this->allocatorCPU.get(), numNodes);
    ncvAssertReturn(h_HaarNodes.isMemAllocated(), false);
    NCVVectorAlloc<HaarFeature64> h_HaarFeatures(*this->allocatorCPU.get(), numFeatures);
    ncvAssertReturn(h_HaarFeatures.isMemAllocated(), false);

    NCVVectorAlloc<HaarStage64> h_HaarStages_2(*this->allocatorCPU.get(), numStages);
    ncvAssertReturn(h_HaarStages_2.isMemAllocated(), false);
    NCVVectorAlloc<HaarClassifierNode128> h_HaarNodes_2(*this->allocatorCPU.get(), numNodes);
    ncvAssertReturn(h_HaarNodes_2.isMemAllocated(), false);
    NCVVectorAlloc<HaarFeature64> h_HaarFeatures_2(*this->allocatorCPU.get(), numFeatures);
    ncvAssertReturn(h_HaarFeatures_2.isMemAllocated(), false);

    HaarClassifierCascadeDescriptor haar;
    HaarClassifierCascadeDescriptor haar_2;

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN

    const std::string testNvbinName = "test.nvbin";
    ncvStat = ncvHaarLoadFromFile_host(this->cascadeName, haar, h_HaarStages, h_HaarNodes, h_HaarFeatures);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);

    ncvStat = ncvHaarStoreNVBIN_host(testNvbinName, haar, h_HaarStages, h_HaarNodes, h_HaarFeatures);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);

    ncvStat = ncvHaarGetClassifierSize(testNvbinName, numStages_2, numNodes_2, numFeatures_2);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);

    ncvStat = ncvHaarLoadFromFile_host(testNvbinName, haar_2, h_HaarStages_2, h_HaarNodes_2, h_HaarFeatures_2);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);

    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN

    if (
    numStages_2 != numStages                                       ||
    numNodes_2 != numNodes                                         ||
    numFeatures_2 != numFeatures                                   ||
    haar.NumStages               != haar_2.NumStages               ||
    haar.NumClassifierRootNodes  != haar_2.NumClassifierRootNodes  ||
    haar.NumClassifierTotalNodes != haar_2.NumClassifierTotalNodes ||
    haar.NumFeatures             != haar_2.NumFeatures             ||
    haar.ClassifierSize.width    != haar_2.ClassifierSize.width    ||
    haar.ClassifierSize.height   != haar_2.ClassifierSize.height   ||
    haar.bNeedsTiltedII          != haar_2.bNeedsTiltedII          ||
    haar.bHasStumpsOnly          != haar_2.bHasStumpsOnly          )
    {
        bLoopVirgin = false;
    }
    if (memcmp(h_HaarStages.ptr(), h_HaarStages_2.ptr(), haar.NumStages * sizeof(HaarStage64)) ||
        memcmp(h_HaarNodes.ptr(), h_HaarNodes_2.ptr(), haar.NumClassifierTotalNodes * sizeof(HaarClassifierNode128)) ||
        memcmp(h_HaarFeatures.ptr(), h_HaarFeatures_2.ptr(), haar.NumFeatures * sizeof(HaarFeature64)) )
    {
        bLoopVirgin = false;
    }
    NCV_SKIP_COND_END

    if (bLoopVirgin)
    {
        rcode = true;
    }

    return rcode;
}


bool TestHaarCascadeLoader::deinit()
{
    return true;
}
