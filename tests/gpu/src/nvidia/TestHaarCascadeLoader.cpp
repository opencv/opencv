/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual 
 * property and proprietary rights in and to this software and 
 * related documentation and any modifications thereto.  
 * Any use, reproduction, disclosure, or distribution of this 
 * software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "TestHaarCascadeLoader.h"
#include "NCVHaarObjectDetection.hpp"


TestHaarCascadeLoader::TestHaarCascadeLoader(std::string testName, std::string cascadeName)
    :
    NCVTestProvider(testName),
    cascadeName(cascadeName)
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
