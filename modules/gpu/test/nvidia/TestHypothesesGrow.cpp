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

#include "TestHypothesesGrow.h"
#include "NCVHaarObjectDetection.hpp"


TestHypothesesGrow::TestHypothesesGrow(std::string testName, NCVTestSourceProvider<Ncv32u> &src,
                                       Ncv32u rectWidth, Ncv32u rectHeight, Ncv32f rectScale, 
                                       Ncv32u maxLenSrc, Ncv32u lenSrc, Ncv32u maxLenDst, Ncv32u lenDst)
    :
    NCVTestProvider(testName),
    src(src),
    rectWidth(rectWidth),
    rectHeight(rectHeight),
    rectScale(rectScale),
    maxLenSrc(maxLenSrc),
    lenSrc(lenSrc),
    maxLenDst(maxLenDst),
    lenDst(lenDst)
{
}


bool TestHypothesesGrow::toString(std::ofstream &strOut)
{
    strOut << "rectWidth=" << rectWidth << std::endl;
    strOut << "rectHeight=" << rectHeight << std::endl;
    strOut << "rectScale=" << rectScale << std::endl;
    strOut << "maxLenSrc=" << maxLenSrc << std::endl;
    strOut << "lenSrc=" << lenSrc << std::endl;
    strOut << "maxLenDst=" << maxLenDst << std::endl;
    strOut << "lenDst=" << lenDst << std::endl;
    return true;
}


bool TestHypothesesGrow::init()
{
    return true;
}


bool TestHypothesesGrow::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    NCVVectorAlloc<Ncv32u> h_vecSrc(*this->allocatorCPU.get(), this->maxLenSrc);
    ncvAssertReturn(h_vecSrc.isMemAllocated(), false);
    NCVVectorAlloc<Ncv32u> d_vecSrc(*this->allocatorGPU.get(), this->maxLenSrc);
    ncvAssertReturn(d_vecSrc.isMemAllocated(), false);

    NCVVectorAlloc<NcvRect32u> h_vecDst(*this->allocatorCPU.get(), this->maxLenDst);
    ncvAssertReturn(h_vecDst.isMemAllocated(), false);
    NCVVectorAlloc<NcvRect32u> d_vecDst(*this->allocatorGPU.get(), this->maxLenDst);
    ncvAssertReturn(d_vecDst.isMemAllocated(), false);
    NCVVectorAlloc<NcvRect32u> h_vecDst_d(*this->allocatorCPU.get(), this->maxLenDst);
    ncvAssertReturn(h_vecDst_d.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());

    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_vecSrc), false);
    memset(h_vecDst.ptr(), 0, h_vecDst.length() * sizeof(NcvRect32u));
    NCVVectorReuse<Ncv32u> h_vecDst_as32u(h_vecDst.getSegment(), lenDst * sizeof(NcvRect32u) / sizeof(Ncv32u));
    ncvAssertReturn(h_vecDst_as32u.isMemReused(), false);
    ncvAssertReturn(this->src.fill(h_vecDst_as32u), false);
    memcpy(h_vecDst_d.ptr(), h_vecDst.ptr(), h_vecDst.length() * sizeof(NcvRect32u));
    NCV_SKIP_COND_END

    ncvStat = h_vecSrc.copySolid(d_vecSrc, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvStat = h_vecDst.copySolid(d_vecDst, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), false);

    Ncv32u h_outElemNum_d = 0;
    Ncv32u h_outElemNum_h = 0;
    NCV_SKIP_COND_BEGIN
    h_outElemNum_d = this->lenDst;
    ncvStat = ncvGrowDetectionsVector_device(d_vecSrc, this->lenSrc,
                                             d_vecDst, h_outElemNum_d, this->maxLenDst,
                                             this->rectWidth, this->rectHeight, this->rectScale, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvStat = d_vecDst.copySolid(h_vecDst_d, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), false);

    h_outElemNum_h = this->lenDst;
    ncvStat = ncvGrowDetectionsVector_host(h_vecSrc, this->lenSrc,
                                           h_vecDst, h_outElemNum_h, this->maxLenDst,
                                           this->rectWidth, this->rectHeight, this->rectScale);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    if (h_outElemNum_d != h_outElemNum_h)
    {
        bLoopVirgin = false;
    }
    else
    {
        if (memcmp(h_vecDst.ptr(), h_vecDst_d.ptr(), this->maxLenDst * sizeof(NcvRect32u)))
        {
            bLoopVirgin = false;
        }
    }
    NCV_SKIP_COND_END

    if (bLoopVirgin)
    {
        rcode = true;
    }

    return rcode;
}


bool TestHypothesesGrow::deinit()
{
    return true;
}
