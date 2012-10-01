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

#if !defined CUDA_DISABLER

#include "TestCompact.h"


TestCompact::TestCompact(std::string testName_, NCVTestSourceProvider<Ncv32u> &src_,
                                             Ncv32u length_, Ncv32u badElem_, Ncv32u badElemPercentage_)
    :
    NCVTestProvider(testName_),
    src(src_),
    length(length_),
    badElem(badElem_),
    badElemPercentage(badElemPercentage_ > 100 ? 100 : badElemPercentage_)
{
}


bool TestCompact::toString(std::ofstream &strOut)
{
    strOut << "length=" << length << std::endl;
    strOut << "badElem=" << badElem << std::endl;
    strOut << "badElemPercentage=" << badElemPercentage << std::endl;
    return true;
}


bool TestCompact::init()
{
    return true;
}


bool TestCompact::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    NCVVectorAlloc<Ncv32u> h_vecSrc(*this->allocatorCPU.get(), this->length);
    ncvAssertReturn(h_vecSrc.isMemAllocated(), false);
    NCVVectorAlloc<Ncv32u> d_vecSrc(*this->allocatorGPU.get(), this->length);
    ncvAssertReturn(d_vecSrc.isMemAllocated(), false);

    NCVVectorAlloc<Ncv32u> h_vecDst(*this->allocatorCPU.get(), this->length);
    ncvAssertReturn(h_vecDst.isMemAllocated(), false);
    NCVVectorAlloc<Ncv32u> d_vecDst(*this->allocatorGPU.get(), this->length);
    ncvAssertReturn(d_vecDst.isMemAllocated(), false);
    NCVVectorAlloc<Ncv32u> h_vecDst_d(*this->allocatorCPU.get(), this->length);
    ncvAssertReturn(h_vecDst_d.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_vecSrc), false);
    for (Ncv32u i=0; i<this->length; i++)
    {
        Ncv32u tmp = (h_vecSrc.ptr()[i]) & 0xFF;
        tmp = tmp * 99 / 255;
        if (tmp < this->badElemPercentage)
        {
            h_vecSrc.ptr()[i] = this->badElem;
        }
    }
    NCV_SKIP_COND_END

    NCVVectorAlloc<Ncv32u> h_dstLen(*this->allocatorCPU.get(), 1);
    ncvAssertReturn(h_dstLen.isMemAllocated(), false);
    Ncv32u bufSize;
    ncvStat = nppsStCompactGetSize_32u(this->length, &bufSize, this->devProp);
    ncvAssertReturn(NPPST_SUCCESS == ncvStat, false);
    NCVVectorAlloc<Ncv8u> d_tmpBuf(*this->allocatorGPU.get(), bufSize);
    ncvAssertReturn(d_tmpBuf.isMemAllocated(), false);

    Ncv32u h_outElemNum_h = 0;

    NCV_SKIP_COND_BEGIN
    ncvStat = h_vecSrc.copySolid(d_vecSrc, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    ncvStat = nppsStCompact_32u(d_vecSrc.ptr(), this->length,
                                d_vecDst.ptr(), h_dstLen.ptr(), this->badElem,
                                d_tmpBuf.ptr(), bufSize, this->devProp);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    ncvStat = d_vecDst.copySolid(h_vecDst_d, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppsStCompact_32u_host(h_vecSrc.ptr(), this->length, h_vecDst.ptr(), &h_outElemNum_h, this->badElem);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    if (h_dstLen.ptr()[0] != h_outElemNum_h)
    {
        bLoopVirgin = false;
    }
    else
    {
        for (Ncv32u i=0; bLoopVirgin && i < h_outElemNum_h; i++)
        {
            if (h_vecDst.ptr()[i] != h_vecDst_d.ptr()[i])
            {
                bLoopVirgin = false;
            }
        }
    }
    NCV_SKIP_COND_END

    if (bLoopVirgin)
    {
        rcode = true;
    }

    return rcode;
}


bool TestCompact::deinit()
{
    return true;
}

#endif /* CUDA_DISABLER */