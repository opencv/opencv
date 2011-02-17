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

#include "TestIntegralImageSquared.h"


TestIntegralImageSquared::TestIntegralImageSquared(std::string testName, NCVTestSourceProvider<Ncv8u> &src,
                                                   Ncv32u width, Ncv32u height)
    :
    NCVTestProvider(testName),
    src(src),
    width(width),
    height(height)
{
}


bool TestIntegralImageSquared::toString(std::ofstream &strOut)
{
    strOut << "width=" << width << std::endl;
    strOut << "height=" << height << std::endl;
    return true;
}


bool TestIntegralImageSquared::init()
{
    return true;
}


bool TestIntegralImageSquared::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    Ncv32u widthSII = this->width + 1;
    Ncv32u heightSII = this->height + 1;

    NCVMatrixAlloc<Ncv8u> d_img(*this->allocatorGPU.get(), this->width, this->height);
    ncvAssertReturn(d_img.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv8u> h_img(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv64u> d_imgSII(*this->allocatorGPU.get(), widthSII, heightSII);
    ncvAssertReturn(d_imgSII.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv64u> h_imgSII(*this->allocatorCPU.get(), widthSII, heightSII);
    ncvAssertReturn(h_imgSII.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv64u> h_imgSII_d(*this->allocatorCPU.get(), widthSII, heightSII);
    ncvAssertReturn(h_imgSII_d.isMemAllocated(), false);

    Ncv32u bufSize;
    ncvStat = nppiStSqrIntegralGetSize_8u64u(NcvSize32u(this->width, this->height), &bufSize, this->devProp);
    ncvAssertReturn(NPPST_SUCCESS == ncvStat, false);
    NCVVectorAlloc<Ncv8u> d_tmpBuf(*this->allocatorGPU.get(), bufSize);
    ncvAssertReturn(d_tmpBuf.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN

    ncvAssertReturn(this->src.fill(h_img), false);

    ncvStat = h_img.copySolid(d_img, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStSqrIntegral_8u64u_C1R(d_img.ptr(), d_img.pitch(),
                                          d_imgSII.ptr(), d_imgSII.pitch(),
                                          NcvSize32u(this->width, this->height),
                                          d_tmpBuf.ptr(), bufSize, this->devProp);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = d_imgSII.copySolid(h_imgSII_d, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStSqrIntegral_8u64u_C1R_host(h_img.ptr(), h_img.pitch(),
                                               h_imgSII.ptr(), h_imgSII.pitch(),
                                               NcvSize32u(this->width, this->height));
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    for (Ncv32u i=0; bLoopVirgin && i < h_img.height() + 1; i++)
    {
        for (Ncv32u j=0; bLoopVirgin && j < h_img.width() + 1; j++)
        {
            if (h_imgSII.ptr()[h_imgSII.stride()*i+j] != h_imgSII_d.ptr()[h_imgSII_d.stride()*i+j])
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


bool TestIntegralImageSquared::deinit()
{
    return true;
}
