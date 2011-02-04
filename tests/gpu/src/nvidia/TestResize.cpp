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

#include <math.h>

#include "TestResize.h"


template <class T>
TestResize<T>::TestResize(std::string testName, NCVTestSourceProvider<T> &src,
                          Ncv32u width, Ncv32u height, Ncv32u scaleFactor, NcvBool bTextureCache)
    :
    NCVTestProvider(testName),
    src(src),
    width(width),
    height(height),
    scaleFactor(scaleFactor),
    bTextureCache(bTextureCache)
{
}


template <class T>
bool TestResize<T>::toString(std::ofstream &strOut)
{
    strOut << "sizeof(T)=" << sizeof(T) << std::endl;
    strOut << "width=" << width << std::endl;
    strOut << "scaleFactor=" << scaleFactor << std::endl;
    strOut << "bTextureCache=" << bTextureCache << std::endl;
    return true;
}


template <class T>
bool TestResize<T>::init()
{
    return true;
}


template <class T>
bool TestResize<T>::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    Ncv32s smallWidth = this->width / this->scaleFactor;
    Ncv32s smallHeight = this->height / this->scaleFactor;
    if (smallWidth == 0 || smallHeight == 0)
    {
        return true;
    }

    NcvSize32u srcSize(this->width, this->height);

    NCVMatrixAlloc<T> d_img(*this->allocatorGPU.get(), this->width, this->height);
    ncvAssertReturn(d_img.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_img(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img.isMemAllocated(), false);

    NCVMatrixAlloc<T> d_small(*this->allocatorGPU.get(), smallWidth, smallHeight);
    ncvAssertReturn(d_small.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_small(*this->allocatorCPU.get(), smallWidth, smallHeight);
    ncvAssertReturn(h_small.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_small_d(*this->allocatorCPU.get(), smallWidth, smallHeight);
    ncvAssertReturn(h_small_d.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_img), false);
    NCV_SKIP_COND_END

    ncvStat = h_img.copySolid(d_img, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_BEGIN
    if (sizeof(T) == sizeof(Ncv32u))
    {
        ncvStat = nppiStDownsampleNearest_32u_C1R((Ncv32u *)d_img.ptr(), d_img.pitch(),
                                                  (Ncv32u *)d_small.ptr(), d_small.pitch(),
                                                  srcSize, this->scaleFactor,
                                                  this->bTextureCache);
    }
    else if (sizeof(T) == sizeof(Ncv64u))
    {
        ncvStat = nppiStDownsampleNearest_64u_C1R((Ncv64u *)d_img.ptr(), d_img.pitch(),
                                                  (Ncv64u *)d_small.ptr(), d_small.pitch(),
                                                  srcSize, this->scaleFactor,
                                                  this->bTextureCache);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect downsample test instance", false);
    }
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_END
    ncvStat = d_small.copySolid(h_small_d, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    NCV_SKIP_COND_BEGIN
    if (sizeof(T) == sizeof(Ncv32u))
    {
        ncvStat = nppiStDownsampleNearest_32u_C1R_host((Ncv32u *)h_img.ptr(), h_img.pitch(),
                                                       (Ncv32u *)h_small.ptr(), h_small.pitch(),
                                                       srcSize, this->scaleFactor);
    }
    else if (sizeof(T) == sizeof(Ncv64u))
    {
        ncvStat = nppiStDownsampleNearest_64u_C1R_host((Ncv64u *)h_img.ptr(), h_img.pitch(),
                                                       (Ncv64u *)h_small.ptr(), h_small.pitch(),
                                                       srcSize, this->scaleFactor);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect downsample test instance", false);
    }
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    //const Ncv64f relEPS = 0.005;
    for (Ncv32u i=0; bLoopVirgin && i < h_small.height(); i++)
    {
        for (Ncv32u j=0; bLoopVirgin && j < h_small.width(); j++)
        {
            if (h_small.ptr()[h_small.stride()*i+j] != h_small_d.ptr()[h_small_d.stride()*i+j])
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


template <class T>
bool TestResize<T>::deinit()
{
    return true;
}


template class TestResize<Ncv32u>;
template class TestResize<Ncv64u>;
