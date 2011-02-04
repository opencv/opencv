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

#include "TestTranspose.h"


template <class T>
TestTranspose<T>::TestTranspose(std::string testName, NCVTestSourceProvider<T> &src,
                                Ncv32u width, Ncv32u height)
    :
    NCVTestProvider(testName),
    src(src),
    width(width),
    height(height)
{
}


template <class T>
bool TestTranspose<T>::toString(std::ofstream &strOut)
{
    strOut << "sizeof(T)=" << sizeof(T) << std::endl;
    strOut << "width=" << width << std::endl;
    return true;
}


template <class T>
bool TestTranspose<T>::init()
{
    return true;
}


template <class T>
bool TestTranspose<T>::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    NcvSize32u srcSize(this->width, this->height);

    NCVMatrixAlloc<T> d_img(*this->allocatorGPU.get(), this->width, this->height);
    ncvAssertReturn(d_img.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_img(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img.isMemAllocated(), false);

    NCVMatrixAlloc<T> d_dst(*this->allocatorGPU.get(), this->height, this->width);
    ncvAssertReturn(d_dst.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_dst(*this->allocatorCPU.get(), this->height, this->width);
    ncvAssertReturn(h_dst.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_dst_d(*this->allocatorCPU.get(), this->height, this->width);
    ncvAssertReturn(h_dst_d.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_img), false);
    NCV_SKIP_COND_END

    ncvStat = h_img.copySolid(d_img, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_BEGIN
    if (sizeof(T) == sizeof(Ncv32u))
    {
        ncvStat = nppiStTranspose_32u_C1R((Ncv32u *)d_img.ptr(), d_img.pitch(),
                                          (Ncv32u *)d_dst.ptr(), d_dst.pitch(),
                                          NcvSize32u(this->width, this->height));
    }
    else if (sizeof(T) == sizeof(Ncv64u))
    {
        ncvStat = nppiStTranspose_64u_C1R((Ncv64u *)d_img.ptr(), d_img.pitch(),
                                        (Ncv64u *)d_dst.ptr(), d_dst.pitch(),
                                        NcvSize32u(this->width, this->height));
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect transpose test instance", false);
    }
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_END
    ncvStat = d_dst.copySolid(h_dst_d, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    NCV_SKIP_COND_BEGIN
    if (sizeof(T) == sizeof(Ncv32u))
    {
        ncvStat = nppiStTranspose_32u_C1R_host((Ncv32u *)h_img.ptr(), h_img.pitch(),
                                               (Ncv32u *)h_dst.ptr(), h_dst.pitch(),
                                               NcvSize32u(this->width, this->height));
    }
    else if (sizeof(T) == sizeof(Ncv64u))
    {
        ncvStat = nppiStTranspose_64u_C1R_host((Ncv64u *)h_img.ptr(), h_img.pitch(),
                                               (Ncv64u *)h_dst.ptr(), h_dst.pitch(),
                                               NcvSize32u(this->width, this->height));
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
    for (Ncv32u i=0; bLoopVirgin && i < this->width; i++)
    {
        for (Ncv32u j=0; bLoopVirgin && j < this->height; j++)
        {
            if (h_dst.ptr()[h_dst.stride()*i+j] != h_dst_d.ptr()[h_dst_d.stride()*i+j])
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
bool TestTranspose<T>::deinit()
{
    return true;
}


template class TestTranspose<Ncv32u>;
template class TestTranspose<Ncv64u>;
