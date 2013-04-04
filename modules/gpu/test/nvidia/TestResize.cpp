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

#if !defined CUDA_DISABLER

#include <math.h>

#include "TestResize.h"


template <class T>
TestResize<T>::TestResize(std::string testName_, NCVTestSourceProvider<T> &src_,
                          Ncv32u width_, Ncv32u height_, Ncv32u scaleFactor_, NcvBool bTextureCache_)
    :
    NCVTestProvider(testName_),
    src(src_),
    width(width_),
    height(height_),
    scaleFactor(scaleFactor_),
    bTextureCache(bTextureCache_)
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
        ncvStat = nppiStDecimate_32u_C1R((Ncv32u *)d_img.ptr(), d_img.pitch(),
                                         (Ncv32u *)d_small.ptr(), d_small.pitch(),
                                         srcSize, this->scaleFactor,
                                         this->bTextureCache);
    }
    else if (sizeof(T) == sizeof(Ncv64u))
    {
        ncvStat = nppiStDecimate_64u_C1R((Ncv64u *)d_img.ptr(), d_img.pitch(),
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
        ncvStat = nppiStDecimate_32u_C1R_host((Ncv32u *)h_img.ptr(), h_img.pitch(),
                                              (Ncv32u *)h_small.ptr(), h_small.pitch(),
                                              srcSize, this->scaleFactor);
    }
    else if (sizeof(T) == sizeof(Ncv64u))
    {
        ncvStat = nppiStDecimate_64u_C1R_host((Ncv64u *)h_img.ptr(), h_img.pitch(),
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

#endif /* CUDA_DISABLER */
