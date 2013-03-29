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

#include "TestTranspose.h"


template <class T>
TestTranspose<T>::TestTranspose(std::string testName_, NCVTestSourceProvider<T> &src_,
                                Ncv32u width_, Ncv32u height_)
    :
    NCVTestProvider(testName_),
    src(src_),
    width(width_),
    height(height_)
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

#endif /* CUDA_DISABLER */
