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


TestIntegralImageSquared::TestIntegralImageSquared(std::string testName_, NCVTestSourceProvider<Ncv8u> &src_,
                                                   Ncv32u width_, Ncv32u height_)
    :
    NCVTestProvider(testName_),
    src(src_),
    width(width_),
    height(height_)
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
