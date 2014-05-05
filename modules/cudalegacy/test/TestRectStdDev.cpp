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


TestRectStdDev::TestRectStdDev(std::string testName_, NCVTestSourceProvider<Ncv8u> &src_,
                               Ncv32u width_, Ncv32u height_, NcvRect32u rect_, Ncv32f scaleFactor_,
                               NcvBool bTextureCache_)
    :
    NCVTestProvider(testName_),
    src(src_),
    width(width_),
    height(height_),
    rect(rect_),
    scaleFactor(scaleFactor_),
    bTextureCache(bTextureCache_)
{
}


bool TestRectStdDev::toString(std::ofstream &strOut)
{
    strOut << "width=" << width << std::endl;
    strOut << "height=" << height << std::endl;
    strOut << "rect=[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]\n";
    strOut << "scaleFactor=" << scaleFactor << std::endl;
    strOut << "bTextureCache=" << bTextureCache << std::endl;
    return true;
}


bool TestRectStdDev::init()
{
    return true;
}


bool TestRectStdDev::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    Ncv32s _normWidth = (Ncv32s)this->width - this->rect.x - this->rect.width + 1;
    Ncv32s _normHeight = (Ncv32s)this->height - this->rect.y - this->rect.height + 1;
    if (_normWidth <= 0 || _normHeight <= 0)
    {
        return true;
    }
    Ncv32u normWidth = (Ncv32u)_normWidth;
    Ncv32u normHeight = (Ncv32u)_normHeight;
    NcvSize32u szNormRoi(normWidth, normHeight);

    Ncv32u widthII = this->width + 1;
    Ncv32u heightII = this->height + 1;
    Ncv32u widthSII = this->width + 1;
    Ncv32u heightSII = this->height + 1;

    NCVMatrixAlloc<Ncv8u> d_img(*this->allocatorGPU.get(), this->width, this->height);
    ncvAssertReturn(d_img.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv8u> h_img(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img.isMemAllocated(), false);

    NCVMatrixAlloc<Ncv32u> d_imgII(*this->allocatorGPU.get(), widthII, heightII);
    ncvAssertReturn(d_imgII.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv32u> h_imgII(*this->allocatorCPU.get(), widthII, heightII);
    ncvAssertReturn(h_imgII.isMemAllocated(), false);

    NCVMatrixAlloc<Ncv64u> d_imgSII(*this->allocatorGPU.get(), widthSII, heightSII);
    ncvAssertReturn(d_imgSII.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv64u> h_imgSII(*this->allocatorCPU.get(), widthSII, heightSII);
    ncvAssertReturn(h_imgSII.isMemAllocated(), false);

    NCVMatrixAlloc<Ncv32f> d_norm(*this->allocatorGPU.get(), normWidth, normHeight);
    ncvAssertReturn(d_norm.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv32f> h_norm(*this->allocatorCPU.get(), normWidth, normHeight);
    ncvAssertReturn(h_norm.isMemAllocated(), false);
    NCVMatrixAlloc<Ncv32f> h_norm_d(*this->allocatorCPU.get(), normWidth, normHeight);
    ncvAssertReturn(h_norm_d.isMemAllocated(), false);

    Ncv32u bufSizeII, bufSizeSII;
    ncvStat = nppiStIntegralGetSize_8u32u(NcvSize32u(this->width, this->height), &bufSizeII, this->devProp);
    ncvAssertReturn(NPPST_SUCCESS == ncvStat, false);
    ncvStat = nppiStSqrIntegralGetSize_8u64u(NcvSize32u(this->width, this->height), &bufSizeSII, this->devProp);
    ncvAssertReturn(NPPST_SUCCESS == ncvStat, false);
    Ncv32u bufSize = bufSizeII > bufSizeSII ? bufSizeII : bufSizeSII;
    NCVVectorAlloc<Ncv8u> d_tmpBuf(*this->allocatorGPU.get(), bufSize);
    ncvAssertReturn(d_tmpBuf.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_img), false);

    ncvStat = h_img.copySolid(d_img, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStIntegral_8u32u_C1R(d_img.ptr(), d_img.pitch(),
                                       d_imgII.ptr(), d_imgII.pitch(),
                                       NcvSize32u(this->width, this->height),
                                       d_tmpBuf.ptr(), bufSize, this->devProp);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStSqrIntegral_8u64u_C1R(d_img.ptr(), d_img.pitch(),
                                          d_imgSII.ptr(), d_imgSII.pitch(),
                                          NcvSize32u(this->width, this->height),
                                          d_tmpBuf.ptr(), bufSize, this->devProp);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStRectStdDev_32f_C1R(d_imgII.ptr(), d_imgII.pitch(),
                                       d_imgSII.ptr(), d_imgSII.pitch(),
                                       d_norm.ptr(), d_norm.pitch(),
                                       szNormRoi, this->rect,
                                       this->scaleFactor,
                                       this->bTextureCache);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = d_norm.copySolid(h_norm_d, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStIntegral_8u32u_C1R_host(h_img.ptr(), h_img.pitch(),
                                          h_imgII.ptr(), h_imgII.pitch(),
                                          NcvSize32u(this->width, this->height));
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStSqrIntegral_8u64u_C1R_host(h_img.ptr(), h_img.pitch(),
                                             h_imgSII.ptr(), h_imgSII.pitch(),
                                             NcvSize32u(this->width, this->height));
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    ncvStat = nppiStRectStdDev_32f_C1R_host(h_imgII.ptr(), h_imgII.pitch(),
                                          h_imgSII.ptr(), h_imgSII.pitch(),
                                          h_norm.ptr(), h_norm.pitch(),
                                          szNormRoi, this->rect,
                                          this->scaleFactor);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    const Ncv64f relEPS = 0.005;
    for (Ncv32u i=0; bLoopVirgin && i < h_norm.height(); i++)
    {
        for (Ncv32u j=0; bLoopVirgin && j < h_norm.width(); j++)
        {
            Ncv64f absErr = fabs(h_norm.ptr()[h_norm.stride()*i+j] - h_norm_d.ptr()[h_norm_d.stride()*i+j]);
            Ncv64f relErr = absErr / h_norm.ptr()[h_norm.stride()*i+j];

            if (relErr > relEPS)
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


bool TestRectStdDev::deinit()
{
    return true;
}
