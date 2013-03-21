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

#include "TestHypothesesGrow.h"
#include "NCVHaarObjectDetection.hpp"


TestHypothesesGrow::TestHypothesesGrow(std::string testName_, NCVTestSourceProvider<Ncv32u> &src_,
                                       Ncv32u rectWidth_, Ncv32u rectHeight_, Ncv32f rectScale_,
                                       Ncv32u maxLenSrc_, Ncv32u lenSrc_, Ncv32u maxLenDst_, Ncv32u lenDst_)
    :
    NCVTestProvider(testName_),
    src(src_),
    rectWidth(rectWidth_),
    rectHeight(rectHeight_),
    rectScale(rectScale_),
    maxLenSrc(maxLenSrc_),
    lenSrc(lenSrc_),
    maxLenDst(maxLenDst_),
    lenDst(lenDst_)
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

#endif /* CUDA_DISABLER */
