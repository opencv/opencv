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


TestHypothesesFilter::TestHypothesesFilter(std::string testName_, NCVTestSourceProvider<Ncv32u> &src_,
                                           Ncv32u numDstRects_, Ncv32u minNeighbors_, Ncv32f eps_)
    :
    NCVTestProvider(testName_),
    src(src_),
    numDstRects(numDstRects_),
    minNeighbors(minNeighbors_),
    eps(eps_)
{
}


bool TestHypothesesFilter::toString(std::ofstream &strOut)
{
    strOut << "numDstRects=" << numDstRects << std::endl;
    strOut << "minNeighbors=" << minNeighbors << std::endl;
    strOut << "eps=" << eps << std::endl;
    return true;
}


bool TestHypothesesFilter::init()
{
    this->canvasWidth = 4096;
    this->canvasHeight = 4096;
    return true;
}


bool compareRects(const NcvRect32u &r1, const NcvRect32u &r2, Ncv32f eps)
{
    double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
    return std::abs((Ncv32s)r1.x - (Ncv32s)r2.x) <= delta &&
        std::abs((Ncv32s)r1.y - (Ncv32s)r2.y) <= delta &&
        std::abs((Ncv32s)r1.x + (Ncv32s)r1.width - (Ncv32s)r2.x - (Ncv32s)r2.width) <= delta &&
        std::abs((Ncv32s)r1.y + (Ncv32s)r1.height - (Ncv32s)r2.y - (Ncv32s)r2.height) <= delta;
}


inline bool operator < (const NcvRect32u &a, const NcvRect32u &b)
{
    return a.x < b.x;
}


bool TestHypothesesFilter::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    NCVVectorAlloc<Ncv32u> h_random32u(*this->allocatorCPU.get(), this->numDstRects * sizeof(NcvRect32u) / sizeof(Ncv32u));
    ncvAssertReturn(h_random32u.isMemAllocated(), false);

    Ncv32u srcSlotSize = 2 * this->minNeighbors + 1;

    NCVVectorAlloc<NcvRect32u> h_vecSrc(*this->allocatorCPU.get(), this->numDstRects*srcSlotSize);
    ncvAssertReturn(h_vecSrc.isMemAllocated(), false);
    NCVVectorAlloc<NcvRect32u> h_vecDst_groundTruth(*this->allocatorCPU.get(), this->numDstRects);
    ncvAssertReturn(h_vecDst_groundTruth.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorCPU.get()->isCounting());

    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_random32u), false);
    Ncv32u randCnt = 0;
    Ncv64f randVal;

    for (Ncv32u i=0; i<this->numDstRects; i++)
    {
        h_vecDst_groundTruth.ptr()[i].x = i * this->canvasWidth / this->numDstRects + this->canvasWidth / (this->numDstRects * 4);
        h_vecDst_groundTruth.ptr()[i].y = i * this->canvasHeight / this->numDstRects + this->canvasHeight / (this->numDstRects * 4);
        h_vecDst_groundTruth.ptr()[i].width = this->canvasWidth / (this->numDstRects * 2);
        h_vecDst_groundTruth.ptr()[i].height = this->canvasHeight / (this->numDstRects * 2);

        Ncv32u numNeighbors = this->minNeighbors + 1 + (Ncv32u)(((1.0 * h_random32u.ptr()[i]) * (this->minNeighbors + 1)) / 0xFFFFFFFF);
        numNeighbors = (numNeighbors > srcSlotSize) ? srcSlotSize : numNeighbors;

        //fill in strong hypotheses                           (2 * ((1.0 * randVal) / 0xFFFFFFFF) - 1)
        for (Ncv32u j=0; j<numNeighbors; j++)
        {
            randVal = (1.0 * h_random32u.ptr()[randCnt++]) / 0xFFFFFFFF; randCnt = randCnt % h_random32u.length();
            h_vecSrc.ptr()[srcSlotSize * i + j].x =
                h_vecDst_groundTruth.ptr()[i].x +
                (Ncv32s)(h_vecDst_groundTruth.ptr()[i].width * this->eps * (randVal - 0.5));
            randVal = (1.0 * h_random32u.ptr()[randCnt++]) / 0xFFFFFFFF; randCnt = randCnt % h_random32u.length();
            h_vecSrc.ptr()[srcSlotSize * i + j].y =
                h_vecDst_groundTruth.ptr()[i].y +
                (Ncv32s)(h_vecDst_groundTruth.ptr()[i].height * this->eps * (randVal - 0.5));
            h_vecSrc.ptr()[srcSlotSize * i + j].width = h_vecDst_groundTruth.ptr()[i].width;
            h_vecSrc.ptr()[srcSlotSize * i + j].height = h_vecDst_groundTruth.ptr()[i].height;
        }

        //generate weak hypotheses (to be removed in processing)
        for (Ncv32u j=numNeighbors; j<srcSlotSize; j++)
        {
            randVal = (1.0 * h_random32u.ptr()[randCnt++]) / 0xFFFFFFFF; randCnt = randCnt % h_random32u.length();
            h_vecSrc.ptr()[srcSlotSize * i + j].x =
                this->canvasWidth + h_vecDst_groundTruth.ptr()[i].x +
                (Ncv32s)(h_vecDst_groundTruth.ptr()[i].width * this->eps * (randVal - 0.5));
            randVal = (1.0 * h_random32u.ptr()[randCnt++]) / 0xFFFFFFFF; randCnt = randCnt % h_random32u.length();
            h_vecSrc.ptr()[srcSlotSize * i + j].y =
                this->canvasHeight + h_vecDst_groundTruth.ptr()[i].y +
                (Ncv32s)(h_vecDst_groundTruth.ptr()[i].height * this->eps * (randVal - 0.5));
            h_vecSrc.ptr()[srcSlotSize * i + j].width = h_vecDst_groundTruth.ptr()[i].width;
            h_vecSrc.ptr()[srcSlotSize * i + j].height = h_vecDst_groundTruth.ptr()[i].height;
        }
    }

    //shuffle
    for (Ncv32u i=0; i<this->numDstRects*srcSlotSize-1; i++)
    {
        Ncv32u randValLocal = h_random32u.ptr()[randCnt++]; randCnt = randCnt % h_random32u.length();
        Ncv32u secondSwap = randValLocal % (this->numDstRects*srcSlotSize-1 - i);
        NcvRect32u tmp = h_vecSrc.ptr()[i + secondSwap];
        h_vecSrc.ptr()[i + secondSwap] = h_vecSrc.ptr()[i];
        h_vecSrc.ptr()[i] = tmp;
    }
    NCV_SKIP_COND_END

    Ncv32u numHypothesesSrc = static_cast<Ncv32u>(h_vecSrc.length());
    NCV_SKIP_COND_BEGIN
    ncvStat = ncvGroupRectangles_host(h_vecSrc, numHypothesesSrc, this->minNeighbors, this->eps, NULL);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    NCV_SKIP_COND_END

    //verification
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    if (numHypothesesSrc != this->numDstRects)
    {
        bLoopVirgin = false;
    }
    else
    {
        std::vector<NcvRect32u> tmpRects(numHypothesesSrc);
        memcpy(&tmpRects[0], h_vecSrc.ptr(), numHypothesesSrc * sizeof(NcvRect32u));
        std::sort(tmpRects.begin(), tmpRects.end());
        for (Ncv32u i=0; i<numHypothesesSrc && bLoopVirgin; i++)
        {
            if (!compareRects(tmpRects[i], h_vecDst_groundTruth.ptr()[i], this->eps))
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


bool TestHypothesesFilter::deinit()
{
    return true;
}
