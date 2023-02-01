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


template <class T>
TestDrawRects<T>::TestDrawRects(std::string testName_, NCVTestSourceProvider<T> &src_,
                                NCVTestSourceProvider<Ncv32u> &src32u_,
                                Ncv32u width_, Ncv32u height_, Ncv32u numRects_, T color_)
    :
    NCVTestProvider(testName_),
    src(src_),
    src32u(src32u_),
    width(width_),
    height(height_),
    numRects(numRects_),
    color(color_)
{
}


template <class T>
bool TestDrawRects<T>::toString(std::ofstream &strOut)
{
    strOut << "sizeof(T)=" << sizeof(T) << std::endl;
    strOut << "width=" << width << std::endl;
    strOut << "height=" << height << std::endl;
    strOut << "numRects=" << numRects << std::endl;
    strOut << "color=" << color << std::endl;
    return true;
}


template <class T>
bool TestDrawRects<T>::init()
{
    return true;
}


template <class T>
bool TestDrawRects<T>::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    NCVMatrixAlloc<T> d_img(*this->allocatorGPU.get(), this->width, this->height);
    ncvAssertReturn(d_img.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_img(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img.isMemAllocated(), false);
    NCVMatrixAlloc<T> h_img_d(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img_d.isMemAllocated(), false);

    NCVVectorAlloc<NcvRect32u> d_rects(*this->allocatorGPU.get(), this->numRects);
    ncvAssertReturn(d_rects.isMemAllocated(), false);
    NCVVectorAlloc<NcvRect32u> h_rects(*this->allocatorCPU.get(), this->numRects);
    ncvAssertReturn(h_rects.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN
    ncvAssertReturn(this->src.fill(h_img), false);
    ncvStat = h_img.copySolid(d_img, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), false);

    //fill vector of rectangles with random rects covering the input
    NCVVectorReuse<Ncv32u> h_rects_as32u(h_rects.getSegment());
    ncvAssertReturn(h_rects_as32u.isMemReused(), false);
    ncvAssertReturn(this->src32u.fill(h_rects_as32u), false);
    for (Ncv32u i=0; i<this->numRects; i++)
    {
        h_rects.ptr()[i].x = (Ncv32u)(((1.0 * h_rects.ptr()[i].x) / RAND_MAX) * (this->width-2));
        h_rects.ptr()[i].y = (Ncv32u)(((1.0 * h_rects.ptr()[i].y) / RAND_MAX) * (this->height-2));
        h_rects.ptr()[i].width = (Ncv32u)(((1.0 * h_rects.ptr()[i].width) / RAND_MAX) * (this->width+10 - h_rects.ptr()[i].x));
        h_rects.ptr()[i].height = (Ncv32u)(((1.0 * h_rects.ptr()[i].height) / RAND_MAX) * (this->height+10 - h_rects.ptr()[i].y));
    }
    ncvStat = h_rects.copySolid(d_rects, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), false);

    if (sizeof(T) == sizeof(Ncv32u))
    {
        ncvStat = ncvDrawRects_32u_device((Ncv32u *)d_img.ptr(), d_img.stride(), this->width, this->height,
                                          (NcvRect32u *)d_rects.ptr(), this->numRects, this->color, 0);
    }
    else if (sizeof(T) == sizeof(Ncv8u))
    {
        ncvStat = ncvDrawRects_8u_device((Ncv8u *)d_img.ptr(), d_img.stride(), this->width, this->height,
                                         (NcvRect32u *)d_rects.ptr(), this->numRects, (Ncv8u)this->color, 0);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect drawrects test instance", false);
    }
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    NCV_SKIP_COND_END

    ncvStat = d_img.copySolid(h_img_d, 0);
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    ncvAssertCUDAReturn(cudaStreamSynchronize(0), false);

    NCV_SKIP_COND_BEGIN
    if (sizeof(T) == sizeof(Ncv32u))
    {
        ncvStat = ncvDrawRects_32u_host((Ncv32u *)h_img.ptr(), h_img.stride(), this->width, this->height,
                                        (NcvRect32u *)h_rects.ptr(), this->numRects, this->color);
    }
    else if (sizeof(T) == sizeof(Ncv8u))
    {
        ncvStat = ncvDrawRects_8u_host((Ncv8u *)h_img.ptr(), h_img.stride(), this->width, this->height,
                                       (NcvRect32u *)h_rects.ptr(), this->numRects, (Ncv8u)this->color);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect drawrects test instance", false);
    }
    ncvAssertReturn(ncvStat == NCV_SUCCESS, false);
    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    //const Ncv64f relEPS = 0.005;
    for (Ncv32u i=0; bLoopVirgin && i < h_img.height(); i++)
    {
        for (Ncv32u j=0; bLoopVirgin && j < h_img.width(); j++)
        {
            if (h_img.ptr()[h_img.stride()*i+j] != h_img_d.ptr()[h_img_d.stride()*i+j])
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
bool TestDrawRects<T>::deinit()
{
    return true;
}


template class TestDrawRects<Ncv8u>;
template class TestDrawRects<Ncv32u>;
