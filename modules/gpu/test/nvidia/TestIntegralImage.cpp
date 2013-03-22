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
#include "TestIntegralImage.h"


template <class T_in, class T_out>
TestIntegralImage<T_in, T_out>::TestIntegralImage(std::string testName_, NCVTestSourceProvider<T_in> &src_,
                                                  Ncv32u width_, Ncv32u height_)
    :
    NCVTestProvider(testName_),
    src(src_),
    width(width_),
    height(height_)
{
}


template <class T_in, class T_out>
bool TestIntegralImage<T_in, T_out>::toString(std::ofstream &strOut)
{
    strOut << "sizeof(T_in)=" << sizeof(T_in) << std::endl;
    strOut << "sizeof(T_out)=" << sizeof(T_out) << std::endl;
    strOut << "width=" << width << std::endl;
    strOut << "height=" << height << std::endl;
    return true;
}


template <class T_in, class T_out>
bool TestIntegralImage<T_in, T_out>::init()
{
    return true;
}


template <class T_in, class T_out>
bool TestIntegralImage<T_in, T_out>::process()
{
    NCVStatus ncvStat;
    bool rcode = false;

    Ncv32u widthII = this->width + 1;
    Ncv32u heightII = this->height + 1;

    NCVMatrixAlloc<T_in> d_img(*this->allocatorGPU.get(), this->width, this->height);
    ncvAssertReturn(d_img.isMemAllocated(), false);
    NCVMatrixAlloc<T_in> h_img(*this->allocatorCPU.get(), this->width, this->height);
    ncvAssertReturn(h_img.isMemAllocated(), false);
    NCVMatrixAlloc<T_out> d_imgII(*this->allocatorGPU.get(), widthII, heightII);
    ncvAssertReturn(d_imgII.isMemAllocated(), false);
    NCVMatrixAlloc<T_out> h_imgII(*this->allocatorCPU.get(), widthII, heightII);
    ncvAssertReturn(h_imgII.isMemAllocated(), false);
    NCVMatrixAlloc<T_out> h_imgII_d(*this->allocatorCPU.get(), widthII, heightII);
    ncvAssertReturn(h_imgII_d.isMemAllocated(), false);

    Ncv32u bufSize;
    if (sizeof(T_in) == sizeof(Ncv8u))
    {
        ncvStat = nppiStIntegralGetSize_8u32u(NcvSize32u(this->width, this->height), &bufSize, this->devProp);
        ncvAssertReturn(NPPST_SUCCESS == ncvStat, false);
    }
    else if (sizeof(T_in) == sizeof(Ncv32f))
    {
        ncvStat = nppiStIntegralGetSize_32f32f(NcvSize32u(this->width, this->height), &bufSize, this->devProp);
        ncvAssertReturn(NPPST_SUCCESS == ncvStat, false);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect integral image test instance", false);
    }

    NCVVectorAlloc<Ncv8u> d_tmpBuf(*this->allocatorGPU.get(), bufSize);
    ncvAssertReturn(d_tmpBuf.isMemAllocated(), false);

    NCV_SET_SKIP_COND(this->allocatorGPU.get()->isCounting());
    NCV_SKIP_COND_BEGIN

    ncvAssertReturn(this->src.fill(h_img), false);

    ncvStat = h_img.copySolid(d_img, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    if (sizeof(T_in) == sizeof(Ncv8u))
    {
        ncvStat = nppiStIntegral_8u32u_C1R((Ncv8u *)d_img.ptr(), d_img.pitch(),
                                           (Ncv32u *)d_imgII.ptr(), d_imgII.pitch(),
                                           NcvSize32u(this->width, this->height),
                                           d_tmpBuf.ptr(), bufSize, this->devProp);
        ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    }
    else if (sizeof(T_in) == sizeof(Ncv32f))
    {
        ncvStat = nppiStIntegral_32f32f_C1R((Ncv32f *)d_img.ptr(), d_img.pitch(),
                                            (Ncv32f *)d_imgII.ptr(), d_imgII.pitch(),
                                            NcvSize32u(this->width, this->height),
                                            d_tmpBuf.ptr(), bufSize, this->devProp);
        ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect integral image test instance", false);
    }

    ncvStat = d_imgII.copySolid(h_imgII_d, 0);
    ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);

    if (sizeof(T_in) == sizeof(Ncv8u))
    {
        ncvStat = nppiStIntegral_8u32u_C1R_host((Ncv8u *)h_img.ptr(), h_img.pitch(),
                                                (Ncv32u *)h_imgII.ptr(), h_imgII.pitch(),
                                                NcvSize32u(this->width, this->height));
        ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    }
    else if (sizeof(T_in) == sizeof(Ncv32f))
    {
        ncvStat = nppiStIntegral_32f32f_C1R_host((Ncv32f *)h_img.ptr(), h_img.pitch(),
                                                 (Ncv32f *)h_imgII.ptr(), h_imgII.pitch(),
                                                 NcvSize32u(this->width, this->height));
        ncvAssertReturn(ncvStat == NPPST_SUCCESS, false);
    }
    else
    {
        ncvAssertPrintReturn(false, "Incorrect integral image test instance", false);
    }

    NCV_SKIP_COND_END

    //bit-to-bit check
    bool bLoopVirgin = true;

    NCV_SKIP_COND_BEGIN
    for (Ncv32u i=0; bLoopVirgin && i < h_img.height() + 1; i++)
    {
        for (Ncv32u j=0; bLoopVirgin && j < h_img.width() + 1; j++)
        {
            if (sizeof(T_in) == sizeof(Ncv8u))
            {
                if (h_imgII.ptr()[h_imgII.stride()*i+j] != h_imgII_d.ptr()[h_imgII_d.stride()*i+j])
                {
                    bLoopVirgin = false;
                }
            }
            else if (sizeof(T_in) == sizeof(Ncv32f))
            {
                if (fabsf((float)h_imgII.ptr()[h_imgII.stride()*i+j] - (float)h_imgII_d.ptr()[h_imgII_d.stride()*i+j]) > 0.01f)
                {
                    bLoopVirgin = false;
                }
            }
            else
            {
                ncvAssertPrintReturn(false, "Incorrect integral image test instance", false);
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


template <class T_in, class T_out>
bool TestIntegralImage<T_in, T_out>::deinit()
{
    return true;
}


template class TestIntegralImage<Ncv8u, Ncv32u>;
template class TestIntegralImage<Ncv32f, Ncv32f>;

#endif /* CUDA_DISABLER */
