/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_PRIVATE_IW_HPP
#define OPENCV_CORE_PRIVATE_IW_HPP

#ifndef __OPENCV_BUILD
#error this is a private header which should not be used from outside of the OpenCV library
#endif


#define IPP_VERSION_X100 (IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR*10 + IPP_VERSION_UPDATE)

#ifdef HAVE_IPP_ICV
#define ICV_BASE
#if IPP_VERSION_X100 >= 201700
#include "ippicv.h"
#else
#include "ipp.h"
#endif
#else
#include "ipp.h"
#endif
#include "iw++/iw.hpp"

#  ifdef HAVE_IPP_IW_LL
#include "iw/iw_ll.h"
#  endif

#include "private.hpp"




namespace cv {

static inline int ippiSuggestThreadsNum(size_t width, size_t height,
                                        size_t elemSize, double multiplier) {
  int threads = cv::getNumThreads();
  if (threads > 1 && height >= 64) {
    size_t opMemory = (int)(width * height * elemSize * multiplier);
    int l2cache = 0;
#if IPP_VERSION_X100 >= 201700
    ippGetL2CacheSize(&l2cache);
#endif
    if (!l2cache)
      l2cache = 1 << 18;

    return IPP_MAX(1, (IPP_MIN((int)(opMemory / l2cache), threads)));
  }
  return 1;
}

static inline int ippiSuggestThreadsNum(const cv::Mat &image,
                                        double multiplier) {
  return ippiSuggestThreadsNum(image.cols, image.rows, image.elemSize(),
                               multiplier);
}

static inline int ippiSuggestThreadsNum(const ::ipp::IwiImage &image,
                                        double multiplier) {
  return ippiSuggestThreadsNum(image.m_size.width, image.m_size.height,
                               image.m_typeSize * image.m_channels, multiplier);
}


static inline ::ipp::IwiSize ippiGetSize(const cv::Size & size)
{
    return ::ipp::IwiSize((IwSize)size.width, (IwSize)size.height);
}

static inline IwiDerivativeType ippiGetDerivType(int dx, int dy, bool nvert)
{
    return (dx == 1 && dy == 0) ? ((nvert)?iwiDerivNVerFirst:iwiDerivVerFirst) :
           (dx == 0 && dy == 1) ? iwiDerivHorFirst :
           (dx == 2 && dy == 0) ? iwiDerivVerSecond :
           (dx == 0 && dy == 2) ? iwiDerivHorSecond :
           (IwiDerivativeType)-1;
}

static inline void ippiGetImage(const cv::Mat &src, ::ipp::IwiImage &dst)
{
    ::ipp::IwiBorderSize inMemBorder;
    if(src.isSubmatrix()) // already have physical border
    {
        cv::Size  origSize;
        cv::Point offset;
        src.locateROI(origSize, offset);

        inMemBorder.left   = (IwSize)offset.x;
        inMemBorder.top    = (IwSize)offset.y;
        inMemBorder.right  = (IwSize)(origSize.width - src.cols - offset.x);
        inMemBorder.bottom = (IwSize)(origSize.height - src.rows - offset.y);
    }

    dst.Init(ippiSize(src.size()), ippiGetDataType(src.depth()), src.channels(), inMemBorder, (void*)src.ptr(), src.step);
}

static inline ::ipp::IwiImage ippiGetImage(const cv::Mat &src)
{
    ::ipp::IwiImage image;
    ippiGetImage(src, image);
    return image;
}

static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ::ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags = 0;
    IppiBorderType border     = ippiGetBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
    if((int)border == -1)
        return (IppiBorderType)0;

    if(!(ocvBorderType & cv::BORDER_ISOLATED))
    {
        if(image.m_inMemSize.left)
        {
            if(image.m_inMemSize.left >= borderSize.left)
                inMemFlags |= ippBorderInMemLeft;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.left = 0;
        if(image.m_inMemSize.top)
        {
            if(image.m_inMemSize.top >= borderSize.top)
                inMemFlags |= ippBorderInMemTop;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.top = 0;
        if(image.m_inMemSize.right)
        {
            if(image.m_inMemSize.right >= borderSize.right)
                inMemFlags |= ippBorderInMemRight;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.right = 0;
        if(image.m_inMemSize.bottom)
        {
            if(image.m_inMemSize.bottom >= borderSize.bottom)
                inMemFlags |= ippBorderInMemBottom;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.bottom = 0;
    }
    else
        borderSize.left = borderSize.right = borderSize.top = borderSize.bottom = 0;

    return (IppiBorderType)(border|inMemFlags);
}


} // namespace cv

//! @endcond

#endif // OPENCV_CORE_PRIVATE_IW_HPP
