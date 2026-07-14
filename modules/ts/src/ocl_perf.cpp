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
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#include "opencv2/ts/ocl_perf.hpp"

namespace cvtest {
namespace ocl {

namespace perf {

void checkDeviceMaxMemoryAllocSize(const Size& size, int type, int factor)
{
    CV_Assert(factor > 0);

    if (!cv::ocl::useOpenCL())
        return;

    size_t memSize = size.area() * CV_ELEM_SIZE(type);
    const cv::ocl::Device& dev = cv::ocl::Device::getDefault();

    if (memSize * factor >= dev.maxMemAllocSize())
        throw ::perf::TestBase::PerfSkipTestException();
}

void randu(InputOutputArray dst)
{
    if (dst.depth() == CV_8U)
        cv::randu(dst, 0, 256);
    else if (dst.depth() == CV_8S)
        cv::randu(dst, -128, 128);
    else if (dst.depth() == CV_16U)
        cv::randu(dst, 0, 1024);
    else if (dst.depth() == CV_32F || dst.depth() == CV_64F || dst.depth() == CV_16F)
        cv::randu(dst, -1.0, 1.0);
    else if (dst.depth() == CV_16S || dst.depth() == CV_32S)
        cv::randu(dst, -4096, 4096);
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported format");
}

} // namespace perf

} } // namespace cvtest::ocl
