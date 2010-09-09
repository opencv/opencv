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
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

#include "precomp.hpp"
#include "npp.h" //TODO: move to the precomp.hpp

using namespace cv;
using namespace cv::gpu;
using namespace std;

#if !defined (HAVE_CUDA)

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

void cv::gpu::add(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());

    dst.create( src1.size(), src1.type() );
    
    CV_DbgAssert(src1.depth() == CV_8U || src1.depth() == CV_32F);
    CV_DbgAssert(src1.channels() == 1 || src1.channels() == 4);

    NppiSize sz;
    sz.width = src1.cols;
    sz.height = src1.rows;

    if (src1.depth() == CV_8U)
    {
        nppiAdd_8u_C1RSfs((const Npp8u*)src1.ptr<char>(), src1.step, 
                          (const Npp8u*)src2.ptr<char>(), src2.step, 
                          (Npp8u*)dst.ptr<char>(), dst.step, sz, 0);
    }
    //TODO: implement other depths
}

#endif /* !defined (HAVE_CUDA) */