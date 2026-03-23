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
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
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
//     and / or other materials provided with the distribution.
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

#ifndef OPENCV_CORE_EXTERNAL_UMAT_ALLOCATOR_HPP
#define OPENCV_CORE_EXTERNAL_UMAT_ALLOCATOR_HPP

#include <opencv2/core.hpp>

namespace cv {

//! Allocator that manages external (user‑supplied) memory for UMat.
class ExternalUMatAllocator : public MatAllocator
{
public:
    ExternalUMatAllocator() = default;
    virtual ~ExternalUMatAllocator() = default;

    virtual UMatData* allocate(int dims, const int* sizes, int type, void* data0,
                               size_t* steps, AccessFlag accessFlags,
                               UMatUsageFlags usageFlags) const CV_OVERRIDE;

    virtual bool allocate(UMatData* u, AccessFlag accessFlags,
                          UMatUsageFlags usageFlags) const CV_OVERRIDE;

    virtual void deallocate(UMatData* u) const CV_OVERRIDE;
    virtual void map(UMatData* u, AccessFlag accessFlags) const CV_OVERRIDE;
    virtual void unmap(UMatData* u) const CV_OVERRIDE;

    virtual void copy(UMatData* srcU, UMatData* dstU, int dims,
                      const size_t sz[], const size_t srcOfs[], const size_t srcStep[],
                      const size_t dstOfs[], const size_t dstStep[], bool sync) const CV_OVERRIDE;
};

//! Returns the global instance of the external memory allocator.
CV_EXPORTS MatAllocator* getExternalUMatAllocator();

} // namespace cv

#endif