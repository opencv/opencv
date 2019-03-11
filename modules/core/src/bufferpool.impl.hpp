// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#ifndef __OPENCV_CORE_BUFFER_POOL_IMPL_HPP__
#define __OPENCV_CORE_BUFFER_POOL_IMPL_HPP__

#include "opencv2/core/bufferpool.hpp"

namespace cv {

class DummyBufferPoolController : public BufferPoolController
{
public:
    DummyBufferPoolController() { }
    virtual ~DummyBufferPoolController() { }

    virtual size_t getReservedSize() const CV_OVERRIDE { return (size_t)-1; }
    virtual size_t getMaxReservedSize() const CV_OVERRIDE { return (size_t)-1; }
    virtual void setMaxReservedSize(size_t size) CV_OVERRIDE { CV_UNUSED(size); }
    virtual void freeAllReservedBuffers() CV_OVERRIDE { }
};

} // namespace

#endif // __OPENCV_CORE_BUFFER_POOL_IMPL_HPP__
