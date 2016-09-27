// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#ifndef OPENCV_CORE_BUFFER_POOL_IMPL_HPP
#define OPENCV_CORE_BUFFER_POOL_IMPL_HPP

#include "opencv2/core/bufferpool.hpp"

namespace cv {

class DummyBufferPoolController : public BufferPoolController
{
public:
    DummyBufferPoolController() { }
    virtual ~DummyBufferPoolController() { }

    virtual size_t getReservedSize() const { return (size_t)-1; }
    virtual size_t getMaxReservedSize() const { return (size_t)-1; }
    virtual void setMaxReservedSize(size_t size) { (void)size; }
    virtual void freeAllReservedBuffers() { }
};

} // namespace

#endif // OPENCV_CORE_BUFFER_POOL_IMPL_HPP
