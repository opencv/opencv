// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#ifndef OPENCV_CORE_BUFFER_POOL_HPP
#define OPENCV_CORE_BUFFER_POOL_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4265)
#endif

namespace cv
{

//! @addtogroup core
//! @{

class BufferPoolController
{
protected:
    ~BufferPoolController() { }
public:
    virtual size_t getReservedSize() const = 0;
    virtual size_t getMaxReservedSize() const = 0;
    virtual void setMaxReservedSize(size_t size) = 0;
    virtual void freeAllReservedBuffers() = 0;
};

//! @}

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // OPENCV_CORE_BUFFER_POOL_HPP
