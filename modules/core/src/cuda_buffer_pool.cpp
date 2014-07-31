/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#ifdef HAVE_CUDA

#include "opencv2/cudev/common.hpp"

/////////////////////////////////////////////////////////////
/// MemoryStack

namespace
{
    class MemoryPool;
}

class cv::cuda::MemoryStack
{
public:
    uchar* requestMemory(size_t size);
    void returnMemory(uchar* ptr);

    uchar* datastart;
    uchar* dataend;
    uchar* tip;

    bool isFree;
    MemoryPool* pool;

#if defined(DEBUG) || defined(_DEBUG)
    std::vector<size_t> allocations;
#endif
};

uchar* cv::cuda::MemoryStack::requestMemory(size_t size)
{
    const size_t freeMem = dataend - tip;

    if (size > freeMem)
        return 0;

    uchar* ptr = tip;

    tip += size;

#if defined(DEBUG) || defined(_DEBUG)
    allocations.push_back(size);
#endif

    return ptr;
}

void cv::cuda::MemoryStack::returnMemory(uchar* ptr)
{
    CV_DbgAssert( ptr >= datastart && ptr < dataend );

#if defined(DEBUG) || defined(_DEBUG)
    const size_t allocSize = tip - ptr;
    CV_Assert( allocSize == allocations.back() );
    allocations.pop_back();
#endif

    tip = ptr;
}

/////////////////////////////////////////////////////////////
/// MemoryPool

namespace
{
    class MemoryPool
    {
    public:
        MemoryPool();

        void initialize(size_t stackSize, int stackCount);
        void release();

        MemoryStack* getFreeMemStack();
        void returnMemStack(MemoryStack* memStack);

    private:
        void initilizeImpl();

        Mutex mtx_;

        bool initialized_;
        size_t stackSize_;
        int stackCount_;

        uchar* mem_;

        std::vector<MemoryStack> stacks_;
    };

    MemoryPool::MemoryPool() : initialized_(false), mem_(0)
    {
        // default : 10 Mb, 5 stacks
        stackSize_ = 10 * 1024 * 1024;
        stackCount_ = 5;
    }

    void MemoryPool::initialize(size_t stackSize, int stackCount)
    {
        AutoLock lock(mtx_);

        release();

        stackSize_ = stackSize;
        stackCount_ = stackCount;

        initilizeImpl();
    }

    void MemoryPool::initilizeImpl()
    {
        const size_t totalSize = stackSize_ * stackCount_;

        if (totalSize > 0)
        {
            cudaError_t err = cudaMalloc(&mem_, totalSize);
            if (err != cudaSuccess)
                return;

            stacks_.resize(stackCount_);

            uchar* ptr = mem_;

            for (int i = 0; i < stackCount_; ++i)
            {
                stacks_[i].datastart = ptr;
                stacks_[i].dataend = ptr + stackSize_;
                stacks_[i].tip = ptr;
                stacks_[i].isFree = true;
                stacks_[i].pool = this;

                ptr += stackSize_;
            }

            initialized_ = true;
        }
    }

    void MemoryPool::release()
    {
        if (mem_)
        {
#if defined(DEBUG) || defined(_DEBUG)
            for (int i = 0; i < stackCount_; ++i)
            {
                CV_DbgAssert( stacks_[i].isFree );
                CV_DbgAssert( stacks_[i].tip == stacks_[i].datastart );
            }
#endif

            cudaFree( mem_ );

            mem_ = 0;
            initialized_ = false;
        }
    }

    MemoryStack* MemoryPool::getFreeMemStack()
    {
        AutoLock lock(mtx_);
        if (!initialized_)
            initilizeImpl();

        if (!mem_)
            return 0;

        for (int i = 0; i < stackCount_; ++i)
        {
            if (stacks_[i].isFree)
            {
                stacks_[i].isFree = false;
                return &stacks_[i];
            }
        }

        return 0;
    }

    void MemoryPool::returnMemStack(MemoryStack* memStack)
    {
        AutoLock lock(mtx_);

        CV_DbgAssert( !memStack->isFree );

#if defined(DEBUG) || defined(_DEBUG)
        bool found = false;
        for (int i = 0; i < stackCount_; ++i)
        {
            if (memStack == &stacks_[i])
            {
                found = true;
                break;
            }
        }
        CV_DbgAssert( found );
#endif

        CV_DbgAssert( memStack->tip == memStack->datastart );

        memStack->isFree = true;
    }
}

/////////////////////////////////////////////////////////////
/// MemoryPoolManager

namespace
{
    Mutex mtx_;
    bool memory_pool_manager_initialized;

    class MemoryPoolManager
    {
    public:
        MemoryPoolManager();
        ~MemoryPoolManager();
        void Init();

        MemoryPool* getPool(int deviceId);

    private:
        std::vector<MemoryPool> pools_;
    } manager;

    //MemoryPoolManager ;

    MemoryPoolManager::MemoryPoolManager()
    {
    }

    void MemoryPoolManager::Init()
    {
        int deviceCount = getCudaEnabledDeviceCount();
        if (deviceCount > 0)
            pools_.resize(deviceCount);
    }

    MemoryPoolManager::~MemoryPoolManager()
    {
        for (size_t i = 0; i < pools_.size(); ++i)
        {
            cudaSetDevice(static_cast<int>(i));
            pools_[i].release();
        }
    }

    MemoryPool* MemoryPoolManager::getPool(int deviceId)
    {
        CV_DbgAssert( deviceId >= 0 && deviceId < static_cast<int>(pools_.size()) );
        return &pools_[deviceId];
    }

    MemoryPool* memPool(int deviceId)
    {
        {
            AutoLock lock(mtx_);
            if (!memory_pool_manager_initialized)
            {
                memory_pool_manager_initialized = true;
                manager.Init();
            }
        }
        return manager.getPool(deviceId);
    }
}

/////////////////////////////////////////////////////////////
/// StackAllocator

namespace
{
    bool enableMemoryPool = true;
}

cv::cuda::StackAllocator::StackAllocator(cudaStream_t stream) : stream_(stream), memStack_(0)
{
    if (enableMemoryPool)
    {
        const int deviceId = getDevice();
        {
            AutoLock lock(mtx_);
            memStack_ = memPool(deviceId)->getFreeMemStack();
        }
        DeviceInfo devInfo(deviceId);
        alignment_ = devInfo.textureAlignment();
    }
}

cv::cuda::StackAllocator::~StackAllocator()
{
    cudaStreamSynchronize(stream_);

    if (memStack_ != 0)
        memStack_->pool->returnMemStack(memStack_);
}

namespace
{
    size_t alignUp(size_t what, size_t alignment)
    {
        size_t alignMask = alignment-1;
        size_t inverseAlignMask = ~alignMask;
        size_t res = (what + alignMask) & inverseAlignMask;
        return res;
    }
}

bool cv::cuda::StackAllocator::allocate(GpuMat* mat, int rows, int cols, size_t elemSize)
{
    if (memStack_ == 0)
        return false;

    size_t pitch, memSize;

    if (rows > 1 && cols > 1)
    {
        pitch = alignUp(cols * elemSize, alignment_);
        memSize = pitch * rows;
    }
    else
    {
        // Single row or single column must be continuous
        pitch = elemSize * cols;
        memSize = alignUp(elemSize * cols * rows, 64);
    }

    uchar* ptr = memStack_->requestMemory(memSize);

    if (ptr == 0)
        return false;

    mat->data = ptr;
    mat->step = pitch;
    mat->refcount = (int*) fastMalloc(sizeof(int));

    return true;
}

void cv::cuda::StackAllocator::free(GpuMat* mat)
{
    if (memStack_ == 0)
        return;

    memStack_->returnMemory(mat->datastart);
    fastFree(mat->refcount);
}

void cv::cuda::setBufferPoolUsage(bool on)
{
    enableMemoryPool = on;
}

void cv::cuda::setBufferPoolConfig(int deviceId, size_t stackSize, int stackCount)
{
    const int currentDevice = getDevice();

    if (deviceId >= 0)
    {
        setDevice(deviceId);
        memPool(deviceId)->initialize(stackSize, stackCount);
    }
    else
    {
        const int deviceCount = getCudaEnabledDeviceCount();

        for (deviceId = 0; deviceId < deviceCount; ++deviceId)
        {
            setDevice(deviceId);
            memPool(deviceId)->initialize(stackSize, stackCount);
        }
    }

    setDevice(currentDevice);
}

/////////////////////////////////////////////////////////////
/// BufferPool

GpuMat cv::cuda::BufferPool::getBuffer(int rows, int cols, int type)
{
    GpuMat buf(allocator_);
    buf.create(rows, cols, type);
    return buf;
}

#endif
