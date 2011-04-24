/*M///////////////////////////////////////////////////////////////////////////////////////
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. 
// 
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2009-2010, NVIDIA Corporation, all rights reserved.
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


#include <ios>
#include <stdarg.h>
#include <vector>
#include "NCV.hpp"


//==============================================================================
//
// Error handling helpers
//
//==============================================================================


static void stdioDebugOutput(const char *msg)
{
    printf("%s", msg);
}


static NCVDebugOutputHandler *debugOutputHandler = stdioDebugOutput;


void ncvDebugOutput(const char *msg, ...)
{
    const int K_DEBUG_STRING_MAXLEN = 1024;
    char buffer[K_DEBUG_STRING_MAXLEN];
    va_list args;
    va_start(args, msg);
    vsnprintf(buffer, K_DEBUG_STRING_MAXLEN, msg, args);
    va_end (args);
    debugOutputHandler(buffer);
}


void ncvSetDebugOutputHandler(NCVDebugOutputHandler *func)
{
    debugOutputHandler = func;
}


//==============================================================================
//
// Memory wrappers and helpers
//
//==============================================================================


Ncv32u alignUp(Ncv32u what, Ncv32u alignment)
{
    Ncv32u alignMask = alignment-1;
    Ncv32u inverseAlignMask = ~alignMask;
    Ncv32u res = (what + alignMask) & inverseAlignMask;
    return res;
}


void NCVMemPtr::clear()
{
    ptr = NULL;
    memtype = NCVMemoryTypeNone;
}


void NCVMemSegment::clear()
{
    begin.clear();
    size = 0;
}


NCVStatus memSegCopyHelper(void *dst, NCVMemoryType dstType, const void *src, NCVMemoryType srcType, size_t sz, cudaStream_t cuStream)
{
    NCVStatus ncvStat;
    switch (dstType)
    {
    case NCVMemoryTypeHostPageable:
    case NCVMemoryTypeHostPinned:
        switch (srcType)
        {
        case NCVMemoryTypeHostPageable:
        case NCVMemoryTypeHostPinned:
            memcpy(dst, src, sz);
            ncvStat = NCV_SUCCESS;
            break;
        case NCVMemoryTypeDevice:
            if (cuStream != 0)
            {
                ncvAssertCUDAReturn(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToHost, cuStream), NCV_CUDA_ERROR);
            }
            else
            {
                ncvAssertCUDAReturn(cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost), NCV_CUDA_ERROR);
            }
            ncvStat = NCV_SUCCESS;
            break;
        default:
            ncvStat = NCV_MEM_RESIDENCE_ERROR;
        }
        break;
    case NCVMemoryTypeDevice:
        switch (srcType)
        {
        case NCVMemoryTypeHostPageable:
        case NCVMemoryTypeHostPinned:
            if (cuStream != 0)
            {
                ncvAssertCUDAReturn(cudaMemcpyAsync(dst, src, sz, cudaMemcpyHostToDevice, cuStream), NCV_CUDA_ERROR);
            }
            else
            {
                ncvAssertCUDAReturn(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice), NCV_CUDA_ERROR);
            }
            ncvStat = NCV_SUCCESS;
            break;
        case NCVMemoryTypeDevice:
            if (cuStream != 0)
            {
                ncvAssertCUDAReturn(cudaMemcpyAsync(dst, src, sz, cudaMemcpyDeviceToDevice, cuStream), NCV_CUDA_ERROR);
            }
            else
            {
                ncvAssertCUDAReturn(cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToDevice), NCV_CUDA_ERROR);
            }
            ncvStat = NCV_SUCCESS;
            break;
        default:
            ncvStat = NCV_MEM_RESIDENCE_ERROR;
        }
        break;
    default:
        ncvStat = NCV_MEM_RESIDENCE_ERROR;
    }

    return ncvStat;
}


NCVStatus memSegCopyHelper2D(void *dst, Ncv32u dstPitch, NCVMemoryType dstType,
                             const void *src, Ncv32u srcPitch, NCVMemoryType srcType,
                             Ncv32u widthbytes, Ncv32u height, cudaStream_t cuStream)
{
    NCVStatus ncvStat;
    switch (dstType)
    {
    case NCVMemoryTypeHostPageable:
    case NCVMemoryTypeHostPinned:
        switch (srcType)
        {
        case NCVMemoryTypeHostPageable:
        case NCVMemoryTypeHostPinned:
            for (Ncv32u i=0; i<height; i++)
            {
                memcpy((char*)dst + i * dstPitch, (char*)src + i * srcPitch, widthbytes);
            }
            ncvStat = NCV_SUCCESS;
            break;
        case NCVMemoryTypeDevice:
            if (cuStream != 0)
            {
                ncvAssertCUDAReturn(cudaMemcpy2DAsync(dst, dstPitch, src, srcPitch, widthbytes, height, cudaMemcpyDeviceToHost, cuStream), NCV_CUDA_ERROR);
            }
            else
            {
                ncvAssertCUDAReturn(cudaMemcpy2D(dst, dstPitch, src, srcPitch, widthbytes, height, cudaMemcpyDeviceToHost), NCV_CUDA_ERROR);
            }
            ncvStat = NCV_SUCCESS;
            break;
        default:
            ncvStat = NCV_MEM_RESIDENCE_ERROR;
        }
        break;
    case NCVMemoryTypeDevice:
        switch (srcType)
        {
        case NCVMemoryTypeHostPageable:
        case NCVMemoryTypeHostPinned:
            if (cuStream != 0)
            {
                ncvAssertCUDAReturn(cudaMemcpy2DAsync(dst, dstPitch, src, srcPitch, widthbytes, height, cudaMemcpyHostToDevice, cuStream), NCV_CUDA_ERROR);
            }
            else
            {
                ncvAssertCUDAReturn(cudaMemcpy2D(dst, dstPitch, src, srcPitch, widthbytes, height, cudaMemcpyHostToDevice), NCV_CUDA_ERROR);
            }
            ncvStat = NCV_SUCCESS;
            break;
        case NCVMemoryTypeDevice:
            if (cuStream != 0)
            {
                ncvAssertCUDAReturn(cudaMemcpy2DAsync(dst, dstPitch, src, srcPitch, widthbytes, height, cudaMemcpyDeviceToDevice, cuStream), NCV_CUDA_ERROR);
            }
            else
            {
                ncvAssertCUDAReturn(cudaMemcpy2D(dst, dstPitch, src, srcPitch, widthbytes, height, cudaMemcpyDeviceToDevice), NCV_CUDA_ERROR);
            }
            ncvStat = NCV_SUCCESS;
            break;
        default:
            ncvStat = NCV_MEM_RESIDENCE_ERROR;
        }
        break;
    default:
        ncvStat = NCV_MEM_RESIDENCE_ERROR;
    }

    return ncvStat;
}


//===================================================================
//
// NCVMemStackAllocator class members implementation
//
//===================================================================


NCVMemStackAllocator::NCVMemStackAllocator(Ncv32u alignment)
    :
    currentSize(0),
    _maxSize(0),
    allocBegin(NULL),
    begin(NULL),
    end(NULL),
    _memType(NCVMemoryTypeNone),
    _alignment(alignment),
    bReusesMemory(false)
{
    NcvBool bProperAlignment = (alignment & (alignment-1)) == 0;
    ncvAssertPrintCheck(bProperAlignment, "NCVMemStackAllocator ctor:: alignment not power of 2");
}


NCVMemStackAllocator::NCVMemStackAllocator(NCVMemoryType memT, size_t capacity, Ncv32u alignment, void *reusePtr)
    :
    currentSize(0),
    _maxSize(0),
    allocBegin(NULL),
    _memType(memT),
    _alignment(alignment)
{
    NcvBool bProperAlignment = (alignment & (alignment-1)) == 0;
    ncvAssertPrintCheck(bProperAlignment, "NCVMemStackAllocator ctor:: _alignment not power of 2");

    allocBegin = NULL;

    if (reusePtr == NULL)
    {
        bReusesMemory = false;
        switch (memT)
        {
        case NCVMemoryTypeDevice:
            ncvAssertCUDAReturn(cudaMalloc(&allocBegin, capacity), );
            break;
        case NCVMemoryTypeHostPinned:
            ncvAssertCUDAReturn(cudaMallocHost(&allocBegin, capacity), );
            break;
        case NCVMemoryTypeHostPageable:
            allocBegin = (Ncv8u *)malloc(capacity);
            break;
        }
    }
    else
    {
        bReusesMemory = true;
        allocBegin = (Ncv8u *)reusePtr;
    }

    if (capacity == 0)
    {
        allocBegin = (Ncv8u *)(0x1);
    }

    if (!isCounting())
    {
        begin = allocBegin;
        end = begin + capacity;
    }
}


NCVMemStackAllocator::~NCVMemStackAllocator()
{
    if (allocBegin != NULL)
    {
        ncvAssertPrintCheck(currentSize == 0, "NCVMemStackAllocator dtor:: not all objects were deallocated properly, forcing destruction");

        if (!bReusesMemory)
        {
            switch (_memType)
            {
            case NCVMemoryTypeDevice:
                ncvAssertCUDAReturn(cudaFree(allocBegin), );
                break;
            case NCVMemoryTypeHostPinned:
                ncvAssertCUDAReturn(cudaFreeHost(allocBegin), );
                break;
            case NCVMemoryTypeHostPageable:
                free(allocBegin);
                break;
            }
        }

        allocBegin = NULL;
    }
}


NCVStatus NCVMemStackAllocator::alloc(NCVMemSegment &seg, size_t size)
{
    seg.clear();
    ncvAssertReturn(isInitialized(), NCV_ALLOCATOR_BAD_ALLOC);

    size = alignUp(size, this->_alignment);
    this->currentSize += size;
    this->_maxSize = std::max(this->_maxSize, this->currentSize);

    if (!isCounting())
    {
        size_t availSize = end - begin;
        ncvAssertReturn(size <= availSize, NCV_ALLOCATOR_INSUFFICIENT_CAPACITY);
    }

    seg.begin.ptr = begin;
    seg.begin.memtype = this->_memType;
    seg.size = size;
    begin += size;

    return NCV_SUCCESS;
}


NCVStatus NCVMemStackAllocator::dealloc(NCVMemSegment &seg)
{
    ncvAssertReturn(isInitialized(), NCV_ALLOCATOR_BAD_ALLOC);
    ncvAssertReturn(seg.begin.memtype == this->_memType, NCV_ALLOCATOR_BAD_DEALLOC);
    ncvAssertReturn(seg.begin.ptr != NULL || isCounting(), NCV_ALLOCATOR_BAD_DEALLOC);
    ncvAssertReturn(seg.begin.ptr == begin - seg.size, NCV_ALLOCATOR_DEALLOC_ORDER);

    currentSize -= seg.size;
    begin -= seg.size;

    seg.clear();

    ncvAssertReturn(allocBegin <= begin, NCV_ALLOCATOR_BAD_DEALLOC);

    return NCV_SUCCESS;
}


NcvBool NCVMemStackAllocator::isInitialized(void) const
{
    return ((this->_alignment & (this->_alignment-1)) == 0) && isCounting() || this->allocBegin != NULL;
}


NcvBool NCVMemStackAllocator::isCounting(void) const
{
    return this->_memType == NCVMemoryTypeNone;
}


NCVMemoryType NCVMemStackAllocator::memType(void) const
{
    return this->_memType;
}


Ncv32u NCVMemStackAllocator::alignment(void) const
{
    return this->_alignment;
}


size_t NCVMemStackAllocator::maxSize(void) const
{
    return this->_maxSize;
}


//===================================================================
//
// NCVMemNativeAllocator class members implementation
//
//===================================================================


NCVMemNativeAllocator::NCVMemNativeAllocator(NCVMemoryType memT, Ncv32u alignment)
    :
    currentSize(0),
    _maxSize(0),
    _memType(memT),
    _alignment(alignment)
{
    ncvAssertPrintReturn(memT != NCVMemoryTypeNone, "NCVMemNativeAllocator ctor:: counting not permitted for this allocator type", );
}


NCVMemNativeAllocator::~NCVMemNativeAllocator()
{
    ncvAssertPrintCheck(currentSize == 0, "NCVMemNativeAllocator dtor:: detected memory leak");
}


NCVStatus NCVMemNativeAllocator::alloc(NCVMemSegment &seg, size_t size)
{
    seg.clear();
    ncvAssertReturn(isInitialized(), NCV_ALLOCATOR_BAD_ALLOC);

    switch (this->_memType)
    {
    case NCVMemoryTypeDevice:
        ncvAssertCUDAReturn(cudaMalloc(&seg.begin.ptr, size), NCV_CUDA_ERROR);
        break;
    case NCVMemoryTypeHostPinned:
        ncvAssertCUDAReturn(cudaMallocHost(&seg.begin.ptr, size), NCV_CUDA_ERROR);
        break;
    case NCVMemoryTypeHostPageable:
        seg.begin.ptr = (Ncv8u *)malloc(size);
        break;
    }

    this->currentSize += alignUp(size, this->_alignment);
    this->_maxSize = std::max(this->_maxSize, this->currentSize);

    seg.begin.memtype = this->_memType;
    seg.size = size;

    return NCV_SUCCESS;
}


NCVStatus NCVMemNativeAllocator::dealloc(NCVMemSegment &seg)
{
    ncvAssertReturn(isInitialized(), NCV_ALLOCATOR_BAD_ALLOC);
    ncvAssertReturn(seg.begin.memtype == this->_memType, NCV_ALLOCATOR_BAD_DEALLOC);
    ncvAssertReturn(seg.begin.ptr != NULL, NCV_ALLOCATOR_BAD_DEALLOC);

    ncvAssertReturn(currentSize >= alignUp(seg.size, this->_alignment), NCV_ALLOCATOR_BAD_DEALLOC);
    currentSize -= alignUp(seg.size, this->_alignment);

    switch (this->_memType)
    {
    case NCVMemoryTypeDevice:
        ncvAssertCUDAReturn(cudaFree(seg.begin.ptr), NCV_CUDA_ERROR);
        break;
    case NCVMemoryTypeHostPinned:
        ncvAssertCUDAReturn(cudaFreeHost(seg.begin.ptr), NCV_CUDA_ERROR);
        break;
    case NCVMemoryTypeHostPageable:
        free(seg.begin.ptr);
        break;
    }

    seg.clear();

    return NCV_SUCCESS;
}


NcvBool NCVMemNativeAllocator::isInitialized(void) const
{
    return (this->_alignment != 0);
}


NcvBool NCVMemNativeAllocator::isCounting(void) const
{
    return false;
}


NCVMemoryType NCVMemNativeAllocator::memType(void) const
{
    return this->_memType;
}


Ncv32u NCVMemNativeAllocator::alignment(void) const
{
    return this->_alignment;
}


size_t NCVMemNativeAllocator::maxSize(void) const
{
    return this->_maxSize;
}


//===================================================================
//
// Time and timer routines
//
//===================================================================


typedef struct _NcvTimeMoment NcvTimeMoment;

#if defined(_WIN32) || defined(_WIN64)

    #include <Windows.h>

    typedef struct _NcvTimeMoment
    {
        LONGLONG moment, freq;
    } NcvTimeMoment;


    static void _ncvQueryMoment(NcvTimeMoment *t)
    {
        QueryPerformanceFrequency((LARGE_INTEGER *)&(t->freq));
        QueryPerformanceCounter((LARGE_INTEGER *)&(t->moment));
    }


    double _ncvMomentToMicroseconds(NcvTimeMoment *t)
    {
        return 1000000.0 * t->moment / t->freq;
    }


    double _ncvMomentsDiffToMicroseconds(NcvTimeMoment *t1, NcvTimeMoment *t2)
    {
        return 1000000.0 * 2 * ((t2->moment) - (t1->moment)) / (t1->freq + t2->freq);
    }


    double _ncvMomentsDiffToMilliseconds(NcvTimeMoment *t1, NcvTimeMoment *t2)
    {
        return 1000.0 * 2 * ((t2->moment) - (t1->moment)) / (t1->freq + t2->freq);
    }

#elif defined(__unix__)

    #include <sys/time.h>

    typedef struct _NcvTimeMoment
    {
        struct timeval tv; 
        struct timezone tz;
    } NcvTimeMoment;


    void _ncvQueryMoment(NcvTimeMoment *t)
    {
        gettimeofday(& t->tv, & t->tz);
    }


    double _ncvMomentToMicroseconds(NcvTimeMoment *t)
    {
        return 1000000.0 * t->tv.tv_sec + (double)t->tv.tv_usec;
    }


    double _ncvMomentsDiffToMicroseconds(NcvTimeMoment *t1, NcvTimeMoment *t2)
    {
        return (((double)t2->tv.tv_sec - (double)t1->tv.tv_sec) * 1000000 + (double)t2->tv.tv_usec - (double)t1->tv.tv_usec);
    }

    double _ncvMomentsDiffToMilliseconds(NcvTimeMoment *t1, NcvTimeMoment *t2)
    {
        return ((double)t2->tv.tv_sec - (double)t1->tv.tv_sec) * 1000;
    }

#endif //#if defined(_WIN32) || defined(_WIN64)


struct _NcvTimer
{
    NcvTimeMoment t1, t2;
};


NcvTimer ncvStartTimer(void)
{
    struct _NcvTimer *t;
    t = (struct _NcvTimer *)malloc(sizeof(struct _NcvTimer));
    _ncvQueryMoment(&t->t1);
    return t;
}


double ncvEndQueryTimerUs(NcvTimer t)
{
    double res;
    _ncvQueryMoment(&t->t2);
    res = _ncvMomentsDiffToMicroseconds(&t->t1, &t->t2);
    free(t);
    return res;
}


double ncvEndQueryTimerMs(NcvTimer t)
{
    double res;
    _ncvQueryMoment(&t->t2);
    res = _ncvMomentsDiffToMilliseconds(&t->t1, &t->t2);
    free(t);
    return res;
}


//===================================================================
//
// Operations with rectangles
//
//===================================================================


//from OpenCV
void groupRectangles(std::vector<NcvRect32u> &hypotheses, int groupThreshold, double eps, std::vector<Ncv32u> *weights);


NCVStatus ncvGroupRectangles_host(NCVVector<NcvRect32u> &hypotheses,
                                  Ncv32u &numHypotheses,
                                  Ncv32u minNeighbors,
                                  Ncv32f intersectEps,
                                  NCVVector<Ncv32u> *hypothesesWeights)
{
    ncvAssertReturn(hypotheses.memType() == NCVMemoryTypeHostPageable ||
                    hypotheses.memType() == NCVMemoryTypeHostPinned, NCV_MEM_RESIDENCE_ERROR);
    if (hypothesesWeights != NULL)
    {
        ncvAssertReturn(hypothesesWeights->memType() == NCVMemoryTypeHostPageable ||
                        hypothesesWeights->memType() == NCVMemoryTypeHostPinned, NCV_MEM_RESIDENCE_ERROR);
    }

    if (numHypotheses == 0)
    {
        return NCV_SUCCESS;
    }

    std::vector<NcvRect32u> rects(numHypotheses);
    memcpy(&rects[0], hypotheses.ptr(), numHypotheses * sizeof(NcvRect32u));

    std::vector<Ncv32u> weights;
    if (hypothesesWeights != NULL)
    {
        groupRectangles(rects, minNeighbors, intersectEps, &weights);
    }
    else
    {
        groupRectangles(rects, minNeighbors, intersectEps, NULL);
    }

    numHypotheses = (Ncv32u)rects.size();
    if (numHypotheses > 0)
    {
        memcpy(hypotheses.ptr(), &rects[0], numHypotheses * sizeof(NcvRect32u));
    }

    if (hypothesesWeights != NULL)
    {
        memcpy(hypothesesWeights->ptr(), &weights[0], numHypotheses * sizeof(Ncv32u));
    }

    return NCV_SUCCESS;
}


template <class T>
static NCVStatus drawRectsWrapperHost(T *h_dst,
                                      Ncv32u dstStride,
                                      Ncv32u dstWidth,
                                      Ncv32u dstHeight,
                                      NcvRect32u *h_rects,
                                      Ncv32u numRects,
                                      T color)
{
    ncvAssertReturn(h_dst != NULL && h_rects != NULL, NCV_NULL_PTR);
    ncvAssertReturn(dstWidth > 0 && dstHeight > 0, NCV_DIMENSIONS_INVALID);
    ncvAssertReturn(dstStride >= dstWidth, NCV_INVALID_STEP);
    ncvAssertReturn(numRects != 0, NCV_SUCCESS);
    ncvAssertReturn(numRects <= dstWidth * dstHeight, NCV_DIMENSIONS_INVALID);

    for (Ncv32u i=0; i<numRects; i++)
    {
        NcvRect32u rect = h_rects[i];

        if (rect.x < dstWidth)
        {
            for (Ncv32u i=rect.y; i<rect.y+rect.height && i<dstHeight; i++)
            {
                h_dst[i*dstStride+rect.x] = color;
            }
        }
        if (rect.x+rect.width-1 < dstWidth)
        {
            for (Ncv32u i=rect.y; i<rect.y+rect.height && i<dstHeight; i++)
            {
                h_dst[i*dstStride+rect.x+rect.width-1] = color;
            }
        }
        if (rect.y < dstHeight)
        {
            for (Ncv32u j=rect.x; j<rect.x+rect.width && j<dstWidth; j++)
            {
                h_dst[rect.y*dstStride+j] = color;
            }
        }
        if (rect.y + rect.height - 1 < dstHeight)
        {
            for (Ncv32u j=rect.x; j<rect.x+rect.width && j<dstWidth; j++)
            {
                h_dst[(rect.y+rect.height-1)*dstStride+j] = color;
            }
        }
    }

    return NCV_SUCCESS;
}


NCVStatus ncvDrawRects_8u_host(Ncv8u *h_dst,
                               Ncv32u dstStride,
                               Ncv32u dstWidth,
                               Ncv32u dstHeight,
                               NcvRect32u *h_rects,
                               Ncv32u numRects,
                               Ncv8u color)
{
    return drawRectsWrapperHost(h_dst, dstStride, dstWidth, dstHeight, h_rects, numRects, color);
}


NCVStatus ncvDrawRects_32u_host(Ncv32u *h_dst,
                                Ncv32u dstStride,
                                Ncv32u dstWidth,
                                Ncv32u dstHeight,
                                NcvRect32u *h_rects,
                                Ncv32u numRects,
                                Ncv32u color)
{
    return drawRectsWrapperHost(h_dst, dstStride, dstWidth, dstHeight, h_rects, numRects, color);
}


const Ncv32u NUMTHREADS_DRAWRECTS = 32;
const Ncv32u NUMTHREADS_DRAWRECTS_LOG2 = 5;


template <class T>
__global__ void drawRects(T *d_dst,
                          Ncv32u dstStride,
                          Ncv32u dstWidth,
                          Ncv32u dstHeight,
                          NcvRect32u *d_rects,
                          Ncv32u numRects,
                          T color)
{
    Ncv32u blockId = blockIdx.y * 65535 + blockIdx.x;
    if (blockId > numRects * 4)
    {
        return;
    }

    NcvRect32u curRect = d_rects[blockId >> 2];
    NcvBool bVertical = blockId & 0x1;
    NcvBool bTopLeft = blockId & 0x2;

    Ncv32u pt0x, pt0y;
    if (bVertical)
    {
        Ncv32u numChunks = (curRect.height + NUMTHREADS_DRAWRECTS - 1) >> NUMTHREADS_DRAWRECTS_LOG2;

        pt0x = bTopLeft ? curRect.x : curRect.x + curRect.width - 1;
        pt0y = curRect.y;

        if (pt0x < dstWidth)
        {
            for (Ncv32u chunkId = 0; chunkId < numChunks; chunkId++)
            {
                Ncv32u ptY = pt0y + chunkId * NUMTHREADS_DRAWRECTS + threadIdx.x;
                if (ptY < pt0y + curRect.height && ptY < dstHeight)
                {
                    d_dst[ptY * dstStride + pt0x] = color;
                }
            }
        }
    }
    else
    {
        Ncv32u numChunks = (curRect.width + NUMTHREADS_DRAWRECTS - 1) >> NUMTHREADS_DRAWRECTS_LOG2;

        pt0x = curRect.x;
        pt0y = bTopLeft ? curRect.y : curRect.y + curRect.height - 1;

        if (pt0y < dstHeight)
        {
            for (Ncv32u chunkId = 0; chunkId < numChunks; chunkId++)
            {
                Ncv32u ptX = pt0x + chunkId * NUMTHREADS_DRAWRECTS + threadIdx.x;
                if (ptX < pt0x + curRect.width && ptX < dstWidth)
                {
                    d_dst[pt0y * dstStride + ptX] = color;
                }
            }
        }
    }
}


template <class T>
static NCVStatus drawRectsWrapperDevice(T *d_dst,
                                        Ncv32u dstStride,
                                        Ncv32u dstWidth,
                                        Ncv32u dstHeight,
                                        NcvRect32u *d_rects,
                                        Ncv32u numRects,
                                        T color,
                                        cudaStream_t cuStream)
{
    ncvAssertReturn(d_dst != NULL && d_rects != NULL, NCV_NULL_PTR);
    ncvAssertReturn(dstWidth > 0 && dstHeight > 0, NCV_DIMENSIONS_INVALID);
    ncvAssertReturn(dstStride >= dstWidth, NCV_INVALID_STEP);
    ncvAssertReturn(numRects <= dstWidth * dstHeight, NCV_DIMENSIONS_INVALID);

    if (numRects == 0)
    {
        return NCV_SUCCESS;
    }

    dim3 grid(numRects * 4);
    dim3 block(NUMTHREADS_DRAWRECTS);
    if (grid.x > 65535)
    {
        grid.y = (grid.x + 65534) / 65535;
        grid.x = 65535;
    }

    drawRects<T><<<grid, block>>>(d_dst, dstStride, dstWidth, dstHeight, d_rects, numRects, color);

    ncvAssertCUDAReturn(cudaGetLastError(), NCV_CUDA_ERROR);

    return NCV_SUCCESS;
}


NCVStatus ncvDrawRects_8u_device(Ncv8u *d_dst,
                                 Ncv32u dstStride,
                                 Ncv32u dstWidth,
                                 Ncv32u dstHeight,
                                 NcvRect32u *d_rects,
                                 Ncv32u numRects,
                                 Ncv8u color,
                                 cudaStream_t cuStream)
{
    return drawRectsWrapperDevice(d_dst, dstStride, dstWidth, dstHeight, d_rects, numRects, color, cuStream);
}


NCVStatus ncvDrawRects_32u_device(Ncv32u *d_dst,
                                  Ncv32u dstStride,
                                  Ncv32u dstWidth,
                                  Ncv32u dstHeight,
                                  NcvRect32u *d_rects,
                                  Ncv32u numRects,
                                  Ncv32u color,
                                  cudaStream_t cuStream)
{
    return drawRectsWrapperDevice(d_dst, dstStride, dstWidth, dstHeight, d_rects, numRects, color, cuStream);
}
