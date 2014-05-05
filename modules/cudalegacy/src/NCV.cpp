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

#include "precomp.hpp"

//==============================================================================
//
// Error handling helpers
//
//==============================================================================

namespace
{
    #define error_entry(entry)  { entry, #entry }

    struct ErrorEntry
    {
        int code;
        const char* str;
    };

    struct ErrorEntryComparer
    {
        int code;
        ErrorEntryComparer(int code_) : code(code_) {}
        bool operator()(const ErrorEntry& e) const { return e.code == code; }
    };

    //////////////////////////////////////////////////////////////////////////
    // NCV errors

    const ErrorEntry ncv_errors [] =
    {
        error_entry( NCV_SUCCESS ),
        error_entry( NCV_UNKNOWN_ERROR ),
        error_entry( NCV_CUDA_ERROR ),
        error_entry( NCV_NPP_ERROR ),
        error_entry( NCV_FILE_ERROR ),
        error_entry( NCV_NULL_PTR ),
        error_entry( NCV_INCONSISTENT_INPUT ),
        error_entry( NCV_TEXTURE_BIND_ERROR ),
        error_entry( NCV_DIMENSIONS_INVALID ),
        error_entry( NCV_INVALID_ROI ),
        error_entry( NCV_INVALID_STEP ),
        error_entry( NCV_INVALID_SCALE ),
        error_entry( NCV_INVALID_SCALE ),
        error_entry( NCV_ALLOCATOR_NOT_INITIALIZED ),
        error_entry( NCV_ALLOCATOR_BAD_ALLOC ),
        error_entry( NCV_ALLOCATOR_BAD_DEALLOC ),
        error_entry( NCV_ALLOCATOR_INSUFFICIENT_CAPACITY ),
        error_entry( NCV_ALLOCATOR_DEALLOC_ORDER ),
        error_entry( NCV_ALLOCATOR_BAD_REUSE ),
        error_entry( NCV_MEM_COPY_ERROR ),
        error_entry( NCV_MEM_RESIDENCE_ERROR ),
        error_entry( NCV_MEM_INSUFFICIENT_CAPACITY ),
        error_entry( NCV_HAAR_INVALID_PIXEL_STEP ),
        error_entry( NCV_HAAR_TOO_MANY_FEATURES_IN_CLASSIFIER ),
        error_entry( NCV_HAAR_TOO_MANY_FEATURES_IN_CASCADE ),
        error_entry( NCV_HAAR_TOO_LARGE_FEATURES ),
        error_entry( NCV_HAAR_XML_LOADING_EXCEPTION ),
        error_entry( NCV_NOIMPL_HAAR_TILTED_FEATURES ),
        error_entry( NCV_WARNING_HAAR_DETECTIONS_VECTOR_OVERFLOW ),
        error_entry( NPPST_SUCCESS ),
        error_entry( NPPST_ERROR ),
        error_entry( NPPST_CUDA_KERNEL_EXECUTION_ERROR ),
        error_entry( NPPST_NULL_POINTER_ERROR ),
        error_entry( NPPST_TEXTURE_BIND_ERROR ),
        error_entry( NPPST_MEMCPY_ERROR ),
        error_entry( NPPST_MEM_ALLOC_ERR ),
        error_entry( NPPST_MEMFREE_ERR ),
        error_entry( NPPST_INVALID_ROI ),
        error_entry( NPPST_INVALID_STEP ),
        error_entry( NPPST_INVALID_SCALE ),
        error_entry( NPPST_MEM_INSUFFICIENT_BUFFER ),
        error_entry( NPPST_MEM_RESIDENCE_ERROR ),
        error_entry( NPPST_MEM_INTERNAL_ERROR )
    };

    const size_t ncv_error_num = sizeof(ncv_errors) / sizeof(ncv_errors[0]);
}

cv::String cv::cuda::getNcvErrorMessage(int code)
{
    size_t idx = std::find_if(ncv_errors, ncv_errors + ncv_error_num, ErrorEntryComparer(code)) - ncv_errors;

    const char* msg = (idx != ncv_error_num) ? ncv_errors[idx].str : "Unknown error code";
    String str = cv::format("%s [Code = %d]", msg, code);

    return str;
}


static void stdDebugOutput(const cv::String &msg)
{
    std::cout << msg.c_str() << std::endl;
}


static NCVDebugOutputHandler *debugOutputHandler = stdDebugOutput;


void ncvDebugOutput(const cv::String &msg)
{
    debugOutputHandler(msg);
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


NCVMemStackAllocator::NCVMemStackAllocator(Ncv32u alignment_) :
    _memType(NCVMemoryTypeNone),
    _alignment(alignment_),
    allocBegin(NULL),
    begin(NULL),
    end(NULL),
    currentSize(0),
    _maxSize(0),
    bReusesMemory(false)
{
    NcvBool bProperAlignment = (alignment_ & (alignment_ - 1)) == 0;
    ncvAssertPrintCheck(bProperAlignment, "NCVMemStackAllocator ctor:: alignment not power of 2");
}


NCVMemStackAllocator::NCVMemStackAllocator(NCVMemoryType memT, size_t capacity, Ncv32u alignment_, void *reusePtr) :
    _memType(memT),
    _alignment(alignment_),
    allocBegin(NULL),
    currentSize(0),
    _maxSize(0)
{
    NcvBool bProperAlignment = (alignment_ & (alignment_ - 1)) == 0;
    ncvAssertPrintCheck(bProperAlignment, "NCVMemStackAllocator ctor:: _alignment not power of 2");
    ncvAssertPrintCheck(memT != NCVMemoryTypeNone, "NCVMemStackAllocator ctor:: Incorrect allocator type");

    allocBegin = NULL;

    if (reusePtr == NULL && capacity != 0)
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
        default:;
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

        if (!bReusesMemory && (allocBegin != (Ncv8u *)(0x1)))
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
            default:;
            }
        }

        allocBegin = NULL;
    }
}


NCVStatus NCVMemStackAllocator::alloc(NCVMemSegment &seg, size_t size)
{
    seg.clear();
    ncvAssertReturn(isInitialized(), NCV_ALLOCATOR_BAD_ALLOC);

    size = alignUp(static_cast<Ncv32u>(size), this->_alignment);
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
    return (((this->_alignment & (this->_alignment-1)) == 0) && isCounting()) || this->allocBegin != NULL;
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


NCVMemNativeAllocator::NCVMemNativeAllocator(NCVMemoryType memT, Ncv32u alignment_) :
    _memType(memT),
    _alignment(alignment_),
    currentSize(0),
    _maxSize(0)
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
    default:;
    }

    this->currentSize += alignUp(static_cast<Ncv32u>(size), this->_alignment);
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

    ncvAssertReturn(currentSize >= alignUp(static_cast<Ncv32u>(seg.size), this->_alignment), NCV_ALLOCATOR_BAD_DEALLOC);
    currentSize -= alignUp(static_cast<Ncv32u>(seg.size), this->_alignment);

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
    default:;
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

#elif defined(__GNUC__)

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

struct RectConvert
{
    cv::Rect operator()(const NcvRect32u& nr) const { return cv::Rect(nr.x, nr.y, nr.width, nr.height); }
    NcvRect32u operator()(const cv::Rect& nr) const
    {
        NcvRect32u rect;
        rect.x = nr.x;
        rect.y = nr.y;
        rect.width = nr.width;
        rect.height = nr.height;
        return rect;
    }
};

static void groupRectangles(std::vector<NcvRect32u> &hypotheses, int groupThreshold, double eps, std::vector<Ncv32u> *weights)
{
#ifndef HAVE_OPENCV_OBJDETECT
    (void) hypotheses;
    (void) groupThreshold;
    (void) eps;
    (void) weights;
    CV_Error(cv::Error::StsNotImplemented, "This functionality requires objdetect module");
#else
    std::vector<cv::Rect> rects(hypotheses.size());
    std::transform(hypotheses.begin(), hypotheses.end(), rects.begin(), RectConvert());

    if (weights)
    {
        std::vector<int> weights_int;
        weights_int.assign(weights->begin(), weights->end());
        cv::groupRectangles(rects, weights_int, groupThreshold, eps);
    }
    else
    {
        cv::groupRectangles(rects, groupThreshold, eps);
    }
    std::transform(rects.begin(), rects.end(), hypotheses.begin(), RectConvert());
    hypotheses.resize(rects.size());
#endif
}



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
            for (Ncv32u each=rect.y; each<rect.y+rect.height && each<dstHeight; each++)
            {
                h_dst[each*dstStride+rect.x] = color;
            }
        }
        if (rect.x+rect.width-1 < dstWidth)
        {
            for (Ncv32u each=rect.y; each<rect.y+rect.height && each<dstHeight; each++)
            {
                h_dst[each*dstStride+rect.x+rect.width-1] = color;
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
