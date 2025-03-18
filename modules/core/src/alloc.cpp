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

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utils/configuration.private.hpp>

#define CV__ALLOCATOR_STATS_LOG(...) CV_LOG_VERBOSE(NULL, 0, "alloc.cpp: " << __VA_ARGS__)
#include "opencv2/core/utils/allocator_stats.impl.hpp"
#undef CV__ALLOCATOR_STATS_LOG

//#define OPENCV_ALLOC_ENABLE_STATISTICS


#ifdef HAVE_POSIX_MEMALIGN
#include <stdlib.h>
#elif defined HAVE_MALLOC_H
#include <malloc.h>
#endif

#ifdef OPENCV_ALLOC_ENABLE_STATISTICS
#define OPENCV_ALLOC_STATISTICS_LIMIT 4096  // don't track buffers less than N bytes
#include <map>
#endif
CV_EXPORTS void alignedFree(void* ptr);
CV_EXPORTS void* alignedMalloc(size_t size, int alignment);
namespace cv {

static void* OutOfMemoryError(size_t size)
{
    CV_Error_(cv::Error::StsNoMem, ("Failed to allocate %llu bytes", (unsigned long long)size));
}

CV_EXPORTS cv::utils::AllocatorStatisticsInterface& getAllocatorStatistics();

static cv::utils::AllocatorStatistics allocator_stats;

cv::utils::AllocatorStatisticsInterface& getAllocatorStatistics()
{
    return allocator_stats;
}

#if defined HAVE_POSIX_MEMALIGN || defined HAVE_MEMALIGN || defined HAVE_WIN32_ALIGNED_MALLOC
static bool readMemoryAlignmentParameter()
{
    bool value = true;
#if defined(__GLIBC__) && defined(__linux__) \
    && !defined(CV_STATIC_ANALYSIS) \
    && !defined(OPENCV_ENABLE_MEMORY_SANITIZER) \
    && !defined(FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION)  /* oss-fuzz */ \
    && !defined(_WIN32)  /* MinGW? */
    {
        // https://github.com/opencv/opencv/issues/15526
        value = false;
    }
#endif
    value = cv::utils::getConfigurationParameterBool("OPENCV_ENABLE_MEMALIGN", value);  // should not call fastMalloc() internally
    // TODO add checks for valgrind, ASAN if value == false
    return value;
}

#if defined _MSC_VER
#pragma warning(suppress:4714)  // preventive: const marked as __forceinline not inlined
static __forceinline
#else
static inline
#endif
bool isAlignedAllocationEnabled()
{
    // use construct on first use idiom https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
    // details: https://github.com/opencv/opencv/issues/15691
    static bool useMemalign = readMemoryAlignmentParameter();
    return useMemalign;
}

// need for this static const is disputed; retaining as it doesn't cause harm
static const bool g_force_initialization_memalign_flag
#if defined __GNUC__
    __attribute__((unused))
#endif
    = isAlignedAllocationEnabled();
#endif

#ifdef OPENCV_ALLOC_ENABLE_STATISTICS
static inline
void* fastMalloc_(size_t size)
#else
void* fastMalloc(size_t size)
#endif
{
#ifdef HAVE_POSIX_MEMALIGN
    if (isAlignedAllocationEnabled())
    {
        void* ptr = NULL;
        if(posix_memalign(&ptr, CV_MALLOC_ALIGN, size))
            ptr = NULL;
        if(!ptr)
            return OutOfMemoryError(size);
        return ptr;
    }
#elif defined HAVE_MEMALIGN
    if (isAlignedAllocationEnabled())
    {
        void* ptr = memalign(CV_MALLOC_ALIGN, size);
        if(!ptr)
            return OutOfMemoryError(size);
        return ptr;
    }
#elif defined HAVE_WIN32_ALIGNED_MALLOC
void* alignedMalloc(size_t size, int alignment)
{
#if defined(HAVE_POSIX_MEMALIGN)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
#elif defined(HAVE_MEMALIGN)
    void* ptr = memalign(alignment, size);
    if (!ptr)
        return nullptr;
    return ptr;
#elif defined(HAVE_WIN32_ALIGNED_MALLOC)
    return _aligned_malloc(size, alignment);
#else
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + alignment);
    if (!udata)
        return nullptr;
    uchar** adata = alignPtr((uchar**)udata + 1, alignment);
    adata[-1] = udata;
    return adata;
#endif
}
    if (isAlignedAllocationEnabled())
    {
        void* ptr = _aligned_malloc(size, CV_MALLOC_ALIGN);
        if(!ptr)
            return OutOfMemoryError(size);
        return ptr;
    }
#endif
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
    if(!udata)
        return OutOfMemoryError(size);
    uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

#ifdef OPENCV_ALLOC_ENABLE_STATISTICS
static inline
void fastFree_(void* ptr)
#else
CV_EXPORTS void fastFree(void* ptr)
#endif
{
#if defined HAVE_POSIX_MEMALIGN || defined HAVE_MEMALIGN
    if (isAlignedAllocationEnabled())
    {
        free(ptr);
        return;
    }
#elif defined HAVE_WIN32_ALIGNED_MALLOC
CV_EXPORTS void alignedFree(void* ptr)
{
    if (ptr) _aligned_free(ptr);
}
   if (isAlignedAllocationEnabled())
    {
        alignedFree(ptr);
        return;
    }
#endif
    if(ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        CV_DbgAssert(udata < (uchar*)ptr &&
               ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*)+CV_MALLOC_ALIGN));
        free(udata);
    }
}

#ifdef OPENCV_ALLOC_ENABLE_STATISTICS

static
Mutex& getAllocationStatisticsMutex()
{
    static Mutex* p_alloc_mutex = allocSingletonNew<Mutex>();
    CV_Assert(p_alloc_mutex);
    return *p_alloc_mutex;
}

static std::map<void*, size_t> allocated_buffers;  // guarded by getAllocationStatisticsMutex()

void* fastMalloc(size_t size)
{
    void* res = fastMalloc_(size);
    if (res && size >= OPENCV_ALLOC_STATISTICS_LIMIT)
    {
        cv::AutoLock lock(getAllocationStatisticsMutex());
        allocated_buffers.insert(std::make_pair(res, size));
        allocator_stats.onAllocate(size);
    }
    return res;
}

void fastFree(void* ptr)
{
    {
        cv::AutoLock lock(getAllocationStatisticsMutex());
        std::map<void*, size_t>::iterator i = allocated_buffers.find(ptr);
        if (i != allocated_buffers.end())
        {
            size_t size = i->second;
            allocator_stats.onFree(size);
            allocated_buffers.erase(i);
        }
    }
    fastFree_(ptr);
}

#endif // OPENCV_ALLOC_ENABLE_STATISTICS

} // namespace

CV_IMPL void* cvAlloc( size_t size )
{
    return cv::fastMalloc( size );
}

CV_IMPL void cvFree_( void* ptr )
{
    cv::fastFree( ptr );
}

/* End of file. */
