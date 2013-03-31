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

/* The header is for internal use and it is likely to change.
   It contains some macro definitions that are used in cxcore, cv, cvaux
   and, probably, other libraries. If you need some of this functionality,
   the safe way is to copy it into your code and rename the macros.
*/
#ifndef __OPENCV_CORE_INTERNAL_HPP__
#define __OPENCV_CORE_INTERNAL_HPP__

#define __BEGIN__ __CV_BEGIN__
#define __END__ __CV_END__
#define EXIT __CV_EXIT__

#ifdef HAVE_IPP
#  include "ipp.h"

CV_INLINE IppiSize ippiSize(int width, int height)
{
    IppiSize size = { width, height };
    return size;
}
#endif

#ifndef IPPI_CALL
#  define IPPI_CALL(func) CV_Assert((func) >= 0)
#endif

#ifdef HAVE_TBB
#  include "tbb/tbb_stddef.h"
#  if TBB_VERSION_MAJOR*100 + TBB_VERSION_MINOR >= 202
#    include "tbb/tbb.h"
#    include "tbb/task.h"
#    undef min
#    undef max
#  else
#    undef HAVE_TBB
#  endif
#endif

#ifdef HAVE_EIGEN
#  if defined __GNUC__ && defined __APPLE__
#    pragma GCC diagnostic ignored "-Wshadow"
#  endif
#  include <Eigen/Core>
#  include "opencv2/core/eigen.hpp"
#endif

#ifdef __cplusplus

namespace cv
{
#ifdef HAVE_TBB

    typedef tbb::blocked_range<int> BlockedRange;

    template<typename Body> static inline
    void parallel_for( const BlockedRange& range, const Body& body )
    {
        tbb::parallel_for(range, body);
    }

    typedef tbb::split Split;

    template<typename Body> static inline
    void parallel_reduce( const BlockedRange& range, Body& body )
    {
        tbb::parallel_reduce(range, body);
    }

    typedef tbb::concurrent_vector<Rect> ConcurrentRectVector;
#else
    class BlockedRange
    {
    public:
        BlockedRange() : _begin(0), _end(0), _grainsize(0) {}
        BlockedRange(int b, int e, int g=1) : _begin(b), _end(e), _grainsize(g) {}
        int begin() const { return _begin; }
        int end() const { return _end; }
        int grainsize() const { return _grainsize; }

    protected:
        int _begin, _end, _grainsize;
    };

    template<typename Body> static inline
    void parallel_for( const BlockedRange& range, const Body& body )
    {
        body(range);
    }
    typedef std::vector<Rect> ConcurrentRectVector;

    class Split {};

    template<typename Body> static inline
    void parallel_reduce( const BlockedRange& range, Body& body )
    {
        body(range);
    }
#endif
} //namespace cv

#define CV_INIT_ALGORITHM(classname, algname, memberinit) \
    static ::cv::Algorithm* create##classname##_hidden() \
    { \
        return new classname; \
    } \
    \
    static ::cv::AlgorithmInfo& classname##_info() \
    { \
        static ::cv::AlgorithmInfo classname##_info_var(algname, create##classname##_hidden); \
        return classname##_info_var; \
    } \
    \
    static ::cv::AlgorithmInfo& classname##_info_auto = classname##_info(); \
    \
    ::cv::AlgorithmInfo* classname::info() const \
    { \
        static volatile bool initialized = false; \
        \
        if( !initialized ) \
        { \
            initialized = true; \
            classname obj; \
            memberinit; \
        } \
        return &classname##_info(); \
    }

#endif //__cplusplus

/* default image row align (in bytes) */
#define  CV_DEFAULT_IMAGE_ROW_ALIGN  4

/* the alignment of all the allocated buffers */
#define  CV_MALLOC_ALIGN    16

/* default alignment for dynamic data strucutures, resided in storages. */
#define  CV_STRUCT_ALIGN    ((int)sizeof(double))

/* default storage block size */
#define  CV_STORAGE_BLOCK_SIZE   ((1<<16) - 128)

/* default memory block for sparse array elements */
#define  CV_SPARSE_MAT_BLOCK    (1<<12)

/* initial hash table size */
#define  CV_SPARSE_HASH_SIZE0    (1<<10)

/* maximal average node_count/hash_size ratio beyond which hash table is resized */
#define  CV_SPARSE_HASH_RATIO    3

/****************************************************************************************\
*                                  Common declarations                                   *
\****************************************************************************************/

#ifdef __GNUC__
#  define CV_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined _MSC_VER
#  define CV_DECL_ALIGNED(x) __declspec(align(x))
#else
#  define CV_DECL_ALIGNED(x)
#endif

#ifndef CV_IMPL
#  define CV_IMPL CV_EXTERN_C
#endif

#define  CV_ORIGIN_TL  0
#define  CV_ORIGIN_BL  1

/* IEEE754 constants and macros */
#define  CV_TOGGLE_FLT(x) ((x)^((int)(x) < 0 ? 0x7fffffff : 0))
#define  CV_TOGGLE_DBL(x) \
    ((x)^((int64)(x) < 0 ? CV_BIG_INT(0x7fffffffffffffff) : 0))

//TODO: remove after dropping sorts
#define  CV_LT(a, b)    ((a) < (b))

CV_INLINE void* cvAlignPtr( const void* ptr, int align CV_DEFAULT(32) )
{
    assert( (align & (align-1)) == 0 );
    return (void*)( ((size_t)ptr + align - 1) & ~(size_t)(align-1) );
}

CV_INLINE int cvAlign( int size, int align )
{
    assert( (align & (align-1)) == 0 && size < INT_MAX );
    return (size + align - 1) & -align;
}

CV_INLINE  CvSize  cvGetMatSize( const CvMat* mat )
{
    CvSize size;
    size.width = mat->cols;
    size.height = mat->rows;
    return size;
}

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))
#define  CV_FLT_TO_FIX(x,n)  cvRound((x)*(1<<(n)))

/****************************************************************************************\
*                     Structures and macros for integration with IPP                     *
\****************************************************************************************/

/* IPP-compatible return codes */
typedef enum CvStatus
{
    CV_BADMEMBLOCK_ERR          = -113,
    CV_INPLACE_NOT_SUPPORTED_ERR= -112,
    CV_UNMATCHED_ROI_ERR        = -111,
    CV_NOTFOUND_ERR             = -110,
    CV_BADCONVERGENCE_ERR       = -109,

    CV_BADDEPTH_ERR             = -107,
    CV_BADROI_ERR               = -106,
    CV_BADHEADER_ERR            = -105,
    CV_UNMATCHED_FORMATS_ERR    = -104,
    CV_UNSUPPORTED_COI_ERR      = -103,
    CV_UNSUPPORTED_CHANNELS_ERR = -102,
    CV_UNSUPPORTED_DEPTH_ERR    = -101,
    CV_UNSUPPORTED_FORMAT_ERR   = -100,

    CV_BADARG_ERR               = -49,  //ipp comp
    CV_NOTDEFINED_ERR           = -48,  //ipp comp

    CV_BADCHANNELS_ERR          = -47,  //ipp comp
    CV_BADRANGE_ERR             = -44,  //ipp comp
    CV_BADSTEP_ERR              = -29,  //ipp comp

    CV_BADFLAG_ERR              =  -12,
    CV_DIV_BY_ZERO_ERR          =  -11, //ipp comp
    CV_BADCOEF_ERR              =  -10,

    CV_BADFACTOR_ERR            =  -7,
    CV_BADPOINT_ERR             =  -6,
    CV_BADSCALE_ERR             =  -4,
    CV_OUTOFMEM_ERR             =  -3,
    CV_NULLPTR_ERR              =  -2,
    CV_BADSIZE_ERR              =  -1,
    CV_NO_ERR                   =   0,
    CV_OK                       =   CV_NO_ERR
}
CvStatus;

#endif // __OPENCV_CORE_INTERNAL_HPP__
