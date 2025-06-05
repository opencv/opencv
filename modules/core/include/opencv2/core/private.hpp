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

#ifndef OPENCV_CORE_PRIVATE_HPP
#define OPENCV_CORE_PRIVATE_HPP

#ifndef __OPENCV_BUILD
#  error this is a private header which should not be used from outside of the OpenCV library
#endif

#include "opencv2/core.hpp"
#include "cvconfig.h"

#include <opencv2/core/utils/trace.hpp>

#ifdef ENABLE_INSTRUMENTATION
#include "opencv2/core/utils/instrumentation.hpp"
#endif

#ifdef HAVE_EIGEN
#  if defined __GNUC__ && defined __APPLE__
#    pragma GCC diagnostic ignored "-Wshadow"
#  endif
#  if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable:4701)  // potentially uninitialized local variable
#    pragma warning(disable:4702)  // unreachable code
#    pragma warning(disable:4714)  // const marked as __forceinline not inlined
#  endif
#  include <Eigen/Core>
#  if defined(_MSC_VER)
#    pragma warning(pop)
#  endif
#  include "opencv2/core/eigen.hpp"
#endif

//! @cond IGNORED

namespace cv
{
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

    // Returns a static string if there is a parallel framework,
    // NULL otherwise.
    CV_EXPORTS const char* currentParallelFramework();
} //namespace cv

/****************************************************************************************\
*                                  Common declarations                                   *
\****************************************************************************************/

/* the alignment of all the allocated buffers */
#define  CV_MALLOC_ALIGN    64

/* IEEE754 constants and macros */
#define  CV_TOGGLE_FLT(x) ((x)^((int)(x) < 0 ? 0x7fffffff : 0))
#define  CV_TOGGLE_DBL(x) ((x)^((int64)(x) < 0 ? CV_BIG_INT(0x7fffffffffffffff) : 0))

static inline void* cvAlignPtr( const void* ptr, int align = 32 )
{
    CV_DbgAssert ( (align & (align-1)) == 0 );
    return (void*)( ((size_t)ptr + align - 1) & ~(size_t)(align-1) );
}

static inline int cvAlign( int size, int align )
{
    CV_DbgAssert( (align & (align-1)) == 0 && size < INT_MAX );
    return (size + align - 1) & -align;
}

#ifdef IPL_DEPTH_8U
static inline cv::Size cvGetMatSize( const CvMat* mat )
{
    return cv::Size(mat->cols, mat->rows);
}
#endif

namespace cv
{
CV_EXPORTS void scalarToRawData(const cv::Scalar& s, void* buf, int type, int unroll_to = 0);

//! Allocate memory buffers which will not be freed, ease filtering memcheck issues. Uses fastMalloc() call.
CV_EXPORTS void* allocSingletonBuffer(size_t size);

//! Allocate memory buffers which will not be freed, ease filtering memcheck issues. Uses fastMalloc() call
template <typename T> static inline
T* allocSingleton(size_t count = 1) { return static_cast<T*>(allocSingletonBuffer(sizeof(T) * count)); }

//! Allocate memory buffers which will not be freed, ease filtering memcheck issues. Uses generic malloc() call.
CV_EXPORTS void* allocSingletonNewBuffer(size_t size);

//! Allocate memory buffers which will not be freed, ease filtering memcheck issues.  Uses generic malloc() call.
template <typename T> static inline
T* allocSingletonNew() { return new(allocSingletonNewBuffer(sizeof(T))) T(); }

} // namespace

#if 1 // TODO: Remove in OpenCV 4.x

// property implementation macros

#define CV_IMPL_PROPERTY_RO(type, name, member) \
    inline type get##name() const { return member; }

#define CV_HELP_IMPL_PROPERTY(r_type, w_type, name, member) \
    CV_IMPL_PROPERTY_RO(r_type, name, member) \
    inline void set##name(w_type val) { member = val; }

#define CV_HELP_WRAP_PROPERTY(r_type, w_type, name, internal_name, internal_obj) \
    r_type get##name() const { return internal_obj.get##internal_name(); } \
    void set##name(w_type val) { internal_obj.set##internal_name(val); }

#define CV_IMPL_PROPERTY(type, name, member) CV_HELP_IMPL_PROPERTY(type, type, name, member)
#define CV_IMPL_PROPERTY_S(type, name, member) CV_HELP_IMPL_PROPERTY(type, const type &, name, member)

#define CV_WRAP_PROPERTY(type, name, internal_name, internal_obj)  CV_HELP_WRAP_PROPERTY(type, type, name, internal_name, internal_obj)
#define CV_WRAP_PROPERTY_S(type, name, internal_name, internal_obj) CV_HELP_WRAP_PROPERTY(type, const type &, name, internal_name, internal_obj)

#define CV_WRAP_SAME_PROPERTY(type, name, internal_obj) CV_WRAP_PROPERTY(type, name, name, internal_obj)
#define CV_WRAP_SAME_PROPERTY_S(type, name, internal_obj) CV_WRAP_PROPERTY_S(type, name, name, internal_obj)

#endif

/****************************************************************************************\
*                     Structures and macros for integration with IPP                     *
\****************************************************************************************/

#define OPENCV_IPP_REDUCE_SIZE 1

// Temporary disabled named IPP region. Accuracy
#define IPP_DISABLE_PYRAMIDS_UP         1 // Different results
#define IPP_DISABLE_PYRAMIDS_DOWN       1 // Different results
#define IPP_DISABLE_PYRAMIDS_BUILD      1 // Different results
#define IPP_DISABLE_WARPAFFINE          1 // Different results
// #define IPP_DISABLE_WARPPERSPECTIVE     0 // Different results
#define IPP_DISABLE_REMAP               1 // Different results
#define IPP_DISABLE_YUV_RGB             1 // accuracy difference
#define IPP_DISABLE_RGB_YUV             1 // breaks OCL accuracy tests
#define IPP_DISABLE_RGB_HSV             1 // breaks OCL accuracy tests
#define IPP_DISABLE_RGB_LAB             1 // breaks OCL accuracy tests
#define IPP_DISABLE_LAB_RGB             1 // breaks OCL accuracy tests
#define IPP_DISABLE_RGB_XYZ             1 // big accuracy difference
#define IPP_DISABLE_XYZ_RGB             1 // big accuracy difference
#define IPP_DISABLE_HOUGH               1 // improper integration/results
#define IPP_DISABLE_FILTER2D_BIG_MASK   1 // different results on masks > 7x7
#define IPP_DISABLE_NORM_8U             1 // accuracy difference in perf test sanity check

// Temporary disabled named IPP region. Performance
#define IPP_DISABLE_PERF_COPYMAKE       1 // performance variations
#define IPP_DISABLE_PERF_TRUE_DIST_MT   1 // cv::distanceTransform OpenCV MT performance is better
#define IPP_DISABLE_PERF_CANNY_MT       1 // cv::Canny OpenCV MT performance is better

#ifdef HAVE_IPP
#include "ippversion.h"
#ifndef IPP_VERSION_UPDATE // prior to 7.1
#define IPP_VERSION_UPDATE 0
#endif

#define IPP_VERSION_X100 (IPP_VERSION_MAJOR * 100 + IPP_VERSION_MINOR*10 + IPP_VERSION_UPDATE)

#ifdef HAVE_IPP_ICV
#define ICV_BASE
#if IPP_VERSION_X100 >= 201700
#include "ippicv.h"
#else
#include "ipp.h"
#endif
#else
#include "ipp.h"
#endif
#ifdef HAVE_IPP_IW
#  if defined(__OPENCV_BUILD) && defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wstrict-prototypes"
#  endif
#  if defined(__OPENCV_BUILD) && defined(__GNUC__) && __GNUC__ >= 5
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsuggest-override"
#  endif
#include "iw++/iw.hpp"
#  ifdef HAVE_IPP_IW_LL
#include "iw/iw_ll.h"
#  endif
#  if defined(__OPENCV_BUILD) && defined(__GNUC__) && __GNUC__ >= 5
#  pragma GCC diagnostic pop
#  endif
#  if defined(__OPENCV_BUILD) && defined(__clang__)
#  pragma clang diagnostic pop
#  endif
#endif

#if IPP_VERSION_X100 >= 201700
#define CV_IPP_MALLOC(SIZE) ippMalloc_L(SIZE)
#else
#define CV_IPP_MALLOC(SIZE) ippMalloc((int)SIZE)
#endif

#define setIppErrorStatus() cv::ipp::setIppStatus(-1, CV_Func, __FILE__, __LINE__)

#if IPP_VERSION_X100 >= 201700
#define ippCPUID_AVX512_SKX (ippCPUID_AVX512F|ippCPUID_AVX512CD|ippCPUID_AVX512VL|ippCPUID_AVX512BW|ippCPUID_AVX512DQ)
#define ippCPUID_AVX512_KNL (ippCPUID_AVX512F|ippCPUID_AVX512CD|ippCPUID_AVX512PF|ippCPUID_AVX512ER)
#else
#define ippCPUID_AVX512_SKX 0xFFFFFFFF
#define ippCPUID_AVX512_KNL 0xFFFFFFFF
#endif

namespace cv
{
namespace ipp
{
CV_EXPORTS   unsigned long long getIppTopFeatures(); // Returns top major enabled IPP feature flag
}
}

static inline IppiSize ippiSize(size_t width, size_t height)
{
    IppiSize size = { (int)width, (int)height };
    return size;
}

static inline IppiSize ippiSize(const cv::Size & _size)
{
    IppiSize size = { _size.width, _size.height };
    return size;
}

#if IPP_VERSION_X100 >= 201700
static inline IppiSizeL ippiSizeL(size_t width, size_t height)
{
    IppiSizeL size = { (IppSizeL)width, (IppSizeL)height };
    return size;
}

static inline IppiSizeL ippiSizeL(const cv::Size & _size)
{
    IppiSizeL size = { _size.width, _size.height };
    return size;
}
#endif

static inline IppiPoint ippiPoint(const cv::Point & _point)
{
    IppiPoint point = { _point.x, _point.y };
    return point;
}

static inline IppiPoint ippiPoint(int x, int y)
{
    IppiPoint point = { x, y };
    return point;
}

static inline IppiBorderType ippiGetBorderType(int borderTypeNI)
{
    return borderTypeNI == cv::BORDER_CONSTANT    ? ippBorderConst   :
           borderTypeNI == cv::BORDER_TRANSPARENT ? ippBorderTransp  :
           borderTypeNI == cv::BORDER_REPLICATE   ? ippBorderRepl    :
           borderTypeNI == cv::BORDER_REFLECT_101 ? ippBorderMirror  :
           (IppiBorderType)-1;
}

static inline IppiMaskSize ippiGetMaskSize(int kx, int ky)
{
    return (kx == 1 && ky == 3) ? ippMskSize1x3 :
           (kx == 1 && ky == 5) ? ippMskSize1x5 :
           (kx == 3 && ky == 1) ? ippMskSize3x1 :
           (kx == 3 && ky == 3) ? ippMskSize3x3 :
           (kx == 5 && ky == 1) ? ippMskSize5x1 :
           (kx == 5 && ky == 5) ? ippMskSize5x5 :
           (IppiMaskSize)-1;
}

static inline IppDataType ippiGetDataType(int depth)
{
    depth = CV_MAT_DEPTH(depth);
    return depth == CV_8U ? ipp8u :
        depth == CV_8S ? ipp8s :
        depth == CV_16U ? ipp16u :
        depth == CV_16S ? ipp16s :
        depth == CV_32S ? ipp32s :
        depth == CV_32F ? ipp32f :
        depth == CV_64F ? ipp64f :
        (IppDataType)-1;
}

static inline int ippiSuggestThreadsNum(size_t width, size_t height, size_t elemSize, double multiplier)
{
    int threads = cv::getNumThreads();
    if(threads > 1 && height >= 64)
    {
        size_t opMemory = (int)(width*height*elemSize*multiplier);
        int l2cache = 0;
#if IPP_VERSION_X100 >= 201700
        ippGetL2CacheSize(&l2cache);
#endif
        if(!l2cache)
            l2cache = 1 << 18;

        return IPP_MAX(1, (IPP_MIN((int)(opMemory/l2cache), threads)));
    }
    return 1;
}

static inline int ippiSuggestThreadsNum(const cv::Mat &image, double multiplier)
{
    return ippiSuggestThreadsNum(image.cols, image.rows, image.elemSize(), multiplier);
}

#ifdef HAVE_IPP_IW
static inline bool ippiCheckAnchor(int x, int y, int kernelWidth, int kernelHeight)
{
    if(x != ((kernelWidth-1)/2) || y != ((kernelHeight-1)/2))
        return 0;
    else
        return 1;
}

static inline ::ipp::IwiSize ippiGetSize(const cv::Size & size)
{
    return ::ipp::IwiSize((IwSize)size.width, (IwSize)size.height);
}

static inline IwiDerivativeType ippiGetDerivType(int dx, int dy, bool nvert)
{
    return (dx == 1 && dy == 0) ? ((nvert)?iwiDerivNVerFirst:iwiDerivVerFirst) :
           (dx == 0 && dy == 1) ? iwiDerivHorFirst :
           (dx == 2 && dy == 0) ? iwiDerivVerSecond :
           (dx == 0 && dy == 2) ? iwiDerivHorSecond :
           (IwiDerivativeType)-1;
}

static inline void ippiGetImage(const cv::Mat &src, ::ipp::IwiImage &dst)
{
    ::ipp::IwiBorderSize inMemBorder;
    if(src.isSubmatrix()) // already have physical border
    {
        cv::Size  origSize;
        cv::Point offset;
        src.locateROI(origSize, offset);

        inMemBorder.left   = (IwSize)offset.x;
        inMemBorder.top    = (IwSize)offset.y;
        inMemBorder.right  = (IwSize)(origSize.width - src.cols - offset.x);
        inMemBorder.bottom = (IwSize)(origSize.height - src.rows - offset.y);
    }

    dst.Init(ippiSize(src.size()), ippiGetDataType(src.depth()), src.channels(), inMemBorder, (void*)src.ptr(), src.step);
}

static inline ::ipp::IwiImage ippiGetImage(const cv::Mat &src)
{
    ::ipp::IwiImage image;
    ippiGetImage(src, image);
    return image;
}

static inline IppiBorderType ippiGetBorder(::ipp::IwiImage &image, int ocvBorderType, ipp::IwiBorderSize &borderSize)
{
    int            inMemFlags = 0;
    IppiBorderType border     = ippiGetBorderType(ocvBorderType & ~cv::BORDER_ISOLATED);
    if((int)border == -1)
        return (IppiBorderType)0;

    if(!(ocvBorderType & cv::BORDER_ISOLATED))
    {
        if(image.m_inMemSize.left)
        {
            if(image.m_inMemSize.left >= borderSize.left)
                inMemFlags |= ippBorderInMemLeft;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.left = 0;
        if(image.m_inMemSize.top)
        {
            if(image.m_inMemSize.top >= borderSize.top)
                inMemFlags |= ippBorderInMemTop;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.top = 0;
        if(image.m_inMemSize.right)
        {
            if(image.m_inMemSize.right >= borderSize.right)
                inMemFlags |= ippBorderInMemRight;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.right = 0;
        if(image.m_inMemSize.bottom)
        {
            if(image.m_inMemSize.bottom >= borderSize.bottom)
                inMemFlags |= ippBorderInMemBottom;
            else
                return (IppiBorderType)0;
        }
        else
            borderSize.bottom = 0;
    }
    else
        borderSize.left = borderSize.right = borderSize.top = borderSize.bottom = 0;

    return (IppiBorderType)(border|inMemFlags);
}

static inline ::ipp::IwValueFloat ippiGetValue(const cv::Scalar &scalar)
{
    return ::ipp::IwValueFloat(scalar[0], scalar[1], scalar[2], scalar[3]);
}

static inline int ippiSuggestThreadsNum(const ::ipp::IwiImage &image, double multiplier)
{
    return ippiSuggestThreadsNum(image.m_size.width, image.m_size.height, image.m_typeSize*image.m_channels, multiplier);
}
#endif

// IPP temporary buffer helper
template<typename T>
class IppAutoBuffer
{
public:
    IppAutoBuffer() { m_size = 0; m_pBuffer = NULL; }
    explicit IppAutoBuffer(size_t size) { m_size = 0; m_pBuffer = NULL; allocate(size); }
    ~IppAutoBuffer() { deallocate(); }
    T* allocate(size_t size)   { if(m_size < size) { deallocate(); m_pBuffer = (T*)CV_IPP_MALLOC(size); m_size = size; } return m_pBuffer; }
    void deallocate() { if(m_pBuffer) { ippFree(m_pBuffer); m_pBuffer = NULL; } m_size = 0; }
    inline T* get() { return (T*)m_pBuffer;}
    inline operator T* () { return (T*)m_pBuffer;}
    inline operator const T* () const { return (const T*)m_pBuffer;}
private:
    // Disable copy operations
    IppAutoBuffer(IppAutoBuffer &) {}
    IppAutoBuffer& operator =(const IppAutoBuffer &) {return *this;}

    size_t m_size;
    T*     m_pBuffer;
};

// Extracts border interpolation type without flags
#if IPP_VERSION_X100 >= 201700
#define IPP_BORDER_INTER(BORDER) (IppiBorderType)((BORDER)&0xF|((((BORDER)&ippBorderInMem) == ippBorderInMem)?ippBorderInMem:0));
#else
#define IPP_BORDER_INTER(BORDER) (IppiBorderType)((BORDER)&0xF);
#endif

#else
#define IPP_VERSION_X100 0
#endif

#if defined HAVE_IPP
#if IPP_VERSION_X100 >= 900
#define IPP_INITIALIZER(FEAT)                           \
{                                                       \
    if(FEAT)                                            \
        ippSetCpuFeatures(FEAT);                        \
    else                                                \
        ippInit();                                      \
}
#elif IPP_VERSION_X100 >= 800
#define IPP_INITIALIZER(FEAT)                           \
{                                                       \
    ippInit();                                          \
}
#else
#define IPP_INITIALIZER(FEAT)                           \
{                                                       \
    ippStaticInit();                                    \
}
#endif

#ifdef CVAPI_EXPORTS
#define IPP_INITIALIZER_AUTO                            \
struct __IppInitializer__                               \
{                                                       \
    __IppInitializer__()                                \
    {IPP_INITIALIZER(cv::ipp::getIppFeatures())}        \
};                                                      \
static struct __IppInitializer__ __ipp_initializer__;
#else
#define IPP_INITIALIZER_AUTO
#endif
#else
#define IPP_INITIALIZER
#define IPP_INITIALIZER_AUTO
#endif

#define CV_IPP_CHECK_COND (cv::ipp::useIPP())
#define CV_IPP_CHECK() if(CV_IPP_CHECK_COND)

#ifdef HAVE_IPP

#ifdef CV_IPP_RUN_VERBOSE
#define CV_IPP_RUN_(condition, func, ...)                                   \
    {                                                                       \
        if (cv::ipp::useIPP() && (condition) && (func))                     \
        {                                                                   \
            printf("%s: IPP implementation is running\n", CV_Func);         \
            fflush(stdout);                                                 \
            CV_IMPL_ADD(CV_IMPL_IPP);                                       \
            return __VA_ARGS__;                                             \
        }                                                                   \
        else                                                                \
        {                                                                   \
            printf("%s: Plain implementation is running\n", CV_Func);       \
            fflush(stdout);                                                 \
        }                                                                   \
    }
#elif defined CV_IPP_RUN_ASSERT
#define CV_IPP_RUN_(condition, func, ...)                                   \
    {                                                                       \
        if (cv::ipp::useIPP() && (condition))                               \
        {                                                                   \
            CV__TRACE_REGION_("IPP:" #func, CV_TRACE_NS::details::REGION_FLAG_IMPL_IPP) \
            if(func)                                                        \
            {                                                               \
                CV_IMPL_ADD(CV_IMPL_IPP);                                   \
            }                                                               \
            else                                                            \
            {                                                               \
                setIppErrorStatus();                                        \
                CV_Error(cv::Error::StsAssert, #func);                      \
            }                                                               \
            return __VA_ARGS__;                                             \
        }                                                                   \
    }
#else
#define CV_IPP_RUN_(condition, func, ...)                                   \
        if (cv::ipp::useIPP() && (condition))                               \
        {                                                                   \
            CV__TRACE_REGION_("IPP:" #func, CV_TRACE_NS::details::REGION_FLAG_IMPL_IPP) \
            if(func)                                                        \
            {                                                               \
                CV_IMPL_ADD(CV_IMPL_IPP);                                   \
                return __VA_ARGS__;                                         \
            }                                                               \
        }
#endif
#else
#define CV_IPP_RUN_(condition, func, ...)
#endif

#define CV_IPP_RUN_FAST(func, ...) CV_IPP_RUN_(true, func, __VA_ARGS__)
#define CV_IPP_RUN(condition, func, ...) CV_IPP_RUN_((condition), (func), __VA_ARGS__)


#ifndef IPPI_CALL
#  define IPPI_CALL(func) CV_Assert((func) >= 0)
#endif

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

#ifdef ENABLE_INSTRUMENTATION
namespace cv
{
namespace instr
{
struct InstrTLSStruct
{
    InstrTLSStruct()
    {
        pCurrentNode = NULL;
    }
    InstrNode* pCurrentNode;
};

class InstrStruct
{
public:
    InstrStruct()
    {
        useInstr    = false;
        flags       = FLAGS_MAPPING;
        maxDepth    = 0;

        rootNode.m_payload = NodeData("ROOT", NULL, 0, NULL, false, TYPE_GENERAL, IMPL_PLAIN);
        tlsStruct.get()->pCurrentNode = &rootNode;
    }

    Mutex mutexCreate;
    Mutex mutexCount;

    bool       useInstr;
    int        flags;
    int        maxDepth;
    InstrNode  rootNode;
    TLSData<InstrTLSStruct> tlsStruct;
};

class CV_EXPORTS IntrumentationRegion
{
public:
    IntrumentationRegion(const char* funName, const char* fileName, int lineNum, void *retAddress, bool alwaysExpand, TYPE instrType = TYPE_GENERAL, IMPL implType = IMPL_PLAIN);
    ~IntrumentationRegion();

private:
    bool    m_disabled; // region status
    uint64  m_regionTicks;
};

CV_EXPORTS InstrStruct& getInstrumentStruct();
InstrTLSStruct&         getInstrumentTLSStruct();
CV_EXPORTS InstrNode*   getCurrentNode();
}
}

#ifdef _WIN32
#define CV_INSTRUMENT_GET_RETURN_ADDRESS _ReturnAddress()
#else
#define CV_INSTRUMENT_GET_RETURN_ADDRESS __builtin_extract_return_addr(__builtin_return_address(0))
#endif

// Instrument region
#define CV_INSTRUMENT_REGION_META(NAME, ALWAYS_EXPAND, TYPE, IMPL)        ::cv::instr::IntrumentationRegion  CVAUX_CONCAT(__instr_region__, __LINE__) (NAME, __FILE__, __LINE__, CV_INSTRUMENT_GET_RETURN_ADDRESS, ALWAYS_EXPAND, TYPE, IMPL);
#define CV_INSTRUMENT_REGION_CUSTOM_META(NAME, ALWAYS_EXPAND, TYPE, IMPL)\
    void *CVAUX_CONCAT(__curr_address__, __LINE__) = [&]() {return CV_INSTRUMENT_GET_RETURN_ADDRESS;}();\
    ::cv::instr::IntrumentationRegion CVAUX_CONCAT(__instr_region__, __LINE__) (NAME, __FILE__, __LINE__, CVAUX_CONCAT(__curr_address__, __LINE__), false, ::cv::instr::TYPE_GENERAL, ::cv::instr::IMPL_PLAIN);
// Instrument functions with non-void return type
#define CV_INSTRUMENT_FUN_RT_META(TYPE, IMPL, ERROR_COND, FUN, ...) ([&]()\
{\
    if(::cv::instr::useInstrumentation()){\
        ::cv::instr::IntrumentationRegion __instr__(#FUN, __FILE__, __LINE__, NULL, false, TYPE, IMPL);\
        try{\
            auto instrStatus = ((FUN)(__VA_ARGS__));\
            if(ERROR_COND){\
                ::cv::instr::getCurrentNode()->m_payload.m_funError = true;\
                CV_INSTRUMENT_MARK_META(IMPL, #FUN " - BadExit");\
            }\
            return instrStatus;\
        }catch(...){\
            ::cv::instr::getCurrentNode()->m_payload.m_funError = true;\
            CV_INSTRUMENT_MARK_META(IMPL, #FUN " - BadExit");\
            throw;\
        }\
    }else{\
        return ((FUN)(__VA_ARGS__));\
    }\
}())
// Instrument functions with void return type
#define CV_INSTRUMENT_FUN_RV_META(TYPE, IMPL, FUN, ...) ([&]()\
{\
    if(::cv::instr::useInstrumentation()){\
        ::cv::instr::IntrumentationRegion __instr__(#FUN, __FILE__, __LINE__, NULL, false, TYPE, IMPL);\
        try{\
            (FUN)(__VA_ARGS__);\
        }catch(...){\
            ::cv::instr::getCurrentNode()->m_payload.m_funError = true;\
            CV_INSTRUMENT_MARK_META(IMPL, #FUN "- BadExit");\
            throw;\
        }\
    }else{\
        (FUN)(__VA_ARGS__);\
    }\
}())
// Instrumentation information marker
#define CV_INSTRUMENT_MARK_META(IMPL, NAME, ...) {::cv::instr::IntrumentationRegion __instr_mark__(NAME, __FILE__, __LINE__, NULL, false, ::cv::instr::TYPE_MARKER, IMPL);}

///// General instrumentation
// General OpenCV region instrumentation macro
#define CV_INSTRUMENT_REGION_();             CV_INSTRUMENT_REGION_META(__FUNCTION__, false, ::cv::instr::TYPE_GENERAL, ::cv::instr::IMPL_PLAIN)
// Custom OpenCV region instrumentation macro
#define CV_INSTRUMENT_REGION_NAME(NAME)     CV_INSTRUMENT_REGION_CUSTOM_META(NAME,  false, ::cv::instr::TYPE_GENERAL, ::cv::instr::IMPL_PLAIN)
// Instrumentation for parallel_for_ or other regions which forks and gathers threads
#define CV_INSTRUMENT_REGION_MT_FORK();      CV_INSTRUMENT_REGION_META(__FUNCTION__, true,  ::cv::instr::TYPE_GENERAL, ::cv::instr::IMPL_PLAIN);

///// IPP instrumentation
// Wrapper region instrumentation macro
#define CV_INSTRUMENT_REGION_IPP();          CV_INSTRUMENT_REGION_META(__FUNCTION__, false, ::cv::instr::TYPE_WRAPPER, ::cv::instr::IMPL_IPP)
// Function instrumentation macro
#define CV_INSTRUMENT_FUN_IPP(FUN, ...)     CV_INSTRUMENT_FUN_RT_META(::cv::instr::TYPE_FUN, ::cv::instr::IMPL_IPP, instrStatus < 0, FUN, __VA_ARGS__)
// Diagnostic markers
#define CV_INSTRUMENT_MARK_IPP(NAME)        CV_INSTRUMENT_MARK_META(::cv::instr::IMPL_IPP, NAME)

///// OpenCL instrumentation
// Wrapper region instrumentation macro
#define CV_INSTRUMENT_REGION_OPENCL();              CV_INSTRUMENT_REGION_META(__FUNCTION__, false, ::cv::instr::TYPE_WRAPPER, ::cv::instr::IMPL_OPENCL)
// OpenCL kernel compilation wrapper
#define CV_INSTRUMENT_REGION_OPENCL_COMPILE(NAME)  CV_INSTRUMENT_REGION_META(NAME, false, ::cv::instr::TYPE_WRAPPER, ::cv::instr::IMPL_OPENCL)
// OpenCL kernel run wrapper
#define CV_INSTRUMENT_REGION_OPENCL_RUN(NAME)      CV_INSTRUMENT_REGION_META(NAME, false, ::cv::instr::TYPE_FUN, ::cv::instr::IMPL_OPENCL)
// Diagnostic markers
#define CV_INSTRUMENT_MARK_OPENCL(NAME)            CV_INSTRUMENT_MARK_META(::cv::instr::IMPL_OPENCL, NAME)
#else
#define CV_INSTRUMENT_REGION_META(...)

#define CV_INSTRUMENT_REGION_();                            CV_TRACE_FUNCTION()
#define CV_INSTRUMENT_REGION_NAME(...)                     CV_TRACE_REGION(__VA_ARGS__)
#define CV_INSTRUMENT_REGION_MT_FORK();

#define CV_INSTRUMENT_REGION_IPP();                         CV__TRACE_REGION_("IPP", CV_TRACE_NS::details::REGION_FLAG_IMPL_IPP)
#define CV_INSTRUMENT_FUN_IPP(FUN, ...) ((FUN)(__VA_ARGS__))
#define CV_INSTRUMENT_MARK_IPP(...)

#define CV_INSTRUMENT_REGION_OPENCL();                      CV__TRACE_REGION_("OpenCL", CV_TRACE_NS::details::REGION_FLAG_IMPL_OPENCL)
#define CV_INSTRUMENT_REGION_OPENCL_COMPILE(...)
#define CV_INSTRUMENT_REGION_OPENCL_RUN(...)
#define CV_INSTRUMENT_MARK_OPENCL(...)
#endif

#ifdef __CV_AVX_GUARD
#define CV_INSTRUMENT_REGION() __CV_AVX_GUARD CV_INSTRUMENT_REGION_();
#else
#define CV_INSTRUMENT_REGION() CV_INSTRUMENT_REGION_();
#endif

namespace cv {

namespace utils {

//! @addtogroup core_utils
//! @{

/** @brief Try to find requested data file

Search directories:

1. Directories passed via `addDataSearchPath()`
2. Check path specified by configuration parameter with "_HINT" suffix (name of environment variable).
3. Check path specified by configuration parameter (name of environment variable).
   If parameter value is not empty and nothing is found then stop searching.
4. Detects build/install path based on:
   a. current working directory (CWD)
   b. and/or binary module location (opencv_core/opencv_world, doesn't work with static linkage)
5. Scan `<source>/{,data}` directories if build directory is detected or the current directory is in source tree.
6. Scan `<install>/share/OpenCV` directory if install directory is detected.

@param relative_path Relative path to data file
@param required Specify "file not found" handling.
       If true, function prints information message and raises cv::Exception.
       If false, function returns empty result
@param configuration_parameter specify configuration parameter name. Default NULL value means "OPENCV_DATA_PATH".
@return Returns path (absolute or relative to the current directory) or empty string if file is not found

@note Implementation is not thread-safe.
*/
CV_EXPORTS
cv::String findDataFile(const cv::String& relative_path, bool required = true,
                        const char* configuration_parameter = NULL);

/** @overload
@param relative_path Relative path to data file
@param configuration_parameter specify configuration parameter name. Default NULL value means "OPENCV_DATA_PATH".
@param search_paths override addDataSearchPath() settings.
@param subdir_paths override addDataSearchSubDirectory() settings.
@return Returns path (absolute or relative to the current directory) or empty string if file is not found

@note Implementation is not thread-safe.
*/
CV_EXPORTS
cv::String findDataFile(const cv::String& relative_path,
                        const char* configuration_parameter,
                        const std::vector<String>* search_paths,
                        const std::vector<String>* subdir_paths);

/** @brief Override default search data path by adding new search location

Use this only to override default behavior
Passed paths are used in LIFO order.

@param path Path to used samples data

@note Implementation is not thread-safe.
*/
CV_EXPORTS void addDataSearchPath(const cv::String& path);

/** @brief Append default search data sub directory

General usage is to add OpenCV modules name (`<opencv_contrib>/modules/<name>/data` -> `modules/<name>/data` + `<name>/data`).
Passed subdirectories are used in LIFO order.

@param subdir samples data sub directory

@note Implementation is not thread-safe.
*/
CV_EXPORTS void addDataSearchSubDirectory(const cv::String& subdir);

/** @brief Retrieve location of OpenCV libraries or current executable
 */
CV_EXPORTS bool getBinLocation(std::string& dst);

#if defined(_WIN32)
/** @brief Retrieve location of OpenCV libraries or current executable

@note WIN32 only
 */
CV_EXPORTS bool getBinLocation(std::wstring& dst);
#endif

//! @}

} // namespace utils
} // namespace cv

//! @endcond

#endif // OPENCV_CORE_PRIVATE_HPP
