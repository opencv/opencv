#ifndef OPENCV_CORE_HAL_INTERFACE_H
#define OPENCV_CORE_HAL_INTERFACE_H

//! @addtogroup core_hal_interface
//! @{

//! @name Return codes
//! @{
#define CV_HAL_ERROR_OK 0
#define CV_HAL_ERROR_NOT_IMPLEMENTED 1
#define CV_HAL_ERROR_UNKNOWN -1
//! @}

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#include <stdbool.h>
#endif

//! @name Data types
//! primitive types
//! - schar  - signed 1 byte integer
//! - uchar  - unsigned 1 byte integer
//! - short  - signed 2 byte integer
//! - ushort - unsigned 2 byte integer
//! - int    - signed 4 byte integer
//! - uint   - unsigned 4 byte integer
//! - int64  - signed 8 byte integer
//! - uint64 - unsigned 8 byte integer
//! @{
#if !defined _MSC_VER && !defined __BORLANDC__
#  if defined __cplusplus && __cplusplus >= 201103L && !defined __APPLE__
#    include <cstdint>
#    ifdef __NEWLIB__
        typedef unsigned int uint;
#    else
        typedef std::uint32_t uint;
#    endif
#  else
#    include <stdint.h>
     typedef uint32_t uint;
#  endif
#else
   typedef unsigned uint;
#endif

typedef signed char schar;

#ifndef __IPL_H__
   typedef unsigned char uchar;
   typedef unsigned short ushort;
#endif

#if defined _MSC_VER || defined __BORLANDC__
   typedef __int64 int64;
   typedef unsigned __int64 uint64;
#  define CV_BIG_INT(n)   n##I64
#  define CV_BIG_UINT(n)  n##UI64
#else
   typedef int64_t int64;
   typedef uint64_t uint64;
#  define CV_BIG_INT(n)   n##LL
#  define CV_BIG_UINT(n)  n##ULL
#endif

#define __CV_MAX_DEPTH_0(m, n)    (m != 7 || n <= 3 ? m : n) /* CV_16F workaround */
#define __CV_MAX_DEPTH_1(d1, d2)  __CV_MAX_DEPTH_0(std::max(d1, d2), std::min(d1, d2))
#define __CV_MAX_DEPTH_2(d1, d2)  __CV_MAX_DEPTH_1(static_cast<int>(d1), static_cast<int>(d2))
#define __CV_MAX_DEPTH_3(d, ...)  __CV_EXPAND(__CV_MAX_DEPTH_2(d, __CV_MAX_DEPTH_2(__VA_ARGS__)))
#define __CV_MAX_DEPTH_4(d, ...)  __CV_EXPAND(__CV_MAX_DEPTH_2(d, __CV_MAX_DEPTH_3(__VA_ARGS__)))
#define CV_MAX_DEPTH(...)         __CV_EXPAND(static_cast<ElemType>(__CV_CAT(__CV_MAX_DEPTH_, __CV_VA_NUM_ARGS(__VA_ARGS__)) (__VA_ARGS__)))

#define __CV_MIN_DEPTH_0(m, n)    (m == 7 && n >= 4 ? m : n) /* CV_16F workaround */
#define __CV_MIN_DEPTH_1(d1, d2)  __CV_MIN_DEPTH_0(std::max(d1, d2), std::min(d1, d2))
#define __CV_MIN_DEPTH_2(d1, d2)  __CV_MIN_DEPTH_1(static_cast<int>(d1), static_cast<int>(d2))
#define __CV_MIN_DEPTH_3(d, ...)  __CV_EXPAND(__CV_MIN_DEPTH_2(d, __CV_MIN_DEPTH_2(__VA_ARGS__)))
#define __CV_MIN_DEPTH_4(d, ...)  __CV_EXPAND(__CV_MIN_DEPTH_2(d, __CV_MIN_DEPTH_3(__VA_ARGS__)))
#define CV_MIN_DEPTH(...)         __CV_EXPAND(static_cast<ElemType>(__CV_CAT(__CV_MIN_DEPTH_, __CV_VA_NUM_ARGS(__VA_ARGS__)) (__VA_ARGS__)))

#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define __CV_MAT_DEPTH(flags)   (static_cast<int>(flags) & CV_MAT_DEPTH_MASK)
#define CV_MAT_DEPTH(flags)     static_cast<ElemType>(__CV_MAT_DEPTH(flags))
#define __CV_MAKETYPE(depth,cn) (__CV_MAT_DEPTH(depth) | (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKETYPE(depth,cn)   static_cast<ElemType>(__CV_MAKETYPE(depth,cn))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_TYPE_SAFE_API
#define CV_TYPE_COMPATIBLE_API
#ifdef CV_DISABLE_TYPE_SAFE_API
#  undef CV_TYPE_SAFE_API
#endif
#ifdef CV_DISABLE_TYPE_COMPATIBLE_API
#  undef CV_TYPE_COMPATIBLE_API
#endif

//Define transnational API availble only when Compatible API is set
#ifdef CV_TYPE_COMPATIBLE_API
#  define CV_TRANSNATIONAL_API
#endif
//TODO: Remove above block after whole library factorization


//#define OPENCV_ENABLE_DEPRECATED_WARNING_ELEMDEPTH_ELEMTYPE_OVERLOAD
#define OPENCV_DISABLE_DEPRECATED_WARNING_INT_ELEMTYPE_OVERLOAD


#if defined(CV_TYPE_SAFE_API) && defined(__cplusplus)
enum MagicFlag {
    CV_MAGIC_FLAG_NONE = 0
};

enum ElemType {
    CV_8U       = 0,
    CV_8S       = 1,
    CV_16U      = 2,
    CV_16S      = 3,
    CV_16F      = 7,
    CV_32S      = 4,
    CV_32F      = 5,
    CV_64F      = 6,

    CV_8UC1 = __CV_MAKETYPE(CV_8U, 1),
    CV_8UC2 = __CV_MAKETYPE(CV_8U, 2),
    CV_8UC3 = __CV_MAKETYPE(CV_8U, 3),
    CV_8UC4 = __CV_MAKETYPE(CV_8U, 4),

    CV_8SC1 = __CV_MAKETYPE(CV_8S, 1),
    CV_8SC2 = __CV_MAKETYPE(CV_8S, 2),
    CV_8SC3 = __CV_MAKETYPE(CV_8S, 3),
    CV_8SC4 = __CV_MAKETYPE(CV_8S, 4),

    CV_16UC1 = __CV_MAKETYPE(CV_16U, 1),
    CV_16UC2 = __CV_MAKETYPE(CV_16U, 2),
    CV_16UC3 = __CV_MAKETYPE(CV_16U, 3),
    CV_16UC4 = __CV_MAKETYPE(CV_16U, 4),

    CV_16SC1 = __CV_MAKETYPE(CV_16S, 1),
    CV_16SC2 = __CV_MAKETYPE(CV_16S, 2),
    CV_16SC3 = __CV_MAKETYPE(CV_16S, 3),
    CV_16SC4 = __CV_MAKETYPE(CV_16S, 4),

    CV_16FC1 = __CV_MAKETYPE(CV_16F, 1),
    CV_16FC2 = __CV_MAKETYPE(CV_16F, 2),
    CV_16FC3 = __CV_MAKETYPE(CV_16F, 3),
    CV_16FC4 = __CV_MAKETYPE(CV_16F, 4),

    CV_32SC1 = __CV_MAKETYPE(CV_32S, 1),
    CV_32SC2 = __CV_MAKETYPE(CV_32S, 2),
    CV_32SC3 = __CV_MAKETYPE(CV_32S, 3),
    CV_32SC4 = __CV_MAKETYPE(CV_32S, 4),

    CV_32FC1 = __CV_MAKETYPE(CV_32F, 1),
    CV_32FC2 = __CV_MAKETYPE(CV_32F, 2),
    CV_32FC3 = __CV_MAKETYPE(CV_32F, 3),
    CV_32FC4 = __CV_MAKETYPE(CV_32F, 4),

    CV_64FC1 = __CV_MAKETYPE(CV_64F, 1),
    CV_64FC2 = __CV_MAKETYPE(CV_64F, 2),
    CV_64FC3 = __CV_MAKETYPE(CV_64F, 3),
    CV_64FC4 = __CV_MAKETYPE(CV_64F, 4),

    CV_SEQ_ELTYPE_PTR = __CV_MAKETYPE(CV_8U, 8 /*sizeof(void*)*/),
};
#define CV_TYPE_AUTO      static_cast<ElemType>(-1)
#define CV_TYPE_UNDEFINED static_cast<ElemType>(-2)


#else // defined(CV_TYPE_SAFE_API) && defined(__cplusplus)

typedef int MagicFlag;
#define CV_MAGIC_FLAG_NONE 0

typedef int ElemType;
#define CV_TYPE_AUTO -1
#define CV_8U       0
#define CV_8S       1
#define CV_16U      2
#define CV_16S      3
#define CV_16F      7
#define CV_32S      4
#define CV_32F      5
#define CV_64F      6

typedef int ElemType;
#define CV_TYPE_AUTO      -1
#define CV_TYPE_UNDEFINED -2

#define CV_SEQ_ELTYPE_PTR CV_MAKE_TYPE(CV_8U, 8 /*sizeof(void*)*/)

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)

#define CV_16FC1 CV_MAKETYPE(CV_16F,1)
#define CV_16FC2 CV_MAKETYPE(CV_16F,2)
#define CV_16FC3 CV_MAKETYPE(CV_16F,3)
#define CV_16FC4 CV_MAKETYPE(CV_16F,4)

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)

#endif // defined(CV_TYPE_SAFE_API) && defined(__cplusplus)

#define CV_USRTYPE1 (void)"CV_USRTYPE1 support has been dropped in OpenCV 4.0"

#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))
#define CV_16FC(n) CV_MAKETYPE(CV_16F,(n))
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))
#define CV_64FC(n) CV_MAKETYPE(CV_64F,(n))

//! @}

//! @name Comparison operation
//! @sa cv::CmpTypes
//! @{
#define CV_HAL_CMP_EQ 0
#define CV_HAL_CMP_GT 1
#define CV_HAL_CMP_GE 2
#define CV_HAL_CMP_LT 3
#define CV_HAL_CMP_LE 4
#define CV_HAL_CMP_NE 5
//! @}

//! @name Border processing modes
//! @sa cv::BorderTypes
//! @{
#define CV_HAL_BORDER_CONSTANT 0
#define CV_HAL_BORDER_REPLICATE 1
#define CV_HAL_BORDER_REFLECT 2
#define CV_HAL_BORDER_WRAP 3
#define CV_HAL_BORDER_REFLECT_101 4
#define CV_HAL_BORDER_TRANSPARENT 5
#define CV_HAL_BORDER_ISOLATED 16
//! @}

//! @name DFT flags
//! @{
#define CV_HAL_DFT_INVERSE        1
#define CV_HAL_DFT_SCALE          2
#define CV_HAL_DFT_ROWS           4
#define CV_HAL_DFT_COMPLEX_OUTPUT 16
#define CV_HAL_DFT_REAL_OUTPUT    32
#define CV_HAL_DFT_TWO_STAGE      64
#define CV_HAL_DFT_STAGE_COLS    128
#define CV_HAL_DFT_IS_CONTINUOUS 512
#define CV_HAL_DFT_IS_INPLACE 1024
//! @}

//! @name SVD flags
//! @{
#define CV_HAL_SVD_NO_UV    1
#define CV_HAL_SVD_SHORT_UV 2
#define CV_HAL_SVD_MODIFY_A 4
#define CV_HAL_SVD_FULL_UV  8
//! @}

//! @name Gemm flags
//! @{
#define CV_HAL_GEMM_1_T 1
#define CV_HAL_GEMM_2_T 2
#define CV_HAL_GEMM_3_T 4
//! @}

//! @}

#endif
