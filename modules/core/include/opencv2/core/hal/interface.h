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
     typedef std::uint32_t uint;
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

#define CV_CN_MAX     512
#define CV_CN_SHIFT   3
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_USRTYPE1 7

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

#define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
#define CV_MAKE_TYPE CV_MAKETYPE

#define CV_8UC1 CV_MAKETYPE(CV_8U,1)
#define CV_8UC2 CV_MAKETYPE(CV_8U,2)
#define CV_8UC3 CV_MAKETYPE(CV_8U,3)
#define CV_8UC4 CV_MAKETYPE(CV_8U,4)
#define CV_8UC(n) CV_MAKETYPE(CV_8U,(n))

#define CV_8SC1 CV_MAKETYPE(CV_8S,1)
#define CV_8SC2 CV_MAKETYPE(CV_8S,2)
#define CV_8SC3 CV_MAKETYPE(CV_8S,3)
#define CV_8SC4 CV_MAKETYPE(CV_8S,4)
#define CV_8SC(n) CV_MAKETYPE(CV_8S,(n))

#define CV_16UC1 CV_MAKETYPE(CV_16U,1)
#define CV_16UC2 CV_MAKETYPE(CV_16U,2)
#define CV_16UC3 CV_MAKETYPE(CV_16U,3)
#define CV_16UC4 CV_MAKETYPE(CV_16U,4)
#define CV_16UC(n) CV_MAKETYPE(CV_16U,(n))

#define CV_16SC1 CV_MAKETYPE(CV_16S,1)
#define CV_16SC2 CV_MAKETYPE(CV_16S,2)
#define CV_16SC3 CV_MAKETYPE(CV_16S,3)
#define CV_16SC4 CV_MAKETYPE(CV_16S,4)
#define CV_16SC(n) CV_MAKETYPE(CV_16S,(n))

#define CV_32SC1 CV_MAKETYPE(CV_32S,1)
#define CV_32SC2 CV_MAKETYPE(CV_32S,2)
#define CV_32SC3 CV_MAKETYPE(CV_32S,3)
#define CV_32SC4 CV_MAKETYPE(CV_32S,4)
#define CV_32SC(n) CV_MAKETYPE(CV_32S,(n))

#define CV_32FC1 CV_MAKETYPE(CV_32F,1)
#define CV_32FC2 CV_MAKETYPE(CV_32F,2)
#define CV_32FC3 CV_MAKETYPE(CV_32F,3)
#define CV_32FC4 CV_MAKETYPE(CV_32F,4)
#define CV_32FC(n) CV_MAKETYPE(CV_32F,(n))

#define CV_64FC1 CV_MAKETYPE(CV_64F,1)
#define CV_64FC2 CV_MAKETYPE(CV_64F,2)
#define CV_64FC3 CV_MAKETYPE(CV_64F,3)
#define CV_64FC4 CV_MAKETYPE(CV_64F,4)
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
