#ifndef OPENCV_CORE_HAL_INTERFACE_H
#define OPENCV_CORE_HAL_INTERFACE_H

#include "cvtype.h"

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
