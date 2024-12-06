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
#include <cstdint>
#else
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
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
typedef int8_t schar;
typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
typedef int64_t int64;
typedef uint64_t uint64;

#define CV_BIG_INT(n)   n##LL
#define CV_BIG_UINT(n)  n##ULL

typedef int16_t cv_hal_f16;
typedef int16_t cv_hal_bf16;
//! @}

#define CV_USRTYPE1 (void)"CV_USRTYPE1 support has been dropped in OpenCV 4.0"

#define CV_CN_MAX     128
#define CV_CN_SHIFT   5
#define CV_DEPTH_MAX  (1 << CV_CN_SHIFT)

#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
#define CV_16F  7
#define CV_16BF 8
#define CV_Bool 9
#define CV_64U  10
#define CV_64S  11
#define CV_32U  12
#define CV_DEPTH_CURR_MAX 13

#define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)
#define CV_IS_INT_TYPE(flags)   (((1 << CV_MAT_DEPTH(flags)) & 0x1e1f) != 0)
#define CV_IS_FLOAT_TYPE(flags) (((1 << CV_MAT_DEPTH(flags)) & 0x1e0) != 0)

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

#define CV_16FC1 CV_MAKETYPE(CV_16F,1)
#define CV_16FC2 CV_MAKETYPE(CV_16F,2)
#define CV_16FC3 CV_MAKETYPE(CV_16F,3)
#define CV_16FC4 CV_MAKETYPE(CV_16F,4)
#define CV_16FC(n) CV_MAKETYPE(CV_16F,(n))

#define CV_64SC1 CV_MAKETYPE(CV_64S,1)
#define CV_64SC2 CV_MAKETYPE(CV_64S,2)
#define CV_64SC3 CV_MAKETYPE(CV_64S,3)
#define CV_64SC4 CV_MAKETYPE(CV_64S,4)
#define CV_64SC(n) CV_MAKETYPE(CV_64S,(n))

#define CV_64UC1 CV_MAKETYPE(CV_64U,1)
#define CV_64UC2 CV_MAKETYPE(CV_64U,2)
#define CV_64UC3 CV_MAKETYPE(CV_64U,3)
#define CV_64UC4 CV_MAKETYPE(CV_64U,4)
#define CV_64UC(n) CV_MAKETYPE(CV_64U,(n))

#define CV_BoolC1 CV_MAKETYPE(CV_Bool,1)
#define CV_BoolC2 CV_MAKETYPE(CV_Bool,2)
#define CV_BoolC3 CV_MAKETYPE(CV_Bool,3)
#define CV_BoolC4 CV_MAKETYPE(CV_Bool,4)
#define CV_BoolC(n) CV_MAKETYPE(CV_Bool,(n))

#define CV_32UC1 CV_MAKETYPE(CV_32U,1)
#define CV_32UC2 CV_MAKETYPE(CV_32U,2)
#define CV_32UC3 CV_MAKETYPE(CV_32U,3)
#define CV_32UC4 CV_MAKETYPE(CV_32U,4)
#define CV_32UC(n) CV_MAKETYPE(CV_32U,(n))

#define CV_16BFC1 CV_MAKETYPE(CV_16BF,1)
#define CV_16BFC2 CV_MAKETYPE(CV_16BF,2)
#define CV_16BFC3 CV_MAKETYPE(CV_16BF,3)
#define CV_16BFC4 CV_MAKETYPE(CV_16BF,4)
#define CV_16BFC(n) CV_MAKETYPE(CV_16BF,(n))

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
