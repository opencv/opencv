// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_CV_DEFS_HPP
#define OPENCV_GAPI_CV_DEFS_HPP

#if defined(GAPI_STANDALONE)

// Simulate OpenCV definitions taken from various
// OpenCV interface headers if G-API is built in a
// standalone mode.

// interface.h:

typedef unsigned char uchar;
typedef          char schar;

typedef unsigned short ushort;

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

// cvdef.h:

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
#define CV_MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)
#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)
#define CV_MAT_CONT_FLAG_SHIFT  14
#define CV_MAT_CONT_FLAG        (1 << CV_MAT_CONT_FLAG_SHIFT)
#define CV_IS_MAT_CONT(flags)   ((flags) & CV_MAT_CONT_FLAG)
#define CV_IS_CONT_MAT          CV_IS_MAT_CONT
#define CV_SUBMAT_FLAG_SHIFT    15
#define CV_SUBMAT_FLAG          (1 << CV_SUBMAT_FLAG_SHIFT)
#define CV_IS_SUBMAT(flags)     ((flags) & CV_MAT_SUBMAT_FLAG)

///** Size of each channel item,
//   0x8442211 = 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
//#define CV_ELEM_SIZE1(type) \
//    ((((sizeof(size_t)<<28)|0x8442211) >> CV_MAT_DEPTH(type)*4) & 15)

#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)

/** 0x3a50 = 11 10 10 01 01 00 00 ~ array of log2(sizeof(arr_type_elem)) */
#define CV_ELEM_SIZE(type) \
    (CV_MAT_CN(type) << ((((sizeof(size_t)/4+1)*16384|0x3a50) >> CV_MAT_DEPTH(type)*2) & 3))

#ifndef CV_OVERRIDE
#  define CV_OVERRIDE override
#endif

// base.h:
namespace cv
{
enum BorderTypes {
    BORDER_CONSTANT    = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
    BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
    BORDER_WRAP        = 3, //!< `cdefgh|abcdefgh|abcdefg`
    BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
    BORDER_TRANSPARENT = 5, //!< `uvwxyz|abcdefgh|ijklmno`

    BORDER_REFLECT101  = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
    BORDER_DEFAULT     = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
    BORDER_ISOLATED    = 16 //!< do not look outside of ROI
};
// imgproc.hpp:
enum InterpolationFlags{
    INTER_NEAREST        = 0,
    INTER_LINEAR         = 1,
    INTER_CUBIC          = 2,
    INTER_AREA           = 3,
    INTER_LANCZOS4       = 4,
    INTER_LINEAR_EXACT   = 5,
    INTER_MAX            = 7,
};
} // namespace cv

static inline int cvFloor( double value )
{
    int i = (int)value;
    return i - (i > value);
}

#endif //  defined(GAPI_STANDALONE)

#endif //  OPENCV_GAPI_CV_DEFS_HPP
