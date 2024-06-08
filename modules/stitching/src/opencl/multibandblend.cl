// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Itseez, Inc, all rights reserved.

//
// Common preprocessors macro
//

//
// TODO: Move this common code into "header" file
//

#ifndef NL // New Line: for preprocessor debugging
#define NL
#endif

#define REF(x) x
#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

//
// All matrixes are come with this description ("name" is a name of matrix):
// * name_CN - number of channels (1,2,3,4)
// * name_DEPTH - numeric value of CV_MAT_DEPTH(type). See CV_8U, CV_32S, etc macro below.
//
// Currently we also pass these attributes (to reduce this macro block):
// * name_T - datatype (int, float, uchar4, float4)
// * name_T1 - datatype for one channel (int, float, uchar).
//   It is equal to result of "T1(name_T)" macro
// * name_TSIZE - CV_ELEM_SIZE(type).
//   We can't use sizeof(name_T) here, because sizeof(float3) is usually equal to 8, not 6.
// * name_T1SIZE - CV_ELEM_SIZE1(type)
//

//
// Usage sample:
//
// #define workType TYPE(float, src_CN)
// #define convertToWorkType CONVERT_TO(workType)
// #define convertWorkTypeToDstType CONVERT(workType, dst_T)
//
// __kernel void kernelFn(DECLARE_MAT_ARG(src), DECLARE_MAT_ARG(dst))
// {
//     const int x = get_global_id(0);
//     const int y = get_global_id(1);
//
//     if (x < srcWidth && y < srcHeight)
//     {
//         int src_byteOffset = MAT_BYTE_OFFSET(src, x, y);
//         int dst_byteOffset = MAT_BYTE_OFFSET(dst, x, y);
//         workType value = convertToWorkType(LOAD_MAT_AT(src, src_byteOffset));
//
//         ... value processing ...
//
//         STORE_MAT_AT(dst, dst_byteOffset, convertWorkTypeToDstType(value));
//     }
// }
//

#define DECLARE_MAT_ARG(name) \
    __global uchar* restrict name ## Ptr, \
    int name ## StepBytes, \
    int name ## Offset, \
    int name ## Height, \
    int name ## Width NL

#define MAT_BYTE_OFFSET(name, x, y) mad24((y)/* + name ## OffsetY*/, name ## StepBytes, ((x)/* + name ## OffsetX*/) * (int)(name ## _TSIZE) + name ## Offset)
#define MAT_RELATIVE_BYTE_OFFSET(name, x, y) mad24(y, name ## StepBytes, (x) * (int)(name ## _TSIZE))

#define __LOAD_MAT_AT(name, byteOffset) *((const __global name ## _T*)(name ## Ptr + (byteOffset)))
#define __vload_CN__(name_cn) vload ## name_cn
#define __vload_CN_(name_cn) __vload_CN__(name_cn)
#define __vload_CN(name) __vload_CN_(name ## _CN)
#define __LOAD_MAT_AT_vload(name, byteOffset) __vload_CN(name)(0, ((const __global name ## _T1*)(name ## Ptr + (byteOffset))))
#define __LOAD_MAT_AT_1 __LOAD_MAT_AT
#define __LOAD_MAT_AT_2 __LOAD_MAT_AT
#define __LOAD_MAT_AT_3 __LOAD_MAT_AT_vload
#define __LOAD_MAT_AT_4 __LOAD_MAT_AT
#define __LOAD_MAT_AT_CN__(name_cn) __LOAD_MAT_AT_ ## name_cn
#define __LOAD_MAT_AT_CN_(name_cn) __LOAD_MAT_AT_CN__(name_cn)
#define __LOAD_MAT_AT_CN(name) __LOAD_MAT_AT_CN_(name ## _CN)
#define LOAD_MAT_AT(name, byteOffset) __LOAD_MAT_AT_CN(name)(name, byteOffset)

#define __STORE_MAT_AT(name, byteOffset, v) *((__global name ## _T*)(name ## Ptr + (byteOffset))) = v
#define __vstore_CN__(name_cn) vstore ## name_cn
#define __vstore_CN_(name_cn) __vstore_CN__(name_cn)
#define __vstore_CN(name) __vstore_CN_(name ## _CN)
#define __STORE_MAT_AT_vstore(name, byteOffset, v) __vstore_CN(name)(v, 0, ((__global name ## _T1*)(name ## Ptr + (byteOffset))))
#define __STORE_MAT_AT_1 __STORE_MAT_AT
#define __STORE_MAT_AT_2 __STORE_MAT_AT
#define __STORE_MAT_AT_3 __STORE_MAT_AT_vstore
#define __STORE_MAT_AT_4 __STORE_MAT_AT
#define __STORE_MAT_AT_CN__(name_cn) __STORE_MAT_AT_ ## name_cn
#define __STORE_MAT_AT_CN_(name_cn) __STORE_MAT_AT_CN__(name_cn)
#define __STORE_MAT_AT_CN(name) __STORE_MAT_AT_CN_(name ## _CN)
#define STORE_MAT_AT(name, byteOffset, v) __STORE_MAT_AT_CN(name)(name, byteOffset, v)

#define T1_uchar uchar
#define T1_uchar2 uchar
#define T1_uchar3 uchar
#define T1_uchar4 uchar
#define T1_char char
#define T1_char2 char
#define T1_char3 char
#define T1_char4 char
#define T1_ushort ushort
#define T1_ushort2 ushort
#define T1_ushort3 ushort
#define T1_ushort4 ushort
#define T1_short short
#define T1_short2 short
#define T1_short3 short
#define T1_short4 short
#define T1_int int
#define T1_int2 int
#define T1_int3 int
#define T1_int4 int
#define T1_float float
#define T1_float2 float
#define T1_float3 float
#define T1_float4 float
#define T1_double double
#define T1_double2 double
#define T1_double3 double
#define T1_double4 double
#define T1(type) REF(CAT(T1_, REF(type)))

#define uchar1 uchar
#define char1 char
#define short1 short
#define ushort1 ushort
#define int1 int
#define float1 float
#define double1 double
#define TYPE(type, cn) REF(CAT(REF(type), REF(cn)))

#define __CONVERT_MODE_uchar_uchar __NO_CONVERT
#define __CONVERT_MODE_uchar_char __CONVERT_sat
#define __CONVERT_MODE_uchar_ushort __CONVERT
#define __CONVERT_MODE_uchar_short __CONVERT
#define __CONVERT_MODE_uchar_int __CONVERT
#define __CONVERT_MODE_uchar_float __CONVERT
#define __CONVERT_MODE_uchar_double __CONVERT
#define __CONVERT_MODE_char_uchar __CONVERT_sat
#define __CONVERT_MODE_char_char __NO_CONVERT
#define __CONVERT_MODE_char_ushort __CONVERT_sat
#define __CONVERT_MODE_char_short __CONVERT
#define __CONVERT_MODE_char_int __CONVERT
#define __CONVERT_MODE_char_float __CONVERT
#define __CONVERT_MODE_char_double __CONVERT
#define __CONVERT_MODE_ushort_uchar __CONVERT_sat
#define __CONVERT_MODE_ushort_char __CONVERT_sat
#define __CONVERT_MODE_ushort_ushort __NO_CONVERT
#define __CONVERT_MODE_ushort_short __CONVERT_sat
#define __CONVERT_MODE_ushort_int __CONVERT
#define __CONVERT_MODE_ushort_float __CONVERT
#define __CONVERT_MODE_ushort_double __CONVERT
#define __CONVERT_MODE_short_uchar __CONVERT_sat
#define __CONVERT_MODE_short_char __CONVERT_sat
#define __CONVERT_MODE_short_ushort __CONVERT_sat
#define __CONVERT_MODE_short_short __NO_CONVERT
#define __CONVERT_MODE_short_int __CONVERT
#define __CONVERT_MODE_short_float __CONVERT
#define __CONVERT_MODE_short_double __CONVERT
#define __CONVERT_MODE_int_uchar __CONVERT_sat
#define __CONVERT_MODE_int_char __CONVERT_sat
#define __CONVERT_MODE_int_ushort __CONVERT_sat
#define __CONVERT_MODE_int_short __CONVERT_sat
#define __CONVERT_MODE_int_int __NO_CONVERT
#define __CONVERT_MODE_int_float __CONVERT
#define __CONVERT_MODE_int_double __CONVERT
#define __CONVERT_MODE_float_uchar __CONVERT_sat_rte
#define __CONVERT_MODE_float_char __CONVERT_sat_rte
#define __CONVERT_MODE_float_ushort __CONVERT_sat_rte
#define __CONVERT_MODE_float_short __CONVERT_sat_rte
#define __CONVERT_MODE_float_int __CONVERT_rte
#define __CONVERT_MODE_float_float __NO_CONVERT
#define __CONVERT_MODE_float_double __CONVERT
#define __CONVERT_MODE_double_uchar __CONVERT_sat_rte
#define __CONVERT_MODE_double_char __CONVERT_sat_rte
#define __CONVERT_MODE_double_ushort __CONVERT_sat_rte
#define __CONVERT_MODE_double_short __CONVERT_sat_rte
#define __CONVERT_MODE_double_int __CONVERT_rte
#define __CONVERT_MODE_double_float __CONVERT
#define __CONVERT_MODE_double_double __NO_CONVERT
#define __CONVERT_MODE(srcType, dstType) CAT(__CONVERT_MODE_, CAT(REF(T1(srcType)), CAT(_, REF(T1(dstType)))))

#define __ROUND_MODE__NO_CONVERT
#define __ROUND_MODE__CONVERT // nothing
#define __ROUND_MODE__CONVERT_rte _rte
#define __ROUND_MODE__CONVERT_sat _sat
#define __ROUND_MODE__CONVERT_sat_rte _sat_rte
#define ROUND_MODE(srcType, dstType) CAT(__ROUND_MODE_, __CONVERT_MODE(srcType, dstType))

#define __CONVERT_ROUND(dstType, roundMode) CAT(CAT(convert_, REF(dstType)), roundMode)
#define __NO_CONVERT(dstType) // nothing
#define __CONVERT(dstType) __CONVERT_ROUND(dstType,)
#define __CONVERT_rte(dstType) __CONVERT_ROUND(dstType,_rte)
#define __CONVERT_sat(dstType) __CONVERT_ROUND(dstType,_sat)
#define __CONVERT_sat_rte(dstType) __CONVERT_ROUND(dstType,_sat_rte)
#define CONVERT(srcType, dstType) REF(__CONVERT_MODE(srcType,dstType))(dstType)
#define CONVERT_TO(dstType) __CONVERT_ROUND(dstType,)

// OpenCV depths
#define CV_8U   0
#define CV_8S   1
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6

//
// End of common preprocessors macro
//



#if defined(DEFINE_feed)

#define workType TYPE(weight_T1, src_CN)

#if src_DEPTH == 3 && src_CN == 3
#define convertSrcToWorkType convert_float3
#else
#define convertSrcToWorkType CONVERT_TO(workType)
#endif

#if dst_DEPTH == 3 && dst_CN == 3
#define convertToDstType convert_short3
#else
#define convertToDstType CONVERT_TO(dst_T) // sat_rte provides incompatible results with CPU path
#endif

__kernel void feed(
        DECLARE_MAT_ARG(src), DECLARE_MAT_ARG(weight),
        DECLARE_MAT_ARG(dst), DECLARE_MAT_ARG(dstWeight)
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < srcWidth && y < srcHeight)
    {
        int src_byteOffset = MAT_BYTE_OFFSET(src, x, y);
        int weight_byteOffset = MAT_BYTE_OFFSET(weight, x, y);
        int dst_byteOffset = MAT_BYTE_OFFSET(dst, x, y);
        int dstWeight_byteOffset = MAT_BYTE_OFFSET(dstWeight, x, y);

        weight_T w = LOAD_MAT_AT(weight, weight_byteOffset);
        workType src_value = convertSrcToWorkType(LOAD_MAT_AT(src, src_byteOffset));
        STORE_MAT_AT(dst, dst_byteOffset, LOAD_MAT_AT(dst, dst_byteOffset) + convertToDstType(src_value * w));
        STORE_MAT_AT(dstWeight, dstWeight_byteOffset, LOAD_MAT_AT(dstWeight, dstWeight_byteOffset) + w);
    }
}

#endif

#if defined(DEFINE_normalizeUsingWeightMap)

#if mat_DEPTH == 3 && mat_CN == 3
#define workType float3
#define convertSrcToWorkType convert_float3
#define convertToDstType convert_short3
#else
#define workType TYPE(weight_T1, mat_CN)
#define convertSrcToWorkType CONVERT_TO(workType)
#define convertToDstType CONVERT_TO(mat_T) // sat_rte provides incompatible results with CPU path
#endif

#if weight_DEPTH >= CV_32F
#define WEIGHT_EPS 1e-5f
#else
#define WEIGHT_EPS 0
#endif

__kernel void normalizeUsingWeightMap(
        DECLARE_MAT_ARG(mat), DECLARE_MAT_ARG(weight)
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < matWidth && y < matHeight)
    {
        int mat_byteOffset = MAT_BYTE_OFFSET(mat, x, y);
        int weight_byteOffset = MAT_BYTE_OFFSET(weight, x, y);

        weight_T w = LOAD_MAT_AT(weight, weight_byteOffset);
        workType value = convertSrcToWorkType(LOAD_MAT_AT(mat, mat_byteOffset));
        value = value / (w + WEIGHT_EPS);
        STORE_MAT_AT(mat, mat_byteOffset, convertToDstType(value));
    }
}

#endif
