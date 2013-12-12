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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <map>

#ifdef HAVE_OPENCL
#include "opencv2/core/opencl/runtime/opencl_core.hpp"
#else
// TODO FIXIT: This file can't be build without OPENCL

/*
  Part of the file is an extract from the standard OpenCL headers from Khronos site.
  Below is the original copyright.
*/

/*******************************************************************************
 * Copyright (c) 2008 - 2012 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#if 0 //defined __APPLE__
#define HAVE_OPENCL 1
#else
#undef HAVE_OPENCL
#endif

#define OPENCV_CL_NOT_IMPLEMENTED -1000

#ifdef HAVE_OPENCL

#if defined __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

static const bool g_haveOpenCL = true;

#else

extern "C" {

struct _cl_platform_id { int dummy; };
struct _cl_device_id { int dummy; };
struct _cl_context { int dummy; };
struct _cl_command_queue { int dummy; };
struct _cl_mem { int dummy; };
struct _cl_program { int dummy; };
struct _cl_kernel { int dummy; };
struct _cl_event { int dummy; };
struct _cl_sampler { int dummy; };

typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;

typedef int cl_int;
typedef unsigned cl_uint;
#if defined (_WIN32) && defined(_MSC_VER)
    typedef __int64 cl_long;
    typedef unsigned __int64 cl_ulong;
#else
    typedef long cl_long;
    typedef unsigned long cl_ulong;
#endif

typedef cl_uint             cl_bool; /* WARNING!  Unlike cl_ types in cl_platform.h, cl_bool is not guaranteed to be the same size as the bool in kernels. */
typedef cl_ulong            cl_bitfield;
typedef cl_bitfield         cl_device_type;
typedef cl_uint             cl_platform_info;
typedef cl_uint             cl_device_info;
typedef cl_bitfield         cl_device_fp_config;
typedef cl_uint             cl_device_mem_cache_type;
typedef cl_uint             cl_device_local_mem_type;
typedef cl_bitfield         cl_device_exec_capabilities;
typedef cl_bitfield         cl_command_queue_properties;
typedef intptr_t            cl_device_partition_property;
typedef cl_bitfield         cl_device_affinity_domain;

typedef intptr_t            cl_context_properties;
typedef cl_uint             cl_context_info;
typedef cl_uint             cl_command_queue_info;
typedef cl_uint             cl_channel_order;
typedef cl_uint             cl_channel_type;
typedef cl_bitfield         cl_mem_flags;
typedef cl_uint             cl_mem_object_type;
typedef cl_uint             cl_mem_info;
typedef cl_bitfield         cl_mem_migration_flags;
typedef cl_uint             cl_image_info;
typedef cl_uint             cl_buffer_create_type;
typedef cl_uint             cl_addressing_mode;
typedef cl_uint             cl_filter_mode;
typedef cl_uint             cl_sampler_info;
typedef cl_bitfield         cl_map_flags;
typedef cl_uint             cl_program_info;
typedef cl_uint             cl_program_build_info;
typedef cl_uint             cl_program_binary_type;
typedef cl_int              cl_build_status;
typedef cl_uint             cl_kernel_info;
typedef cl_uint             cl_kernel_arg_info;
typedef cl_uint             cl_kernel_arg_address_qualifier;
typedef cl_uint             cl_kernel_arg_access_qualifier;
typedef cl_bitfield         cl_kernel_arg_type_qualifier;
typedef cl_uint             cl_kernel_work_group_info;
typedef cl_uint             cl_event_info;
typedef cl_uint             cl_command_type;
typedef cl_uint             cl_profiling_info;


typedef struct _cl_image_format {
    cl_channel_order        image_channel_order;
    cl_channel_type         image_channel_data_type;
} cl_image_format;

typedef struct _cl_image_desc {
    cl_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    cl_uint                 num_mip_levels;
    cl_uint                 num_samples;
    cl_mem                  buffer;
} cl_image_desc;

typedef struct _cl_buffer_region {
    size_t                  origin;
    size_t                  size;
} cl_buffer_region;


//////////////////////////////////////////////////////////

#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                         -1
#define CL_DEVICE_NOT_AVAILABLE                     -2
#define CL_COMPILER_NOT_AVAILABLE                   -3
#define CL_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define CL_OUT_OF_RESOURCES                         -5
#define CL_OUT_OF_HOST_MEMORY                       -6
#define CL_PROFILING_INFO_NOT_AVAILABLE             -7
#define CL_MEM_COPY_OVERLAP                         -8
#define CL_IMAGE_FORMAT_MISMATCH                    -9
#define CL_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define CL_BUILD_PROGRAM_FAILURE                    -11
#define CL_MAP_FAILURE                              -12
#define CL_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define CL_COMPILE_PROGRAM_FAILURE                  -15
#define CL_LINKER_NOT_AVAILABLE                     -16
#define CL_LINK_PROGRAM_FAILURE                     -17
#define CL_DEVICE_PARTITION_FAILED                  -18
#define CL_KERNEL_ARG_INFO_NOT_AVAILABLE            -19

#define CL_INVALID_VALUE                            -30
#define CL_INVALID_DEVICE_TYPE                      -31
#define CL_INVALID_PLATFORM                         -32
#define CL_INVALID_DEVICE                           -33
#define CL_INVALID_CONTEXT                          -34
#define CL_INVALID_QUEUE_PROPERTIES                 -35
#define CL_INVALID_COMMAND_QUEUE                    -36
#define CL_INVALID_HOST_PTR                         -37
#define CL_INVALID_MEM_OBJECT                       -38
#define CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define CL_INVALID_IMAGE_SIZE                       -40
#define CL_INVALID_SAMPLER                          -41
#define CL_INVALID_BINARY                           -42
#define CL_INVALID_BUILD_OPTIONS                    -43
#define CL_INVALID_PROGRAM                          -44
#define CL_INVALID_PROGRAM_EXECUTABLE               -45
#define CL_INVALID_KERNEL_NAME                      -46
#define CL_INVALID_KERNEL_DEFINITION                -47
#define CL_INVALID_KERNEL                           -48
#define CL_INVALID_ARG_INDEX                        -49
#define CL_INVALID_ARG_VALUE                        -50
#define CL_INVALID_ARG_SIZE                         -51
#define CL_INVALID_KERNEL_ARGS                      -52
#define CL_INVALID_WORK_DIMENSION                   -53
#define CL_INVALID_WORK_GROUP_SIZE                  -54
#define CL_INVALID_WORK_ITEM_SIZE                   -55
#define CL_INVALID_GLOBAL_OFFSET                    -56
#define CL_INVALID_EVENT_WAIT_LIST                  -57
#define CL_INVALID_EVENT                            -58
#define CL_INVALID_OPERATION                        -59
#define CL_INVALID_GL_OBJECT                        -60
#define CL_INVALID_BUFFER_SIZE                      -61
#define CL_INVALID_MIP_LEVEL                        -62
#define CL_INVALID_GLOBAL_WORK_SIZE                 -63
#define CL_INVALID_PROPERTY                         -64
#define CL_INVALID_IMAGE_DESCRIPTOR                 -65
#define CL_INVALID_COMPILER_OPTIONS                 -66
#define CL_INVALID_LINKER_OPTIONS                   -67
#define CL_INVALID_DEVICE_PARTITION_COUNT           -68

/*#define CL_VERSION_1_0                              1
#define CL_VERSION_1_1                              1
#define CL_VERSION_1_2                              1*/

#define CL_FALSE                                    0
#define CL_TRUE                                     1
#define CL_BLOCKING                                 CL_TRUE
#define CL_NON_BLOCKING                             CL_FALSE

#define CL_PLATFORM_PROFILE                         0x0900
#define CL_PLATFORM_VERSION                         0x0901
#define CL_PLATFORM_NAME                            0x0902
#define CL_PLATFORM_VENDOR                          0x0903
#define CL_PLATFORM_EXTENSIONS                      0x0904

#define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define CL_DEVICE_TYPE_CPU                          (1 << 1)
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
#define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF
#define CL_DEVICE_TYPE                              0x1000
#define CL_DEVICE_VENDOR_ID                         0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS                 0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE               0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES               0x1005
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR       0x1006
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT      0x1007
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT        0x1008
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG       0x1009
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT      0x100A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE     0x100B
#define CL_DEVICE_MAX_CLOCK_FREQUENCY               0x100C
#define CL_DEVICE_ADDRESS_BITS                      0x100D
#define CL_DEVICE_MAX_READ_IMAGE_ARGS               0x100E
#define CL_DEVICE_MAX_WRITE_IMAGE_ARGS              0x100F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE                0x1010
#define CL_DEVICE_IMAGE2D_MAX_WIDTH                 0x1011
#define CL_DEVICE_IMAGE2D_MAX_HEIGHT                0x1012
#define CL_DEVICE_IMAGE3D_MAX_WIDTH                 0x1013
#define CL_DEVICE_IMAGE3D_MAX_HEIGHT                0x1014
#define CL_DEVICE_IMAGE3D_MAX_DEPTH                 0x1015
#define CL_DEVICE_IMAGE_SUPPORT                     0x1016
#define CL_DEVICE_MAX_PARAMETER_SIZE                0x1017
#define CL_DEVICE_MAX_SAMPLERS                      0x1018
#define CL_DEVICE_MEM_BASE_ADDR_ALIGN               0x1019
#define CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE          0x101A
#define CL_DEVICE_SINGLE_FP_CONFIG                  0x101B
#define CL_DEVICE_GLOBAL_MEM_CACHE_TYPE             0x101C
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE         0x101D
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE             0x101E
#define CL_DEVICE_GLOBAL_MEM_SIZE                   0x101F
#define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE          0x1020
#define CL_DEVICE_MAX_CONSTANT_ARGS                 0x1021
#define CL_DEVICE_LOCAL_MEM_TYPE                    0x1022
#define CL_DEVICE_LOCAL_MEM_SIZE                    0x1023
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT          0x1024
#define CL_DEVICE_PROFILING_TIMER_RESOLUTION        0x1025
#define CL_DEVICE_ENDIAN_LITTLE                     0x1026
#define CL_DEVICE_AVAILABLE                         0x1027
#define CL_DEVICE_COMPILER_AVAILABLE                0x1028
#define CL_DEVICE_EXECUTION_CAPABILITIES            0x1029
#define CL_DEVICE_QUEUE_PROPERTIES                  0x102A
#define CL_DEVICE_NAME                              0x102B
#define CL_DEVICE_VENDOR                            0x102C
#define CL_DRIVER_VERSION                           0x102D
#define CL_DEVICE_PROFILE                           0x102E
#define CL_DEVICE_VERSION                           0x102F
#define CL_DEVICE_EXTENSIONS                        0x1030
#define CL_DEVICE_PLATFORM                          0x1031
#define CL_DEVICE_DOUBLE_FP_CONFIG                  0x1032
#define CL_DEVICE_HALF_FP_CONFIG                    0x1033
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF       0x1034
#define CL_DEVICE_HOST_UNIFIED_MEMORY               0x1035
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR          0x1036
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT         0x1037
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_INT           0x1038
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG          0x1039
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT         0x103A
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE        0x103B
#define CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF          0x103C
#define CL_DEVICE_OPENCL_C_VERSION                  0x103D
#define CL_DEVICE_LINKER_AVAILABLE                  0x103E
#define CL_DEVICE_BUILT_IN_KERNELS                  0x103F
#define CL_DEVICE_IMAGE_MAX_BUFFER_SIZE             0x1040
#define CL_DEVICE_IMAGE_MAX_ARRAY_SIZE              0x1041
#define CL_DEVICE_PARENT_DEVICE                     0x1042
#define CL_DEVICE_PARTITION_MAX_SUB_DEVICES         0x1043
#define CL_DEVICE_PARTITION_PROPERTIES              0x1044
#define CL_DEVICE_PARTITION_AFFINITY_DOMAIN         0x1045
#define CL_DEVICE_PARTITION_TYPE                    0x1046
#define CL_DEVICE_REFERENCE_COUNT                   0x1047
#define CL_DEVICE_PREFERRED_INTEROP_USER_SYNC       0x1048
#define CL_DEVICE_PRINTF_BUFFER_SIZE                0x1049
#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT             0x104A
#define CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT      0x104B

#define CL_FP_DENORM                                (1 << 0)
#define CL_FP_INF_NAN                               (1 << 1)
#define CL_FP_ROUND_TO_NEAREST                      (1 << 2)
#define CL_FP_ROUND_TO_ZERO                         (1 << 3)
#define CL_FP_ROUND_TO_INF                          (1 << 4)
#define CL_FP_FMA                                   (1 << 5)
#define CL_FP_SOFT_FLOAT                            (1 << 6)
#define CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT         (1 << 7)

#define CL_NONE                                     0x0
#define CL_READ_ONLY_CACHE                          0x1
#define CL_READ_WRITE_CACHE                         0x2
#define CL_LOCAL                                    0x1
#define CL_GLOBAL                                   0x2
#define CL_EXEC_KERNEL                              (1 << 0)
#define CL_EXEC_NATIVE_KERNEL                       (1 << 1)
#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)

#define CL_CONTEXT_REFERENCE_COUNT                  0x1080
#define CL_CONTEXT_DEVICES                          0x1081
#define CL_CONTEXT_PROPERTIES                       0x1082
#define CL_CONTEXT_NUM_DEVICES                      0x1083
#define CL_CONTEXT_PLATFORM                         0x1084
#define CL_CONTEXT_INTEROP_USER_SYNC                0x1085

#define CL_DEVICE_PARTITION_EQUALLY                 0x1086
#define CL_DEVICE_PARTITION_BY_COUNTS               0x1087
#define CL_DEVICE_PARTITION_BY_COUNTS_LIST_END      0x0
#define CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN      0x1088
#define CL_DEVICE_AFFINITY_DOMAIN_NUMA                     (1 << 0)
#define CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE                 (1 << 1)
#define CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE                 (1 << 2)
#define CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE                 (1 << 3)
#define CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE                 (1 << 4)
#define CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE       (1 << 5)
#define CL_QUEUE_CONTEXT                            0x1090
#define CL_QUEUE_DEVICE                             0x1091
#define CL_QUEUE_REFERENCE_COUNT                    0x1092
#define CL_QUEUE_PROPERTIES                         0x1093
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_USE_HOST_PTR                         (1 << 3)
#define CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)
// reserved                                         (1 << 6)
#define CL_MEM_HOST_WRITE_ONLY                      (1 << 7)
#define CL_MEM_HOST_READ_ONLY                       (1 << 8)
#define CL_MEM_HOST_NO_ACCESS                       (1 << 9)
#define CL_MIGRATE_MEM_OBJECT_HOST                  (1 << 0)
#define CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED     (1 << 1)

#define CL_R                                        0x10B0
#define CL_A                                        0x10B1
#define CL_RG                                       0x10B2
#define CL_RA                                       0x10B3
#define CL_RGB                                      0x10B4
#define CL_RGBA                                     0x10B5
#define CL_BGRA                                     0x10B6
#define CL_ARGB                                     0x10B7
#define CL_INTENSITY                                0x10B8
#define CL_LUMINANCE                                0x10B9
#define CL_Rx                                       0x10BA
#define CL_RGx                                      0x10BB
#define CL_RGBx                                     0x10BC
#define CL_DEPTH                                    0x10BD
#define CL_DEPTH_STENCIL                            0x10BE

#define CL_SNORM_INT8                               0x10D0
#define CL_SNORM_INT16                              0x10D1
#define CL_UNORM_INT8                               0x10D2
#define CL_UNORM_INT16                              0x10D3
#define CL_UNORM_SHORT_565                          0x10D4
#define CL_UNORM_SHORT_555                          0x10D5
#define CL_UNORM_INT_101010                         0x10D6
#define CL_SIGNED_INT8                              0x10D7
#define CL_SIGNED_INT16                             0x10D8
#define CL_SIGNED_INT32                             0x10D9
#define CL_UNSIGNED_INT8                            0x10DA
#define CL_UNSIGNED_INT16                           0x10DB
#define CL_UNSIGNED_INT32                           0x10DC
#define CL_HALF_FLOAT                               0x10DD
#define CL_FLOAT                                    0x10DE
#define CL_UNORM_INT24                              0x10DF

#define CL_MEM_OBJECT_BUFFER                        0x10F0
#define CL_MEM_OBJECT_IMAGE2D                       0x10F1
#define CL_MEM_OBJECT_IMAGE3D                       0x10F2
#define CL_MEM_OBJECT_IMAGE2D_ARRAY                 0x10F3
#define CL_MEM_OBJECT_IMAGE1D                       0x10F4
#define CL_MEM_OBJECT_IMAGE1D_ARRAY                 0x10F5
#define CL_MEM_OBJECT_IMAGE1D_BUFFER                0x10F6

#define CL_MEM_TYPE                                 0x1100
#define CL_MEM_FLAGS                                0x1101
#define CL_MEM_SIZE                                 0x1102
#define CL_MEM_HOST_PTR                             0x1103
#define CL_MEM_MAP_COUNT                            0x1104
#define CL_MEM_REFERENCE_COUNT                      0x1105
#define CL_MEM_CONTEXT                              0x1106
#define CL_MEM_ASSOCIATED_MEMOBJECT                 0x1107
#define CL_MEM_OFFSET                               0x1108

#define CL_IMAGE_FORMAT                             0x1110
#define CL_IMAGE_ELEMENT_SIZE                       0x1111
#define CL_IMAGE_ROW_PITCH                          0x1112
#define CL_IMAGE_SLICE_PITCH                        0x1113
#define CL_IMAGE_WIDTH                              0x1114
#define CL_IMAGE_HEIGHT                             0x1115
#define CL_IMAGE_DEPTH                              0x1116
#define CL_IMAGE_ARRAY_SIZE                         0x1117
#define CL_IMAGE_BUFFER                             0x1118
#define CL_IMAGE_NUM_MIP_LEVELS                     0x1119
#define CL_IMAGE_NUM_SAMPLES                        0x111A

#define CL_ADDRESS_NONE                             0x1130
#define CL_ADDRESS_CLAMP_TO_EDGE                    0x1131
#define CL_ADDRESS_CLAMP                            0x1132
#define CL_ADDRESS_REPEAT                           0x1133
#define CL_ADDRESS_MIRRORED_REPEAT                  0x1134

#define CL_FILTER_NEAREST                           0x1140
#define CL_FILTER_LINEAR                            0x1141

#define CL_SAMPLER_REFERENCE_COUNT                  0x1150
#define CL_SAMPLER_CONTEXT                          0x1151
#define CL_SAMPLER_NORMALIZED_COORDS                0x1152
#define CL_SAMPLER_ADDRESSING_MODE                  0x1153
#define CL_SAMPLER_FILTER_MODE                      0x1154

#define CL_MAP_READ                                 (1 << 0)
#define CL_MAP_WRITE                                (1 << 1)
#define CL_MAP_WRITE_INVALIDATE_REGION              (1 << 2)

#define CL_PROGRAM_REFERENCE_COUNT                  0x1160
#define CL_PROGRAM_CONTEXT                          0x1161
#define CL_PROGRAM_NUM_DEVICES                      0x1162
#define CL_PROGRAM_DEVICES                          0x1163
#define CL_PROGRAM_SOURCE                           0x1164
#define CL_PROGRAM_BINARY_SIZES                     0x1165
#define CL_PROGRAM_BINARIES                         0x1166
#define CL_PROGRAM_NUM_KERNELS                      0x1167
#define CL_PROGRAM_KERNEL_NAMES                     0x1168
#define CL_PROGRAM_BUILD_STATUS                     0x1181
#define CL_PROGRAM_BUILD_OPTIONS                    0x1182
#define CL_PROGRAM_BUILD_LOG                        0x1183
#define CL_PROGRAM_BINARY_TYPE                      0x1184
#define CL_PROGRAM_BINARY_TYPE_NONE                 0x0
#define CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT      0x1
#define CL_PROGRAM_BINARY_TYPE_LIBRARY              0x2
#define CL_PROGRAM_BINARY_TYPE_EXECUTABLE           0x4

#define CL_BUILD_SUCCESS                            0
#define CL_BUILD_NONE                               -1
#define CL_BUILD_ERROR                              -2
#define CL_BUILD_IN_PROGRESS                        -3

#define CL_KERNEL_FUNCTION_NAME                     0x1190
#define CL_KERNEL_NUM_ARGS                          0x1191
#define CL_KERNEL_REFERENCE_COUNT                   0x1192
#define CL_KERNEL_CONTEXT                           0x1193
#define CL_KERNEL_PROGRAM                           0x1194
#define CL_KERNEL_ATTRIBUTES                        0x1195
#define CL_KERNEL_ARG_ADDRESS_QUALIFIER             0x1196
#define CL_KERNEL_ARG_ACCESS_QUALIFIER              0x1197
#define CL_KERNEL_ARG_TYPE_NAME                     0x1198
#define CL_KERNEL_ARG_TYPE_QUALIFIER                0x1199
#define CL_KERNEL_ARG_NAME                          0x119A
#define CL_KERNEL_ARG_ADDRESS_GLOBAL                0x119B
#define CL_KERNEL_ARG_ADDRESS_LOCAL                 0x119C
#define CL_KERNEL_ARG_ADDRESS_CONSTANT              0x119D
#define CL_KERNEL_ARG_ADDRESS_PRIVATE               0x119E
#define CL_KERNEL_ARG_ACCESS_READ_ONLY              0x11A0
#define CL_KERNEL_ARG_ACCESS_WRITE_ONLY             0x11A1
#define CL_KERNEL_ARG_ACCESS_READ_WRITE             0x11A2
#define CL_KERNEL_ARG_ACCESS_NONE                   0x11A3
#define CL_KERNEL_ARG_TYPE_NONE                     0
#define CL_KERNEL_ARG_TYPE_CONST                    (1 << 0)
#define CL_KERNEL_ARG_TYPE_RESTRICT                 (1 << 1)
#define CL_KERNEL_ARG_TYPE_VOLATILE                 (1 << 2)
#define CL_KERNEL_WORK_GROUP_SIZE                   0x11B0
#define CL_KERNEL_COMPILE_WORK_GROUP_SIZE           0x11B1
#define CL_KERNEL_LOCAL_MEM_SIZE                    0x11B2
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define CL_KERNEL_PRIVATE_MEM_SIZE                  0x11B4
#define CL_KERNEL_GLOBAL_WORK_SIZE                  0x11B5

#define CL_EVENT_COMMAND_QUEUE                      0x11D0
#define CL_EVENT_COMMAND_TYPE                       0x11D1
#define CL_EVENT_REFERENCE_COUNT                    0x11D2
#define CL_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#define CL_EVENT_CONTEXT                            0x11D4

#define CL_COMMAND_NDRANGE_KERNEL                   0x11F0
#define CL_COMMAND_TASK                             0x11F1
#define CL_COMMAND_NATIVE_KERNEL                    0x11F2
#define CL_COMMAND_READ_BUFFER                      0x11F3
#define CL_COMMAND_WRITE_BUFFER                     0x11F4
#define CL_COMMAND_COPY_BUFFER                      0x11F5
#define CL_COMMAND_READ_IMAGE                       0x11F6
#define CL_COMMAND_WRITE_IMAGE                      0x11F7
#define CL_COMMAND_COPY_IMAGE                       0x11F8
#define CL_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define CL_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define CL_COMMAND_MAP_BUFFER                       0x11FB
#define CL_COMMAND_MAP_IMAGE                        0x11FC
#define CL_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define CL_COMMAND_MARKER                           0x11FE
#define CL_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define CL_COMMAND_RELEASE_GL_OBJECTS               0x1200
#define CL_COMMAND_READ_BUFFER_RECT                 0x1201
#define CL_COMMAND_WRITE_BUFFER_RECT                0x1202
#define CL_COMMAND_COPY_BUFFER_RECT                 0x1203
#define CL_COMMAND_USER                             0x1204
#define CL_COMMAND_BARRIER                          0x1205
#define CL_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
#define CL_COMMAND_FILL_BUFFER                      0x1207
#define CL_COMMAND_FILL_IMAGE                       0x1208

#define CL_COMPLETE                                 0x0
#define CL_RUNNING                                  0x1
#define CL_SUBMITTED                                0x2
#define CL_QUEUED                                   0x3
#define CL_BUFFER_CREATE_TYPE_REGION                0x1220

#define CL_PROFILING_COMMAND_QUEUED                 0x1280
#define CL_PROFILING_COMMAND_SUBMIT                 0x1281
#define CL_PROFILING_COMMAND_START                  0x1282
#define CL_PROFILING_COMMAND_END                    0x1283

#define CL_CALLBACK CV_STDCALL

static volatile bool g_haveOpenCL = false;
static const char* oclFuncToCheck = "clEnqueueReadBufferRect";

#if defined(__APPLE__)
#include <dlfcn.h>

static void* initOpenCLAndLoad(const char* funcname)
{
    static bool initialized = false;
    static void* handle = 0;
    if (!handle)
    {
        if(!initialized)
        {
            const char* oclpath = getenv("OPENCV_OPENCL_RUNTIME");
            oclpath = oclpath && strlen(oclpath) > 0 ? oclpath :
                "/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL";
            handle = dlopen(oclpath, RTLD_LAZY);
            initialized = true;
            g_haveOpenCL = handle != 0 && dlsym(handle, oclFuncToCheck) != 0;
            if( g_haveOpenCL )
                fprintf(stderr, "Succesffuly loaded OpenCL v1.1+ runtime from %s\n", oclpath);
            else
                fprintf(stderr, "Failed to load OpenCL runtime\n");
        }
        if(!handle)
            return 0;
    }

    return funcname && handle ? dlsym(handle, funcname) : 0;
}

#elif defined WIN32 || defined _WIN32

#ifndef _WIN32_WINNT           // This is needed for the declaration of TryEnterCriticalSection in winbase.h with Visual Studio 2005 (and older?)
  #define _WIN32_WINNT 0x0400  // http://msdn.microsoft.com/en-us/library/ms686857(VS.85).aspx
#endif
#include <windows.h>
#if (_WIN32_WINNT >= 0x0602)
  #include <synchapi.h>
#endif
#undef small
#undef min
#undef max
#undef abs

static void* initOpenCLAndLoad(const char* funcname)
{
    static bool initialized = false;
    static HMODULE handle = 0;
    if (!handle)
    {
        if(!initialized)
        {
            handle = LoadLibraryA("OpenCL.dll");
            initialized = true;
            g_haveOpenCL = handle != 0 && GetProcAddress(handle, oclFuncToCheck) != 0;
        }
        if(!handle)
            return 0;
    }

    return funcname ? (void*)GetProcAddress(handle, funcname) : 0;
}

#elif defined(__linux)

#include <dlfcn.h>
#include <stdio.h>

static void* initOpenCLAndLoad(const char* funcname)
{
    static bool initialized = false;
    static void* handle = 0;
    if (!handle)
    {
        if(!initialized)
        {
            handle = dlopen("libOpenCL.so", RTLD_LAZY);
            if(!handle)
                handle = dlopen("libCL.so", RTLD_LAZY);
            initialized = true;
            g_haveOpenCL = handle != 0 && dlsym(handle, oclFuncToCheck) != 0;
        }
        if(!handle)
            return 0;
    }

    return funcname ? (void*)dlsym(handle, funcname) : 0;
}

#else

static void* initOpenCLAndLoad(const char*)
{
    return 0;
}

#endif


#define OCL_FUNC(rettype, funcname, argsdecl, args) \
    typedef rettype (CV_STDCALL * funcname##_t) argsdecl; \
    static rettype funcname argsdecl \
    { \
        static funcname##_t funcname##_p = 0; \
        if( !funcname##_p ) \
        { \
            funcname##_p = (funcname##_t)initOpenCLAndLoad(#funcname); \
            if( !funcname##_p ) \
                return OPENCV_CL_NOT_IMPLEMENTED; \
        } \
        return funcname##_p args; \
    }


#define OCL_FUNC_P(rettype, funcname, argsdecl, args) \
    typedef rettype (CV_STDCALL * funcname##_t) argsdecl; \
    static rettype funcname argsdecl \
    { \
        static funcname##_t funcname##_p = 0; \
        if( !funcname##_p ) \
        { \
            funcname##_p = (funcname##_t)initOpenCLAndLoad(#funcname); \
            if( !funcname##_p ) \
            { \
                if( errcode_ret ) \
                    *errcode_ret = OPENCV_CL_NOT_IMPLEMENTED; \
                return 0; \
            } \
        } \
        return funcname##_p args; \
    }

OCL_FUNC(cl_int, clGetPlatformIDs,
    (cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms),
    (num_entries, platforms, num_platforms))

OCL_FUNC(cl_int, clGetPlatformInfo,
    (cl_platform_id platform, cl_platform_info param_name,
    size_t param_value_size, void * param_value,
    size_t * param_value_size_ret),
    (platform, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clGetDeviceInfo,
         (cl_device_id device,
          cl_device_info param_name,
          size_t param_value_size,
          void * param_value,
          size_t * param_value_size_ret),
         (device, param_name, param_value_size, param_value, param_value_size_ret))


OCL_FUNC(cl_int, clGetDeviceIDs,
    (cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id * devices,
    cl_uint * num_devices),
    (platform, device_type, num_entries, devices, num_devices))

OCL_FUNC_P(cl_context, clCreateContext,
    (const cl_context_properties * properties,
    cl_uint num_devices,
    const cl_device_id * devices,
    void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
    void * user_data,
    cl_int * errcode_ret),
    (properties, num_devices, devices, pfn_notify, user_data, errcode_ret))

OCL_FUNC(cl_int, clReleaseContext, (cl_context context), (context))

/*
OCL_FUNC(cl_int, clRetainContext, (cl_context context), (context))

OCL_FUNC_P(cl_context, clCreateContextFromType,
    (const cl_context_properties * properties,
    cl_device_type device_type,
    void (CL_CALLBACK * pfn_notify)(const char *, const void *, size_t, void *),
    void * user_data,
    cl_int * errcode_ret),
    (properties, device_type, pfn_notify, user_data, errcode_ret))

OCL_FUNC(cl_int, clGetContextInfo,
    (cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void * param_value,
    size_t * param_value_size_ret),
    (context, param_name, param_value_size,
    param_value, param_value_size_ret))
*/
OCL_FUNC_P(cl_command_queue, clCreateCommandQueue,
    (cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int * errcode_ret),
    (context, device, properties, errcode_ret))

OCL_FUNC(cl_int, clReleaseCommandQueue, (cl_command_queue command_queue), (command_queue))

OCL_FUNC_P(cl_mem, clCreateBuffer,
    (cl_context context,
    cl_mem_flags flags,
    size_t size,
    void * host_ptr,
    cl_int * errcode_ret),
    (context, flags, size, host_ptr, errcode_ret))

/*
OCL_FUNC(cl_int, clRetainCommandQueue, (cl_command_queue command_queue), (command_queue))

OCL_FUNC(cl_int, clGetCommandQueueInfo,
 (cl_command_queue command_queue,
 cl_command_queue_info param_name,
 size_t param_value_size,
 void * param_value,
 size_t * param_value_size_ret),
 (command_queue, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC_P(cl_mem, clCreateSubBuffer,
    (cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void * buffer_create_info,
    cl_int * errcode_ret),
    (buffer, flags, buffer_create_type, buffer_create_info, errcode_ret))

OCL_FUNC_P(cl_mem, clCreateImage,
    (cl_context context,
    cl_mem_flags flags,
    const cl_image_format * image_format,
    const cl_image_desc * image_desc,
    void * host_ptr,
    cl_int * errcode_ret),
    (context, flags, image_format, image_desc, host_ptr, errcode_ret))

OCL_FUNC(cl_int, clGetSupportedImageFormats,
 (cl_context context,
 cl_mem_flags flags,
 cl_mem_object_type image_type,
 cl_uint num_entries,
 cl_image_format * image_formats,
 cl_uint * num_image_formats),
 (context, flags, image_type, num_entries, image_formats, num_image_formats))

OCL_FUNC(cl_int, clGetMemObjectInfo,
 (cl_mem memobj,
 cl_mem_info param_name,
 size_t param_value_size,
 void * param_value,
 size_t * param_value_size_ret),
 (memobj, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clGetImageInfo,
 (cl_mem image,
 cl_image_info param_name,
 size_t param_value_size,
 void * param_value,
 size_t * param_value_size_ret),
 (image, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clCreateKernelsInProgram,
 (cl_program program,
 cl_uint num_kernels,
 cl_kernel * kernels,
 cl_uint * num_kernels_ret),
 (program, num_kernels, kernels, num_kernels_ret))

OCL_FUNC(cl_int, clRetainKernel, (cl_kernel kernel), (kernel))

OCL_FUNC(cl_int, clGetKernelArgInfo,
 (cl_kernel kernel,
 cl_uint arg_indx,
 cl_kernel_arg_info param_name,
 size_t param_value_size,
 void * param_value,
 size_t * param_value_size_ret),
 (kernel, arg_indx, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clEnqueueReadImage,
 (cl_command_queue command_queue,
 cl_mem image,
 cl_bool blocking_read,
 const size_t * origin[3],
 const size_t * region[3],
 size_t row_pitch,
 size_t slice_pitch,
 void * ptr,
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event),
 (command_queue, image, blocking_read, origin, region,
 row_pitch, slice_pitch,
 ptr,
 num_events_in_wait_list,
 event_wait_list,
 event))

OCL_FUNC(cl_int, clEnqueueWriteImage,
 (cl_command_queue command_queue,
 cl_mem image,
 cl_bool blocking_write,
 const size_t * origin[3],
 const size_t * region[3],
 size_t input_row_pitch,
 size_t input_slice_pitch,
 const void * ptr,
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event),
 (command_queue, image, blocking_write, origin, region, input_row_pitch,
 input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueFillImage,
 (cl_command_queue command_queue,
 cl_mem image,
 const void * fill_color,
 const size_t * origin[3],
 const size_t * region[3],
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event),
 (command_queue, image, fill_color, origin, region,
 num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueCopyImage,
 (cl_command_queue command_queue,
 cl_mem src_image,
 cl_mem dst_image,
 const size_t * src_origin[3],
 const size_t * dst_origin[3],
 const size_t * region[3],
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event),
 (command_queue, src_image, dst_image, src_origin, dst_origin,
 region, num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueCopyImageToBuffer,
 (cl_command_queue command_queue,
 cl_mem src_image,
 cl_mem dst_buffer,
 const size_t * src_origin[3],
 const size_t * region[3],
 size_t dst_offset,
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event),
 (command_queue, src_image, dst_buffer, src_origin, region, dst_offset,
 num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueCopyBufferToImage,
 (cl_command_queue command_queue,
 cl_mem src_buffer,
 cl_mem dst_image,
 size_t src_offset,
 const size_t * dst_origin[3],
 const size_t * region[3],
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event),
 (command_queue, src_buffer, dst_image, src_offset, dst_origin,
 region, num_events_in_wait_list, event_wait_list, event))


OCL_FUNC_P(void*, clEnqueueMapImage,
 (cl_command_queue command_queue,
 cl_mem image,
 cl_bool blocking_map,
 cl_map_flags map_flags,
 const size_t * origin[3],
 const size_t * region[3],
 size_t * image_row_pitch,
 size_t * image_slice_pitch,
 cl_uint num_events_in_wait_list,
 const cl_event * event_wait_list,
 cl_event * event,
 cl_int * errcode_ret),
 (command_queue, image, blocking_map, map_flags, origin, region,
 image_row_pitch, image_slice_pitch, num_events_in_wait_list,
 event_wait_list, event, errcode_ret))

OCL_FUNC(cl_int, clRetainProgram, (cl_program program), (program))

OCL_FUNC(cl_int, clGetKernelInfo,
 (cl_kernel kernel,
 cl_kernel_info param_name,
 size_t param_value_size,
 void * param_value,
 size_t * param_value_size_ret),
 (kernel, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clRetainMemObject, (cl_mem memobj), (memobj))

*/

OCL_FUNC(cl_int, clReleaseMemObject, (cl_mem memobj), (memobj))


OCL_FUNC_P(cl_program, clCreateProgramWithSource,
    (cl_context context,
    cl_uint count,
    const char ** strings,
    const size_t * lengths,
    cl_int * errcode_ret),
    (context, count, strings, lengths, errcode_ret))

OCL_FUNC_P(cl_program, clCreateProgramWithBinary,
    (cl_context context,
    cl_uint num_devices,
    const cl_device_id * device_list,
    const size_t * lengths,
    const unsigned char ** binaries,
    cl_int * binary_status,
    cl_int * errcode_ret),
    (context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret))

OCL_FUNC(cl_int, clReleaseProgram, (cl_program program), (program))

OCL_FUNC(cl_int, clBuildProgram,
    (cl_program program,
    cl_uint num_devices,
    const cl_device_id * device_list,
    const char * options,
    void (CL_CALLBACK * pfn_notify)(cl_program, void *),
    void * user_data),
    (program, num_devices, device_list, options, pfn_notify, user_data))

OCL_FUNC(cl_int, clGetProgramInfo,
    (cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void * param_value,
    size_t * param_value_size_ret),
    (program, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clGetProgramBuildInfo,
    (cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void * param_value,
    size_t * param_value_size_ret),
    (program, device, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC_P(cl_kernel, clCreateKernel,
    (cl_program program,
    const char * kernel_name,
    cl_int * errcode_ret),
    (program, kernel_name, errcode_ret))

OCL_FUNC(cl_int, clReleaseKernel, (cl_kernel kernel), (kernel))

OCL_FUNC(cl_int, clSetKernelArg,
    (cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void * arg_value),
    (kernel, arg_index, arg_size, arg_value))

OCL_FUNC(cl_int, clGetKernelWorkGroupInfo,
    (cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void * param_value,
    size_t * param_value_size_ret),
    (kernel, device, param_name, param_value_size, param_value, param_value_size_ret))

OCL_FUNC(cl_int, clFinish, (cl_command_queue command_queue), (command_queue))

OCL_FUNC(cl_int, clEnqueueReadBuffer,
    (cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void * ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, buffer, blocking_read, offset, size, ptr,
    num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueReadBufferRect,
    (cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t * buffer_offset,
    const size_t * host_offset,
    const size_t * region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void * ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, buffer, blocking_read, buffer_offset, host_offset, region, buffer_row_pitch,
    buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list,
    event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueWriteBuffer,
    (cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void * ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, buffer, blocking_write, offset, size, ptr,
    num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueWriteBufferRect,
    (cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t * buffer_offset,
    const size_t * host_offset,
    const size_t * region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void * ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, buffer, blocking_write, buffer_offset, host_offset,
    region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch,
    host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event))

/*OCL_FUNC(cl_int, clEnqueueFillBuffer,
    (cl_command_queue command_queue,
    cl_mem buffer,
    const void * pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, buffer, pattern, pattern_size, offset, size,
    num_events_in_wait_list, event_wait_list, event))*/

OCL_FUNC(cl_int, clEnqueueCopyBuffer,
    (cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, src_buffer, dst_buffer, src_offset, dst_offset,
    size, num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueCopyBufferRect,
    (cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t * src_origin,
    const size_t * dst_origin,
    const size_t * region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, src_buffer, dst_buffer, src_origin, dst_origin,
    region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
    num_events_in_wait_list, event_wait_list, event))

OCL_FUNC_P(void*, clEnqueueMapBuffer,
    (cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event,
    cl_int * errcode_ret),
    (command_queue, buffer, blocking_map, map_flags, offset, size,
    num_events_in_wait_list, event_wait_list, event, errcode_ret))

OCL_FUNC(cl_int, clEnqueueUnmapMemObject,
    (cl_command_queue command_queue,
    cl_mem memobj,
    void * mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueNDRangeKernel,
    (cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t * global_work_offset,
    const size_t * global_work_size,
    const size_t * local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, kernel, work_dim, global_work_offset, global_work_size,
    local_work_size, num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clEnqueueTask,
    (cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event * event),
    (command_queue, kernel, num_events_in_wait_list, event_wait_list, event))

OCL_FUNC(cl_int, clSetEventCallback,
    (cl_event event,
    cl_int command_exec_callback_type ,
    void (CL_CALLBACK  *pfn_event_notify) (cl_event event, cl_int event_command_exec_status, void *user_data),
    void *user_data),
    (event, command_exec_callback_type, pfn_event_notify, user_data))

OCL_FUNC(cl_int, clReleaseEvent, (cl_event event), (event))

}

#endif

#ifndef CL_VERSION_1_2
#define CL_VERSION_1_2
#endif

#endif

namespace cv { namespace ocl {

struct UMat2D
{
    UMat2D(const UMat& m)
    {
        offset = (int)m.offset;
        step = (int)m.step;
        rows = m.rows;
        cols = m.cols;
    }
    int offset;
    int step;
    int rows;
    int cols;
};

struct UMat3D
{
    UMat3D(const UMat& m)
    {
        offset = (int)m.offset;
        step = (int)m.step.p[1];
        slicestep = (int)m.step.p[0];
        slices = (int)m.size.p[0];
        rows = m.size.p[1];
        cols = m.size.p[2];
    }
    int offset;
    int slicestep;
    int step;
    int slices;
    int rows;
    int cols;
};

// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
static uint64 crc64( const uchar* data, size_t size, uint64 crc0=0 )
{
    static uint64 table[256];
    static bool initialized = false;

    if( !initialized )
    {
        for( int i = 0; i < 256; i++ )
        {
            uint64 c = i;
            for( int j = 0; j < 8; j++ )
                c = ((c & 1) ? CV_BIG_UINT(0xc96c5795d7870f42) : 0) ^ (c >> 1);
            table[i] = c;
        }
        initialized = true;
    }

    uint64 crc = ~crc0;
    for( size_t idx = 0; idx < size; idx++ )
        crc = table[(uchar)crc ^ data[idx]] ^ (crc >> 8);

    return ~crc;
}

struct HashKey
{
    typedef uint64 part;
    HashKey(part _a, part _b) : a(_a), b(_b) {}
    part a, b;
};

inline bool operator == (const HashKey& h1, const HashKey& h2)
{
    return h1.a == h2.a && h1.b == h2.b;
}

inline bool operator < (const HashKey& h1, const HashKey& h2)
{
    return h1.a < h2.a || (h1.a == h2.a && h1.b < h2.b);
}

static bool g_isInitialized = false;
static bool g_isOpenCLAvailable = false;
bool haveOpenCL()
{
    if (!g_isInitialized)
    {
        if (!g_isInitialized)
        {
            try
            {
                cl_uint n = 0;
                cl_int err = ::clGetPlatformIDs(0, NULL, &n);
                if (err != CL_SUCCESS)
                    g_isOpenCLAvailable = false;
                else
                    g_isOpenCLAvailable = true;
            }
            catch (...)
            {
                g_isOpenCLAvailable = false;
            }
            g_isInitialized = true;
        }
    }
    return g_isOpenCLAvailable;
}

bool useOpenCL()
{
    TLSData* data = TLSData::get();
    if( data->useOpenCL < 0 )
        data->useOpenCL = (int)haveOpenCL();
    return data->useOpenCL > 0;
}

void setUseOpenCL(bool flag)
{
    if( haveOpenCL() )
    {
        TLSData* data = TLSData::get();
        data->useOpenCL = flag ? 1 : 0;
    }
}

void finish2()
{
    Queue::getDefault().finish();
}

#define IMPLEMENT_REFCOUNTABLE() \
    void addref() { CV_XADD(&refcount, 1); } \
    void release() { if( CV_XADD(&refcount, -1) == 1 ) delete this; } \
    int refcount

class Platform
{
public:
    Platform();
    ~Platform();
    Platform(const Platform& p);
    Platform& operator = (const Platform& p);

    void* ptr() const;
    static Platform& getDefault();
protected:
    struct Impl;
    Impl* p;
};

struct Platform::Impl
{
    Impl()
    {
        refcount = 1;
        handle = 0;
        initialized = false;
    }

    ~Impl() {}

    void init()
    {
        if( !initialized )
        {
            //cl_uint num_entries
            cl_uint n = 0;
            if( clGetPlatformIDs(1, &handle, &n) < 0 || n == 0 )
                handle = 0;
            if( handle != 0 )
            {
                char buf[1000];
                size_t len = 0;
                clGetPlatformInfo(handle, CL_PLATFORM_VENDOR, sizeof(buf), buf, &len);
                buf[len] = '\0';
                vendor = String(buf);
            }

            initialized = true;
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_platform_id handle;
    String vendor;
    bool initialized;
};

Platform::Platform()
{
    p = 0;
}

Platform::~Platform()
{
    if(p)
        p->release();
}

Platform::Platform(const Platform& pl)
{
    p = (Impl*)pl.p;
    if(p)
        p->addref();
}

Platform& Platform::operator = (const Platform& pl)
{
    Impl* newp = (Impl*)pl.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

void* Platform::ptr() const
{
    return p ? p->handle : 0;
}

Platform& Platform::getDefault()
{
    static Platform p;
    if( !p.p )
    {
        p.p = new Impl;
        p.p->init();
    }
    return p;
}

///////////////////////////////////////////////////////////////////////////////////

struct Device::Impl
{
    Impl(void* d)
    {
        handle = (cl_device_id)d;
    }

    template<typename _TpCL, typename _TpOut>
    _TpOut getProp(cl_device_info prop) const
    {
        _TpCL temp=_TpCL();
        size_t sz = 0;

        return clGetDeviceInfo(handle, prop, sizeof(temp), &temp, &sz) >= 0 &&
            sz == sizeof(temp) ? _TpOut(temp) : _TpOut();
    }

    bool getBoolProp(cl_device_info prop) const
    {
        cl_bool temp = CL_FALSE;
        size_t sz = 0;

        return clGetDeviceInfo(handle, prop, sizeof(temp), &temp, &sz) >= 0 &&
            sz == sizeof(temp) ? temp != 0 : false;
    }

    String getStrProp(cl_device_info prop) const
    {
        char buf[1024];
        size_t sz=0;
        return clGetDeviceInfo(handle, prop, sizeof(buf)-16, buf, &sz) >= 0 &&
            sz < sizeof(buf) ? String(buf) : String();
    }

    IMPLEMENT_REFCOUNTABLE();
    cl_device_id handle;
};


Device::Device()
{
    p = 0;
}

Device::Device(void* d)
{
    p = 0;
    set(d);
}

Device::Device(const Device& d)
{
    p = d.p;
    if(p)
        p->addref();
}

Device& Device::operator = (const Device& d)
{
    Impl* newp = (Impl*)d.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Device::~Device()
{
    if(p)
        p->release();
}

void Device::set(void* d)
{
    if(p)
        p->release();
    p = new Impl(d);
}

void* Device::ptr() const
{
    return p ? p->handle : 0;
}

String Device::name() const
{ return p ? p->getStrProp(CL_DEVICE_NAME) : String(); }

String Device::extensions() const
{ return p ? p->getStrProp(CL_DEVICE_EXTENSIONS) : String(); }

String Device::vendor() const
{ return p ? p->getStrProp(CL_DEVICE_VENDOR) : String(); }

String Device::OpenCL_C_Version() const
{ return p ? p->getStrProp(CL_DEVICE_OPENCL_C_VERSION) : String(); }

String Device::OpenCLVersion() const
{ return p ? p->getStrProp(CL_DEVICE_EXTENSIONS) : String(); }

String Device::driverVersion() const
{ return p ? p->getStrProp(CL_DRIVER_VERSION) : String(); }

int Device::type() const
{ return p ? p->getProp<cl_device_type, int>(CL_DEVICE_TYPE) : 0; }

int Device::addressBits() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_ADDRESS_BITS) : 0; }

bool Device::available() const
{ return p ? p->getBoolProp(CL_DEVICE_AVAILABLE) : false; }

bool Device::compilerAvailable() const
{ return p ? p->getBoolProp(CL_DEVICE_COMPILER_AVAILABLE) : false; }

bool Device::linkerAvailable() const
#ifdef CL_VERSION_1_2
{ return p ? p->getBoolProp(CL_DEVICE_LINKER_AVAILABLE) : false; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

int Device::doubleFPConfig() const
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_DOUBLE_FP_CONFIG) : 0; }

int Device::singleFPConfig() const
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_SINGLE_FP_CONFIG) : 0; }

int Device::halfFPConfig() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<cl_device_fp_config, int>(CL_DEVICE_HALF_FP_CONFIG) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

bool Device::endianLittle() const
{ return p ? p->getBoolProp(CL_DEVICE_ENDIAN_LITTLE) : false; }

bool Device::errorCorrectionSupport() const
{ return p ? p->getBoolProp(CL_DEVICE_ERROR_CORRECTION_SUPPORT) : false; }

int Device::executionCapabilities() const
{ return p ? p->getProp<cl_device_exec_capabilities, int>(CL_DEVICE_EXECUTION_CAPABILITIES) : 0; }

size_t Device::globalMemCacheSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) : 0; }

int Device::globalMemCacheType() const
{ return p ? p->getProp<cl_device_mem_cache_type, int>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE) : 0; }

int Device::globalMemCacheLineSize() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) : 0; }

size_t Device::globalMemSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_GLOBAL_MEM_SIZE) : 0; }

size_t Device::localMemSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_LOCAL_MEM_SIZE) : 0; }

int Device::localMemType() const
{ return p ? p->getProp<cl_device_local_mem_type, int>(CL_DEVICE_LOCAL_MEM_TYPE) : 0; }

bool Device::hostUnifiedMemory() const
{ return p ? p->getBoolProp(CL_DEVICE_HOST_UNIFIED_MEMORY) : false; }

bool Device::imageSupport() const
{ return p ? p->getBoolProp(CL_DEVICE_IMAGE_SUPPORT) : false; }

size_t Device::image2DMaxWidth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH) : 0; }

size_t Device::image2DMaxHeight() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT) : 0; }

size_t Device::image3DMaxWidth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH) : 0; }

size_t Device::image3DMaxHeight() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT) : 0; }

size_t Device::image3DMaxDepth() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH) : 0; }

size_t Device::imageMaxBufferSize() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

size_t Device::imageMaxArraySize() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif

int Device::maxClockFrequency() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_CLOCK_FREQUENCY) : 0; }

int Device::maxComputeUnits() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_COMPUTE_UNITS) : 0; }

int Device::maxConstantArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_CONSTANT_ARGS) : 0; }

size_t Device::maxConstantBufferSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) : 0; }

size_t Device::maxMemAllocSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_MEM_ALLOC_SIZE) : 0; }

size_t Device::maxParameterSize() const
{ return p ? p->getProp<cl_ulong, size_t>(CL_DEVICE_MAX_PARAMETER_SIZE) : 0; }

int Device::maxReadImageArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_READ_IMAGE_ARGS) : 0; }

int Device::maxWriteImageArgs() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS) : 0; }

int Device::maxSamplers() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_SAMPLERS) : 0; }

size_t Device::maxWorkGroupSize() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE) : 0; }

int Device::maxWorkItemDims() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) : 0; }

void Device::maxWorkItemSizes(size_t* sizes) const
{
    if(p)
    {
        const int MAX_DIMS = 32;
        size_t retsz = 0;
        clGetDeviceInfo(p->handle, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                MAX_DIMS*sizeof(sizes[0]), &sizes[0], &retsz);
    }
}

int Device::memBaseAddrAlign() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) : 0; }

int Device::nativeVectorWidthChar() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR) : 0; }

int Device::nativeVectorWidthShort() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT) : 0; }

int Device::nativeVectorWidthInt() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT) : 0; }

int Device::nativeVectorWidthLong() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG) : 0; }

int Device::nativeVectorWidthFloat() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) : 0; }

int Device::nativeVectorWidthDouble() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE) : 0; }

int Device::nativeVectorWidthHalf() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) : 0; }

int Device::preferredVectorWidthChar() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) : 0; }

int Device::preferredVectorWidthShort() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) : 0; }

int Device::preferredVectorWidthInt() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) : 0; }

int Device::preferredVectorWidthLong() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) : 0; }

int Device::preferredVectorWidthFloat() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) : 0; }

int Device::preferredVectorWidthDouble() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) : 0; }

int Device::preferredVectorWidthHalf() const
{ return p ? p->getProp<cl_uint, int>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) : 0; }

size_t Device::printfBufferSize() const
#ifdef CL_VERSION_1_2
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_PRINTF_BUFFER_SIZE) : 0; }
#else
{ CV_REQUIRE_OPENCL_1_2_ERROR; }
#endif


size_t Device::profilingTimerResolution() const
{ return p ? p->getProp<size_t, size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION) : 0; }

const Device& Device::getDefault()
{
    const Context2& ctx = Context2::getDefault();
    int idx = TLSData::get()->device;
    return ctx.device(idx);
}

/////////////////////////////////////////////////////////////////////////////////////////

struct Context2::Impl
{
    Impl(int dtype0)
    {
        refcount = 1;
        handle = 0;

        cl_int retval = 0;
        cl_platform_id pl = (cl_platform_id)Platform::getDefault().ptr();
        cl_context_properties prop[] =
        {
            CL_CONTEXT_PLATFORM, (cl_context_properties)pl,
            0
        };

        cl_uint i, nd0 = 0, nd = 0;
        int dtype = dtype0 & 15;
        clGetDeviceIDs( pl, dtype, 0, 0, &nd0 );
        if(retval < 0)
            return;
        AutoBuffer<void*> dlistbuf(nd0*2+1);
        cl_device_id* dlist = (cl_device_id*)(void**)dlistbuf;
        cl_device_id* dlist_new = dlist + nd0;
        clGetDeviceIDs(	pl, dtype, nd0, dlist, &nd0 );
        String name0;

        for(i = 0; i < nd0; i++)
        {
            Device d(dlist[i]);
            if( !d.available() || !d.compilerAvailable() )
                continue;
            if( dtype0 == Device::TYPE_DGPU && d.hostUnifiedMemory() )
                continue;
            if( dtype0 == Device::TYPE_IGPU && !d.hostUnifiedMemory() )
                continue;
            String name = d.name();
            if( nd != 0 && name != name0 )
                continue;
            name0 = name;
            dlist_new[nd++] = dlist[i];
        }

        if(nd == 0)
            return;

        // !!! in the current implementation force the number of devices to 1 !!!
        nd = 1;

        handle = clCreateContext(prop, nd, dlist_new, 0, 0, &retval);
        bool ok = handle != 0 && retval >= 0;
        if( ok )
        {
            devices.resize(nd);
            for( i = 0; i < nd; i++ )
                devices[i].set(dlist_new[i]);
        }
    }

    ~Impl()
    {
        if(handle)
            clReleaseContext(handle);
        devices.clear();
    }

    Program getProg(const ProgramSource2& src,
                    const String& buildflags, String& errmsg)
    {
        String prefix = Program::getPrefix(buildflags);
        HashKey k(src.hash(), crc64((const uchar*)prefix.c_str(), prefix.size()));
        phash_t::iterator it = phash.find(k);
        if( it != phash.end() )
            return it->second;
        //String filename = format("%08x%08x_%08x%08x.clb2",
        Program prog(src, buildflags, errmsg);
        if(prog.ptr())
            phash.insert(std::pair<HashKey,Program>(k, prog));
        return prog;
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_context handle;
    std::vector<Device> devices;
    bool initialized;

    typedef ProgramSource2::hash_t hash_t;

    struct HashKey
    {
        HashKey(hash_t _a, hash_t _b) : a(_a), b(_b) {}
        bool operator < (const HashKey& k) const { return a < k.a || (a == k.a && b < k.b); }
        bool operator == (const HashKey& k) const { return a == k.a && b == k.b; }
        bool operator != (const HashKey& k) const { return a != k.a || b != k.b; }
        hash_t a, b;
    };
    typedef std::map<HashKey, Program> phash_t;
    phash_t phash;
};


Context2::Context2()
{
    p = 0;
}

Context2::Context2(int dtype)
{
    p = 0;
    create(dtype);
}

bool Context2::create(int dtype0)
{
    if( !haveOpenCL() )
        return false;
    if(p)
        p->release();
    p = new Impl(dtype0);
    if(!p->handle)
    {
        delete p;
        p = 0;
    }
    return p != 0;
}

Context2::~Context2()
{
    p->release();
}

Context2::Context2(const Context2& c)
{
    p = (Impl*)c.p;
    if(p)
        p->addref();
}

Context2& Context2::operator = (const Context2& c)
{
    Impl* newp = (Impl*)c.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

void* Context2::ptr() const
{
    return p->handle;
}

size_t Context2::ndevices() const
{
    return p ? p->devices.size() : 0;
}

const Device& Context2::device(size_t idx) const
{
    static Device dummy;
    return !p || idx >= p->devices.size() ? dummy : p->devices[idx];
}

Context2& Context2::getDefault()
{
    static Context2 ctx;
    if( !ctx.p && haveOpenCL() )
    {
        // do not create new Context2 right away.
        // First, try to retrieve existing context of the same type.
        // In its turn, Platform::getContext() may call Context2::create()
        // if there is no such context.
        ctx.create(Device::TYPE_ACCELERATOR);
        if(!ctx.p)
            ctx.create(Device::TYPE_DGPU);
        if(!ctx.p)
            ctx.create(Device::TYPE_IGPU);
        if(!ctx.p)
            ctx.create(Device::TYPE_CPU);
    }

    return ctx;
}

Program Context2::getProg(const ProgramSource2& prog,
                         const String& buildopts, String& errmsg)
{
    return p ? p->getProg(prog, buildopts, errmsg) : Program();
}

struct Queue::Impl
{
    Impl(const Context2& c, const Device& d)
    {
        refcount = 1;
        const Context2* pc = &c;
        cl_context ch = (cl_context)pc->ptr();
        if( !ch )
        {
            pc = &Context2::getDefault();
            ch = (cl_context)pc->ptr();
        }
        cl_device_id dh = (cl_device_id)d.ptr();
        if( !dh )
            dh = (cl_device_id)pc->device(0).ptr();
        cl_int retval = 0;
        handle = clCreateCommandQueue(ch, dh, 0, &retval);
    }

    ~Impl()
    {
#ifdef _WIN32
        if (!cv::__termination)
#endif
        {
            if(handle)
            {
                clFinish(handle);
                clReleaseCommandQueue(handle);
            }
        }
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_command_queue handle;
    bool initialized;
};

Queue::Queue()
{
    p = 0;
}

Queue::Queue(const Context2& c, const Device& d)
{
    p = 0;
    create(c, d);
}

Queue::Queue(const Queue& q)
{
    p = q.p;
    if(p)
        p->addref();
}

Queue& Queue::operator = (const Queue& q)
{
    Impl* newp = (Impl*)q.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Queue::~Queue()
{
    if(p)
        p->release();
}

bool Queue::create(const Context2& c, const Device& d)
{
    if(p)
        p->release();
    p = new Impl(c, d);
    return p->handle != 0;
}

void Queue::finish()
{
    if(p && p->handle)
        clFinish(p->handle);
}

void* Queue::ptr() const
{
    return p ? p->handle : 0;
}

Queue& Queue::getDefault()
{
    Queue& q = TLSData::get()->oclQueue;
    if( !q.p && haveOpenCL() )
        q.create(Context2::getDefault());
    return q;
}

static cl_command_queue getQueue(const Queue& q)
{
    cl_command_queue qq = (cl_command_queue)q.ptr();
    if(!qq)
        qq = (cl_command_queue)Queue::getDefault().ptr();
    return qq;
}

KernelArg::KernelArg()
    : flags(0), m(0), obj(0), sz(0), wscale(1)
{
}

KernelArg::KernelArg(int _flags, UMat* _m, int _wscale, const void* _obj, size_t _sz)
    : flags(_flags), m(_m), obj(_obj), sz(_sz), wscale(_wscale)
{
}

KernelArg KernelArg::Constant(const Mat& m)
{
    CV_Assert(m.isContinuous());
    return KernelArg(CONSTANT, 0, 1, m.data, m.total()*m.elemSize());
}


struct Kernel::Impl
{
    Impl(const char* kname, const Program& prog)
    {
        e = 0; refcount = 1;
        cl_program ph = (cl_program)prog.ptr();
        cl_int retval = 0;
        handle = ph != 0 ?
            clCreateKernel(ph, kname, &retval) : 0;
        for( int i = 0; i < MAX_ARRS; i++ )
            u[i] = 0;
        haveTempDstUMats = false;
    }

    void cleanupUMats()
    {
        for( int i = 0; i < MAX_ARRS; i++ )
            if( u[i] )
            {
                if( CV_XADD(&u[i]->urefcount, -1) == 1 )
                    u[i]->currAllocator->deallocate(u[i]);
                u[i] = 0;
            }
        nu = 0;
        haveTempDstUMats = false;
    }

    void addUMat(const UMat& m, bool dst)
    {
        CV_Assert(nu < MAX_ARRS && m.u && m.u->urefcount > 0);
        u[nu] = m.u;
        CV_XADD(&m.u->urefcount, 1);
        nu++;
        if(dst && m.u->tempUMat())
            haveTempDstUMats = true;
    }

    void finit()
    {
        cleanupUMats();
        if(e) { clReleaseEvent(e); e = 0; }
        release();
    }

    ~Impl()
    {
        if(handle)
            clReleaseKernel(handle);
    }

    IMPLEMENT_REFCOUNTABLE();

    cl_kernel handle;
    cl_event e;
    enum { MAX_ARRS = 16 };
    UMatData* u[MAX_ARRS];
    int nu;
    bool haveTempDstUMats;
};

}}

extern "C"
{
static void CL_CALLBACK oclCleanupCallback(cl_event, cl_int, void *p)
{
    ((cv::ocl::Kernel::Impl*)p)->finit();
}

}

namespace cv { namespace ocl {

Kernel::Kernel()
{
    p = 0;
}

Kernel::Kernel(const char* kname, const Program& prog)
{
    p = 0;
    create(kname, prog);
}

Kernel::Kernel(const char* kname, const ProgramSource2& src,
               const String& buildopts, String* errmsg)
{
    p = 0;
    create(kname, src, buildopts, errmsg);
}

Kernel::Kernel(const Kernel& k)
{
    p = k.p;
    if(p)
        p->addref();
}

Kernel& Kernel::operator = (const Kernel& k)
{
    Impl* newp = (Impl*)k.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Kernel::~Kernel()
{
    if(p)
        p->release();
}

bool Kernel::create(const char* kname, const Program& prog)
{
    if(p)
        p->release();
    p = new Impl(kname, prog);
    if(p->handle == 0)
    {
        p->release();
        p = 0;
    }
    return p != 0;
}

bool Kernel::create(const char* kname, const ProgramSource2& src,
                    const String& buildopts, String* errmsg)
{
    if(p)
    {
        p->release();
        p = 0;
    }
    String tempmsg;
    if( !errmsg ) errmsg = &tempmsg;
    const Program& prog = Context2::getDefault().getProg(src, buildopts, *errmsg);
    return create(kname, prog);
}

void* Kernel::ptr() const
{
    return p ? p->handle : 0;
}

bool Kernel::empty() const
{
    return ptr() == 0;
}

int Kernel::set(int i, const void* value, size_t sz)
{
    CV_Assert(i >= 0);
    if( i == 0 )
        p->cleanupUMats();
    if( !p || !p->handle || clSetKernelArg(p->handle, (cl_uint)i, sz, value) < 0 )
        return -1;
    return i+1;
}

int Kernel::set(int i, const UMat& m)
{
    return set(i, KernelArg(KernelArg::READ_WRITE, (UMat*)&m, 0, 0));
}

int Kernel::set(int i, const KernelArg& arg)
{
    CV_Assert( i >= 0 );
    if( !p || !p->handle )
        return -1;
    if( i == 0 )
        p->cleanupUMats();
    if( arg.m )
    {
        int accessFlags = ((arg.flags & KernelArg::READ_ONLY) ? ACCESS_READ : 0) +
                          ((arg.flags & KernelArg::WRITE_ONLY) ? ACCESS_WRITE : 0);
        bool ptronly = (arg.flags & KernelArg::PTR_ONLY) != 0;
        cl_mem h = (cl_mem)arg.m->handle(accessFlags);

        if (ptronly)
            clSetKernelArg(p->handle, (cl_uint)i++, sizeof(h), &h);
        else if( arg.m->dims <= 2 )
        {
            UMat2D u2d(*arg.m);
            clSetKernelArg(p->handle, (cl_uint)i, sizeof(h), &h);
            clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u2d.step), &u2d.step);
            clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u2d.offset), &u2d.offset);
            i += 3;

            if( !(arg.flags & KernelArg::NO_SIZE) )
            {
                int cols = u2d.cols*arg.wscale;
                clSetKernelArg(p->handle, (cl_uint)i, sizeof(u2d.rows), &u2d.rows);
                clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(cols), &cols);
                i += 2;
            }
        }
        else
        {
            UMat3D u3d(*arg.m);
            clSetKernelArg(p->handle, (cl_uint)i, sizeof(h), &h);
            clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u3d.slicestep), &u3d.slicestep);
            clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u3d.step), &u3d.step);
            clSetKernelArg(p->handle, (cl_uint)(i+3), sizeof(u3d.offset), &u3d.offset);
            i += 4;
            if( !(arg.flags & KernelArg::NO_SIZE) )
            {
                int cols = u3d.cols*arg.wscale;
                clSetKernelArg(p->handle, (cl_uint)i, sizeof(u3d.slices), &u3d.rows);
                clSetKernelArg(p->handle, (cl_uint)(i+1), sizeof(u3d.rows), &u3d.rows);
                clSetKernelArg(p->handle, (cl_uint)(i+2), sizeof(u3d.cols), &cols);
                i += 3;
            }
        }
        p->addUMat(*arg.m, (accessFlags & ACCESS_WRITE) != 0);
        return i;
    }
    clSetKernelArg(p->handle, (cl_uint)i, arg.sz, arg.obj);
    return i+1;
}


bool Kernel::run(int dims, size_t _globalsize[], size_t _localsize[],
                 bool sync, const Queue& q)
{
    if(!p || !p->handle || p->e != 0)
        return false;

    cl_command_queue qq = getQueue(q);
    size_t offset[CV_MAX_DIM] = {0}, globalsize[CV_MAX_DIM] = {1,1,1};
    size_t total = 1;
    CV_Assert(_globalsize != 0);
    for (int i = 0; i < dims; i++)
    {
        size_t val = _localsize ? _localsize[i] :
            dims == 1 ? 64 : dims == 2 ? (16>>i) : dims == 3 ? (8>>(int)(i>0)) : 1;
        CV_Assert( val > 0 );
        total *= _globalsize[i];
        globalsize[i] = ((_globalsize[i] + val - 1)/val)*val;
    }
    if( total == 0 )
        return true;
    if( p->haveTempDstUMats )
        sync = true;
    cl_int retval = clEnqueueNDRangeKernel(qq, p->handle, (cl_uint)dims,
                                           offset, globalsize, _localsize, 0, 0,
                                           sync ? 0 : &p->e);
    if( sync || retval < 0 )
    {
        clFinish(qq);
        p->cleanupUMats();
    }
    else
    {
        p->addref();
        clSetEventCallback(p->e, CL_COMPLETE, oclCleanupCallback, p);
    }
    return retval >= 0;
}

bool Kernel::runTask(bool sync, const Queue& q)
{
    if(!p || !p->handle || p->e != 0)
        return false;

    cl_command_queue qq = getQueue(q);
    cl_int retval = clEnqueueTask(qq, p->handle, 0, 0, sync ? 0 : &p->e);
    if( sync || retval < 0 )
    {
        clFinish(qq);
        p->cleanupUMats();
    }
    else
    {
        p->addref();
        clSetEventCallback(p->e, CL_COMPLETE, oclCleanupCallback, p);
    }
    return retval >= 0;
}


size_t Kernel::workGroupSize() const
{
    if(!p)
        return 0;
    size_t val = 0, retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(val), &val, &retsz) >= 0 ? val : 0;
}

bool Kernel::compileWorkGroupSize(size_t wsz[]) const
{
    if(!p || !wsz)
        return 0;
    size_t retsz = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                    sizeof(wsz[0]*3), wsz, &retsz) >= 0;
}

size_t Kernel::localMemSize() const
{
    if(!p)
        return 0;
    size_t retsz = 0;
    cl_ulong val = 0;
    cl_device_id dev = (cl_device_id)Device::getDefault().ptr();
    return clGetKernelWorkGroupInfo(p->handle, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(val), &val, &retsz) >= 0 ? (size_t)val : 0;
}

////////////////////////////////////////////////////////////////////////////////////////

struct Program::Impl
{
    Impl(const ProgramSource2& _src,
         const String& _buildflags, String& errmsg)
    {
        refcount = 1;
        const Context2& ctx = Context2::getDefault();
        src = _src;
        buildflags = _buildflags;
        const String& srcstr = src.source();
        const char* srcptr = srcstr.c_str();
        size_t srclen = srcstr.size();
        cl_int retval = 0;

        handle = clCreateProgramWithSource((cl_context)ctx.ptr(), 1, &srcptr, &srclen, &retval);
        if( handle && retval >= 0 )
        {
            int i, n = (int)ctx.ndevices();
            AutoBuffer<void*> deviceListBuf(n+1);
            void** deviceList = deviceListBuf;
            for( i = 0; i < n; i++ )
                deviceList[i] = ctx.device(i).ptr();

            retval = clBuildProgram(handle, n,
                                    (const cl_device_id*)deviceList,
                                    buildflags.c_str(), 0, 0);
            if( retval < 0 )
            {
                size_t retsz = 0;
                retval = clGetProgramBuildInfo(handle, (cl_device_id)deviceList[0],
                                               CL_PROGRAM_BUILD_LOG, 0, 0, &retsz);
                if( retval >= 0 && retsz > 1 )
                {
                    AutoBuffer<char> bufbuf(retsz + 16);
                    char* buf = bufbuf;
                    retval = clGetProgramBuildInfo(handle, (cl_device_id)deviceList[0],
                                                   CL_PROGRAM_BUILD_LOG, retsz+1, buf, &retsz);
                    if( retval >= 0 )
                    {
                        errmsg = String(buf);
                        CV_Error_(Error::StsAssert, ("OpenCL program can not be built: %s", errmsg.c_str()));
                    }
                }
            }
            CV_Assert(retval >= 0);
        }
    }

    Impl(const String& _buf, const String& _buildflags)
    {
        refcount = 1;
        handle = 0;
        buildflags = _buildflags;
        if(_buf.empty())
            return;
        String prefix0 = Program::getPrefix(buildflags);
        const Context2& ctx = Context2::getDefault();
        const Device& dev = Device::getDefault();
        const char* pos0 = _buf.c_str();
        const char* pos1 = strchr(pos0, '\n');
        if(!pos1)
            return;
        const char* pos2 = strchr(pos1+1, '\n');
        if(!pos2)
            return;
        const char* pos3 = strchr(pos2+1, '\n');
        if(!pos3)
            return;
        size_t prefixlen = (pos3 - pos0)+1;
        String prefix(pos0, prefixlen);
        if( prefix != prefix0 )
            return;
        const uchar* bin = (uchar*)(pos3+1);
        void* devid = dev.ptr();
        size_t codelen = _buf.length() - prefixlen;
        cl_int binstatus = 0, retval = 0;
        handle = clCreateProgramWithBinary((cl_context)ctx.ptr(), 1, (cl_device_id*)&devid,
                                           &codelen, &bin, &binstatus, &retval);
    }

    String store()
    {
        if(!handle)
            return String();
        size_t progsz = 0, retsz = 0;
        String prefix = Program::getPrefix(buildflags);
        size_t prefixlen = prefix.length();
        if(clGetProgramInfo(handle, CL_PROGRAM_BINARY_SIZES, sizeof(progsz), &progsz, &retsz) < 0)
            return String();
        AutoBuffer<uchar> bufbuf(prefixlen + progsz + 16);
        uchar* buf = bufbuf;
        memcpy(buf, prefix.c_str(), prefixlen);
        buf += prefixlen;
        if(clGetProgramInfo(handle, CL_PROGRAM_BINARIES, sizeof(buf), &buf, &retsz) < 0)
            return String();
        buf[progsz] = (uchar)'\0';
        return String((const char*)(uchar*)bufbuf, prefixlen + progsz);
    }

    ~Impl()
    {
        if( handle )
            clReleaseProgram(handle);
    }

    IMPLEMENT_REFCOUNTABLE();

    ProgramSource2 src;
    String buildflags;
    cl_program handle;
};


Program::Program() { p = 0; }

Program::Program(const ProgramSource2& src,
        const String& buildflags, String& errmsg)
{
    p = 0;
    create(src, buildflags, errmsg);
}

Program::Program(const Program& prog)
{
    p = prog.p;
    if(p)
        p->addref();
}

Program& Program::operator = (const Program& prog)
{
    Impl* newp = (Impl*)prog.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

Program::~Program()
{
    if(p)
        p->release();
}

bool Program::create(const ProgramSource2& src,
            const String& buildflags, String& errmsg)
{
    if(p)
        p->release();
    p = new Impl(src, buildflags, errmsg);
    if(!p->handle)
    {
        p->release();
        p = 0;
    }
    return p != 0;
}

const ProgramSource2& Program::source() const
{
    static ProgramSource2 dummy;
    return p ? p->src : dummy;
}

void* Program::ptr() const
{
    return p ? p->handle : 0;
}

bool Program::read(const String& bin, const String& buildflags)
{
    if(p)
        p->release();
    p = new Impl(bin, buildflags);
    return p->handle != 0;
}

bool Program::write(String& bin) const
{
    if(!p)
        return false;
    bin = p->store();
    return !bin.empty();
}

String Program::getPrefix() const
{
    if(!p)
        return String();
    return getPrefix(p->buildflags);
}

String Program::getPrefix(const String& buildflags)
{
    const Context2& ctx = Context2::getDefault();
    const Device& dev = ctx.device(0);
    return format("name=%s\ndriver=%s\nbuildflags=%s\n",
                  dev.name().c_str(), dev.driverVersion().c_str(), buildflags.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////

struct ProgramSource2::Impl
{
    Impl(const char* _src)
    {
        init(String(_src));
    }
    Impl(const String& _src)
    {
        init(_src);
    }
    void init(const String& _src)
    {
        refcount = 1;
        src = _src;
        h = crc64((uchar*)src.c_str(), src.size());
    }

    IMPLEMENT_REFCOUNTABLE();
    String src;
    ProgramSource2::hash_t h;
};


ProgramSource2::ProgramSource2()
{
    p = 0;
}

ProgramSource2::ProgramSource2(const char* prog)
{
    p = new Impl(prog);
}

ProgramSource2::ProgramSource2(const String& prog)
{
    p = new Impl(prog);
}

ProgramSource2::~ProgramSource2()
{
    if(p)
        p->release();
}

ProgramSource2::ProgramSource2(const ProgramSource2& prog)
{
    p = prog.p;
    if(p)
        p->addref();
}

ProgramSource2& ProgramSource2::operator = (const ProgramSource2& prog)
{
    Impl* newp = (Impl*)prog.p;
    if(newp)
        newp->addref();
    if(p)
        p->release();
    p = newp;
    return *this;
}

const String& ProgramSource2::source() const
{
    static String dummy;
    return p ? p->src : dummy;
}

ProgramSource2::hash_t ProgramSource2::hash() const
{
    return p ? p->h : 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////

class OpenCLAllocator : public MatAllocator
{
public:
    OpenCLAllocator() { matStdAllocator = Mat::getStdAllocator(); }

    UMatData* defaultAllocate(int dims, const int* sizes, int type, void* data, size_t* step, int flags) const
    {
        UMatData* u = matStdAllocator->allocate(dims, sizes, type, data, step, flags);
        u->urefcount = 1;
        u->refcount = 0;
        return u;
    }

    void getBestFlags(const Context2& ctx, int /*flags*/, int& createFlags, int& flags0) const
    {
        const Device& dev = ctx.device(0);
        createFlags = CL_MEM_READ_WRITE;

        if( dev.hostUnifiedMemory() )
            flags0 = 0;
        else
            flags0 = UMatData::COPY_ON_MAP;
    }

    UMatData* allocate(int dims, const int* sizes, int type,
                       void* data, size_t* step, int flags) const
    {
        if(!useOpenCL())
            return defaultAllocate(dims, sizes, type, data, step, flags);
        CV_Assert(data == 0);
        size_t total = CV_ELEM_SIZE(type);
        for( int i = dims-1; i >= 0; i-- )
        {
            if( step )
                step[i] = total;
            total *= sizes[i];
        }

        Context2& ctx = Context2::getDefault();
        int createFlags = 0, flags0 = 0;
        getBestFlags(ctx, flags, createFlags, flags0);

        cl_int retval = 0;
        void* handle = clCreateBuffer((cl_context)ctx.ptr(),
                                      createFlags, total, 0, &retval);
        if( !handle || retval < 0 )
            return defaultAllocate(dims, sizes, type, data, step, flags);
        UMatData* u = new UMatData(this);
        u->data = 0;
        u->size = total;
        u->handle = handle;
        u->urefcount = 1;
        u->flags = flags0;

        return u;
    }

    bool allocate(UMatData* u, int accessFlags) const
    {
        if(!u)
            return false;

        UMatDataAutoLock lock(u);

        if(u->handle == 0)
        {
            CV_Assert(u->origdata != 0);
            Context2& ctx = Context2::getDefault();
            int createFlags = 0, flags0 = 0;
            getBestFlags(ctx, accessFlags, createFlags, flags0);

            cl_context ctx_handle = (cl_context)ctx.ptr();
            cl_int retval = 0;
            int tempUMatFlags = UMatData::TEMP_UMAT;
            u->handle = clCreateBuffer(ctx_handle, CL_MEM_USE_HOST_PTR|createFlags,
                                       u->size, u->origdata, &retval);
            if((!u->handle || retval < 0) && !(accessFlags & ACCESS_FAST))
            {
                u->handle = clCreateBuffer(ctx_handle, CL_MEM_COPY_HOST_PTR|createFlags,
                                           u->size, u->origdata, &retval);
                tempUMatFlags = UMatData::TEMP_COPIED_UMAT;
            }
            if(!u->handle || retval < 0)
                return false;
            u->prevAllocator = u->currAllocator;
            u->currAllocator = this;
            u->flags |= tempUMatFlags;
        }
        if(accessFlags & ACCESS_WRITE)
            u->markHostCopyObsolete(true);
        CV_XADD(&u->urefcount, 1);
        return true;
    }

    /*void sync(UMatData* u) const
    {
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
        UMatDataAutoLock lock(u);

        if( u->hostCopyObsolete() && u->handle && u->refcount > 0 && u->origdata)
        {
            if( u->tempCopiedUMat() )
            {
                clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                    u->size, u->origdata, 0, 0, 0);
            }
            else
            {
                cl_int retval = 0;
                void* data = clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                                (CL_MAP_READ | CL_MAP_WRITE),
                                                0, u->size, 0, 0, 0, &retval);
                clEnqueueUnmapMemObject(q, (cl_mem)u->handle, data, 0, 0, 0);
                clFinish(q);
            }
            u->markHostCopyObsolete(false);
        }
        else if( u->copyOnMap() && u->deviceCopyObsolete() && u->data )
        {
            clEnqueueWriteBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                 u->size, u->data, 0, 0, 0);
        }
    }*/

    void deallocate(UMatData* u) const
    {
        if(!u)
            return;

        // TODO: !!! when we add Shared Virtual Memory Support,
        // this function (as well as the others) should be corrected
        CV_Assert(u->handle != 0 && u->urefcount == 0);
        if(u->tempUMat())
        {
            UMatDataAutoLock lock(u);
            if( u->hostCopyObsolete() && u->refcount > 0 )
            {
                cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
                if( u->tempCopiedUMat() )
                {
                    clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                        u->size, u->origdata, 0, 0, 0);
                }
                else
                {
                    cl_int retval = 0;
                    void* data = clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                                    (CL_MAP_READ | CL_MAP_WRITE),
                                                    0, u->size, 0, 0, 0, &retval);
                    clEnqueueUnmapMemObject(q, (cl_mem)u->handle, data, 0, 0, 0);
                    clFinish(q);
                }
            }
            u->markHostCopyObsolete(false);
            clReleaseMemObject((cl_mem)u->handle);
            u->handle = 0;
            u->currAllocator = u->prevAllocator;
            if(u->data && u->copyOnMap() && !(u->flags & UMatData::USER_ALLOCATED))
                fastFree(u->data);
            u->data = u->origdata;
            if(u->refcount == 0)
                u->currAllocator->deallocate(u);
        }
        else
        {
            CV_Assert(u->refcount == 0);
            if(u->data && u->copyOnMap() && !(u->flags & UMatData::USER_ALLOCATED))
            {
                fastFree(u->data);
                u->data = 0;
            }
            clReleaseMemObject((cl_mem)u->handle);
            u->handle = 0;
            delete u;
        }
    }

    void map(UMatData* u, int accessFlags) const
    {
        if(!u)
            return;

        CV_Assert( u->handle != 0 );

        UMatDataAutoLock autolock(u);

        if(accessFlags & ACCESS_WRITE)
            u->markDeviceCopyObsolete(true);

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        if( u->refcount == 0 )
        {
            if( !u->copyOnMap() )
            {
                CV_Assert(u->data == 0);
                // because there can be other map requests for the same UMat with different access flags,
                // we use the universal (read-write) access mode.
                cl_int retval = 0;
                u->data = (uchar*)clEnqueueMapBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                                     (CL_MAP_READ | CL_MAP_WRITE),
                                                     0, u->size, 0, 0, 0, &retval);
                if(u->data && retval >= 0)
                {
                    u->markHostCopyObsolete(false);
                    return;
                }

                // if map failed, switch to copy-on-map mode for the particular buffer
                u->flags |= UMatData::COPY_ON_MAP;
            }

            if(!u->data)
            {
                u->data = (uchar*)fastMalloc(u->size);
                u->markHostCopyObsolete(true);
            }
        }

        if( (accessFlags & ACCESS_READ) != 0 && u->hostCopyObsolete() )
        {
            CV_Assert( clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                           u->size, u->data, 0, 0, 0) >= 0 );
            u->markHostCopyObsolete(false);
        }
    }

    void unmap(UMatData* u) const
    {
        if(!u)
            return;

        CV_Assert(u->handle != 0);

        UMatDataAutoLock autolock(u);

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();
        cl_int retval = 0;
        if( !u->copyOnMap() && u->data )
        {
            CV_Assert( (retval = clEnqueueUnmapMemObject(q,
                                (cl_mem)u->handle, u->data, 0, 0, 0)) >= 0 );
            clFinish(q);
            u->data = 0;
        }
        else if( u->copyOnMap() && u->deviceCopyObsolete() )
        {
            CV_Assert( (retval = clEnqueueWriteBuffer(q, (cl_mem)u->handle, CL_TRUE, 0,
                                u->size, u->data, 0, 0, 0)) >= 0 );
        }
        u->markDeviceCopyObsolete(false);
        u->markHostCopyObsolete(false);
    }

    bool checkContinuous(int dims, const size_t sz[],
                         const size_t srcofs[], const size_t srcstep[],
                         const size_t dstofs[], const size_t dststep[],
                         size_t& total, size_t new_sz[],
                         size_t& srcrawofs, size_t new_srcofs[], size_t new_srcstep[],
                         size_t& dstrawofs, size_t new_dstofs[], size_t new_dststep[]) const
    {
        bool iscontinuous = true;
        srcrawofs = srcofs ? srcofs[dims-1] : 0;
        dstrawofs = dstofs ? dstofs[dims-1] : 0;
        total = sz[dims-1];
        for( int i = dims-2; i >= 0; i-- )
        {
            if( i >= 0 && (total != srcstep[i] || total != dststep[i]) )
                iscontinuous = false;
            total *= sz[i];
            if( srcofs )
                srcrawofs += srcofs[i]*srcstep[i];
            if( dstofs )
                dstrawofs += dstofs[i]*dststep[i];
        }

        if( !iscontinuous )
        {
            // OpenCL uses {x, y, z} order while OpenCV uses {z, y, x} order.
            if( dims == 2 )
            {
                new_sz[0] = sz[1]; new_sz[1] = sz[0]; new_sz[2] = 1;
                // we assume that new_... arrays are initialized by caller
                // with 0's, so there is no else branch
                if( srcofs )
                {
                    new_srcofs[0] = srcofs[1];
                    new_srcofs[1] = srcofs[0];
                    new_srcofs[2] = 0;
                }

                if( dstofs )
                {
                    new_dstofs[0] = dstofs[1];
                    new_dstofs[1] = dstofs[0];
                    new_dstofs[2] = 0;
                }

                new_srcstep[0] = srcstep[0]; new_srcstep[1] = 0;
                new_dststep[0] = dststep[0]; new_dststep[1] = 0;
            }
            else
            {
                // we could check for dims == 3 here,
                // but from user perspective this one is more informative
                CV_Assert(dims <= 3);
                new_sz[0] = sz[2]; new_sz[1] = sz[1]; new_sz[2] = sz[0];
                if( srcofs )
                {
                    new_srcofs[0] = srcofs[2];
                    new_srcofs[1] = srcofs[1];
                    new_srcofs[2] = srcofs[0];
                }

                if( dstofs )
                {
                    new_dstofs[0] = dstofs[2];
                    new_dstofs[1] = dstofs[1];
                    new_dstofs[2] = dstofs[0];
                }

                new_srcstep[0] = srcstep[1]; new_srcstep[1] = srcstep[0];
                new_dststep[0] = dststep[1]; new_dststep[1] = dststep[0];
            }
        }
        return iscontinuous;
    }

    void download(UMatData* u, void* dstptr, int dims, const size_t sz[],
                  const size_t srcofs[], const size_t srcstep[],
                  const size_t dststep[]) const
    {
        if(!u)
            return;
        UMatDataAutoLock autolock(u);

        if( u->data && !u->hostCopyObsolete() )
        {
            Mat::getStdAllocator()->download(u, dstptr, dims, sz, srcofs, srcstep, dststep);
            return;
        }
        CV_Assert( u->handle != 0 );

        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, srcofs, srcstep, 0, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);
        if( iscontinuous )
        {
            CV_Assert( clEnqueueReadBuffer(q, (cl_mem)u->handle, CL_TRUE,
                                           srcrawofs, total, dstptr, 0, 0, 0) >= 0 );
        }
        else
        {
            CV_Assert( clEnqueueReadBufferRect(q, (cl_mem)u->handle, CL_TRUE,
                            new_srcofs, new_dstofs, new_sz, new_srcstep[0], new_srcstep[1],
                            new_dststep[0], new_dststep[1], dstptr, 0, 0, 0) >= 0 );
        }
    }

    void upload(UMatData* u, const void* srcptr, int dims, const size_t sz[],
                const size_t dstofs[], const size_t dststep[],
                const size_t srcstep[]) const
    {
        if(!u)
            return;

        // there should be no user-visible CPU copies of the UMat which we are going to copy to
        CV_Assert(u->refcount == 0 || u->tempUMat());

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, 0, srcstep, dstofs, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

        UMatDataAutoLock autolock(u);

        // if there is cached CPU copy of the GPU matrix,
        // we could use it as a destination.
        // we can do it in 2 cases:
        //    1. we overwrite the whole content
        //    2. we overwrite part of the matrix, but the GPU copy is out-of-date
        if( u->data && (u->hostCopyObsolete() <= u->deviceCopyObsolete() || total == u->size))
        {
            Mat::getStdAllocator()->upload(u, srcptr, dims, sz, dstofs, dststep, srcstep);
            u->markHostCopyObsolete(false);
            u->markDeviceCopyObsolete(true);
            return;
        }

        CV_Assert( u->handle != 0 );
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        if( iscontinuous )
        {
            int crc = 0;
            for( size_t i = 0; i < total; i++ )
                crc ^= ((uchar*)srcptr)[i];
            CV_Assert( clEnqueueWriteBuffer(q, (cl_mem)u->handle,
                CL_TRUE, dstrawofs, total, srcptr, 0, 0, 0) >= 0 );
        }
        else
        {
            CV_Assert( clEnqueueWriteBufferRect(q, (cl_mem)u->handle, CL_TRUE,
                new_dstofs, new_srcofs, new_sz, new_dststep[0], new_dststep[1],
                new_srcstep[0], new_srcstep[1], srcptr, 0, 0, 0) >= 0 );
        }

        u->markHostCopyObsolete(true);
        u->markDeviceCopyObsolete(false);

        clFinish(q);
    }

    void copy(UMatData* src, UMatData* dst, int dims, const size_t sz[],
              const size_t srcofs[], const size_t srcstep[],
              const size_t dstofs[], const size_t dststep[], bool _sync) const
    {
        if(!src || !dst)
            return;

        size_t total = 0, new_sz[] = {0, 0, 0};
        size_t srcrawofs = 0, new_srcofs[] = {0, 0, 0}, new_srcstep[] = {0, 0, 0};
        size_t dstrawofs = 0, new_dstofs[] = {0, 0, 0}, new_dststep[] = {0, 0, 0};

        bool iscontinuous = checkContinuous(dims, sz, srcofs, srcstep, dstofs, dststep,
                                            total, new_sz,
                                            srcrawofs, new_srcofs, new_srcstep,
                                            dstrawofs, new_dstofs, new_dststep);

        UMatDataAutoLock src_autolock(src);
        UMatDataAutoLock dst_autolock(dst);

        if( !src->handle || (src->data && src->hostCopyObsolete() <= src->deviceCopyObsolete()) )
        {
            upload(dst, src->data + srcrawofs, dims, sz, dstofs, dststep, srcstep);
            return;
        }
        if( !dst->handle || (dst->data && dst->hostCopyObsolete() <= dst->deviceCopyObsolete()) )
        {
            download(src, dst->data + dstrawofs, dims, sz, srcofs, srcstep, dststep);
            dst->markHostCopyObsolete(false);
            dst->markDeviceCopyObsolete(true);
            return;
        }

        // there should be no user-visible CPU copies of the UMat which we are going to copy to
        CV_Assert(dst->refcount == 0);
        cl_command_queue q = (cl_command_queue)Queue::getDefault().ptr();

        if( iscontinuous )
        {
            CV_Assert( clEnqueueCopyBuffer(q, (cl_mem)src->handle, (cl_mem)dst->handle,
                                           srcrawofs, dstrawofs, total, 0, 0, 0) >= 0 );
        }
        else
        {
            cl_int retval;
            CV_Assert( (retval = clEnqueueCopyBufferRect(q, (cl_mem)src->handle, (cl_mem)dst->handle,
                                               new_srcofs, new_dstofs, new_sz,
                                               new_srcstep[0], new_srcstep[1],
                                               new_dststep[0], new_dststep[1],
                                               0, 0, 0)) >= 0 );
        }

        dst->markHostCopyObsolete(true);
        dst->markDeviceCopyObsolete(false);

        if( _sync )
            clFinish(q);
    }

    MatAllocator* matStdAllocator;
};

MatAllocator* getOpenCLAllocator()
{
    static OpenCLAllocator allocator;
    return &allocator;
}

const char* typeToStr(int t)
{
    static const char* tab[]=
    {
        "uchar", "uchar2", "uchar3", "uchar4",
        "char", "char2", "char3", "char4",
        "ushort", "ushort2", "ushort3", "ushort4",
        "short", "short2", "short3", "short4",
        "int", "int2", "int3", "int4",
        "float", "float2", "float3", "float4",
        "double", "double2", "double3", "double4",
        "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(t);
    return cn > 4 ? "?" : tab[CV_MAT_DEPTH(t)*4 + cn-1];
}

const char* memopTypeToStr(int t)
{
    static const char* tab[]=
    {
        "uchar", "uchar2", "uchar3", "uchar4",
        "uchar", "uchar2", "uchar3", "uchar4",
        "ushort", "ushort2", "ushort3", "ushort4",
        "ushort", "ushort2", "ushort3", "ushort4",
        "int", "int2", "int3", "int4",
        "int", "int2", "int3", "int4",
        "int2", "int4", "?", "int8",
        "?", "?", "?", "?"
    };
    int cn = CV_MAT_CN(t);
    return cn > 4 ? "?" : tab[CV_MAT_DEPTH(t)*4 + cn-1];
}

const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf)
{
    if( sdepth == ddepth )
        return "noconvert";
    const char *typestr = typeToStr(CV_MAKETYPE(ddepth, cn));
    if( ddepth >= CV_32F ||
        (ddepth == CV_32S && sdepth < CV_32S) ||
        (ddepth == CV_16S && sdepth <= CV_8S) ||
        (ddepth == CV_16U && sdepth == CV_8U))
    {
        sprintf(buf, "convert_%s", typestr);
    }
    else if( sdepth >= CV_32F )
    {
        sprintf(buf, "convert_%s%s_rte", typestr, (ddepth < CV_32S ? "_sat" : ""));
    }
    else
    {
        sprintf(buf, "convert_%s_sat", typestr);
    }
    return buf;
}

}}
