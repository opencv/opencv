/*******************************************************************************
 * Copyright (c) 2008-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/
/*****************************************************************************\

Copyright (c) 2013-2020 Intel Corporation All Rights Reserved.

THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

File Name: cl_ext_intel.h

Abstract:

Notes:

\*****************************************************************************/

#ifndef __CL_EXT_INTEL_H
#define __CL_EXT_INTEL_H

#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/***************************************
* cl_intel_thread_local_exec extension *
****************************************/

#define cl_intel_thread_local_exec 1

#define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL      (((cl_bitfield)1) << 31)

/***********************************************
* cl_intel_device_partition_by_names extension *
************************************************/

#define cl_intel_device_partition_by_names 1

#define CL_DEVICE_PARTITION_BY_NAMES_INTEL          0x4052
#define CL_PARTITION_BY_NAMES_LIST_END_INTEL        -1

/************************************************
* cl_intel_accelerator extension                *
* cl_intel_motion_estimation extension          *
* cl_intel_advanced_motion_estimation extension *
*************************************************/

#define cl_intel_accelerator 1
#define cl_intel_motion_estimation 1
#define cl_intel_advanced_motion_estimation 1

typedef struct _cl_accelerator_intel* cl_accelerator_intel;
typedef cl_uint cl_accelerator_type_intel;
typedef cl_uint cl_accelerator_info_intel;

typedef struct _cl_motion_estimation_desc_intel {
    cl_uint mb_block_type;
    cl_uint subpixel_mode;
    cl_uint sad_adjust_mode;
    cl_uint search_path_type;
} cl_motion_estimation_desc_intel;

/* error codes */
#define CL_INVALID_ACCELERATOR_INTEL                              -1094
#define CL_INVALID_ACCELERATOR_TYPE_INTEL                         -1095
#define CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL                   -1096
#define CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL                   -1097

/* cl_accelerator_type_intel */
#define CL_ACCELERATOR_TYPE_MOTION_ESTIMATION_INTEL               0x0

/* cl_accelerator_info_intel */
#define CL_ACCELERATOR_DESCRIPTOR_INTEL                           0x4090
#define CL_ACCELERATOR_REFERENCE_COUNT_INTEL                      0x4091
#define CL_ACCELERATOR_CONTEXT_INTEL                              0x4092
#define CL_ACCELERATOR_TYPE_INTEL                                 0x4093

/* cl_motion_detect_desc_intel flags */
#define CL_ME_MB_TYPE_16x16_INTEL                                 0x0
#define CL_ME_MB_TYPE_8x8_INTEL                                   0x1
#define CL_ME_MB_TYPE_4x4_INTEL                                   0x2

#define CL_ME_SUBPIXEL_MODE_INTEGER_INTEL                         0x0
#define CL_ME_SUBPIXEL_MODE_HPEL_INTEL                            0x1
#define CL_ME_SUBPIXEL_MODE_QPEL_INTEL                            0x2

#define CL_ME_SAD_ADJUST_MODE_NONE_INTEL                          0x0
#define CL_ME_SAD_ADJUST_MODE_HAAR_INTEL                          0x1

#define CL_ME_SEARCH_PATH_RADIUS_2_2_INTEL                        0x0
#define CL_ME_SEARCH_PATH_RADIUS_4_4_INTEL                        0x1
#define CL_ME_SEARCH_PATH_RADIUS_16_12_INTEL                      0x5

#define CL_ME_SKIP_BLOCK_TYPE_16x16_INTEL                         0x0
#define CL_ME_CHROMA_INTRA_PREDICT_ENABLED_INTEL                  0x1
#define CL_ME_LUMA_INTRA_PREDICT_ENABLED_INTEL                    0x2
#define CL_ME_SKIP_BLOCK_TYPE_8x8_INTEL                           0x4

#define CL_ME_FORWARD_INPUT_MODE_INTEL                            0x1
#define CL_ME_BACKWARD_INPUT_MODE_INTEL                           0x2
#define CL_ME_BIDIRECTION_INPUT_MODE_INTEL                        0x3

#define CL_ME_BIDIR_WEIGHT_QUARTER_INTEL                          16
#define CL_ME_BIDIR_WEIGHT_THIRD_INTEL                            21
#define CL_ME_BIDIR_WEIGHT_HALF_INTEL                             32
#define CL_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL                        43
#define CL_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL                    48

#define CL_ME_COST_PENALTY_NONE_INTEL                             0x0
#define CL_ME_COST_PENALTY_LOW_INTEL                              0x1
#define CL_ME_COST_PENALTY_NORMAL_INTEL                           0x2
#define CL_ME_COST_PENALTY_HIGH_INTEL                             0x3

#define CL_ME_COST_PRECISION_QPEL_INTEL                           0x0
#define CL_ME_COST_PRECISION_HPEL_INTEL                           0x1
#define CL_ME_COST_PRECISION_PEL_INTEL                            0x2
#define CL_ME_COST_PRECISION_DPEL_INTEL                           0x3

#define CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL                  0x0
#define CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL                0x1
#define CL_ME_LUMA_PREDICTOR_MODE_DC_INTEL                        0x2
#define CL_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL        0x3

#define CL_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL       0x4
#define CL_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL                     0x4
#define CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL            0x5
#define CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL           0x6
#define CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL             0x7
#define CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL             0x8

#define CL_ME_CHROMA_PREDICTOR_MODE_DC_INTEL                      0x0
#define CL_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL              0x1
#define CL_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL                0x2
#define CL_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL                   0x3

/* cl_device_info */
#define CL_DEVICE_ME_VERSION_INTEL                                0x407E

#define CL_ME_VERSION_LEGACY_INTEL                                0x0
#define CL_ME_VERSION_ADVANCED_VER_1_INTEL                        0x1
#define CL_ME_VERSION_ADVANCED_VER_2_INTEL                        0x2

extern CL_API_ENTRY cl_accelerator_intel CL_API_CALL
clCreateAcceleratorINTEL(
    cl_context                   context,
    cl_accelerator_type_intel    accelerator_type,
    size_t                       descriptor_size,
    const void*                  descriptor,
    cl_int*                      errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_accelerator_intel (CL_API_CALL *clCreateAcceleratorINTEL_fn)(
    cl_context                   context,
    cl_accelerator_type_intel    accelerator_type,
    size_t                       descriptor_size,
    const void*                  descriptor,
    cl_int*                      errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetAcceleratorInfoINTEL(
    cl_accelerator_intel         accelerator,
    cl_accelerator_info_intel    param_name,
    size_t                       param_value_size,
    void*                        param_value,
    size_t*                      param_value_size_ret) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetAcceleratorInfoINTEL_fn)(
    cl_accelerator_intel         accelerator,
    cl_accelerator_info_intel    param_name,
    size_t                       param_value_size,
    void*                        param_value,
    size_t*                      param_value_size_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainAcceleratorINTEL(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clRetainAcceleratorINTEL_fn)(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseAcceleratorINTEL(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clReleaseAcceleratorINTEL_fn)(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

/******************************************
* cl_intel_simultaneous_sharing extension *
*******************************************/

#define cl_intel_simultaneous_sharing 1

#define CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL            0x4104
#define CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL        0x4105

/***********************************
* cl_intel_egl_image_yuv extension *
************************************/

#define cl_intel_egl_image_yuv 1

#define CL_EGL_YUV_PLANE_INTEL                           0x4107

/********************************
* cl_intel_packed_yuv extension *
*********************************/

#define cl_intel_packed_yuv 1

#define CL_YUYV_INTEL                                    0x4076
#define CL_UYVY_INTEL                                    0x4077
#define CL_YVYU_INTEL                                    0x4078
#define CL_VYUY_INTEL                                    0x4079

/********************************************
* cl_intel_required_subgroup_size extension *
*********************************************/

#define cl_intel_required_subgroup_size 1

#define CL_DEVICE_SUB_GROUP_SIZES_INTEL                  0x4108
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL                   0x4109
#define CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL           0x410A

/****************************************
* cl_intel_driver_diagnostics extension *
*****************************************/

#define cl_intel_driver_diagnostics 1

typedef cl_uint cl_diagnostics_verbose_level;

#define CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL                0x4106

#define CL_CONTEXT_DIAGNOSTICS_LEVEL_ALL_INTEL           ( 0xff )
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL          ( 1 )
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL           ( 1 << 1 )
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL       ( 1 << 2 )

/********************************
* cl_intel_planar_yuv extension *
*********************************/

#define CL_NV12_INTEL                                       0x410E

#define CL_MEM_NO_ACCESS_INTEL                              ( 1 << 24 )
#define CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL              ( 1 << 25 )

#define CL_DEVICE_PLANAR_YUV_MAX_WIDTH_INTEL                0x417E
#define CL_DEVICE_PLANAR_YUV_MAX_HEIGHT_INTEL               0x417F

/*******************************************************
* cl_intel_device_side_avc_motion_estimation extension *
********************************************************/

#define CL_DEVICE_AVC_ME_VERSION_INTEL                      0x410B
#define CL_DEVICE_AVC_ME_SUPPORTS_TEXTURE_SAMPLER_USE_INTEL 0x410C
#define CL_DEVICE_AVC_ME_SUPPORTS_PREEMPTION_INTEL          0x410D

#define CL_AVC_ME_VERSION_0_INTEL                           0x0   /* No support. */
#define CL_AVC_ME_VERSION_1_INTEL                           0x1   /* First supported version. */

#define CL_AVC_ME_MAJOR_16x16_INTEL                         0x0
#define CL_AVC_ME_MAJOR_16x8_INTEL                          0x1
#define CL_AVC_ME_MAJOR_8x16_INTEL                          0x2
#define CL_AVC_ME_MAJOR_8x8_INTEL                           0x3

#define CL_AVC_ME_MINOR_8x8_INTEL                           0x0
#define CL_AVC_ME_MINOR_8x4_INTEL                           0x1
#define CL_AVC_ME_MINOR_4x8_INTEL                           0x2
#define CL_AVC_ME_MINOR_4x4_INTEL                           0x3

#define CL_AVC_ME_MAJOR_FORWARD_INTEL                       0x0
#define CL_AVC_ME_MAJOR_BACKWARD_INTEL                      0x1
#define CL_AVC_ME_MAJOR_BIDIRECTIONAL_INTEL                 0x2

#define CL_AVC_ME_PARTITION_MASK_ALL_INTEL                  0x0
#define CL_AVC_ME_PARTITION_MASK_16x16_INTEL                0x7E
#define CL_AVC_ME_PARTITION_MASK_16x8_INTEL                 0x7D
#define CL_AVC_ME_PARTITION_MASK_8x16_INTEL                 0x7B
#define CL_AVC_ME_PARTITION_MASK_8x8_INTEL                  0x77
#define CL_AVC_ME_PARTITION_MASK_8x4_INTEL                  0x6F
#define CL_AVC_ME_PARTITION_MASK_4x8_INTEL                  0x5F
#define CL_AVC_ME_PARTITION_MASK_4x4_INTEL                  0x3F

#define CL_AVC_ME_SEARCH_WINDOW_EXHAUSTIVE_INTEL            0x0
#define CL_AVC_ME_SEARCH_WINDOW_SMALL_INTEL                 0x1
#define CL_AVC_ME_SEARCH_WINDOW_TINY_INTEL                  0x2
#define CL_AVC_ME_SEARCH_WINDOW_EXTRA_TINY_INTEL            0x3
#define CL_AVC_ME_SEARCH_WINDOW_DIAMOND_INTEL               0x4
#define CL_AVC_ME_SEARCH_WINDOW_LARGE_DIAMOND_INTEL         0x5
#define CL_AVC_ME_SEARCH_WINDOW_RESERVED0_INTEL             0x6
#define CL_AVC_ME_SEARCH_WINDOW_RESERVED1_INTEL             0x7
#define CL_AVC_ME_SEARCH_WINDOW_CUSTOM_INTEL                0x8
#define CL_AVC_ME_SEARCH_WINDOW_16x12_RADIUS_INTEL          0x9
#define CL_AVC_ME_SEARCH_WINDOW_4x4_RADIUS_INTEL            0x2
#define CL_AVC_ME_SEARCH_WINDOW_2x2_RADIUS_INTEL            0xa

#define CL_AVC_ME_SAD_ADJUST_MODE_NONE_INTEL                0x0
#define CL_AVC_ME_SAD_ADJUST_MODE_HAAR_INTEL                0x2

#define CL_AVC_ME_SUBPIXEL_MODE_INTEGER_INTEL               0x0
#define CL_AVC_ME_SUBPIXEL_MODE_HPEL_INTEL                  0x1
#define CL_AVC_ME_SUBPIXEL_MODE_QPEL_INTEL                  0x3

#define CL_AVC_ME_COST_PRECISION_QPEL_INTEL                 0x0
#define CL_AVC_ME_COST_PRECISION_HPEL_INTEL                 0x1
#define CL_AVC_ME_COST_PRECISION_PEL_INTEL                  0x2
#define CL_AVC_ME_COST_PRECISION_DPEL_INTEL                 0x3

#define CL_AVC_ME_BIDIR_WEIGHT_QUARTER_INTEL                0x10
#define CL_AVC_ME_BIDIR_WEIGHT_THIRD_INTEL                  0x15
#define CL_AVC_ME_BIDIR_WEIGHT_HALF_INTEL                   0x20
#define CL_AVC_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL              0x2B
#define CL_AVC_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL          0x30

#define CL_AVC_ME_BORDER_REACHED_LEFT_INTEL                 0x0
#define CL_AVC_ME_BORDER_REACHED_RIGHT_INTEL                0x2
#define CL_AVC_ME_BORDER_REACHED_TOP_INTEL                  0x4
#define CL_AVC_ME_BORDER_REACHED_BOTTOM_INTEL               0x8

#define CL_AVC_ME_SKIP_BLOCK_PARTITION_16x16_INTEL          0x0
#define CL_AVC_ME_SKIP_BLOCK_PARTITION_8x8_INTEL            0x4000

#define CL_AVC_ME_SKIP_BLOCK_16x16_FORWARD_ENABLE_INTEL     ( 0x1 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_16x16_BACKWARD_ENABLE_INTEL    ( 0x2 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_16x16_DUAL_ENABLE_INTEL        ( 0x3 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_FORWARD_ENABLE_INTEL       ( 0x55 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_BACKWARD_ENABLE_INTEL      ( 0xAA << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_DUAL_ENABLE_INTEL          ( 0xFF << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_0_FORWARD_ENABLE_INTEL     ( 0x1 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_0_BACKWARD_ENABLE_INTEL    ( 0x2 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_1_FORWARD_ENABLE_INTEL     ( 0x1 << 26 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_1_BACKWARD_ENABLE_INTEL    ( 0x2 << 26 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_2_FORWARD_ENABLE_INTEL     ( 0x1 << 28 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_2_BACKWARD_ENABLE_INTEL    ( 0x2 << 28 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_3_FORWARD_ENABLE_INTEL     ( 0x1 << 30 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_3_BACKWARD_ENABLE_INTEL    ( 0x2 << 30 )

#define CL_AVC_ME_BLOCK_BASED_SKIP_4x4_INTEL                0x00
#define CL_AVC_ME_BLOCK_BASED_SKIP_8x8_INTEL                0x80

#define CL_AVC_ME_INTRA_16x16_INTEL                         0x0
#define CL_AVC_ME_INTRA_8x8_INTEL                           0x1
#define CL_AVC_ME_INTRA_4x4_INTEL                           0x2

#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_16x16_INTEL     0x6
#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_8x8_INTEL       0x5
#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_4x4_INTEL       0x3

#define CL_AVC_ME_INTRA_NEIGHBOR_LEFT_MASK_ENABLE_INTEL         0x60
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_MASK_ENABLE_INTEL        0x10
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_RIGHT_MASK_ENABLE_INTEL  0x8
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_LEFT_MASK_ENABLE_INTEL   0x4

#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL            0x0
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL          0x1
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DC_INTEL                  0x2
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL  0x3
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL 0x4
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL               0x4
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL      0x5
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL     0x6
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL       0x7
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL       0x8
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_DC_INTEL                0x0
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL        0x1
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL          0x2
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL             0x3

#define CL_AVC_ME_FRAME_FORWARD_INTEL                       0x1
#define CL_AVC_ME_FRAME_BACKWARD_INTEL                      0x2
#define CL_AVC_ME_FRAME_DUAL_INTEL                          0x3

#define CL_AVC_ME_SLICE_TYPE_PRED_INTEL                     0x0
#define CL_AVC_ME_SLICE_TYPE_BPRED_INTEL                    0x1
#define CL_AVC_ME_SLICE_TYPE_INTRA_INTEL                    0x2

#define CL_AVC_ME_INTERLACED_SCAN_TOP_FIELD_INTEL           0x0
#define CL_AVC_ME_INTERLACED_SCAN_BOTTOM_FIELD_INTEL        0x1

/*******************************************
* cl_intel_unified_shared_memory extension *
********************************************/

/* These APIs are in sync with Revision O of the cl_intel_unified_shared_memory spec! */

#define cl_intel_unified_shared_memory 1

/* cl_device_info */
#define CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL                   0x4190
#define CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL                 0x4191
#define CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL   0x4192
#define CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL    0x4193
#define CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL          0x4194

typedef cl_bitfield cl_device_unified_shared_memory_capabilities_intel;

/* cl_device_unified_shared_memory_capabilities_intel - bitfield */
#define CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL                   (1 << 0)
#define CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL            (1 << 1)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL        (1 << 2)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL (1 << 3)

typedef cl_properties cl_mem_properties_intel;

/* cl_mem_properties_intel */
#define CL_MEM_ALLOC_FLAGS_INTEL        0x4195

typedef cl_bitfield cl_mem_alloc_flags_intel;

/* cl_mem_alloc_flags_intel - bitfield */
#define CL_MEM_ALLOC_WRITE_COMBINED_INTEL               (1 << 0)

typedef cl_uint cl_mem_info_intel;

/* cl_mem_alloc_info_intel */
#define CL_MEM_ALLOC_TYPE_INTEL         0x419A
#define CL_MEM_ALLOC_BASE_PTR_INTEL     0x419B
#define CL_MEM_ALLOC_SIZE_INTEL         0x419C
#define CL_MEM_ALLOC_DEVICE_INTEL       0x419D
/* Enum values 0x419E-0x419F are reserved for future queries. */

typedef cl_uint cl_unified_shared_memory_type_intel;

/* cl_unified_shared_memory_type_intel */
#define CL_MEM_TYPE_UNKNOWN_INTEL       0x4196
#define CL_MEM_TYPE_HOST_INTEL          0x4197
#define CL_MEM_TYPE_DEVICE_INTEL        0x4198
#define CL_MEM_TYPE_SHARED_INTEL        0x4199

typedef cl_uint cl_mem_advice_intel;

/* cl_mem_advice_intel */
/* Enum values 0x4208-0x420F are reserved for future memory advices. */

/* cl_kernel_exec_info */
#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL      0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL    0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL    0x4202
#define CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL                  0x4203

/* cl_command_type */
#define CL_COMMAND_MEMFILL_INTEL        0x4204
#define CL_COMMAND_MEMCPY_INTEL         0x4205
#define CL_COMMAND_MIGRATEMEM_INTEL     0x4206
#define CL_COMMAND_MEMADVISE_INTEL      0x4207

extern CL_API_ENTRY void* CL_API_CALL
clHostMemAllocINTEL(
            cl_context context,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef CL_API_ENTRY void* (CL_API_CALL *
clHostMemAllocINTEL_fn)(
            cl_context context,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

extern CL_API_ENTRY void* CL_API_CALL
clDeviceMemAllocINTEL(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef CL_API_ENTRY void* (CL_API_CALL *
clDeviceMemAllocINTEL_fn)(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

extern CL_API_ENTRY void* CL_API_CALL
clSharedMemAllocINTEL(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef CL_API_ENTRY void* (CL_API_CALL *
clSharedMemAllocINTEL_fn)(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clMemFreeINTEL(
            cl_context context,
            void* ptr);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clMemFreeINTEL_fn)(
            cl_context context,
            void* ptr);

extern CL_API_ENTRY cl_int CL_API_CALL
clGetMemAllocInfoINTEL(
            cl_context context,
            const void* ptr,
            cl_mem_info_intel param_name,
            size_t param_value_size,
            void* param_value,
            size_t* param_value_size_ret);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clGetMemAllocInfoINTEL_fn)(
            cl_context context,
            const void* ptr,
            cl_mem_info_intel param_name,
            size_t param_value_size,
            void* param_value,
            size_t* param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgMemPointerINTEL(
            cl_kernel kernel,
            cl_uint arg_index,
            const void* arg_value);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clSetKernelArgMemPointerINTEL_fn)(
            cl_kernel kernel,
            cl_uint arg_index,
            const void* arg_value);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(       /* Deprecated */
            cl_command_queue command_queue,
            void* dst_ptr,
            cl_int value,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemsetINTEL_fn)(   /* Deprecated */
            cl_command_queue command_queue,
            void* dst_ptr,
            cl_int value,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemFillINTEL(
            cl_command_queue command_queue,
            void* dst_ptr,
            const void* pattern,
            size_t pattern_size,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemFillINTEL_fn)(
            cl_command_queue command_queue,
            void* dst_ptr,
            const void* pattern,
            size_t pattern_size,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemcpyINTEL(
            cl_command_queue command_queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemcpyINTEL_fn)(
            cl_command_queue command_queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

#ifdef CL_VERSION_1_2

/* Because these APIs use cl_mem_migration_flags, they require
   OpenCL 1.2: */

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemINTEL(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_migration_flags flags,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMigrateMemINTEL_fn)(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_migration_flags flags,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemAdviseINTEL(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_advice_intel advice,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemAdviseINTEL_fn)(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_advice_intel advice,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

/***************************************************
* cl_intel_create_buffer_with_properties extension *
****************************************************/

#define cl_intel_create_buffer_with_properties 1

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBufferWithPropertiesINTEL(
    cl_context   context,
    const cl_mem_properties_intel* properties,
    cl_mem_flags flags,
    size_t       size,
    void *       host_ptr,
    cl_int *     errcode_ret) CL_EXT_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *
clCreateBufferWithPropertiesINTEL_fn)(
    cl_context   context,
    const cl_mem_properties_intel* properties,
    cl_mem_flags flags,
    size_t       size,
    void *       host_ptr,
    cl_int *     errcode_ret) CL_EXT_SUFFIX__VERSION_1_0;

/******************************************
* cl_intel_mem_channel_property extension *
*******************************************/

#define CL_MEM_CHANNEL_INTEL            0x4213

#ifdef __cplusplus
}
#endif

#endif /* __CL_EXT_INTEL_H */
