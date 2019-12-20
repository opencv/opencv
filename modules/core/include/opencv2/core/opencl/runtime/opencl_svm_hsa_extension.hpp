/* See LICENSE file in the root OpenCV directory */

#ifndef OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_HSA_EXTENSION_HPP
#define OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_HSA_EXTENSION_HPP

#if defined(HAVE_OPENCL_SVM)
#include "opencl_core.hpp"

#ifndef CL_DEVICE_SVM_CAPABILITIES_AMD
//
//  Part of the file is an extract from the cl_ext.h file from AMD APP SDK package.
//  Below is the original copyright.
//
/*******************************************************************************
 * Copyright (c) 2008-2013 The Khronos Group Inc.
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

/*******************************************
 * Shared Virtual Memory (SVM) extension
 *******************************************/
typedef cl_bitfield                      cl_device_svm_capabilities_amd;
typedef cl_bitfield                      cl_svm_mem_flags_amd;
typedef cl_uint                          cl_kernel_exec_info_amd;

/* cl_device_info */
#define CL_DEVICE_SVM_CAPABILITIES_AMD                     0x1053
#define CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT_AMD  0x1054

/* cl_device_svm_capabilities_amd */
#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_AMD             (1 << 0)
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER_AMD               (1 << 1)
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM_AMD               (1 << 2)
#define CL_DEVICE_SVM_ATOMICS_AMD                         (1 << 3)

/* cl_svm_mem_flags_amd */
#define CL_MEM_SVM_FINE_GRAIN_BUFFER_AMD                  (1 << 10)
#define CL_MEM_SVM_ATOMICS_AMD                            (1 << 11)

/* cl_mem_info */
#define CL_MEM_USES_SVM_POINTER_AMD                       0x1109

/* cl_kernel_exec_info_amd */
#define CL_KERNEL_EXEC_INFO_SVM_PTRS_AMD                  0x11B6
#define CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM_AMD     0x11B7

/* cl_command_type */
#define CL_COMMAND_SVM_FREE_AMD                           0x1209
#define CL_COMMAND_SVM_MEMCPY_AMD                         0x120A
#define CL_COMMAND_SVM_MEMFILL_AMD                        0x120B
#define CL_COMMAND_SVM_MAP_AMD                            0x120C
#define CL_COMMAND_SVM_UNMAP_AMD                          0x120D

typedef CL_API_ENTRY void*
(CL_API_CALL * clSVMAllocAMD_fn)(
    cl_context            /* context */,
    cl_svm_mem_flags_amd  /* flags */,
    size_t                /* size */,
    unsigned int          /* alignment */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY void
(CL_API_CALL * clSVMFreeAMD_fn)(
    cl_context  /* context */,
    void*       /* svm_pointer */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clEnqueueSVMFreeAMD_fn)(
    cl_command_queue /* command_queue */,
    cl_uint          /* num_svm_pointers */,
    void**           /* svm_pointers */,
    void (CL_CALLBACK *)( /*pfn_free_func*/
        cl_command_queue /* queue */,
        cl_uint          /* num_svm_pointers */,
        void**           /* svm_pointers */,
        void*            /* user_data */),
    void*             /* user_data */,
    cl_uint           /* num_events_in_wait_list */,
    const cl_event*   /* event_wait_list */,
    cl_event*         /* event */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clEnqueueSVMMemcpyAMD_fn)(
    cl_command_queue /* command_queue */,
    cl_bool          /* blocking_copy */,
    void*            /* dst_ptr */,
    const void*      /* src_ptr */,
    size_t           /* size */,
    cl_uint          /* num_events_in_wait_list */,
    const cl_event*  /* event_wait_list */,
    cl_event*        /* event */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clEnqueueSVMMemFillAMD_fn)(
    cl_command_queue /* command_queue */,
    void*            /* svm_ptr */,
    const void*      /* pattern */,
    size_t           /* pattern_size */,
    size_t           /* size */,
    cl_uint          /* num_events_in_wait_list */,
    const cl_event*  /* event_wait_list */,
    cl_event*        /* event */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clEnqueueSVMMapAMD_fn)(
    cl_command_queue /* command_queue */,
    cl_bool          /* blocking_map */,
    cl_map_flags     /* map_flags */,
    void*            /* svm_ptr */,
    size_t           /* size */,
    cl_uint          /* num_events_in_wait_list */,
    const cl_event*  /* event_wait_list */,
    cl_event*        /* event */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clEnqueueSVMUnmapAMD_fn)(
    cl_command_queue /* command_queue */,
    void*            /* svm_ptr */,
    cl_uint          /* num_events_in_wait_list */,
    const cl_event*  /* event_wait_list */,
    cl_event*        /* event */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clSetKernelArgSVMPointerAMD_fn)(
    cl_kernel     /* kernel */,
    cl_uint       /* arg_index */,
    const void *  /* arg_value */
) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clSetKernelExecInfoAMD_fn)(
     cl_kernel                /* kernel */,
     cl_kernel_exec_info_amd  /* param_name */,
     size_t                   /* param_value_size */,
     const void *             /* param_value */
) CL_EXT_SUFFIX__VERSION_1_2;

#endif

#endif // HAVE_OPENCL_SVM

#endif // OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_HSA_EXTENSION_HPP
