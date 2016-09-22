/* See LICENSE file in the root OpenCV directory */

#ifndef OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_2_0_HPP
#define OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_2_0_HPP

#if defined(HAVE_OPENCL_SVM)
#include "opencl_core.hpp"

#include "opencl_svm_definitions.hpp"

#ifndef HAVE_OPENCL_STATIC

#undef clSVMAlloc
#define clSVMAlloc clSVMAlloc_pfn
#undef clSVMFree
#define clSVMFree clSVMFree_pfn
#undef clSetKernelArgSVMPointer
#define clSetKernelArgSVMPointer clSetKernelArgSVMPointer_pfn
#undef clSetKernelExecInfo
//#define clSetKernelExecInfo clSetKernelExecInfo_pfn
#undef clEnqueueSVMFree
//#define clEnqueueSVMFree clEnqueueSVMFree_pfn
#undef clEnqueueSVMMemcpy
#define clEnqueueSVMMemcpy clEnqueueSVMMemcpy_pfn
#undef clEnqueueSVMMemFill
#define clEnqueueSVMMemFill clEnqueueSVMMemFill_pfn
#undef clEnqueueSVMMap
#define clEnqueueSVMMap clEnqueueSVMMap_pfn
#undef clEnqueueSVMUnmap
#define clEnqueueSVMUnmap clEnqueueSVMUnmap_pfn

extern CL_RUNTIME_EXPORT void* (CL_API_CALL *clSVMAlloc)(cl_context context, cl_svm_mem_flags flags, size_t size, unsigned int alignment);
extern CL_RUNTIME_EXPORT void (CL_API_CALL *clSVMFree)(cl_context context, void* svm_pointer);
extern CL_RUNTIME_EXPORT cl_int (CL_API_CALL *clSetKernelArgSVMPointer)(cl_kernel kernel, cl_uint arg_index, const void* arg_value);
//extern CL_RUNTIME_EXPORT void* (CL_API_CALL *clSetKernelExecInfo)(cl_kernel kernel, cl_kernel_exec_info param_name, size_t param_value_size, const void* param_value);
//extern CL_RUNTIME_EXPORT cl_int (CL_API_CALL *clEnqueueSVMFree)(cl_command_queue command_queue, cl_uint num_svm_pointers, void* svm_pointers[],
//        void (CL_CALLBACK *pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data), void* user_data,
//        cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
extern CL_RUNTIME_EXPORT cl_int (CL_API_CALL *clEnqueueSVMMemcpy)(cl_command_queue command_queue, cl_bool blocking_copy, void* dst_ptr, const void* src_ptr, size_t size,
        cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
extern CL_RUNTIME_EXPORT cl_int (CL_API_CALL *clEnqueueSVMMemFill)(cl_command_queue command_queue, void* svm_ptr, const void* pattern, size_t pattern_size, size_t size,
        cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
extern CL_RUNTIME_EXPORT cl_int (CL_API_CALL *clEnqueueSVMMap)(cl_command_queue command_queue, cl_bool blocking_map, cl_map_flags map_flags, void* svm_ptr, size_t size,
        cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
extern CL_RUNTIME_EXPORT cl_int (CL_API_CALL *clEnqueueSVMUnmap)(cl_command_queue command_queue, void* svm_ptr,
        cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);

#endif // HAVE_OPENCL_STATIC

#endif // HAVE_OPENCL_SVM

#endif // OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_2_0_HPP
