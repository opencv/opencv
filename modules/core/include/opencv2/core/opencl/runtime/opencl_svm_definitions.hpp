/* See LICENSE file in the root OpenCV directory */

#ifndef __OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_DEFINITIONS_HPP__
#define __OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_DEFINITIONS_HPP__

#if defined(HAVE_OPENCL_SVM)
#if defined(CL_VERSION_2_0)

// OpenCL 2.0 contains SVM definitions

#else

typedef cl_bitfield cl_device_svm_capabilities;
typedef cl_bitfield cl_svm_mem_flags;
typedef cl_uint     cl_kernel_exec_info;

//
// TODO Add real values after OpenCL 2.0 release
//

#ifndef CL_DEVICE_SVM_CAPABILITIES
#define CL_DEVICE_SVM_CAPABILITIES 0x1053

#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER             (1 << 0)
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER               (1 << 1)
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM               (1 << 2)
#define CL_DEVICE_SVM_ATOMICS                         (1 << 3)
#endif

#ifndef CL_MEM_SVM_FINE_GRAIN_BUFFER
#define CL_MEM_SVM_FINE_GRAIN_BUFFER (1 << 10)
#endif

#ifndef CL_MEM_SVM_ATOMICS
#define CL_MEM_SVM_ATOMICS (1 << 11)
#endif


#endif // CL_VERSION_2_0
#endif // HAVE_OPENCL_SVM

#endif // __OPENCV_CORE_OCL_RUNTIME_OPENCL_SVM_DEFINITIONS_HPP__
