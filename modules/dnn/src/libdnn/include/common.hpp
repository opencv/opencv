#ifndef _OPENCV_GREENTEA_COMMON_HPP_
#define _OPENCV_GREENTEA_COMMON_HPP_
#include "../../precomp.hpp"
#include "../../caffe/glog_emulator.hpp"
#include <opencv2/core/opencl/runtime/opencl_core.hpp>

namespace greentea {

#ifdef HAVE_OPENCL
#ifdef USE_INDEX_64
#define int_tp int64_t
#define uint_tp uint64_t
#else
#define int_tp int32_t
#define uint_tp uint32_t
#endif // USE_INDEX_64

#define ALIGN(val,N) (( (val) + (N) - 1 ) & ~( (N) - 1 ))
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// Macro to select the single (_float) or double (_double) precision kernel
#define CL_KERNEL_SELECT(kernel) kernel "_float"

#define OCL_CHECK(condition) \
    do { \
        cl_int error = (condition); \
        CHECK_EQ(error, CL_SUCCESS) << " " << greentea::clGetErrorString(error); \
    } while (0)

const char* clGetErrorString(cl_int error);
bool IsBeignet();
void AllocateMemory(void** ptr, uint_tp size, int_tp flags);
bool CheckCapability(std::string cap);

#endif // HAVE_OPENCL

}
#endif
