#include "../../precomp.hpp"
#include "common.hpp"

using namespace cv;

namespace greentea
{

#ifdef HAVE_OPENCL
bool IsBeignet()
{
    return ocl::Device::getDefault().OpenCL_C_Version().find("beignet") != std::string::npos;
}

void AllocateMemory(void** ptr, uint_tp size, int_tp flags)
{
    ocl::Context ctx = ocl::Context::getDefault();
    *ptr = (void*)clCreateBuffer((cl_context)ctx.ptr(),    // NOLINT
                                 flags,
                                 size, nullptr, nullptr);
}

bool CheckCapability(std::string cap)
{
    String extsstr = ocl::Device::getDefault().extensions();
    return extsstr.find(cap) != String::npos;
}

const char* clGetErrorString(cl_int error)
{
    switch (error)
    {
    case   0: return "CL_SUCCESS";
    case  -1: return "CL_DEVICE_NOT_FOUND";
    case  -2: return "CL_DEVICE_NOT_AVAILABLE";
    case  -3: return "CL_COMPILER_NOT_AVAILABLE";
    case  -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case  -5: return "CL_OUT_OF_RESOURCES";
    case  -6: return "CL_OUT_OF_HOST_MEMORY";
    case  -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case  -8: return "CL_MEM_COPY_OVERLAP";
    case  -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1024: return "clBLAS: Functionality is not implemented";
    case -1023: return "clBLAS: Library is not initialized yet";
    case -1022: return "clBLAS: Matrix A is not a valid memory object";
    case -1021: return "clBLAS: Matrix B is not a valid memory object";
    case -1020: return "clBLAS: Matrix C is not a valid memory object";
    case -1019: return "clBLAS: Vector X is not a valid memory object";
    case -1018: return "clBLAS: Vector Y is not a valid memory object";
    case -1017: return "clBLAS: An input dimension (M:N:K) is invalid";
    case -1016: return "clBLAS: Leading dimension A must not be less than the "
                       "size of the first dimension";
    case -1015: return "clBLAS: Leading dimension B must not be less than the "
                       "size of the second dimension";
    case -1014: return "clBLAS: Leading dimension C must not be less than the "
                       "size of the third dimension";
    case -1013: return "clBLAS: The increment for a vector X must not be 0";
    case -1012: return "clBLAS: The increment for a vector Y must not be 0";
    case -1011: return "clBLAS: The memory object for Matrix A is too small";
    case -1010: return "clBLAS: The memory object for Matrix B is too small";
    case -1009: return "clBLAS: The memory object for Matrix C is too small";
    case -1008: return "clBLAS: The memory object for Vector X is too small";
    case -1007: return "clBLAS: The memory object for Vector Y is too small";
    default: return "Unknown OpenCL error";
    }
}
#endif // HAVE_OPENCL

}  // namespace greentea
