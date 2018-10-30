#ifndef OPENCV_GAPI_GPU_CORE_API_HPP
#define OPENCV_GAPI_GPU_CORE_API_HPP

#include <opencv2/core/cvdef.h>     // CV_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace core {
namespace gpu {

CV_EXPORTS GKernelPackage kernels();

} // namespace gpu
} // namespace core
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_GPU_CORE_API_HPP
