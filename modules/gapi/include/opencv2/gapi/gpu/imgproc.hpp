#ifndef OPENCV_GAPI_GPU_IMGPROC_API_HPP
#define OPENCV_GAPI_GPU_IMGPROC_API_HPP

#include <opencv2/core/cvdef.h>     // CV_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace imgproc {
namespace gpu {

CV_EXPORTS GKernelPackage kernels();

} // namespace gpu
} // namespace imgproc
} // namespace gapi
} // namespace cv


#endif // OPENCV_GAPI_GPU_IMGPROC_API_HPP
