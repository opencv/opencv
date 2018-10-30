#ifndef OPENCV_GAPI_GGPUIMGPROC_HPP
#define OPENCV_GAPI_GGPUIMGPROC_HPP

#include <map>
#include <string>

#include "opencv2/gapi/gpu/ggpukernel.hpp"

namespace cv { namespace gimpl {

// NB: This is what a "Kernel Package" from the origianl Wiki doc should be.
void loadGPUImgProc(std::map<std::string, cv::GGPUKernel> &kmap);

}}

#endif // OPENCV_GAPI_GGPUIMGPROC_HPP
