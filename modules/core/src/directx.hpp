// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_SRC_DIRECTX_HPP
#define OPENCV_CORE_SRC_DIRECTX_HPP

#ifndef HAVE_DIRECTX
#error Invalid build configuration
#endif

namespace cv {
namespace directx {
namespace internal {

struct OpenCLDirectXImpl;
OpenCLDirectXImpl* createDirectXImpl();
void deleteDirectXImpl(OpenCLDirectXImpl**);
OpenCLDirectXImpl* getDirectXImpl(ocl::Context& ctx);

}}} // namespace internal

#endif  // OPENCV_CORE_SRC_DIRECTX_HPP
