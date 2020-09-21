// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#if defined(__EMSCRIPTEN__) && defined(DAWN_EMSDK)
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu_cpp.h>
#else
#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn_native/DawnNative.h>
#endif  // HAVE_WEBGPU
#endif  //__EMSCRIPTEN__
namespace cv { namespace dnn { namespace webgpu {
#if defined(HAVE_WEBGPU) || (defined(DAWN_EMSDK) && defined(__EMSCRIPTEN__))

    wgpu::Device createCppDawnDevice();

#endif   //HAVE_WEBGPU

}}}     // namespace cv::dnn::webgpu
