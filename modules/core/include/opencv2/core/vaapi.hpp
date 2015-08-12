// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2015, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_CORE_VAAPI_HPP__
#define __OPENCV_CORE_VAAPI_HPP__

#ifndef __cplusplus
#  error vaapi.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "ocl.hpp"

#if defined(HAVE_VAAPI)
# include "va/va.h"
#else  // HAVE_VAAPI
# if !defined(_VA_H_)
    typedef void* VADisplay;
    typedef unsigned int VASurfaceID;
# endif // !_VA_H_
#endif // HAVE_VAAPI

namespace cv { namespace vaapi {

/** @addtogroup core_vaapi
This section describes CL-VA (VA-API) interoperability.

To enable CL-VA interoperability support, configure OpenCV using CMake with WITH_VAAPI=ON . Currently VA-API is
supported on Linux only. You should also install Intel Media Server Studio (MSS) to use this feature. You may
have to specify the path(s) to MSS components for cmake in environment variables: VAAPI_MSDK_ROOT for Media SDK
(default is "/opt/intel/mediasdk"), and VAAPI_IOCL_ROOT for Intel OpenCL (default is "/opt/intel/opencl").

To use VA-API interoperability you should first create VADisplay (libva), and then call initializeContextFromVA()
function to create OpenCL context and set up interoperability.
*/
//! @{

/////////////////// CL-VA Interoperability Functions ///////////////////

namespace ocl {
using namespace cv::ocl;

// TODO static functions in the Context class
/** @brief Creates OpenCL context from VA.
@param display - VADisplay for which CL interop should be established.
@return Returns reference to OpenCL Context
 */
CV_EXPORTS Context& initializeContextFromVA(VADisplay display);

} // namespace cv::vaapi::ocl

/** @brief Converts InputArray to VASurfaceID object.
@param src     - source InputArray.
@param surface - destination VASurfaceID object.
@param size    - size of image represented by VASurfaceID object.
 */
CV_EXPORTS void convertToVASurface(InputArray src, VASurfaceID surface, Size size);

/** @brief Converts VASurfaceID object to OutputArray.
@param surface - source VASurfaceID object.
@param size    - size of image represented by VASurfaceID object.
@param dst     - destination OutputArray.
 */
CV_EXPORTS void convertFromVASurface(VASurfaceID surface, Size size, OutputArray dst);

//! @}

}} // namespace cv::vaapi

#endif /* __OPENCV_CORE_VAAPI_HPP__ */
