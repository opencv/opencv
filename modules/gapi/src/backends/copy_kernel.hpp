//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#ifndef OPENCV_GAPI_COPY_KERNEL_HPP
#define OPENCV_GAPI_COPY_KERNEL_HPP

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/streaming/copy.hpp>

namespace cv {
namespace gimpl {

// Manually implement G-API's user kernel convetions with this structure
struct GCopyKernel: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GCopy;
    static GKernelImpl kernel();
    static gapi::GBackend backend();
};

} // namespace gimpl
} // namespace cv

#endif // OPENCV_GAPI_COPY_KERNEL_HPP
