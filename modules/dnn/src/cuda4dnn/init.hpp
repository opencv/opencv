// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_INIT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_INIT_HPP

#include "csl/error.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <opencv2/core/cuda.hpp>
#include <sstream>

namespace cv { namespace dnn { namespace cuda4dnn {

    void checkVersions()
    {
        // https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#programming-model
        // cuDNN API Compatibility
        // Beginning in cuDNN 7, the binary compatibility of a patch and minor releases is maintained as follows:
        //     Any patch release x.y.z is forward or backward-compatible with applications built against another cuDNN patch release x.y.w (meaning, of the same major and minor version number, but having w!=z).
        //     cuDNN minor releases beginning with cuDNN 7 are binary backward-compatible with applications built against the same or earlier patch release (meaning, an application built against cuDNN 7.x is binary compatible with cuDNN library 7.y, where y>=x).
        //     Applications compiled with a cuDNN version 7.y are not guaranteed to work with 7.x release when y > x.
        auto cudnn_bversion = cudnnGetVersion();
        auto cudnn_major_bversion = cudnn_bversion / 1000, cudnn_minor_bversion = cudnn_bversion % 1000 / 100;
        if (cudnn_major_bversion != CUDNN_MAJOR || cudnn_minor_bversion < CUDNN_MINOR)
        {
            std::ostringstream oss;
            oss << "cuDNN reports version " << cudnn_major_bversion << "." << cudnn_minor_bversion << " which is not compatible with the version " << CUDNN_MAJOR << "." << CUDNN_MINOR << " with which OpenCV was built";
            CV_LOG_WARNING(NULL, oss.str().c_str());
        }
    }

    int getDeviceCount()
    {
        return cuda::getCudaEnabledDeviceCount();
    }

    int getDevice()
    {
        int device_id = -1;
        CUDA4DNN_CHECK_CUDA(cudaGetDevice(&device_id));
        return device_id;
    }

    bool isDeviceCompatible()
    {
        int device_id = getDevice();
        if (device_id < 0)
            return false;

        int major = 0, minor = 0;
        CUDA4DNN_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
        CUDA4DNN_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));

        if (cv::cuda::TargetArchs::hasEqualOrLessPtx(major, minor))
            return true;

        for (int i = minor; i >= 0; i--)
            if (cv::cuda::TargetArchs::hasBin(major, i))
                return true;

        return false;
    }

    bool doesDeviceSupportFP16()
    {
        int device_id = getDevice();
        if (device_id < 0)
            return false;

        int major = 0, minor = 0;
        CUDA4DNN_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
        CUDA4DNN_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));

        int version = major * 10 + minor;
        if (version < 53)
            return false;
        return true;
    }

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_INIT_HPP */
