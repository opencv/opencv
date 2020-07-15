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
        int cudart_version = 0;
        CUDA4DNN_CHECK_CUDA(cudaRuntimeGetVersion(&cudart_version));
        if (cudart_version != CUDART_VERSION)
        {
            std::ostringstream oss;
            oss << "CUDART reports version " << cudart_version << " which does not match with the version " << CUDART_VERSION << " with which OpenCV was built";
            CV_LOG_WARNING(NULL, oss.str().c_str());
        }

        auto cudnn_version = cudnnGetVersion();
        if (cudnn_version != CUDNN_VERSION)
        {
            std::ostringstream oss;
            oss << "cuDNN reports version " << cudnn_version << " which does not match with the version " << CUDNN_VERSION << " with which OpenCV was built";
            CV_LOG_WARNING(NULL, oss.str().c_str());
        }

        auto cudnn_cudart_version = cudnnGetCudartVersion();
        if (cudart_version != cudnn_cudart_version)
        {
            std::ostringstream oss;
            oss << "CUDART version " << cudnn_cudart_version << " reported by cuDNN " << cudnn_version << " does not match with the version reported by CUDART " << cudart_version;
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
