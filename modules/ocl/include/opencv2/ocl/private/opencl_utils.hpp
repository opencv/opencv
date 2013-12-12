/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_OCL_PRIVATE_OPENCL_UTILS_HPP__
#define __OPENCV_OCL_PRIVATE_OPENCL_UTILS_HPP__

#include "opencv2/core/opencl/runtime/opencl_core.hpp"
#include <vector>
#include <string>

namespace cl_utils {

inline cl_int getPlatforms(std::vector<cl_platform_id>& platforms)
{
    cl_uint n = 0;

    cl_int err = ::clGetPlatformIDs(0, NULL, &n);
    if (err != CL_SUCCESS)
        return err;

    platforms.clear(); platforms.resize(n);
    err = ::clGetPlatformIDs(n, &platforms[0], NULL);
    if (err != CL_SUCCESS)
        return err;

    return CL_SUCCESS;
}

inline cl_int getDevices(cl_platform_id platform, cl_device_type type, std::vector<cl_device_id>& devices)
{
    cl_uint n = 0;

    cl_int err = ::clGetDeviceIDs(platform, type, 0, NULL, &n);
    if (err != CL_SUCCESS)
        return err;

    devices.clear(); devices.resize(n);
    err = ::clGetDeviceIDs(platform, type, n, &devices[0], NULL);
    if (err != CL_SUCCESS)
        return err;

    return CL_SUCCESS;
}




template <typename Functor, typename ObjectType, typename T>
inline cl_int getScalarInfo(Functor f, ObjectType obj, cl_uint name, T& param)
{
    return f(obj, name, sizeof(T), &param, NULL);
}

template <typename Functor, typename ObjectType>
inline cl_int getStringInfo(Functor f, ObjectType obj, cl_uint name, std::string& param)
{
    ::size_t required;
    cl_int err = f(obj, name, 0, NULL, &required);
    if (err != CL_SUCCESS)
        return err;

    param.clear();
    if (required > 0)
    {
        std::vector<char> buf(required + 1, char(0));
        err = f(obj, name, required, &buf[0], NULL);
        if (err != CL_SUCCESS)
            return err;
        param = &buf[0];
    }

    return CL_SUCCESS;
};

} // namespace cl_utils

#endif // __OPENCV_OCL_PRIVATE_OPENCL_UTILS_HPP__
