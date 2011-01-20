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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <functional>

using namespace cv;
using namespace cv::gpu;


#if !defined (HAVE_CUDA)

CV_EXPORTS int cv::gpu::getCudaEnabledDeviceCount() { return 0; }
CV_EXPORTS string cv::gpu::getDeviceName(int /*device*/)  { throw_nogpu(); return 0; } 
CV_EXPORTS void cv::gpu::setDevice(int /*device*/) { throw_nogpu(); } 
CV_EXPORTS int cv::gpu::getDevice() { throw_nogpu(); return 0; } 
CV_EXPORTS void cv::gpu::getComputeCapability(int /*device*/, int& /*major*/, int& /*minor*/) { throw_nogpu(); } 
CV_EXPORTS int cv::gpu::getNumberOfSMs(int /*device*/) { throw_nogpu(); return 0; } 
CV_EXPORTS void cv::gpu::getGpuMemInfo(size_t& /*free*/, size_t& /*total*/)  { throw_nogpu(); } 
CV_EXPORTS bool cv::gpu::hasNativeDoubleSupport(int /*device*/) { throw_nogpu(); return false; }
CV_EXPORTS bool cv::gpu::hasAtomicsSupport(int /*device*/) { throw_nogpu(); return false; }
CV_EXPORTS bool cv::gpu::hasPtxVersion(int major, int minor) { throw_nogpu(); return false; }
CV_EXPORTS bool cv::gpu::hasLessOrEqualPtxVersion(int major, int minor) { return false; }
CV_EXPORTS bool cv::gpu::hasGreaterOrEqualPtxVersion(int major, int minor) { return false; }
CV_EXPORTS bool cv::gpu::hasCubinVersion(int major, int minor) { return false; }
CV_EXPORTS bool cv::gpu::hasGreaterOrEqualCubinVersion(int major, int minor) { return false; }
CV_EXPORTS bool cv::gpu::hasVersion(int major, int minor) { return false; }
CV_EXPORTS bool cv::gpu::hasGreaterOrEqualVersion(int major, int minor) { return false; }
CV_EXPORTS bool cv::gpu::isCompatibleWith(int device) { throw_nogpu(); return false; }


#else /* !defined (HAVE_CUDA) */

CV_EXPORTS int cv::gpu::getCudaEnabledDeviceCount()
{
    int count;
    cudaSafeCall( cudaGetDeviceCount( &count ) );
    return count;
}


CV_EXPORTS string cv::gpu::getDeviceName(int device)
{
    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties( &prop, device) );
    return prop.name;
}


CV_EXPORTS void cv::gpu::setDevice(int device)
{
    cudaSafeCall( cudaSetDevice( device ) );
}


CV_EXPORTS int cv::gpu::getDevice()
{
    int device;    
    cudaSafeCall( cudaGetDevice( &device ) );
    return device;
}


CV_EXPORTS void cv::gpu::getComputeCapability(int device, int& major, int& minor)
{
    cudaDeviceProp prop;    
    cudaSafeCall( cudaGetDeviceProperties( &prop, device) );

    major = prop.major;
    minor = prop.minor;
}


CV_EXPORTS int cv::gpu::getNumberOfSMs(int device)
{
    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties( &prop, device ) );
    return prop.multiProcessorCount;
}


CV_EXPORTS void cv::gpu::getGpuMemInfo(size_t& free, size_t& total)
{
    cudaSafeCall( cudaMemGetInfo( &free, &total ) );
}


CV_EXPORTS bool cv::gpu::hasNativeDoubleSupport(int device)
{
    int major, minor;
    getComputeCapability(device, major, minor);
    return major > 1 || (major == 1 && minor >= 3);
}


CV_EXPORTS bool cv::gpu::hasAtomicsSupport(int device) 
{
    int major, minor;
    getComputeCapability(device, major, minor);
    return major > 1 || (major == 1 && minor >= 1);
}


namespace 
{
    template <typename Comparer>
    bool compare(const std::string& str, int x, Comparer cmp)
    {
        if (str.find_first_not_of(" ") == string::npos)
            return false;

        std::stringstream stream(str);
        int val;

        while (!stream.eof())
        {
            stream >> val;
            if (cmp(val, x))
                return true;
        }

        return false;
    }
}


CV_EXPORTS bool cv::gpu::hasPtxVersion(int major, int minor)
{
    return ::compare(OPENCV_ARCH_PTX, major * 10 + minor, std::equal_to<int>());
}


CV_EXPORTS bool cv::gpu::hasLessOrEqualPtxVersion(int major, int minor)
{
    return ::compare(OPENCV_ARCH_PTX, major * 10 + minor, 
                     std::less_equal<int>());
}


CV_EXPORTS bool cv::gpu::hasGreaterOrEqualPtxVersion(int major, int minor)
{
    return ::compare(OPENCV_ARCH_PTX, major * 10 + minor, 
                     std::greater_equal<int>());
}


CV_EXPORTS bool cv::gpu::hasCubinVersion(int major, int minor)
{
    return ::compare(OPENCV_ARCH_GPU, major * 10 + minor, std::equal_to<int>());
}


CV_EXPORTS bool cv::gpu::hasGreaterOrEqualCubinVersion(int major, int minor)
{
    return ::compare(OPENCV_ARCH_GPU, major * 10 + minor, 
                     std::greater_equal<int>());
}


CV_EXPORTS bool cv::gpu::hasVersion(int major, int minor)
{
    return hasPtxVersion(major, minor) || hasCubinVersion(major, minor);
}


CV_EXPORTS bool cv::gpu::hasGreaterOrEqualVersion(int major, int minor)
{
    return hasGreaterOrEqualPtxVersion(major, minor) || 
           hasGreaterOrEqualCubinVersion(major, minor);
}


CV_EXPORTS bool cv::gpu::isCompatibleWith(int device)
{
    // According to the CUDA C Programming Guide Version 3.2: "PTX code 
    // produced for some specific compute capability can always be compiled to
    // binary code of greater or equal compute capability". 

    int major, minor;
    getComputeCapability(device, major, minor);

    // Check PTX compatibility
    if (hasLessOrEqualPtxVersion(major, minor))
        return true;

    // Check CUBIN compatibility
    for (int i = minor; i >= 0; --i)
        if (hasCubinVersion(major, i))
            return true;

    return false;
}

#endif

