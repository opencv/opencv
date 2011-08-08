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

using namespace cv;
using namespace cv::gpu;


namespace 
{
    // Compares value to set using the given comparator. Returns true if
    // there is at least one element x in the set satisfying to: x cmp value
    // predicate.
    template <typename Comparer>
    bool compareToSet(const std::string& set_as_str, int value, Comparer cmp)
    {
        if (set_as_str.find_first_not_of(" ") == string::npos)
            return false;

        std::stringstream stream(set_as_str);
        int cur_value;

        while (!stream.eof())
        {
            stream >> cur_value;
            if (cmp(cur_value, value))
                return true;
        }

        return false;
    }
}


bool cv::gpu::TargetArchs::builtWith(cv::gpu::FeatureSet feature_set)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_FEATURES, feature_set, std::greater_equal<int>());
#else
	(void)feature_set;
	return false;
#endif
}


bool cv::gpu::TargetArchs::has(int major, int minor)
{
    return hasPtx(major, minor) || hasBin(major, minor);
}


bool cv::gpu::TargetArchs::hasPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, std::equal_to<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasBin(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_BIN, major * 10 + minor, std::equal_to<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrLessPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, 
                     std::less_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrGreater(int major, int minor)
{
    return hasEqualOrGreaterPtx(major, minor) ||
           hasEqualOrGreaterBin(major, minor);
}


bool cv::gpu::TargetArchs::hasEqualOrGreaterPtx(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_PTX, major * 10 + minor, 
                     std::greater_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


bool cv::gpu::TargetArchs::hasEqualOrGreaterBin(int major, int minor)
{
#if defined (HAVE_CUDA)
    return ::compareToSet(CUDA_ARCH_BIN, major * 10 + minor, 
                     std::greater_equal<int>());
#else
	(void)major;
	(void)minor;
	return false;
#endif
}


#if !defined (HAVE_CUDA)

int cv::gpu::getCudaEnabledDeviceCount() { return 0; }
void cv::gpu::setDevice(int) { throw_nogpu(); } 
int cv::gpu::getDevice() { throw_nogpu(); return 0; }
void cv::gpu::resetDevice() { throw_nogpu(); }
size_t cv::gpu::DeviceInfo::freeMemory() const { throw_nogpu(); return 0; }
size_t cv::gpu::DeviceInfo::totalMemory() const { throw_nogpu(); return 0; }
bool cv::gpu::DeviceInfo::supports(cv::gpu::FeatureSet) const { throw_nogpu(); return false; }
bool cv::gpu::DeviceInfo::isCompatible() const { throw_nogpu(); return false; }
void cv::gpu::DeviceInfo::query() { throw_nogpu(); }
void cv::gpu::DeviceInfo::queryMemory(size_t&, size_t&) const { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

int cv::gpu::getCudaEnabledDeviceCount()
{
    int count;
    cudaSafeCall( cudaGetDeviceCount( &count ) );
    return count;
}


void cv::gpu::setDevice(int device)
{
    cudaSafeCall( cudaSetDevice( device ) );
}


int cv::gpu::getDevice()
{
    int count;
    cudaError_t error = cudaGetDeviceCount( &count );

    if (error == cudaErrorInsufficientDriver)
        return -1;

    if (error == cudaErrorNoDevice)
        return 0;

    cudaSafeCall(error);
    return count;
}


void cv::gpu::resetDevice()
{
    cudaSafeCall( cudaDeviceReset() );
}


size_t cv::gpu::DeviceInfo::freeMemory() const
{
    size_t free_memory, total_memory;
    queryMemory(free_memory, total_memory);
    return free_memory;
}


size_t cv::gpu::DeviceInfo::totalMemory() const
{
    size_t free_memory, total_memory;
    queryMemory(free_memory, total_memory);
    return total_memory;
}


bool cv::gpu::DeviceInfo::supports(cv::gpu::FeatureSet feature_set) const
{
    int version = majorVersion() * 10 + minorVersion();
    return version >= feature_set;
}


bool cv::gpu::DeviceInfo::isCompatible() const
{
    // Check PTX compatibility
    if (TargetArchs::hasEqualOrLessPtx(majorVersion(), minorVersion()))
        return true;

    // Check BIN compatibility
    for (int i = minorVersion(); i >= 0; --i)
        if (TargetArchs::hasBin(majorVersion(), i))
            return true;

    return false;
}


void cv::gpu::DeviceInfo::query()
{
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, device_id_));
    name_ = prop.name;
    multi_processor_count_ = prop.multiProcessorCount;
    majorVersion_ = prop.major;
    minorVersion_ = prop.minor;
}


void cv::gpu::DeviceInfo::queryMemory(size_t& free_memory, size_t& total_memory) const
{
    int prev_device_id = getDevice();
    if (prev_device_id != device_id_)
        setDevice(device_id_);

    cudaSafeCall(cudaMemGetInfo(&free_memory, &total_memory));

    if (prev_device_id != device_id_)
        setDevice(prev_device_id);
}

#endif

