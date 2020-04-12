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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"
#include <CL/cl.h>

#ifdef HAVE_OPENCL

namespace opencv_test
{
namespace ocl
{

typedef tuple<Size, MatType, int> FilterParams;
typedef TestBaseWithParam<FilterParams> FilterFixture;

////////////// custom filter ////////////

struct _cl_filter
{
    /// @brief throws std::runtime_error on OpenCL failure
    _cl_filter(uint8_t _radius);
    ~_cl_filter();

    void
    filter(cv::InputArray _input, cv::OutputArray &_output, std::size_t _radius);

private:
    std::vector<cl_context_properties> context_properties_;
    cl_context context_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel h_filter_ = nullptr;
    cl_kernel v_filter_ = nullptr;
    cl_command_queue queue_ = nullptr;
};

inline void
clThrowOnError(cl_uint _error, const std::string &_message = "")
{
    if (_error != CL_SUCCESS)
        throw std::runtime_error("cl error with the code " + _message + " " +
                                 std::to_string(_error));
}

#define THROW_ON_ERROR(error) clThrowOnError(error, std::to_string(__LINE__));

inline cl_image_desc
_create_image_desc(cv::InputArray _image) noexcept
{
    return {CL_MEM_OBJECT_IMAGE2D,
            _image.cols(),
            _image.rows(),
            0,
            0,
            0,
            0,
            0,
            0,
            nullptr};
}

cl_platform_id
_get_platform_id()
{
    // get the platform id count
    cl_uint platform_id_count = 0;
    clThrowOnError(clGetPlatformIDs(0, nullptr, &platform_id_count));

    if (platform_id_count == 0)
        throw std::runtime_error("failed to get a platform id");

    // get the platform ids
    std::vector<cl_platform_id> platform_ids(platform_id_count);
    clThrowOnError(
        clGetPlatformIDs(platform_id_count, platform_ids.data(), nullptr));

    return platform_ids.front();
}

std::vector<cl_device_id>
_get_device_id(cl_platform_id _platform_id)
{
    // get the device ids
    cl_uint device_id_count = 0;
    clThrowOnError(clGetDeviceIDs(_platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr,
                                  &device_id_count));

    if (device_id_count == 0)
        throw std::runtime_error("failed to get device id count");
    std::vector<cl_device_id> device_ids(device_id_count);
    clThrowOnError(clGetDeviceIDs(_platform_id, CL_DEVICE_TYPE_GPU,
                                  device_id_count, device_ids.data(), nullptr));

    if (device_ids.empty())
        throw std::runtime_error("failed to get device ids");
    return device_ids;
}

std::string
_load_kernel(const std::string &_file)
{
    std::ifstream in(_file);
    std::string result((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
    std::cout << "loaded following kernel:\n"
              << result << std::endl;
    return result;
}

cl_program
_create_program(const std::string &source, cl_context context)
{
    size_t lengths[1] = {source.size()};
    const char *sources[1] = {source.data()};

    cl_int error = 0;
    cl_program program =
        clCreateProgramWithSource(context, 1, sources, lengths, &error);
    clThrowOnError(error);

    return program;
}

inline std::string
print_threshold(std::size_t _threshold) noexcept
{
    return "-D THRESHOLD=" + std::to_string(_threshold);
}

inline std::string
print_radius(std::size_t _radius) noexcept
{
    return "-D RADIUS=" + std::to_string(_radius) +
           " -D POW_RADIUS=" + std::to_string(std::pow(_radius, 2));
}

constexpr const char *horizontal_filter = "HorizontalFilter";
constexpr const char *vertical_filter = "VerticalFilter";
constexpr const char *my_kernel =
    "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
    "CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
    "#define IS_SET(value) value[0] > 200\n"
    "#define WITHIN(value, iter) value + pown((float)(iter), 2) < POW_RADIUS\n"
    "__kernel void\n"
    "HorizontalFilter(__read_only image2d_t inputImage,\n"
    "__write_only image2d_t outputImage, unsigned int _radius)\n"
    "{\n"
    "const int2 currentPose = {get_global_id(0), get_global_id(1)};\n"
    "for (int ii = 0; ii != _radius; ++ii)\n"
    "{\n"
    "const int2 offset = {ii, 0};\n"
    "if (IS_SET(read_imageui(inputImage, sampler, currentPose + offset)) ||\n"
    "IS_SET(read_imageui(inputImage, sampler, currentPose - offset)))\n"
    "{\n"
    "write_imagef(outputImage, currentPose, (float4)(pown((float)(ii), 2), 0, 0, 0));\n"
    "return;\n"
    "}\n"
    "}\n"
    "write_imagef(outputImage, currentPose, (float4)(POW_RADIUS + 1, 0, 0, 0));\n"
    "}\n"
    "\n"
    "__kernel void\n"
    "VerticalFilter(__read_only image2d_t inputImage,\n"
    "__write_only image2d_t outputImage, unsigned int _radius)\n"
    "{\n"
    "const int2 currentPose = (int2)(get_global_id(0), get_global_id(1));\n"
    "for (unsigned int ii = 0; ii != RADIUS; ++ii)\n"
    "{\n"
    "const int2 offset = {0, ii};\n"
    "if (WITHIN(read_imagef(inputImage, sampler, currentPose + offset)[0],ii) ||\n"
    "WITHIN(read_imagef(inputImage, sampler, currentPose - offset)[0], ii))\n"
    "{\n"
    "write_imageui(outputImage, currentPose, (uint4)(0, 0, 0, 0));\n"
    "return;\n"
    "}\n"
    "}\n"
    "write_imageui(outputImage, currentPose, (uint4)(255, 0, 0, 0));\n"
    "}\n";

_cl_filter::_cl_filter(uint8_t _radius)
{
    // get the platform id and the corresponding device ids
    const auto platform_id = _get_platform_id();
    const auto device_ids = _get_device_id(platform_id);

    // setup the context
    context_properties_ = {CL_CONTEXT_PLATFORM,
                           reinterpret_cast<cl_context_properties>(platform_id),
                           0, 0};
    cl_int error;
    context_ = clCreateContext(context_properties_.data(), device_ids.size(),
                               device_ids.data(), nullptr, nullptr, &error);
    clThrowOnError(error);

    // Create the program
    // todo fix this path
    program_ = _create_program(my_kernel, context_);

    std::string args = print_threshold(100) + " " + print_radius(_radius);
    // std::cout << "building with following args\n"
    //           << args << std::endl;
    clThrowOnError(clBuildProgram(program_, device_ids.size(), device_ids.data(),
                                  args.data(), nullptr, nullptr));
    // Create kernel(s)
    h_filter_ = clCreateKernel(program_, horizontal_filter, &error);
    clThrowOnError(error);

    v_filter_ = clCreateKernel(program_, vertical_filter, &error);
    clThrowOnError(error);

    // Create queue
    queue_ = clCreateCommandQueueWithProperties(context_, device_ids.front(),
                                                nullptr, &error);
    clThrowOnError(error);
}

_cl_filter::~_cl_filter()
{
    // dismantle the cl data in reversed order
    clReleaseCommandQueue(queue_);
    clReleaseKernel(h_filter_);
    clReleaseKernel(v_filter_);
    clReleaseProgram(program_);
    clReleaseContext(context_);
}

void _cl_filter::filter(cv::InputArray input, cv::OutputArray _output,
                        std::size_t _radius)
{
    const auto image_desc = _create_image_desc(input);
    const auto read_only_and_copy_host = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    // note: assume one channel and
    const cl_image_format format_ = {CL_R, CL_UNSIGNED_INT8};

    cl_int error;
    // create the images
    cl_mem input_image =
        clCreateImage(context_, read_only_and_copy_host, &format_, &image_desc,
                      const_cast<unsigned char *>(input.getMat().data), &error);
    clThrowOnError(error);

    const cl_image_format float_format = {CL_R, CL_FLOAT};
    cl_mem temp_image = clCreateImage(context_, CL_MEM_READ_WRITE, &float_format,
                                      &image_desc, nullptr, &error);

    clThrowOnError(error);
    cl_mem output_image = clCreateImage(context_, CL_MEM_WRITE_ONLY, &format_,
                                        &image_desc, nullptr, &error);
    clThrowOnError(error);

    // setup kernel args
    size_t argc = 0;
    THROW_ON_ERROR(
        clSetKernelArg(h_filter_, argc++, sizeof(cl_mem), &input_image));
    THROW_ON_ERROR(
        clSetKernelArg(h_filter_, argc++, sizeof(cl_mem), &temp_image));
    THROW_ON_ERROR(
        clSetKernelArg(h_filter_, argc++, sizeof(unsigned int), &_radius));

    // Run the processing on the device
    std::size_t origin[3] = {0};
    std::size_t size[3] = {input.cols(), input.rows(), 1};
    cl_event h_event;
    THROW_ON_ERROR(clEnqueueNDRangeKernel(queue_, h_filter_, 2, nullptr, size,
                                          nullptr, 0, nullptr, &h_event));

    argc = 0;
    THROW_ON_ERROR(
        clSetKernelArg(v_filter_, argc++, sizeof(cl_mem), &temp_image));
    THROW_ON_ERROR(
        clSetKernelArg(v_filter_, argc++, sizeof(cl_mem), &output_image));
    THROW_ON_ERROR(
        clSetKernelArg(v_filter_, argc++, sizeof(unsigned int), &_radius));

    cl_event v_event;
    error = clEnqueueNDRangeKernel(queue_, v_filter_, 2, nullptr, size, nullptr, 1,
                                   &h_event, nullptr);
    clThrowOnError(error);

    // read the output
    error = clEnqueueReadImage(queue_, output_image, CL_TRUE, origin, size, 0, 0,
                               _output.getMat().data, 0, nullptr, nullptr);
    clThrowOnError(error);

    // release the allocated memory
    error = clReleaseMemObject(temp_image);
    error = clReleaseMemObject(output_image);
    error = clReleaseMemObject(input_image);
}

///////////// Dilate ////////////////////

typedef FilterFixture DilateFixture;

OCL_PERF_TEST_P(DilateFixture, Dilate,
                ::testing::Combine(OCL_TEST_SIZES, ::testing::Values(CV_8UC1), OCL_PERF_ENUM(3, 5, 10, 20, 50, 100)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);
    const Mat ker = getStructuringElement(MORPH_ELLIPSE, Size(ksize, ksize));

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    OCL_TEST_CYCLE()
    cv::dilate(src, dst, ker);

    SANITY_CHECK(dst);
}

typedef FilterFixture SepDilateFixture;

OCL_PERF_TEST_P(SepDilateFixture, Dilate,
                ::testing::Combine(OCL_TEST_SIZES, ::testing::Values(CV_8UC1), OCL_PERF_ENUM(3, 5, 10, 20, 50, 100)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    // note: my impl uses radius and not diameter
    const int type = get<1>(params), ksize = get<2>(params) / 2 + 1;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    // lets not exclude the ocl compilation -
    // I'm cheating here since my compilation takes way longer then
    // the impl of opencv
    _cl_filter filter(ksize);

    auto run_filter = [&]() {
        filter.filter(src, dst, ksize);
    };

    OCL_TEST_CYCLE() run_filter();

    SANITY_CHECK(dst);
}

OCL_PERF_TEST_P(SepDilateFixture, MaxDilate,
                ::testing::Combine(OCL_TEST_SIZES, ::testing::Values(CV_8UC1), OCL_PERF_ENUM(3, 5, 10, 20, 50, 100)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    // note: my impl uses radius and not diameter
    const int type = get<1>(params), ksize = get<2>(params) / 2 + 1;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    // since my impl escapes early, lets make a "fair" game and use the worst-case
    // scenario
    Mat src(srcSize, type, Scalar(0)), dst(srcSize, type, Scalar(0));

    // lets not exclude the ocl compilation -
    // I'm cheating here since my compilation takes way longer then
    // the impl of opencv
    _cl_filter filter(ksize);

    auto run_filter = [&]() {
        filter.filter(src, dst, ksize);
    };

    OCL_TEST_CYCLE()
    run_filter();

    SANITY_CHECK(dst);
}

} // namespace ocl
} // namespace opencv_test

#endif // HAVE_OPENCL
