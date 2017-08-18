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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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

#include "../../precomp.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <assert.h>
#include "common.hpp"
#include "ocl4dnn.hpp"
#include "opencl_kernels_dnn.hpp"
#include "math_functions.hpp"

#if defined WIN32 || defined _WIN32
#include <windows.h>
#include <direct.h>
#endif

#ifdef HAVE_OPENCL
namespace cv { namespace dnn { namespace ocl4dnn {
template<typename Dtype>
OCL4DNNConvSpatial<Dtype>::OCL4DNNConvSpatial(OCL4DNNConvConfig config)
{
    bias_term_ = config.bias_term;
    int32_t dims = config.in_shape.size();
    int32_t spatial_dims = config.kernel.size();

    channels_   = config.in_shape[dims - spatial_dims - 1];
    num_output_ = config.out_shape[dims - spatial_dims - 1];
    group_ = config.group;

    std::vector<int32_t> pad_;
    std::vector<int32_t> stride_;
    std::vector<int32_t> dilation_;
    std::vector<int32_t> kernel_shape_;
    std::vector<int32_t> im_in_shape_;
    std::vector<int32_t> im_out_shape_;

    for (int32_t i = 0; i < spatial_dims; ++i) {
        kernel_shape_.push_back(config.kernel[i]);
        pad_.push_back(config.pad[i]);
        stride_.push_back(config.stride[i]);
        dilation_.push_back(config.dilation[i]);
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    bestKernelConfig = NULL;
    prev_kernel_type_ = -1;
    bias_ = NULL;
    tuned_ = false;
    swizzled_weights_ = NULL;
    kernel_dim_ = channels_ / group_;
    out_spatial_dim_ = 1;

    int32_t in_spatial_dim_ = 1;
    for (int32_t i = 0; i < spatial_dims; ++i) {
        kernel_dim_ *= config.kernel[i];
        in_spatial_dim_ *= config.in_shape[dims - spatial_dims + i];
        out_spatial_dim_ *= config.out_shape[dims - spatial_dims + i];
    }

    // assumption: spatial dimension is 2.
    kernel_h_ = kernel_shape_[0];
    kernel_w_ = kernel_shape_[1];
    pad_h_ = pad_[0];
    pad_w_ = pad_[1];
    stride_h_ = stride_[0];
    stride_w_ = stride_[1];
    dilation_h_ = dilation_[0];
    dilation_w_ = dilation_[1];
    M_ = num_output_ / group_;
    K_ = channels_ * kernel_h_ * kernel_w_ / group_;
    height_ = im_in_shape_[0];
    width_ = im_in_shape_[1];
    output_h_ = im_out_shape_[0];
    output_w_ = im_out_shape_[1];
    bottom_dim_ = channels_ * in_spatial_dim_;
    top_dim_ = num_output_ * out_spatial_dim_;
    auto_tuning_ = false;

    if (std::getenv("OPENCV_OCL4DNN_KERNEL_CONFIG_PATH")) {
        cache_path_ << std::getenv("OPENCV_OCL4DNN_KERNEL_CONFIG_PATH");
    }

    bool hasCacheDir = false;
#if defined WIN32 || defined _WIN32
    if (cache_path_.str().empty() && std::getenv("USERPROFILE"))
        cache_path_ << std::getenv("USERPROFILE");

    struct _stat file_stat;
    cache_path_ << "\\spatialkernels\\";
    hasCacheDir = _stat(cache_path_.str().c_str(), &file_stat) == 0 &&
                  ((_S_IFDIR & file_stat.st_mode) != 0);
#else
    if (cache_path_.str().empty() && std::getenv("HOME"))
        cache_path_ << std::getenv("HOME");

    struct stat file_stat;
    cache_path_ << "/spatialkernels/";
    hasCacheDir = stat(cache_path_.str().c_str(), &file_stat) == 0 &&
                  S_ISDIR(file_stat.st_mode);
#endif

    if (!hasCacheDir) {
#if defined WIN32 || defined _WIN32
        int result = _mkdir(cache_path_.str().c_str());
#else
        int result = mkdir(cache_path_.str().c_str(), 0755);
#endif
        hasCacheDir = (result == -1) ? false : true;
    }
    if (!hasCacheDir) {
        std::cout << "Failed to create cache directory: "
                  << cache_path_.str()
                  << ", disable auto-tuning." << std::endl;
    }
    auto_tuning_ = (getenv("OPENCV_OCL4DNN_ENABLE_AUTO_TUNING") != NULL)
                   && hasCacheDir;
}

template<typename Dtype>
OCL4DNNConvSpatial<Dtype>::~OCL4DNNConvSpatial()
{
    if (swizzled_weights_) {
        clReleaseMemObject((cl_mem)swizzled_weights_);
    }
    if (bestKernelConfig) {
        delete bestKernelConfig;
    }
}

template<typename Dtype>
std::string OCL4DNNConvSpatial<Dtype>::generateHeader()
{
    std::stringstream ss;

    if (std::is_same<Dtype, double>::value) {
        // Test/enable KHR 64 bit (double)
        ss << "#if defined(cl_khr_fp64)" << std::endl;
        ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
        ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;

        // Test/enable AMD 64 bit (double)
        ss << "#elif defined(cl_amd_fp64)" << std::endl;
        ss << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable" << std::endl;
        ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;
        ss << "#endif" << std::endl;
    }

    // Test/enable 32 bit atomics
    ss << "#if defined(cl_khr_int32_base_atomics)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable"
       << std::endl;
    ss << "#define ATOMICS_32_AVAILABLE" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#if defined(cl_khr_global_int32_base_atomics)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable"
       << std::endl;
    ss << "#define ATOMICS_32_AVAILABLE" << std::endl;
    ss << "#endif" << std::endl;

    // 64 bit integers
    if (sizeof(int32_t) == 8 || std::is_same<Dtype, double>::value) {
        // Test/enable 64 bit atomics
        ss << "#if defined(cl_khr_int64_base_atomics)" << std::endl;
        ss << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
           << std::endl;
        ss << "#define ATOMICS_64_AVAILABLE" << std::endl;
        ss << "#endif" << std::endl;
    }

    if (std::is_same<Dtype, double>::value) {
        ss << "#define Dtype double" << std::endl;
        ss << "#define Dtype1 double" << std::endl;
        // double2, double4, double8, double16
        for (int32_t i = 2; i <= 16; i *= 2) {
            ss << "#define Dtype" << i << " double" << i << std::endl;
        }
    } else {
        ss << "#define Dtype float" << std::endl;
        ss << "#define Dtype1 float" << std::endl;
        // float2, float4, float8, float16
        for (int32_t i = 2; i <= 16; i *= 2) {
            ss << "#define Dtype" << i << " float" << i << std::endl;
        }
    }

    if (sizeof(int32_t) == 8) {
        ss << "#define int32_t long" << std::endl;
        ss << "#define uint32_t unsigned long" << std::endl;
        ss << "#define int32_tc long" << std::endl;
        ss << "#define uint32_tc unsigned long" << std::endl;
    } else {
        ss << "#define int32_t int" << std::endl;
        ss << "#define uint32_t unsigned int" << std::endl;
        ss << "#define int32_tc int" << std::endl;
        ss << "#define uint32_tc unsigned int" << std::endl;
    }

    return ss.str();
}

template<typename Dtype>
std::string OCL4DNNConvSpatial<Dtype>::generateDefs()
{
    std::stringstream ss;

    ss << "#define __CAT(x, y) x##y" << std::endl;
    ss << "#define CAT(x, y) __CAT(x, y)" << std::endl;
    ss << "#define LOOP0(VAR, STMT)" << std::endl;
    ss << "#define LOOP1(VAR, STMT) (STMT); (VAR)++;" << std::endl;
    ss << "#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))"
       << std::endl;

    addDef(ss, "KERNEL_WIDTH", kernel_w_);
    addDef(ss, "KERNEL_HEIGHT" , kernel_h_);
    addDef(ss, "STRIDE_X", stride_w_);
    addDef(ss, "STRIDE_Y", stride_h_);
    addDef(ss, "DILATION_X", dilation_w_);
    addDef(ss, "DILATION_Y", dilation_h_);
    addDef(ss, "INPUT_PAD_W", pad_w_);
    addDef(ss, "INPUT_PAD_H", pad_h_);

    return ss.str();
}

typedef enum {
    KERNEL_TYPE_INTEL_IDLF = 2,
    KERNEL_TYPE_BASIC = 4,
    KERNEL_TYPE_GEMM_LIKE = 5
} ocl4dnnConvSpatialKernelType_t;

template<typename Dtype>
std::string OCL4DNNConvSpatial<Dtype>::generateKernels(int32_t kernelType,
                                                      int32_t blockM,
                                                      int32_t blockK,
                                                      int32_t blockN)
{
    std::stringstream ss;
    std::stringstream opts;
    std::string kernelUKey;
    int32_t simd_size;

    if (kernelType == KERNEL_TYPE_INTEL_IDLF) {
        simd_size = blockN;
        kernelUKey = generateSpecificKey(2, blockM, blockK, 1);

        // kernel name
        kernel_name_ = "IDLF_";
        kernel_name_ += kernelUKey.c_str();
        if (simd_size == 16)
            kernel_name_ += "_SIMD16";
        else
            kernel_name_ += "_SIMD8";

        // options
        opts << "-cl-fast-relaxed-math -D convolve_simd=" << kernel_name_;
        if (isBeignet())
            opts << " -D__BEIGNET__ ";
        else
            opts << " -cl-no-subgroup-ifp ";
        options_ = opts.str();

        // defs
        int32_t output_width = output_w_;
        int32_t output_height = output_h_;
        int32_t output_block_width = blockM;
        int32_t output_block_height = blockK;
        const int32_t last_block_width = (output_width % output_block_width == 0) ?
                                        output_block_width : output_width % output_block_width;
        const int32_t last_block_height = (output_height % output_block_height == 0) ?
                                         output_block_height : output_height % output_block_height;
        int tile_x = (((output_block_width - 1) * stride_w_ + kernel_w_ * dilation_w_) + 3) & ~3;
        int tile_y = (output_block_height -1) * stride_h_ + kernel_h_ * dilation_h_;
        int tile_y_stride = (4 * simd_size) / tile_x;
        int invec_size = (tile_y + tile_y_stride - 1) / tile_y_stride;

        addDef(ss, "SIMD_SIZE", simd_size);
        addDef(ss, "filter_qualifier", "__global");
        addDef(ss, "OUT_BLOCK_WIDTH", output_block_width);
        addDef(ss, "OUT_BLOCK_HEIGHT", output_block_height);
        addDef(ss, "LAST_BLOCK_WIDTH", last_block_width);
        addDef(ss, "LAST_BLOCK_HEIGHT", last_block_height);
        addDef(ss, "INPUT_DEPTH", channels_ / group_);
        addDef(ss, "TOTAL_INPUT_DEPTH_SIZE", channels_);
        addDef(ss, "TOTAL_OUTPUT_DEPTH", num_output_);
        addDef(ss, "INPUT_START_X", 0);
        addDef(ss, "INPUT_START_Y", 0);
        addDef(ss, "INPUT_START_Z", 0);
        addDef(ss, "NUM_FILTERS", M_);
        addDef(ss, "OUT_BUFF_OFFSET", 0);
        addDef(ss, "TILE_X", tile_x);
        addDef(ss, "TILE_Y", tile_y);
        addDef(ss, "TILE_Y_STRIDE", tile_y_stride);
        addDef(ss, "INVEC_SIZE", invec_size);
        addDef(ss, "ALIGNED_NUM_FILTERS", ALIGN(M_, simd_size));
        addDef(ss, "OUT_BLOCK_SIZE", (output_block_width*output_block_height));

        // kernel source
        // Each work-item computes
        // a OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT region of one output map.
        // Each work-group (which will be mapped to 1 SIMD16/SIMD8 EU thread)
        // will compute 16/8 different feature maps,
        // but each feature map is for the same region of the imput image.
        // NDRange:  (output_width+pad)/ OUT_BLOCK_WIDTH,
        //           (output_height+pad)/OUT_BLOCK_HEIGHT,
        //           NUM_FILTERS/OUT_BLOCK_DEPTH
        // NOTE: for beignet
        // this reqd_work_group_size does not guarantee that
        // SIMD16/8 mode will be used,
        // the compiler could choose to use two SIMD8 threads,
        // and if that happens the code will break.
        ss << "#define activation_function(x) (x)" << std::endl;
        ss << "__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))" << std::endl;
        ss << "kernel void" << std::endl;
        ss << "convolve_simd(" << std::endl;
        ss << "__global float* inputs_base," << std::endl;
        ss << "filter_qualifier float* weights_base," << std::endl;
        ss << "__global float* biases_base," << std::endl;
        ss << "__global float* outputs_base," << std::endl;
        ss << "const ushort input_width," << std::endl;
        ss << "const ushort input_height," << std::endl;
        ss << "const ushort output_width," << std::endl;
        ss << "const ushort output_height)" << std::endl;
        ss << "{" << std::endl;
        ss << "__global float* outputs = outputs_base;" << std::endl;
        ss << "__global float* inputs = inputs_base;" << std::endl;
        ss << "filter_qualifier float* weights = weights_base;" << std::endl;
        ss << "__global float* biases = biases_base;" << std::endl;
        // oc = Output Column
        ss << "uint32_t oc = get_global_id(0) * OUT_BLOCK_WIDTH;" << std::endl;
        // or = Output Row
        ss << "uint32_t or = get_global_id(1) * OUT_BLOCK_HEIGHT;" << std::endl;
        // fm = Feature Map = od = Output Depth
        ss << "uint32_t fm = get_global_id(2);" << std::endl;
        ss << "uint32_t fmg = get_group_id(2);" << std::endl;
        ss << "uint32_t lid = get_local_id(2);" << std::endl;
        ss << "float out[OUT_BLOCK_SIZE];" << std::endl;
        ss << "int32_t in_addr;" << std::endl;
        // find weights adress of given neuron (lid is index)
        ss << "uint32_t weight_addr = (fmg % (ALIGNED_NUM_FILTERS/SIMD_SIZE)) * "
           << "INPUT_DEPTH * KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE + lid;"
           << std::endl;
        ss << "for(int32_t i=0;i<OUT_BLOCK_SIZE;i++) {" << std::endl;
        ss << "out[i]=0.0f;" << std::endl;
        ss << "}" << std::endl;
        ss << "uint32_t num_in_batch = ( fm ) / ALIGNED_NUM_FILTERS;" << std::endl;
        ss << "uint32_t input_batch_offset = "
           << "num_in_batch * input_height * input_width * TOTAL_INPUT_DEPTH_SIZE;"
           << std::endl;
        ss << "int curr_y = or * STRIDE_Y + INPUT_START_Y + (lid / (TILE_X/4));"
           << std::endl;
        ss << "int curr_x = oc * STRIDE_X + INPUT_START_X + (lid % (TILE_X/4)) * 4;"
           << std::endl;
        ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
        ss << "int saved_y = curr_y;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "in_addr = "
           << "input_batch_offset + INPUT_START_Z * input_height * input_width"
            // y tile offset
           << "+  (curr_y - INPUT_PAD_H) * input_width"
            // x tile offset
           << "+   curr_x - INPUT_PAD_W;"
           << std::endl;
        ss << "union {" << std::endl;
        ss << "float4 in_vec[INVEC_SIZE];" << std::endl;
        ss << "float in_array[INVEC_SIZE * 4];" << std::endl;
        ss << "} in_buf;" << std::endl;
        ss << "for(int32_t kd = 0; kd < INPUT_DEPTH; kd++)" << std::endl;
        ss << "{" << std::endl;
        ss << "int32_t in_offset = in_addr;" << std::endl;
        ss << "int32_t reg = 0;" << std::endl;
        ss << "LOOP(INVEC_SIZE, reg," << std::endl;
        ss << "{" << std::endl;
        ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
        ss << "if (curr_y >= INPUT_PAD_H && "
           << "curr_y < input_height + INPUT_PAD_H && "
           << "curr_x + 3 >= INPUT_PAD_W && "
           << "curr_x < input_width + INPUT_PAD_W) {" << std::endl;
        ss << "if (curr_x < INPUT_PAD_W) {" << std::endl;
        ss << "in_buf.in_vec[reg].s0 = 0;" << std::endl;
        ss << "if (curr_x + 1 >= INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s1 = *(inputs + in_offset + 1);" << std::endl;
        ss << "else" << std::endl;
        ss << "in_buf.in_vec[reg].s1 = 0;" << std::endl;
        ss << "if (curr_x + 2 >= INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s2 = *(inputs + in_offset + 2);" << std::endl;
        ss << "else" << std::endl;
        ss << "in_buf.in_vec[reg].s2 = 0;" << std::endl;
        ss << "in_buf.in_vec[reg].s3 = *(inputs + in_offset + 3);" << std::endl;
        ss << "} else {" << std::endl;
        // read SIMD_SIZE elements
        ss << "in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);"
           << std::endl;
        ss << "if (curr_x + 1 >= input_width + INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s1 = 0;" << std::endl;
        ss << "if (curr_x + 2 >= input_width + INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s2 = 0;" << std::endl;
        ss << "if (curr_x + 3 >= input_width + INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s3 = 0;" << std::endl;
        ss << "}" << std::endl;
        ss << "} else {" << std::endl;
        ss << "in_buf.in_vec[reg] = 0;" << std::endl;
        ss << "}" << std::endl;
        ss << "curr_y += TILE_Y_STRIDE;" << std::endl;
        ss << "#else" << std::endl;
        // read SIMD_SIZE elements
        ss << "in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);"
           << std::endl;
        ss << "#endif" << std::endl;
        ss << "in_offset += input_width * TILE_Y_STRIDE;" << std::endl;
        ss << "});" << std::endl;
        ss << "in_addr += input_height * input_width;" << std::endl;
        ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
        ss << "curr_y = saved_y;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
        ss << "#define WEIGHT_PREF 8" << std::endl;
        ss << "#else" << std::endl;
        ss << "#define WEIGHT_PREF 1" << std::endl;
        ss << "#endif" << std::endl;
        ss << "union {" << std::endl;
        ss << "float w[WEIGHT_PREF];" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
        ss << "uint8 ui8;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "} weight_buf;" << std::endl;
        ss << "int32_t w_idx=0;" << std::endl;
        ss << "uint32_t orig_weight_addr = weight_addr;" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
        ss << "weight_buf.ui8 = "
           << "intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "weight_addr += SIMD_SIZE * WEIGHT_PREF;" << std::endl;
        ss << "#else" << std::endl;
        ss << "weight_buf.w[0] = as_float("
           << "intel_sub_group_block_read((__global uint *)&weights[weight_addr]));"
           << std::endl;
        ss << "weight_addr += SIMD_SIZE * 1;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#define BLOCK_IN(n) "
           << "sub_group_broadcast("
           << "in_buf.in_array[((n)%4) + ((n) / (TILE_Y_STRIDE * TILE_X)) * 4], "
           << "(((n) % (TILE_Y_STRIDE * TILE_X))/4))" << std::endl;
        // kr = Kernel Row
        ss << "int32_t kr = 0;" << std::endl;
        ss << "LOOP(KERNEL_HEIGHT, kr," << std::endl;
        ss << "{" << std::endl;
        // kc = Kernel Column
        ss << "int32_t kc = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH, kc," << std::endl;
        ss << "{" << std::endl;
        ss << "for(int32_t br=0; br < OUT_BLOCK_HEIGHT; br++) {" << std::endl;
        ss << "for(int32_t bc=0; bc < OUT_BLOCK_WIDTH; bc++) {" << std::endl;
        ss << "float input = BLOCK_IN((br * STRIDE_Y + kr * DILATION_Y) * "
           << "TILE_X + bc * STRIDE_X + kc * DILATION_X);" << std::endl;
        ss << "out[br * OUT_BLOCK_WIDTH + bc] = "
           << "mad(weight_buf.w[w_idx % WEIGHT_PREF], "
           << "input, out[br * OUT_BLOCK_WIDTH + bc]);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT > WEIGHT_PREF" << std::endl;
        // We assume KERNEL_WIDTH is equal to KERNEL_HEIGHT here.
        ss << "if ((w_idx + 1) % WEIGHT_PREF == 0" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 != 0" << std::endl;
        ss << "&& ((w_idx + 1) <= (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF))"
           << std::endl;
        ss << "#endif" << std::endl;
        ss << ") {" << std::endl;
        ss << "weight_buf.ui8 = "
           << "intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);"
           << std::endl;
        // weights must be stored in just the right SIMD swizzled format
        // for this to work, see host code for details.
        ss << "weight_addr += SIMD_SIZE * WEIGHT_PREF;" << std::endl;
        ss << "}" << std::endl;
        ss << "#if KERNEL_WIDTH*KERNEL_HEIGHT % 8 == 0" << std::endl;
        // need to do nothing
        ss << "#else" << std::endl;
        ss << "else if ((w_idx + 1) %  WEIGHT_PREF == 0 && "
           << "((w_idx + 1) > (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF)))"
           << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 1" << std::endl;
        ss << "weight_buf.w[0] = weights[weight_addr];" << std::endl;
        ss << "#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 2" << std::endl;
        ss << "weight_buf.ui8.s01 = "
           << "intel_sub_group_block_read2((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 <= 4" << std::endl;
        ss << "weight_buf.ui8.s0123 = "
           << "intel_sub_group_block_read4((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "#else" << std::endl;
        ss << "weight_buf.ui8 = "
           << "intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "#endif" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#endif" << std::endl;
        ss << "++w_idx;" << std::endl;
        ss << "});" << std::endl;
        ss << "});" << std::endl;
        ss << "weight_addr = "
           << "orig_weight_addr + KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE;"
           << std::endl;
        ss << "}" << std::endl;
        // dead code to work around possible compiler bug.
        ss << "if (ALIGNED_NUM_FILTERS != NUM_FILTERS && fm > 0xfffffffeul) {"
           << std::endl;
        ss << "outputs[0] = BLOCK_IN(fm % SIMD_SIZE);" << std::endl;
        ss << "}" << std::endl;
        ss << "fm = fm % ALIGNED_NUM_FILTERS;" << std::endl;
        ss << "if ((ALIGNED_NUM_FILTERS == NUM_FILTERS || fm < NUM_FILTERS)) {"
           << std::endl;
        ss << "uint32_t out_addr = "
           << "OUT_BUFF_OFFSET + "
           << "( num_in_batch * TOTAL_OUTPUT_DEPTH + fm ) * "
           << "output_width * output_height;"
           << std::endl;
        ss << "out_addr += or * output_width + oc;" << std::endl;
        ss << "float bias = biases[fm];" << std::endl;
        ss << "for(uint32_t r = 0; r < OUT_BLOCK_HEIGHT; r++) {" << std::endl;
        ss << "if (r + or >= output_height) break;" << std::endl;
        ss << "for(uint32_t c = 0; c < OUT_BLOCK_WIDTH; c++) {" << std::endl;
        ss << "if (c + oc >= output_width) break;" << std::endl;
        ss << "outputs[out_addr + r * output_width + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
    } else if (kernelType == KERNEL_TYPE_BASIC) {
        kernelUKey = generateSpecificKey(4, blockM, blockK, blockN);
        kernel_name_ = "BASIC_";
        kernel_name_ += kernelUKey.c_str();

        // opts
        opts << " -cl-fast-relaxed-math -D CFMultiNoPadding=" << kernel_name_;
        if (isBeignet())
            opts << " -D__BEIGNET__ ";
        else
            opts << " -cl-no-subgroup-ifp ";
        options_ = opts.str();

        // defs
        addDef(ss, "CHANNELS", channels_ / group_);
        addDef(ss, "APPLY_BIAS", bias_term_);
        addDef(ss, "OUTPUT_Z", M_);
        addDef(ss, "ZPAR", 1);

        // kernel
        ss << "#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) "
           << "do { (_dst_)[(_offset_)] = (_data_);} while(0)" << std::endl;
        ss << "__kernel void CFMultiNoPadding(" << std::endl;
        ss << "__global Dtype* image_data," << std::endl;
        ss << "int32_t image_offset," << std::endl;
        ss << "__global Dtype* kernel_data, " << std::endl;
        ss << "int32_t kernel_offset," << std::endl;
        ss << "__global Dtype* bias," << std::endl;
        ss << "const int32_t bias_offset," << std::endl;
        ss << "__global Dtype* convolved_image, " << std::endl;
        ss << "const int32_t convolved_image_offset," << std::endl;
        ss << "const ushort input_width," << std::endl;
        ss << "const ushort input_height," << std::endl;
        ss << "const ushort output_width," << std::endl;
        ss << "const ushort output_height," << std::endl;
        ss << "const ushort pad_w," << std::endl;
        ss << "const ushort pad_h) {" << std::endl;
        ss << "const int32_t outputX = get_global_id(0);" << std::endl;
        ss << "const int32_t outputY = get_global_id(1);" << std::endl;
        ss << "const int32_t kernelNum = get_global_id(2)*ZPAR;" << std::endl;
        ss << "if(outputX < output_width && outputY < output_height)" << std::endl;
        ss << "{" << std::endl;
        ss << "Dtype sum[ZPAR];" << std::endl;
        ss << "for(int32_t kern =0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "sum[kern] = 0.0f;" << std::endl;
        ss << "}" << std::endl;
        ss << "const int32_t org_y = outputY * STRIDE_Y - pad_h;" << std::endl;
        ss << "const int32_t org_x = outputX * STRIDE_X - pad_w;" << std::endl;
        ss << "const int32_t currentKernelOffset = "
           << "kernel_offset + kernelNum*KERNEL_HEIGHT*KERNEL_WIDTH*CHANNELS;"
           << std::endl;
        ss << "const int32_t biasIndex=bias_offset + kernelNum;" << std::endl;
        ss << "const int32_t local_image_offset = org_y*input_width + org_x;"
           << std::endl;
        ss << "const int32_t imageSize = input_width*input_height;" << std::endl;
        ss << "__global Dtype* image_dataPtrFloat = "
           << "(image_data + (image_offset + local_image_offset));" << std::endl;
        ss << "__global Dtype* kernel_dataPtrFloat = "
           << "(kernel_data + (currentKernelOffset));" << std::endl;
        ss << "for(int32_t c = 0; c < CHANNELS; c++)" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int32_t y = 0; y < KERNEL_HEIGHT; y++)" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int32_t x = 0; x < KERNEL_WIDTH; x++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if(!(org_y + y * DILATION_Y >= 0 && "
           << "org_y + y * DILATION_Y < input_height && "
           << "org_x + x * DILATION_X >= 0 && "
           << "org_x + x * DILATION_X < input_width))" << std::endl;
        ss << "{" << std::endl;
        ss << "continue;" << std::endl;
        ss << "}" << std::endl;
        ss << "for(int32_t kern =0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "sum[kern] += image_dataPtrFloat[x * DILATION_X] * "
           << "kernel_dataPtrFloat[kern*KERNEL_HEIGHT*KERNEL_WIDTH*CHANNELS + x];"
           << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "image_dataPtrFloat += input_width * DILATION_Y;" << std::endl;
        ss << "kernel_dataPtrFloat += KERNEL_WIDTH;" << std::endl;
        ss << "}" << std::endl;
        ss << "image_dataPtrFloat += "
           << "imageSize - input_width*KERNEL_HEIGHT*DILATION_Y;" << std::endl;
        ss << "}" << std::endl;
        ss << "if(APPLY_BIAS == 1)" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int32_t kern = 0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if(kernelNum+kern < OUTPUT_Z)" << std::endl;
        ss << "{" << std::endl;
        ss << "int32_t offset = convolved_image_offset + "
           << "(kernelNum+kern)*output_height*output_width + "
           << "outputY*output_width + outputX;" << std::endl;
        ss << "ACTIVATION_FUNCTION(convolved_image, offset, sum[kern] + "
           << "bias[biasIndex +kern]);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "else" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int32_t kern = 0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if(kernelNum+kern < OUTPUT_Z)" << std::endl;
        ss << "{" << std::endl;
        ss << "int32_t offset = convolved_image_offset + "
           << "(kernelNum+kern)*output_height*output_width + "
           << "outputY*output_width + outputX;" << std::endl;
        ss << "ACTIVATION_FUNCTION(convolved_image, offset, sum[kern]);"
           << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
    }
    return ss.str();
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::generateKernelSrc()
{
    std::stringstream ss;

    ss << generateHeader();
    ss << generateDefs();
    ss << generateKernels(kernelType_, blockM_, blockK_, blockN_);

    kernel_ = ss.str();
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::Forward(const UMat& bottom,
                                        const UMat& weight,
                                        const UMat& bias, UMat& top,
                                        int32_t batch_size)
{
    cl_int ret;
    bottom_data_ = (Dtype*)bottom.handle(ACCESS_READ);
    top_data_ = (Dtype*)top.handle(ACCESS_WRITE);
    weight_ = (Dtype*)weight.handle(ACCESS_READ);
    bias_ = (Dtype*)bias.handle(ACCESS_READ);
    num_ = batch_size;

    prepareKernel();
    ret = convolve(bottom_data_, top_data_, 0, num_, bestKernelConfig);

    return ret == CL_SUCCESS ? true : false;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::tune(Dtype* top_data,
                                    const Dtype* weight,
                                    const Dtype* bias,
                                    const Dtype* bottom_data,
                                    int32_t batch_size)
{
    cl_int err;
    Dtype *verify_data;
    ocl::Context &ctx = ocl::Context::getDefault();

    verify_data =
        reinterpret_cast<Dtype*>(clCreateBuffer((cl_context)ctx.ptr(),
                                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                batch_size * num_output_ * out_spatial_dim_ * sizeof(Dtype),
                                                NULL, &err));
    CHECK_EQ(err, CL_SUCCESS) << "Failed to create verify buffer." << std::endl;

    calculateBenchmark(bottom_data, weight, bias, verify_data);
    setupConvolution(bottom_data, top_data, verify_data);
    clReleaseMemObject((cl_mem)verify_data);
    CHECK_EQ(tuned_, true) << "Spatial convolution auto-tuning failed.";
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::calculateBenchmark(const Dtype* bottom,
                                                  const Dtype* w,
                                                  const Dtype* bias,
                                                  Dtype* verify_data)
{
    createBasicKernel(1, 1, 1);
    kernel_index_ = kernelQueue.size() - 1;
    convolve(bottom, verify_data, 0, num_, kernelQueue[kernel_index_]);
    ocl::Queue &queue = ocl::Queue::getDefault();
    clEnqueueCopyBuffer((cl_command_queue)queue.ptr(),
                        (cl_mem)top_data_,
                        (cl_mem)verify_data, 0, 0,
                        sizeof(float) * num_ * top_dim_, 0, NULL, NULL);
    phash.erase(kernelQueue[kernel_index_]->kernelName);
    kernelQueue.pop_back();
    return;
}

#define dbg
#ifdef dbg
#define dbgPrint(x) (x)
#else
#define dbgPrint(x)
#endif

// For large enough input size, we do not need to tune kernels for different
// size. The reason is with large input size, there will be enough work items
// to feed al the EUs.
// FIXME for the gemm like convolution, switch back to eaxct image size.

#define TUNING_SIZE(x) ((x) > 256 ? 256 : (ALIGN(x, 16)))

// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
template<typename Dtype>
uint64 OCL4DNNConvSpatial<Dtype>::crc64(const uchar* data, size_t size, uint64 crc0)
{
    static uint64 table[256];
    static bool initialized = false;

    if( !initialized )
    {
        for( int i = 0; i < 256; i++ )
        {
            uint64 c = i;
            for( int j = 0; j < 8; j++ )
                c = ((c & 1) ? CV_BIG_UINT(0xc96c5795d7870f42) : 0) ^ (c >> 1);
            table[i] = c;
        }
        initialized = true;
    }

    uint64 crc = ~crc0;
    for( size_t idx = 0; idx < size; idx++ )
        crc = table[(uchar)crc ^ data[idx]] ^ (crc >> 8);

    return ~crc;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::generateKey()
{
    std::stringstream keyBuilder;
    // FIXME: to support fuse?
    keyBuilder << kernel_w_ << "_"
               << kernel_h_ << "_"
               << channels_ << "_"
               << group_ << "_"
               << stride_h_ << "_"
               << stride_w_ << "_"
               << dilation_h_ << "_"
               << dilation_w_ << "_"
               << bias_term_ << "_"
               << TUNING_SIZE(width_) << "_"
               << TUNING_SIZE(height_) << "_"
               << pad_w_ << "_"
               << pad_h_ << "_"
               << num_ << "_"
               << M_;

    std::string prefix = ocl::Device::getDefault().name() +
                         ocl::Device::getDefault().vendorName() +
                         ocl::Device::getDefault().driverVersion() +
                         std::to_string(ocl::Device::getDefault().maxComputeUnits());
    prefix = prefix + keyBuilder.str();
    key_ = std::to_string(crc64((uchar*)prefix.c_str(), prefix.size()));
    short_key_ = keyBuilder.str();
}

template<typename Dtype>
std::string OCL4DNNConvSpatial<Dtype>::generateSpecificKey(int32_t type, int32_t blockWidth,
                                                            int32_t blockHeight, int32_t blockDepth)
{
    std::stringstream keyBuilder;
    keyBuilder << short_key_
               << "_" << type
               << "_" << blockWidth
               << "_" << blockHeight
               << "_" << blockDepth;
    return keyBuilder.str();
}

template<typename Dtype>
void interleaveMatrix(Dtype* mem_dst, const Dtype *mem,
                      int r, int c, int interleavedRows, int nonInterleavedRows,
                      int blockWidth, int rowAlignment )
{
    CHECK_EQ(interleavedRows % 2, 0) <<
             "interleaveMatrix only supports even values for interleavedRows.";

    size_t memSize = r * c * sizeof(float);
    size_t dstSize = memSize *
                     (interleavedRows + nonInterleavedRows * 2) /
                     (interleavedRows + nonInterleavedRows);
    memset(mem_dst, 0, dstSize);    // NOLINT

    const int xStride = blockWidth;
    const int yStride = c * 2;
    const Dtype *pSrc = mem;
    Dtype* pDst = mem_dst;
    for (int y = 0; y < r;) {
        for (int rows = 0; rows < interleavedRows; rows += 2) {
            if ( y >= r ) break;
            if ((c % xStride) == 0) {
                for (int x = 0; x < c / xStride; x++) {
                    memcpy(pDst + x * xStride * 2,                         // NOLINT
                           pSrc + x * xStride,     xStride * sizeof(Dtype));
                    memcpy(pDst + x * xStride * 2 + xStride,               // NOLINT
                           pSrc + x * xStride + c, xStride * sizeof(Dtype));
                }
            } else {
                const int count = c / xStride;
                int x = 0;
                for (; x < count - 1; x++) {
                    memcpy(pDst + x * xStride * 2,                          // NOLINT
                           pSrc + x * xStride, xStride * sizeof(Dtype));
                    memcpy(pDst + x * xStride * 2 + xStride,                // NOLINT
                           pSrc + x * xStride + c, xStride * sizeof(Dtype));
                }
                memcpy(pDst + x * xStride * 2,                            // NOLINT
                       pSrc + x * xStride, xStride * sizeof(Dtype));
            }
            pSrc += yStride;
            pDst += yStride;
            y += 2;
        }

        for (int rows = 0; rows < nonInterleavedRows; rows++) {
            if (y >= r) break;
            const int stride = rowAlignment;
            int remaining = c;
            for (int x = 0; x < c; x += stride) {
                if (remaining >= stride) {
                    memcpy(pDst + x * 2, pSrc + x, stride * sizeof(Dtype));    // NOLINT
                    remaining -=stride;
                } else {
                    memcpy(pDst + x * 2, pSrc + x, remaining * sizeof(Dtype));  // NOLINT
                }
            }
            pSrc += yStride / 2;
            pDst += yStride;
            y++;
        }
    }
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::swizzleWeights(const Dtype *bottom,
                                              const Dtype *top,
                                              int32_t swizzled_factor,
                                              bool interleave)
{
    // Simply skip the weight swizzle if we already got a swizzled_weights_
    // in test phase and not in auto tuning
    // This requires we always call convolve again with the winner configuration
    // during the auto tuning stage.
    if (tuned_ && swizzled_weights_ != NULL)
        return;

    cl_int err;
    ocl::Context ocl_ctx = ocl::Context::getDefault();
    if (swizzled_weights_ == NULL) {
        swizzled_weights_ = reinterpret_cast<Dtype*>(clCreateBuffer((cl_context)ocl_ctx.ptr(),
                                                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                                    sizeof(Dtype) *
                                                                    ((num_output_ + 15) & ~15) *
                                                                    channels_ * kernel_h_ * ((kernel_w_ + 1) & ~1),
                                                                    NULL, &err));
        CHECK_EQ(err, CL_SUCCESS) << "Failed to create swizzled_weights buffer.";
    }

    ocl::Queue queue = ocl::Queue::getDefault();

    if (!interleave) {
        ocl::Kernel oclk_copy_weight(CL_KERNEL_SELECT("copyWeightsSwizzled"), cv::ocl::dnn::conv_spatial_helper_oclsrc);
        cl_uint argIdx = 0;

        int32_t channels = channels_ / group_;
        oclk_copy_weight.set(argIdx++, (cl_mem) weight_);
        oclk_copy_weight.set(argIdx++, (cl_mem) swizzled_weights_);
        oclk_copy_weight.set(argIdx++, kernel_w_);
        oclk_copy_weight.set(argIdx++, kernel_h_);
        oclk_copy_weight.set(argIdx++, channels);
        oclk_copy_weight.set(argIdx++, this->num_output_);
        oclk_copy_weight.set(argIdx++, swizzled_factor);
        const size_t global_work_size_Copy[3] = {
            (size_t) (ALIGN(this->num_output_, swizzled_factor) * channels * kernel_w_ * kernel_h_), 1, 1 };

        OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                         (cl_kernel)oclk_copy_weight.ptr(), 3, NULL,
                                         global_work_size_Copy, NULL, 0, NULL,
                                         NULL));
    } else {
        Dtype* cpu_weight =
            reinterpret_cast<Dtype*>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                        (cl_mem)weight_, true, CL_MAP_READ, 0,
                                                        sizeof(Dtype) * num_output_ * kernel_dim_ * group_,
                                                        0, NULL, NULL, NULL));

        // assumption: kernel dimesion is 2
        Dtype* cpu_swizzled_weight =
            reinterpret_cast<Dtype*>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                        (cl_mem)swizzled_weights_,
                                                        true, CL_MAP_WRITE, 0,
                                                        sizeof(Dtype) *
                                                        ((num_output_ + 15) & ~15) *
                                                        channels_ * kernel_h_ * ((kernel_w_ + 1) & ~1),
                                                        0, NULL, NULL, NULL));

        int interleavedRows = (kernel_w_ / 2) * 2;
        int nonInterleavedRows = kernel_w_ % 2;
        int blockWidth = swizzled_factor;  // should equal to simd size.
        int rowAlignment = 32;
        size_t interleaved_filter_size = M_ * kernel_w_ * kernel_h_ * channels_ * sizeof(Dtype);
        Dtype * tmpSwizzledWeight = reinterpret_cast<Dtype*>(malloc(interleaved_filter_size));
        CHECK_EQ(tmpSwizzledWeight != NULL, true) << "Failed to allocate temporary swizzled weight";
        for (int od = 0; od < M_; od++)
            for (int id = 0; id < channels_; id++)
                for (int r = 0; r < kernel_h_; r++)
                    for (int c = 0; c < kernel_w_; c++)
                        tmpSwizzledWeight[((id * kernel_h_ + r)* kernel_w_ + c) * M_ + od] =
                            cpu_weight[((od * channels_ + id) * kernel_h_ + r)*kernel_w_+c];
        interleaveMatrix(cpu_swizzled_weight,
                         tmpSwizzledWeight,
                         kernel_w_ * kernel_h_ * channels_, M_,
                         interleavedRows,
                         nonInterleavedRows,
                         blockWidth,
                         rowAlignment);

        clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                                (cl_mem)weight_,
                                cpu_weight, 0, NULL,
                                NULL);
        clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                                (cl_mem)swizzled_weights_,
                                cpu_swizzled_weight, 0, NULL,
                                NULL);
        free(tmpSwizzledWeight);
    }
}

template<>
void OCL4DNNConvSpatial<float>::computeGlobalSize(int32_t batch,
                                                 int32_t* wio,    // work item output size
                                                 size_t* lSize,  // local size
                                                 size_t* gSize)  // global size
{
    gSize[0] = ceil((fmax(static_cast<float>(output_w_) / wio[0], 1.0)) / lSize[0]) * lSize[0];
    gSize[1] = ceil((fmax(static_cast<float>(output_h_) / wio[1], 1.0)) / lSize[1]) * lSize[1];
    gSize[2] = ceil(static_cast<float>((ceil(static_cast<float>(M_) * batch / wio[2]))) / lSize[2]) * lSize[2];
}

template<>
bool OCL4DNNConvSpatial<float>::createBasicKernel(int32_t blockWidth,
                                                   int32_t blockHeight, int32_t blockDepth)
{
    int32_t workItemOutput[3];
    workItemOutput[0] = 1;
    workItemOutput[1] = 1;
    workItemOutput[2] = 1;

    kernelType_ = 4;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = blockDepth;
    generateKernelSrc();
    compileKernel();

    size_t localSize[3] = { 1, 1, 1 };
    size_t globalSize[3];

    computeGlobalSize(1, workItemOutput, localSize, globalSize);
    kernelQueue.push_back(new kernelConfig(kernel_name_, globalSize, localSize, workItemOutput,
                                           false, false, true, 4));

    return true;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setBufferKernelArg(const Dtype *bottom, const Dtype *top,
                                                  ocl::Kernel *kernel,
                                                  const cl_uint &argIdx,
                                                  ocl::Context *ctx,
                                                  cl_mem buffer, size_t offset,
                                                  size_t size, bool readOnly,
                                                  bool preserved)
{

    if (offset == 0) {
        kernel->set(argIdx, (cl_mem)buffer);
        return;
    }

    if (preserved &&
        subBufferMap.find(std::make_tuple(buffer, offset, size)) != subBufferMap.end()) {
        kernel->set(argIdx, (cl_mem)(subBufferMap.find(std::make_tuple(buffer, offset, size))->second));
        return;
    }
    cl_buffer_region region;
    region.origin = offset * sizeof(Dtype);
    region.size = size * sizeof(Dtype);
    cl_mem_flags memFlags = readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
    cl_int error;
    cl_mem sub_buffer = clCreateSubBuffer(buffer, memFlags,
                                          CL_BUFFER_CREATE_TYPE_REGION,
                                          &region, &error);
    CHECK_EQ(error, CL_SUCCESS) << "Failed to create sub buffer." << std::endl;
    if (error != CL_SUCCESS) {
        dbgPrint(std::cout << "Failed to create sub buffer (" << error << ")." << std::endl);
        throw(error);
    }
    kernel->set(argIdx, (cl_mem)sub_buffer);
    if (preserved)
        subBufferMap.insert(std::make_pair(std::make_tuple(buffer, offset, size),
                            sub_buffer));
    else
        tmpSubBuffers.push_back(sub_buffer);
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::cleanTmpSubBuffers(const Dtype *bottom, const Dtype *top)
{
    for (auto &buffer : tmpSubBuffers)
        clReleaseMemObject(buffer);
    tmpSubBuffers.clear();
}

template<>
cl_int OCL4DNNConvSpatial<float>::convolve(const float *bottom, const float *top,
                                          int32_t index,
                                          int32_t numImages, kernelConfig* config)
{
    ocl::Context ctx = ocl::Context::getDefault();
    ocl::Program program;
    phash_t::iterator it = phash.find(config->kernelName);
    if (it != phash.end())
        program = it->second;
    else
        return CL_INVALID_PROGRAM;
    ocl::Kernel kernel(config->kernelName.c_str(), program);
    cl_int err = CL_SUCCESS;

    int32_t bias_offset;
    if (config->kernelType == 2) {
        swizzleWeights(bottom, top, config->workItem_output[2], false);
        size_t total_bottom_size = bottom_dim_ * numImages;
        size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
        size_t total_bias_size = M_ * group_;
        size_t total_top_size = top_dim_ * numImages;
        for (int32_t g = 0; g < group_; ++g) {
            bias_offset = M_ * g;
            int32_t image_offset = width_ * height_ * (channels_ / group_) * g;
            int32_t output_image_offset = output_w_ * output_h_ * M_ * g;

            int32_t kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;
            cl_uint argIdx = 0;

            try {
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bottom_data_,
                                   image_offset,
                                   total_bottom_size - image_offset,
                                   true, false);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) swizzled_weights_,
                                   kernel_offset,
                                   total_kernel_size - kernel_offset,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bias_,
                                   bias_offset,
                                   total_bias_size - bias_offset,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) top_data_,
                                   output_image_offset,
                                   total_top_size - output_image_offset,
                                   false, false);
            } catch (int e) {
                err = e;
            }

            if (err == CL_SUCCESS) {
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                ocl::Queue queue = ocl::Queue::getDefault();
                err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                             (cl_kernel)kernel.ptr(), 3,
                                             NULL,
                                             config->global_work_size,
                                             config->local_work_size, 0, NULL,
                                             NULL);
            }
            if (err != CL_SUCCESS)
                break;
        }

        if (group_ > 1) {
            cleanTmpSubBuffers(bottom, top);
        }
        if (err != CL_SUCCESS)
            return err;
    } else if (config->kernelType == 5) {
        swizzleWeights(bottom, top, config->workItem_output[1], true);
        size_t total_bottom_size = bottom_dim_ * numImages;
        size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
        size_t total_bias_size = M_ * group_;
        size_t total_top_size = top_dim_ * numImages;
        for (int32_t g = 0; g < group_; ++g) {
            bias_offset = M_ * g;
            int32_t image_offset = width_ * height_ * (channels_ / group_) * g;
            int32_t output_image_offset = output_w_ * output_h_ * M_ * g;

            cl_uint argIdx = 0;
            int32_t kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;
            try {
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bottom_data_,
                                   image_offset,
                                   total_bottom_size - image_offset,
                                   true, false);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) swizzled_weights_,
                                   kernel_offset,
                                   total_kernel_size - kernel_offset,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bias_,
                                   bias_offset,
                                   total_bias_size - bias_offset,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) top_data_,
                                   output_image_offset,
                                   total_top_size - output_image_offset,
                                   false, false);
            } catch (int e) {
                err = e;
            }

            if (err == CL_SUCCESS) {
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                int out_pitch_y = output_w_ * output_h_;
                int out_pitch_z = out_pitch_y * M_;
                int aligned_input_size = height_ * width_ * channels_ / group_;
                int slice_pitch = width_ * height_;
                kernel.set(argIdx++, (uint32_t)out_pitch_y);
                kernel.set(argIdx++, (uint32_t)out_pitch_z);
                kernel.set(argIdx++, (uint32_t)aligned_input_size);
                kernel.set(argIdx++, (uint32_t)slice_pitch);

                int blockM = config->workItem_output[0];
                int blockK = config->workItem_output[1];
                int blockN = config->workItem_output[2];
                int alignedFilterWidth = ALIGN(M_, blockN);
                int alignedExpandHeight = ALIGN(output_w_ * output_h_, blockM);
                int globalWorkSizeDX = blockN;
                int globalWorkSizeDY = blockM;
                size_t sgemm_m = alignedExpandHeight;
                size_t sgemm_n = alignedFilterWidth;
                size_t gx = (size_t) ceil( (float) sgemm_n /
                        (float) globalWorkSizeDX );
                size_t gy = (size_t) ceil( (float) sgemm_m /
                        (float) globalWorkSizeDY );
                gy = ALIGN(gy, blockK);
                size_t global_size[3] = { gx, gy, config->global_work_size[2] };

                ocl::Queue queue = ocl::Queue::getDefault();
                err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                             (cl_kernel)kernel.ptr(), 3,
                                             NULL,
                                             global_size,
                                             config->local_work_size, 0, NULL,
                                             NULL);
                OCL_CHECK(err);
            }
            if (err != CL_SUCCESS)
                break;
        }

        if (group_ > 1) {
            cleanTmpSubBuffers(bottom, top);
        }
        if (err != CL_SUCCESS)
            return err;
    } else {
        for (int32_t n = 0; n < numImages; ++n) {
            for (int32_t g = 0; g < group_; ++g) {
                bias_offset = M_ * g;
                int32_t image_offset = n * bottom_dim_
                    + width_ * height_ * (channels_ / group_) * g;
                int32_t output_image_offset = n * top_dim_
                    + output_w_ * output_h_ * M_ * g;

                cl_uint argIdx = 0;
                int32_t kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;

                kernel.set(argIdx++, (cl_mem) bottom_data_);
                kernel.set(argIdx++, image_offset);
                kernel.set(argIdx++, (cl_mem) weight_);
                kernel.set(argIdx++, kernel_offset);
                kernel.set(argIdx++, (cl_mem) bias_);
                kernel.set(argIdx++, bias_offset);
                kernel.set(argIdx++, (cl_mem) top_data_);
                kernel.set(argIdx++, output_image_offset);
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                kernel.set(argIdx++, (uint16_t)pad_w_);
                kernel.set(argIdx++, (uint16_t)pad_h_);
                ocl::Queue queue = ocl::Queue::getDefault();
                if (config->use_null_local) {
                    err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                                 (cl_kernel)kernel.ptr(), 3,
                                                 NULL,
                                                 config->global_work_size, NULL, 0, NULL,
                                                 NULL);
                } else {
                    err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                                 (cl_kernel)kernel.ptr(), 3,
                                                 NULL,
                                                 config->global_work_size,
                                                 config->local_work_size, 0, NULL,
                                                 NULL);
                }

                if (err != CL_SUCCESS)
                    return err;
            }
        }
    }

    return err;
}

template<>
float OCL4DNNConvSpatial<float>::timedConvolve(const float *bottom,
                                              const float *top,
                                              int32_t index,
                                              int32_t numImages,
                                              kernelConfig* config)
{
    // warm up.
    bool saved_tuned = tuned_;
    tuned_ = false;
    convolve(bottom, top, index, num_, config);
    cv::ocl::Timer timer;
    timer.start();
    cl_int err;
    dbgPrint(std::cout << "Benchmarking kernel: " << config->kernelName << std::endl);
    tuned_ = true;
    int loop_cnt = 4;
    for (int i = 0; i < loop_cnt; i++) {
        err = convolve(bottom, top, index, num_, config);
        if (err != CL_SUCCESS)
            break;
    }
    tuned_ = saved_tuned;
    timer.stop();
    if (err != CL_SUCCESS) {
        config->tested = true;
        config->verified = false;
        dbgPrint(std::cout << "convolution failed with error code " << err << std::endl);
        return 1e5;
    }

    float elapsedTime = timer.milliSeconds() / loop_cnt;
    #ifdef dbg
    double out_w = output_w_;
    double out_h = output_h_;
    double out_z = M_;
    double k_w = kernel_w_;
    double k_h = kernel_h_;
    double k_z = channels_;
    double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z)*num_;
    std::cout << "\tEstimated Gflops:" << ((totalFlops/1000)/1000)/1000
              << std::endl;
    std::cout << "\tEstimated GFLOPS/S: " << (((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime)
              << std::endl;
    #if 0
    std::cout << "Estimated utilization: " <<
        ((((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime))/880.0
        << std::endl;
    #endif
    #endif
    return elapsedTime;
}

template<>
bool OCL4DNNConvSpatial<float>::verifyResult(const float *bottom, float *top,
                                             int32_t index,
                                             int32_t numImages, const float *verify_blob, kernelConfig* config)
{

    uint32_t verificationFail = 0;

    if (config->verified)
        return true;
    else if (config->tested)
        return false;

    ocl4dnnSet(0, numImages * top_dim_, (float) 0, (cl_mem) top, 0);
    config->executionTime = timedConvolve(bottom, top, index, numImages, config);
    const float *verify_data;
    float *data;
    float *tmp_verify_data;
    ocl::Queue queue = ocl::Queue::getDefault();
    data = reinterpret_cast<float *>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                        (cl_mem)top, true, CL_MAP_READ,
                                                        0, sizeof(float) * numImages * top_dim_,
                                                        0, NULL, NULL, NULL));
    tmp_verify_data =
        reinterpret_cast<float *>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                     (cl_mem)verify_blob, true, CL_MAP_READ,
                                                     0, sizeof(float) * numImages * top_dim_,
                                                     0, NULL, NULL, NULL));
    verify_data = tmp_verify_data;

    for (int32_t n = 0; n < numImages; ++n) {
        for (int32_t g = 0; g < group_; ++g) {
            int32_t output_image_offset = n * top_dim_ + output_w_ * output_h_ * M_ * g;
            for (int out_ch = 0; out_ch < M_ && !verificationFail; out_ch++)
                for (int h = 0; h < output_h_ && !verificationFail; h++)
                    for (int w = 0; w < output_w_; w++) {
                        size_t offset = output_image_offset + out_ch * output_w_ * output_h_ + h * output_w_ + w;
                        if (fabs(data[offset] - verify_data[offset]) > 0.1 * fabs(verify_data[offset]) &&
                            !(fabs(verify_data[offset]) < 1.e-3 &&
                            fabs(data[offset] - verify_data[offset]) < 1.e-4))
                        {
                            dbgPrint(printf("test verification failed @ image %d group %d"
                                            "out_ch %d h %d w %d got %G expected %G\n",
                                            n, g, out_ch, h, w, data[offset], verify_data[offset]));
                            verificationFail = 1;
                            goto out;
                        }
                    }
        }
    }

out:
    clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                            (cl_mem)top, data, 0, NULL, NULL);
    clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                            (cl_mem)verify_blob, tmp_verify_data, 0, NULL, NULL);
    if (verificationFail == 1)
        return false;
    else
        return true;
}

template<typename Dtype>
ocl::Program OCL4DNNConvSpatial<Dtype>::compileKernel()
{
    String errmsg;
    ocl::Context ctx = ocl::Context::getDefault();
    ocl::ProgramSource src(kernel_.c_str());
    ocl::Program program = ctx.getProg(src, options_, errmsg);
    if (!kernel_name_.empty())
        phash.insert(std::pair<std::string, ocl::Program>(kernel_name_, program));
    return program;
}

template<>
std::string OCL4DNNConvSpatial<float>::programEntryToString(ocl::ProgramSource& src)
{
    return src.source();
}

template<>
bool OCL4DNNConvSpatial<float>::createGEMMLikeConvKernel(int32_t blockM,
                                                         int32_t blockK,
                                                         int32_t blockN)
{
    std::stringstream optionsString;
    std::string kernelUKey = generateSpecificKey(5,
                                                 blockM,
                                                 blockK,
                                                 blockN);
    int workItemOutput[3] = { blockM, blockK, blockN };
    int simd_size = blockK;
    int num_batches = num_;
    int output_width = output_w_;
    int output_height = output_h_;
    int alignedFilterWidth = ALIGN(M_, blockN);
    int alignedExpandHeight = ALIGN(output_width * output_height, blockM);
    int globalWorkSizeDX = blockN;
    int globalWorkSizeDY = blockM;
    size_t sgemm_m = alignedExpandHeight;
    size_t sgemm_n = alignedFilterWidth;
    size_t gx = (size_t) ceil( (float) sgemm_n / (float) globalWorkSizeDX );
    size_t gy = (size_t) ceil( (float) sgemm_m / (float) globalWorkSizeDY );
    gy = ALIGN(gy, blockK);
    size_t gz = num_batches;
    size_t global_size[3] = { gx, gy, gz };
    size_t local_size[3] = { 1, static_cast<size_t>(simd_size), 1 };

    kernel_name_ = "U_GEMM_LIKE_CONV_";
    kernel_name_ += kernelUKey.c_str();
    if (blockK == 8)
        kernel_name_ += "_SIMD8";
    else
        kernel_name_ += "_SIMD16";
    std::stringstream kernelDef;
    kernelDef << "GEMM_LIKE_CONV_" << blockN << "_" << blockM;
    if (blockK == 16)
        kernelDef << "_SIMD16";

    // Build list of options and defines
    optionsString.str("");
    optionsString << "-cl-fast-relaxed-math " << " -D " << kernelDef.str()
        << " -D Conv_Interleaved=" << kernel_name_.c_str();
    optionsString <<
        " -cl-mad-enable" <<
        " -DKERNEL_WIDTH=" << kernel_w_ <<
        " -DKERNEL_HEIGHT=" << kernel_h_ <<
        " -DSTRIDE_X=" << stride_w_ <<
        " -DSTRIDE_Y=" << stride_h_ <<
        " -DDILATION_X=" << dilation_w_ <<
        " -DDILATION_Y=" << dilation_h_ <<
        " -DINPUT_DEPTH=" << channels_ <<
        " -DWIDTH1=" << M_ <<
        " -DOUT_PADDING_LEFT=" << 0 <<
        " -DOUT_PADDING_HEIGHT=" << 0 <<
        " -DOUT_DEPTH=" << M_ <<
        " -DNUM_BATCHES=" << num_ <<
        " -DDY=" << globalWorkSizeDY <<
        " -DDX=" << globalWorkSizeDX <<
        " -DKERNEL_WIDTH_DIV2=" << kernel_w_ / 2 <<
        " -DKERNEL_SLICE_DIV2=" << (kernel_w_ * kernel_h_) / 2 <<
        " -DTILE_N_LAST=" << M_ % 32 <<
        " -DTILE_N_LAST_DIV8=" << (M_ % 32) / 8 <<
        " -DINPUT_PAD_W=" << pad_w_ << " -DINPUT_PAD_H=" << pad_h_;
    options_ = optionsString.str();
    kernel_ = programEntryToString(ocl::dnn::conv_layer_spatial_oclsrc);
    ocl::Program program = compileKernel();

    size_t workgroupSize_used;
    ocl::Kernel kernel(kernel_name_.c_str(), program);
    cl_int err = clGetKernelWorkGroupInfo((cl_kernel)kernel.ptr(),
                                          (cl_device_id)ocl::Device::getDefault().ptr(),
                                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                          sizeof(size_t), &workgroupSize_used,
                                          NULL);

    if (workgroupSize_used != simd_size) {
        phash.erase(kernel_name_);
        return false;
    }

    if (err == CL_SUCCESS) {
        kernelQueue.push_back(new kernelConfig(kernel_name_,
                                               global_size,
                                               local_size,
                                               workItemOutput,
                                               false,
                                               true,
                                               false,
                                               5));
        return true;
    } else {
        phash.erase(kernel_name_);
        return false;
    }
}

template<>
bool OCL4DNNConvSpatial<float>::setupIDLF(int32_t blockWidth,
                                         int32_t blockHeight,
                                         int32_t simd_size)
{
    int32_t workItemOutput[3] = { blockWidth, blockHeight, simd_size };
    const int32_t num_output_maps = M_;
    int32_t output_width = output_w_;
    int32_t output_height = output_h_;
    int32_t output_block_width = blockWidth;
    int32_t output_block_height = blockHeight;
    int32_t num_batches = num_;

    size_t global_size[3] = {
        (size_t) (output_width + output_block_width - 1) / output_block_width,
        (size_t) (output_height + output_block_height - 1) / output_block_height,
        (size_t) num_batches * ALIGN(num_output_maps, simd_size) };
    size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };

    kernelType_ = KERNEL_TYPE_INTEL_IDLF;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = simd_size;

    generateKernelSrc();
    ocl::Program program = compileKernel();

    // ClKernel kernel;
    size_t workgroupSize_used;
    ocl::Kernel kernel(kernel_name_.c_str(), program);
    cl_int err = clGetKernelWorkGroupInfo((cl_kernel)kernel.ptr(),
                                          (cl_device_id)ocl::Device::getDefault().ptr(),
                                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                          sizeof(size_t), &workgroupSize_used,
                                          NULL);

    if (workgroupSize_used != simd_size) {
        phash.erase(kernel_name_);
        return false;
    }

    if (err == CL_SUCCESS) {
        kernelQueue.push_back(new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                                               false, true, false, 2));
        return true;
    } else {
        phash.erase(kernel_name_);
        return false;
    }
}

template<>
bool OCL4DNNConvSpatial<float>::tuneLocalSize(const float *bottom,
                                             const float *top,
                                             kernelConfig* config)
{
    if (config->use_null_local || !config->autoTune)
        return true;

    float fastestTime = 999999990000000000000000000.0f;
    uint32_t multiplier = 4;
    uint32_t localSize[3] = { 1, 1, 1 };

    int32_t skip = 0;
    cv::ocl::Timer timer;
    bool allFailed = true;
    for (int32_t z = 0; z <= 16; z++) {
        for (int32_t y = 0; y <= 16; y++) {
            for (int32_t x = 1; x <= 16; x++) {
                timer.start();
                skip = 0;

                if (config->autoTune) {
                    config->local_work_size[0] =
                        (multiplier * x == 0) ? 1 : multiplier * x;
                    config->local_work_size[1] =
                        (multiplier * y == 0) ? 1 : multiplier * y;
                    config->local_work_size[2] =
                        (multiplier * z == 0) ? 1 : multiplier * z;

                    computeGlobalSize(1, config->workItem_output,
                                      config->local_work_size,
                                      config->global_work_size);
                }
                if (config->workItem_output[2] * config->global_work_size[2] != M_)
                    break;

                if (config->swizzle_weights)
                    z = 32;

                int32_t err = 0;
                err = convolve(bottom, top, 0, 1, config);

                if (err != CL_SUCCESS)
                    skip = 1;

                if (skip) {
                    timer.stop();
                    break;
                }
                timer.stop();
                allFailed = false;
                float elapsedTime = timer.milliSeconds();

                if (elapsedTime < fastestTime) {
                    fastestTime = elapsedTime;
                    localSize[0] = config->local_work_size[0];
                    localSize[1] = config->local_work_size[1];
                    localSize[2] = config->local_work_size[2];
                }
            }
        }
    }
    if (allFailed) {
        // 1,1,1 is never a good local size and no need to test at all.
        dbgPrint(std::cout << "Can't find good local size for " << config->kernelName << std::endl);
        return false;
    }

    dbgPrint(std::cout << "Best local size[" << localSize[0] << "]["
             << localSize[1] << "]["<< localSize[2] << "]: " << fastestTime
             << " Kernel_h: " << kernel_h_ << " kernel_w_: " << kernel_w_
             << " stride_w: " << stride_w_ << " pad_w_: " << pad_w_ << std::endl);

    if (config->autoTune) {
        for (int32_t li = 0; li < 3; li++)
            config->local_work_size[li] = localSize[li];

        computeGlobalSize(1, config->workItem_output,
                          config->local_work_size,
                          config->global_work_size);
    }
    return true;
}

template<>
void OCL4DNNConvSpatial<float>::createConvolutionKernel(int32_t kernelType,
                                                       int32_t blockWidth,
                                                       int32_t blockHeight,
                                                       int32_t blockDepth)
{
    if (kernelType == 2)
        setupIDLF(blockWidth, blockHeight, blockDepth);
    else if (kernelType == 4)
        createBasicKernel(blockWidth, blockHeight, blockDepth);
    else if (kernelType == 5)
        createGEMMLikeConvKernel(blockWidth, blockHeight, blockDepth);
    else
        assert(0);
}

template<>
void OCL4DNNConvSpatial<float>::setupConvolution(const float *bottom, float *top,
                                                 const float *verify_blob)
{
    if (auto_tuning_ && ocl::Device::getDefault().intelSubgroupsSupport()) {
        /* IDLF kernels are using Intel specific extension which make
           them intel only. */
        // Generates static key_
        int max_compute_units = ocl::Device::getDefault().maxComputeUnits();
        int kernelCnt = 0;
        if (this->group_ == 1 && ((M_ % 8 == 0) && (M_ % 32 != 24))) {
            createConvolutionKernel(5, 1, 8, 32);
            createConvolutionKernel(5, 2, 8, 32);
            if (kernel_w_ < 4 && M_ % 32 == 0)
                createConvolutionKernel(5, 1, 16, 32);
        }

        for (int simd_size = 8; simd_size <= 16; simd_size += 8) {
            if (simd_size == 8 && !((this->group_ == 1 || M_ % 8 == 0)))
                continue;
            if (simd_size == 16 && !(this->group_ == 1 || M_ % 16 == 0))
                continue;
            int width_max, height_max, block_size_max;
            if (simd_size == 8) {
                width_max = 16;
                height_max = 16;
                block_size_max = 64;
            } else {
                width_max = 14;
                height_max = 14;
                block_size_max = 32;
            }
            for (uint32_t width = width_max; width > 0; width--) {
                int candidate = 0;
                if (width > output_w_)
                    continue;
                for (uint32_t height = height_max; height > 0; height--) {
                    if (width * height > block_size_max || height > output_h_)
                        continue;
                    // Only when the work items count is less than the device
                    // max work items or the M_ is less than 16, we will tune
                    // for simd 8.
                    if (simd_size == 8 &&
                        M_ >= 16 &&
                        ((num_ * M_ * output_w_ * output_h_ / static_cast<float>(width * height)) >=
                        max_compute_units * 7 * 16))
                        continue;
                    int tile_x = (kernel_w_ * dilation_w_ + (width - 1) * stride_w_ + 3) & ~3;
                    int tile_y = kernel_h_ * dilation_h_ + (height - 1) * stride_h_;
                    if (tile_x > (4 * simd_size))
                        continue;
                    int tile_y_stride = (4 * simd_size) / tile_x;

                    if ((tile_y + tile_y_stride - 1) / tile_y_stride < 4) {
                        createConvolutionKernel(2, width, height, simd_size);
                        candidate++;
                    }
                    if (candidate >= 4 && height == 2)
                        break;
                }
                kernelCnt += candidate;
                if (kernelCnt >= 12 && width == 2)
                    break;
            }
        }
    }
    for (int32_t x = 0; x < kernelQueue.size(); x++) {
        if (tuneLocalSize(bottom, top, kernelQueue[x])) {
            kernelQueue[x]->executionTime = timedConvolve(bottom, top, bottom_index_,
                                                           num_, kernelQueue[x]);
        } else {
            // skip those kernels without a good local size.
            kernelQueue[x]->verified = false;
            kernelQueue[x]->tested = true;
        }
        #ifdef TEST_ALL_KERNELS
        if (kernelQueue[x]->tested == false) {
            bool verified = verifyResult(bottom, top, bottom_index_, num_,
                                          verify_blob, kernelQueue[x]);
            if (verified == false) {
                dbgPrint(std::cout << "Kernel "
                         << kernelQueue[x]->kernelName
                         << " failed verification" << std::endl);
                dbgPrint(std::cout << "kernelQueue[x]->workItem_output[0]: "
                         << kernelQueue[x]->workItem_output[0] << " "
                         << "kernelQueue[x]->workItem_output[1]: "
                         << kernelQueue[x]->workItem_output[1] << " "
                         << "kernelQueue[x]->workItem_output[2]: "
                         << kernelQueue[x]->workItem_output[2] << " "
                         << "kernelQueue[x]->kernelType: "
                         << kernelQueue[x]->kernelType << " "
                         << "kernelQueue[x]->global_work_size[0]: "
                         << kernelQueue[x]->global_work_size[0] << " "
                         << "kernelQueue[x]->global_work_size[1]: "
                         << kernelQueue[x]->global_work_size[1] << " "
                         << "kernelQueue[x]->global_work_size[2]: "
                         << kernelQueue[x]->global_work_size[2] << " "
                         << "kernelQueue[x]->local_work_size[0]: "
                         << kernelQueue[x]->local_work_size[0] << " "
                         << "kernelQueue[x]->local_work_size[1]: "
                         << kernelQueue[x]->local_work_size[1] << " "
                         << "kernelQueue[x]->local_work_size[2]: "
                         << kernelQueue[x]->local_work_size[2] << " "
                         << kernelQueue[x]->swizzle_weights << " "
                         << kernelQueue[x]->use_null_local << std::endl);
            } else {
                dbgPrint(std::cout << "Kernel "
                         << kernelQueue[x]->kernelName
                         << " pass verification" << std::endl);
            }
        }
        #endif
    }
    int32_t failures = 0;
    bool verification = false;
    if (kernelQueue.size()) {
        while (failures < kernelQueue.size()) {
            int32_t fastestKernel = -1;
            float fastestTime = 999999990000000000000000000.0f;

            for (int32_t x = 0; x < kernelQueue.size(); x++) {
                if (kernelQueue[x]->executionTime < fastestTime &&
                    kernelQueue[x]->tested == false) {
                    fastestKernel = x;
                    fastestTime = kernelQueue[x]->executionTime;
                }
            }
            if (fastestKernel < 0) break;
            // Test fastest kernel
            bool verified = verifyResult(bottom, top, bottom_index_, num_,
                                          verify_blob, kernelQueue[fastestKernel]);
            if (verified == true) {
                kernelQueue[fastestKernel]->verified = true;
                kernel_index_ = fastestKernel;
                verification = true;
                break;
            } else {
                kernelQueue[fastestKernel]->tested = true;
                dbgPrint(std::cout << "Kernel " <<
                         kernelQueue[fastestKernel]->kernelName <<
                         " failed verification" << std::endl);
                failures++;
            }
        }
    }
    if (verification) {
        dbgPrint(std::cout << "Kernel <" << kernelQueue[kernel_index_]->kernelName <<
                 "> passed verification" << std::endl);
    } else {
        if (auto_tuning_)
            dbgPrint(std::cout << "Verification was not successful, " <<
                     "fallback to basic kernel" << std::endl);
        else
            dbgPrint(std::cout << "Auto-tuning disabled, fallback to basic kernel" << std::endl);
        createBasicKernel(1, 1, 1);
        kernel_index_ = kernelQueue.size() - 1;
        verification = verifyResult(bottom, top, bottom_index_, num_,
                                     verify_blob, kernelQueue[kernel_index_]);
        CHECK_EQ(verification, true) << "Basic kernel failed verification." << std::endl;
    }
    this->bestKernelConfig = kernelQueue[kernel_index_];

    dbgPrint(std::cout << "Convolution Time:" << kernelQueue[kernel_index_]->executionTime << std::endl);

    if (bestKernelConfig->kernelType != 2 && bestKernelConfig->kernelType != 5)
        swizzled_weights_ = NULL;

    for (int32_t x = 0; x < kernelQueue.size(); x++) {
        if (x != kernel_index_) {
            phash.erase(kernelQueue[x]->kernelName);
            delete kernelQueue[x];
        }
    }
    kernelQueue.clear();

    tuned_ = true;

    if (auto_tuning_) {
        std::string outputFile;
        outputFile = cache_path_.str() + key_;
        std::ifstream cachedKernel(outputFile.c_str());
        std::ofstream outputKernel;
        outputKernel.open(outputFile.c_str());
        outputKernel << bestKernelConfig->workItem_output[0] << " "
                     << bestKernelConfig->workItem_output[1] << " "
                     << bestKernelConfig->workItem_output[2] << " "
                     << bestKernelConfig->kernelType << " "
                     << bestKernelConfig->local_work_size[0] << " "
                     << bestKernelConfig->local_work_size[1] << " "
                     << bestKernelConfig->local_work_size[2] << " "
                     << bestKernelConfig->swizzle_weights << " "
                     << bestKernelConfig->use_null_local << " ";
        outputKernel.close();
    }
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::prepareKernel()
{
    std::string previous_key = key_;

    generateKey();
    if (key_.compare(previous_key) == 0 && bestKernelConfig != NULL)
        return;

    if (bestKernelConfig)
    {
        prev_kernel_type_ = bestKernelConfig->kernelType;
        phash.erase(bestKernelConfig->kernelName);
        delete bestKernelConfig;
        bestKernelConfig = NULL;
    }

    if (loadCachedConfig())
        return;

    tune(top_data_, weight_, bias_, bottom_data_, num_);
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::loadCachedConfig()
{
    // Find cached kernel configuration
    std::string outputFile;
    outputFile = cache_path_.str() + key_;
    std::ifstream cachedKernel(outputFile.c_str());
    if (cachedKernel)
    {
        int32_t x, y, z, type;
        cachedKernel >> x;
        cachedKernel >> y;
        cachedKernel >> z;
        cachedKernel >> type;
        if (type == 2)
        {
            if (z == 1)
                z = 16;
            CHECK_EQ(z == 16 || z == 8, true) << "invalid SIMD size" << std::endl;
        }
        createConvolutionKernel(type, x, y, z);
        kernel_index_ = kernelQueue.size() - 1;
        if (kernel_index_ == -1) {
            std::cerr << "Failed to get kernel from cached configurations."
                      << std::endl;
            std::cerr << "Deleting broken cache file and try tuning again..."
                      << std::endl;
            std::string bakFile = outputFile + ".bak";
            std::rename(outputFile.c_str(), bakFile.c_str());
            return false;
        }
        bestKernelConfig = kernelQueue[kernel_index_];
        kernelQueue.clear();
        cachedKernel >> bestKernelConfig->local_work_size[0];
        cachedKernel >> bestKernelConfig->local_work_size[1];
        cachedKernel >> bestKernelConfig->local_work_size[2];
        cachedKernel >> bestKernelConfig->swizzle_weights;
        cachedKernel >> bestKernelConfig->use_null_local;
        // If kernel type changed to type 2 or 4, we need to reset the swizzled
        // weights pointer to invalidate the previous swizzled weights data.
        if (prev_kernel_type_ != bestKernelConfig->kernelType &&
            (bestKernelConfig->kernelType == 2 || bestKernelConfig->kernelType == 5))
        {
            if (swizzled_weights_)
                clReleaseMemObject((cl_mem)swizzled_weights_);
            swizzled_weights_ = NULL;
        }
        tuned_ = true;
        return true;
    }
    else
        return false;
}

template class OCL4DNNConvSpatial<float>;
} // namespace ocl4dnn
}
}
#endif // HAVE_OPENCL
