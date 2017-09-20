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
#include "default_kernel_config.hpp"

#if defined WIN32 || defined _WIN32
#include <windows.h>
#include <direct.h>
#endif

#ifdef HAVE_OPENCL
namespace cv { namespace dnn { namespace ocl4dnn {
static cv::Mutex kernelConfigMutex;
typedef std::map<std::string, std::string> kernel_hash_t;
static kernel_hash_t kernelConfigMap;
static bool defaultConfigLoaded = false;

template<typename Dtype>
OCL4DNNConvSpatial<Dtype>::OCL4DNNConvSpatial(OCL4DNNConvConfig config)
{
    bias_term_ = config.bias_term;
    int dims = config.in_shape.size();
    int spatial_dims = 2;

    channels_   = config.in_shape[dims - spatial_dims - 1];
    num_output_ = config.out_shape[dims - spatial_dims - 1];
    group_ = config.group;

    std::vector<int32_t> pad_;
    std::vector<int32_t> stride_;
    std::vector<int32_t> dilation_;
    std::vector<int32_t> kernel_shape_;
    std::vector<int32_t> im_in_shape_;
    std::vector<int32_t> im_out_shape_;

    for (int i = 0; i < spatial_dims; ++i) {
        kernel_shape_.push_back(i == 0 ? config.kernel.height : config.kernel.width);
        pad_.push_back(i == 0 ? config.pad.height : config.pad.width);
        stride_.push_back(i == 0 ? config.stride.height : config.stride.width);
        dilation_.push_back(i == 0 ? config.dilation.height : config.dilation.width);
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    prev_kernel_type_ = -1;
    bias_ = NULL;
    tuned_ = false;
    kernel_dim_ = channels_ / group_;
    out_spatial_dim_ = 1;

    int in_spatial_dim_ = 1;
    for (int i = 0; i < spatial_dims; ++i) {
        kernel_dim_ *= i == 0 ? config.kernel.height : config.kernel.width;
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
    auto_tuning_ = hasCacheDir && getenv("OPENCV_OCL4DNN_ENABLE_AUTO_TUNING");
}

template<typename Dtype>
OCL4DNNConvSpatial<Dtype>::~OCL4DNNConvSpatial()
{
    if (!swizzled_weights_umat.empty()) {
        swizzled_weights_umat.release();
    }
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::collectCommonInformation()
{
    if (cv::traits::Depth<Dtype>::value == CV_64F)
    {
        addDef("DOUBLE_SUPPORT");
        addDef("Dtype", "double");
        addDef("Dtype_ID", CV_64F);
    }
    else
    {
        addDef("Dtype", "float");
        addDef("Dtype_ID", CV_32F);
    }

    addDef(sizeof(int32_t) == 8 ? "SYSTEM_INT_64BIT" : "SYSTEM_INT_32BIT");
}

typedef enum {
    KERNEL_TYPE_INTEL_IDLF = 2,
    KERNEL_TYPE_BASIC = 4,
    KERNEL_TYPE_GEMM_LIKE = 5
} ocl4dnnConvSpatialKernelType_t;

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setupKernelDetails(int32_t kernelType,
                                                   int32_t blockM,
                                                   int32_t blockK,
                                                   int32_t blockN)
{
    std::string kernelUKey;
    int32_t simd_size;

    if (kernelType == KERNEL_TYPE_INTEL_IDLF) {
        simd_size = blockN;
        kernelUKey = generateSpecificKey(KERNEL_TYPE_INTEL_IDLF, blockM, blockK, 1);

        // kernel name
        kernel_name_ = "IDLF_";
        kernel_name_ += kernelUKey;
        if (simd_size == 16)
            kernel_name_ += "_SIMD16";
        else
            kernel_name_ += "_SIMD8";

        // options
        options_ << " -cl-fast-relaxed-math -D KERNEL_IDLF -D convolve_simd=" << kernel_name_;
        if (build_option_check())
            options_ << " -cl-no-subgroup-ifp ";

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

        addDef("SIMD_SIZE", simd_size);
        addDef("filter_qualifier", "__global");
        addDef("OUT_BLOCK_WIDTH", output_block_width);
        addDef("OUT_BLOCK_HEIGHT", output_block_height);
        addDef("LAST_BLOCK_WIDTH", last_block_width);
        addDef("LAST_BLOCK_HEIGHT", last_block_height);
        addDef("INPUT_DEPTH", channels_ / group_);
        addDef("TOTAL_INPUT_DEPTH_SIZE", channels_);
        addDef("TOTAL_OUTPUT_DEPTH", num_output_);
        addDef("INPUT_START_X", 0);
        addDef("INPUT_START_Y", 0);
        addDef("INPUT_START_Z", 0);
        addDef("NUM_FILTERS", M_);
        addDef("OUT_BUFF_OFFSET", 0);
        addDef("TILE_X", tile_x);
        addDef("TILE_Y", tile_y);
        addDef("TILE_Y_STRIDE", tile_y_stride);
        addDef("INVEC_SIZE", invec_size);
        addDef("ALIGNED_NUM_FILTERS", (int)alignSize(M_, simd_size));
        addDef("OUT_BLOCK_SIZE", (output_block_width*output_block_height));

        src_ = cv::ocl::dnn::conv_layer_spatial_oclsrc;
    }
    else if (kernelType == KERNEL_TYPE_BASIC)
    {
        addDef("KERNEL_BASIC");

        kernelUKey = generateSpecificKey(KERNEL_TYPE_BASIC, blockM, blockK, blockN);
        kernel_name_ = "BASIC_";
        kernel_name_ += kernelUKey;

        // opts
        options_ << " -cl-fast-relaxed-math -D ConvolveBasic=" << kernel_name_;
        if (build_option_check())
            options_ << " -cl-no-subgroup-ifp ";

        // defs
        addDef("CHANNELS", channels_ / group_);
        addDef("APPLY_BIAS", bias_term_);
        addDef("OUTPUT_Z", M_);
        addDef("ZPAR", 1);

        src_ = cv::ocl::dnn::conv_layer_spatial_oclsrc;
    }
    else if (kernelType == KERNEL_TYPE_GEMM_LIKE)
    {
        simd_size = blockK;
        kernelUKey = generateSpecificKey(KERNEL_TYPE_GEMM_LIKE, blockM, blockK, blockN);

        kernel_name_ = "U_GEMM_LIKE_CONV_";
        kernel_name_ += kernelUKey.c_str();
        kernel_name_ += (blockK == 8) ? "_SIMD8" : "_SIMD16";
        std::stringstream kernelDef;
        kernelDef << "GEMM_LIKE_CONV_" << blockN << "_" << blockM;
        if (blockK == 16)
            kernelDef << "_SIMD16";

        // Build list of options and defines
        options_ << " -cl-fast-relaxed-math " << " -D " << kernelDef.str()
            << " -D Conv_Interleaved=" << kernel_name_.c_str();
        options_ << " -cl-mad-enable";

        addDef("INPUT_DEPTH", channels_);
        addDef("WIDTH1", M_);
        addDef("OUT_PADDING_LEFT", 0);
        addDef("OUT_PADDING_HEIGHT", 0);
        addDef("OUT_DEPTH", M_);
        addDef("NUM_BATCHES", num_);
        addDef("DY", blockM);
        addDef("DX", blockN);
        addDef("KERNEL_WIDTH_DIV2", kernel_w_ / 2);
        addDef("KERNEL_SLICE_DIV2", (kernel_w_ * kernel_h_) / 2);
        addDef("TILE_N_LAST", M_ % 32);
        addDef("TILE_N_LAST_DIV8", (M_ % 32) / 8);
        src_ = ocl::dnn::conv_layer_spatial_oclsrc;
    }
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setupKernel()
{
    collectCommonInformation();

    addDef("KERNEL_WIDTH", kernel_w_);
    addDef("KERNEL_HEIGHT" , kernel_h_);
    addDef("STRIDE_X", stride_w_);
    addDef("STRIDE_Y", stride_h_);
    addDef("DILATION_X", dilation_w_);
    addDef("DILATION_Y", dilation_h_);
    if (kernelType_ != KERNEL_TYPE_BASIC)
    {
        addDef("INPUT_PAD_W", pad_w_);
        addDef("INPUT_PAD_H", pad_h_);
    }

    setupKernelDetails(kernelType_, blockM_, blockK_, blockN_);
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::Forward(const UMat& bottom,
                                        const UMat& weight,
                                        const UMat& bias,
                                        UMat& top,
                                        int32_t numImages)
{
    int32_t total_bottom_size = bottom.total();
    int32_t total_kernel_size = weight.total();
    int32_t total_bias_size   = bias.total();
    int32_t total_top_size    = top.total();

    UMat bottom_reshaped = bottom.reshape(1, 1, &total_bottom_size);
    UMat top_reshaped    = top.reshape(1, 1, &total_top_size);
    UMat weight_reshaped = weight.reshape(1, 1, &total_kernel_size);
    UMat bias_reshaped;
    if (bias_term_)
        bias_reshaped = bias.reshape(1, 1, &total_bias_size);
    num_ = numImages;

    prepareKernel(bottom_reshaped, top_reshaped, weight_reshaped, bias_reshaped, numImages);
    return convolve(bottom_reshaped, top_reshaped, weight_reshaped, bias_reshaped, numImages, bestKernelConfig);
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::calculateBenchmark(UMat &bottom, UMat &verifyTop,
                                                   UMat &weight, UMat &bias,
                                                   int32_t numImages)
{
    options_.str(""); options_.clear(); // clear contents and state flags
    createBasicKernel(1, 1, 1);
    kernel_index_ = kernelQueue.size() - 1;
    convolve(bottom, verifyTop, weight, bias, numImages, kernelQueue[kernel_index_]);
    CV_Assert(phash.find(kernelQueue[kernel_index_]->kernelName) != phash.end());
    //unloadProgram(kernelQueue[kernel_index_]->kernelName);
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

#define TUNING_SIZE(x) ((x) > 256 ? 256 : (alignSize(x, 16)))

// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
static
uint64 crc64(const uchar* data, size_t size, uint64 crc0 = 0)
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
                         cv::format("%d", ocl::Device::getDefault().maxComputeUnits());

    prefix = prefix + keyBuilder.str();
    key_ = cv::format("%08llx", crc64((uchar*)prefix.c_str(), prefix.size()));
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
bool OCL4DNNConvSpatial<Dtype>::swizzleWeight(UMat &weight,
                                              int32_t swizzled_factor,
                                              bool interleave)
{
    // Simply skip the weight swizzle if we already got a swizzled_weights_
    // in test phase and not in auto tuning
    // This requires we always call convolve again with the winner configuration
    // during the auto tuning stage.
    if (tuned_ && !swizzled_weights_umat.empty())
        return true;

    ocl::Context ocl_ctx = ocl::Context::getDefault();
    if (swizzled_weights_umat.empty())
        swizzled_weights_umat.create(1, ((num_output_ + 15) & ~15) * channels_ *
                                     kernel_h_ * ((kernel_w_ + 1) & ~1), CV_32FC1);

    ocl::Queue queue = ocl::Queue::getDefault();
    if (!interleave) {
        cl_uint argIdx = 0;
        int32_t channels = channels_ / group_;

        ocl::Kernel oclk_copy_weight(CL_KERNEL_SELECT("copyWeightsSwizzled"),
                                     cv::ocl::dnn::conv_spatial_helper_oclsrc);
        if (oclk_copy_weight.empty())
            return false;

        oclk_copy_weight.set(argIdx++, ocl::KernelArg::PtrReadOnly(weight));
        oclk_copy_weight.set(argIdx++, ocl::KernelArg::PtrWriteOnly(swizzled_weights_umat));
        oclk_copy_weight.set(argIdx++, kernel_w_);
        oclk_copy_weight.set(argIdx++, kernel_h_);
        oclk_copy_weight.set(argIdx++, channels);
        oclk_copy_weight.set(argIdx++, num_output_);
        oclk_copy_weight.set(argIdx++, swizzled_factor);

        size_t global_work_size_copy[3] = {
            (size_t) (alignSize(num_output_, swizzled_factor) * channels * kernel_w_ * kernel_h_), 1, 1 };

        if (!oclk_copy_weight.run(3, global_work_size_copy, NULL, false))
        {
            std::cout << "Swizzle kernel run failed." << std::endl;
            return false;
        }
    } else {
        // assumption: kernel dimesion is 2
        Mat weightMat = weight.getMat(ACCESS_READ);
        Dtype* cpu_weight = (Dtype *)weightMat.ptr<float>();
        Mat swizzledWeightMat = swizzled_weights_umat.getMat(ACCESS_WRITE);
        Dtype* cpu_swizzled_weight = (Dtype *)swizzledWeightMat.ptr<float>();

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
        free(tmpSwizzledWeight);
    }
    return true;
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
    int32_t workItemOutput[3] = {1, 1, 1};

    kernelType_ = KERNEL_TYPE_BASIC;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = blockDepth;
    setupKernel();

    ocl::Program program = compileKernel();
    if (program.ptr())
    {
        size_t localSize[3] = { 1, 1, 1 };
        size_t globalSize[3];
        computeGlobalSize(1, workItemOutput, localSize, globalSize);
        kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &globalSize[0], &localSize[0], &workItemOutput[0],
                                                    false, false, true, KERNEL_TYPE_BASIC));
        return true;
    }
    else
        return false;
}

template<>
bool OCL4DNNConvSpatial<float>::convolve(UMat &bottom, UMat &top,
                                         UMat &weight, UMat &bias,
                                         int32_t numImages, kernelConfig* config)
{
    ocl::Queue queue = ocl::Queue::getDefault();
    ocl::Program program;
    phash_t::iterator it = phash.find(config->kernelName);
    if (it != phash.end())
        program = it->second;
    else
        return false;

    int32_t bias_offset;

    if (config->kernelType == KERNEL_TYPE_INTEL_IDLF) {
        if (!swizzleWeight(weight, config->workItem_output[2], false))
            return false;
        size_t total_bottom_size = bottom_dim_ * numImages;
        size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
        size_t total_bias_size = M_ * group_;
        size_t total_top_size = top_dim_ * numImages;
        for (int32_t g = 0; g < group_; ++g) {
            bias_offset = M_ * g;
            int32_t image_offset = width_ * height_ * (channels_ / group_) * g;
            int32_t output_image_offset = output_w_ * output_h_ * M_ * g;
            int32_t kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;

            ocl::Kernel kernel(config->kernelName.c_str(), program);
            if (kernel.empty())
                return false;

            cl_uint argIdx = 0;
            cl_buffer_region region;
            cl_int error;

            cl_mem img_buffer = NULL;
            if (image_offset)
            {
                region.origin = image_offset * sizeof(float);
                region.size = (total_bottom_size - image_offset) * sizeof(float);
                img_buffer = clCreateSubBuffer((cl_mem)bottom.handle(ACCESS_READ), CL_MEM_READ_ONLY,
                                               CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, img_buffer);
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
            }

            cl_mem kernel_buffer = NULL;
            if (kernel_offset)
            {
                region.origin = kernel_offset * sizeof(float);
                region.size = (total_kernel_size - kernel_offset) * sizeof(float);
                kernel_buffer = clCreateSubBuffer((cl_mem)swizzled_weights_umat.handle(ACCESS_READ),
                                                  CL_MEM_READ_ONLY,
                                                  CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, kernel_buffer);
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(swizzled_weights_umat));
            }

            cl_mem bias_buffer = NULL;
            if (bias_offset)
            {
                region.origin = bias_offset * sizeof(float);
                region.size = (total_bias_size - bias_offset) * sizeof(float);
                bias_buffer = clCreateSubBuffer((cl_mem)bias.handle(ACCESS_READ), CL_MEM_READ_ONLY,
                                                CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, bias_buffer);
            }
            else
            {
                if (bias_term_)
                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias));
                else
                    kernel.set(argIdx++, (void *)NULL);
            }

            cl_mem out_buffer = NULL;
            if (output_image_offset)
            {
                region.origin = output_image_offset * sizeof(float);
                region.size = (total_top_size - output_image_offset) * sizeof(float);
                out_buffer = clCreateSubBuffer((cl_mem)top.handle(ACCESS_READ), CL_MEM_WRITE_ONLY,
                                               CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, out_buffer);
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
            }

            kernel.set(argIdx++, (uint16_t)width_);
            kernel.set(argIdx++, (uint16_t)height_);
            kernel.set(argIdx++, (uint16_t)output_w_);
            kernel.set(argIdx++, (uint16_t)output_h_);
            if (!kernel.run(3, config->global_work_size, config->local_work_size, false))
            {
                std::cout << "IDLF kernel run failed." << std::endl;
                return false;
            }
            clReleaseMemObject(img_buffer);
            clReleaseMemObject(kernel_buffer);
            clReleaseMemObject(bias_buffer);
            clReleaseMemObject(out_buffer);
        }
    } else if (config->kernelType == KERNEL_TYPE_GEMM_LIKE) {
        if (!swizzleWeight(weight, config->workItem_output[1], true))
            return false;
        size_t total_bottom_size = bottom_dim_ * numImages;
        size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
        size_t total_bias_size = M_ * group_;
        size_t total_top_size = top_dim_ * numImages;
        for (int32_t g = 0; g < group_; ++g) {
            bias_offset = M_ * g;
            int32_t image_offset = width_ * height_ * (channels_ / group_) * g;
            int32_t output_image_offset = output_w_ * output_h_ * M_ * g;
            int32_t kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;

            ocl::Kernel kernel(config->kernelName.c_str(), program);
            if (kernel.empty())
                return false;

            cl_uint argIdx = 0;
            cl_buffer_region region;
            cl_int error;

            cl_mem img_buffer = NULL;
            if (image_offset)
            {
                region.origin = image_offset * sizeof(float);
                region.size = (total_bottom_size - image_offset) * sizeof(float);
                img_buffer = clCreateSubBuffer((cl_mem)bottom.handle(ACCESS_READ), CL_MEM_READ_ONLY,
                                               CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, img_buffer);
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
            }

            cl_mem kernel_buffer = NULL;
            if (kernel_offset)
            {
                region.origin = kernel_offset * sizeof(float);
                region.size = (total_kernel_size - kernel_offset) * sizeof(float);
                kernel_buffer = clCreateSubBuffer((cl_mem)swizzled_weights_umat.handle(ACCESS_READ),
                                                  CL_MEM_READ_ONLY,
                                                  CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, kernel_buffer);
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(swizzled_weights_umat));
            }

            cl_mem bias_buffer = NULL;
            if (bias_offset)
            {
                region.origin = bias_offset * sizeof(float);
                region.size = (total_bias_size - bias_offset) * sizeof(float);
                bias_buffer = clCreateSubBuffer((cl_mem)bias.handle(ACCESS_READ), CL_MEM_READ_ONLY,
                                                CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, bias_buffer);
            }
            else
            {
                if (bias_term_)
                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias));
                else
                    kernel.set(argIdx++, (void *)NULL);
            }

            cl_mem out_buffer = NULL;
            if (output_image_offset)
            {
                region.origin = output_image_offset * sizeof(float);
                region.size = (total_top_size - output_image_offset) * sizeof(float);
                out_buffer = clCreateSubBuffer((cl_mem)top.handle(ACCESS_READ), CL_MEM_WRITE_ONLY,
                                               CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
                if (error)
                {
                    std::cout << "Failed to create sub buffer." << std::endl;
                    return false;
                }
                kernel.set(argIdx++, out_buffer);
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
            }

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
            int alignedFilterWidth = alignSize(M_, blockN);
            int alignedExpandHeight = alignSize(output_w_ * output_h_, blockM);
            int globalWorkSizeDX = blockN;
            int globalWorkSizeDY = blockM;
            size_t sgemm_m = alignedExpandHeight;
            size_t sgemm_n = alignedFilterWidth;
            size_t gx = (size_t) ceil( (float) sgemm_n /
                    (float) globalWorkSizeDX );
            size_t gy = (size_t) ceil( (float) sgemm_m /
                    (float) globalWorkSizeDY );
            gy = alignSize(gy, blockK);
            size_t global_size[3] = { gx, gy, config->global_work_size[2] };

            if (!kernel.run(3, global_size, config->local_work_size, false))
            {
                std::cout << "GEMM like kernel run failed." << std::endl;
                return false;
            }
            clReleaseMemObject(img_buffer);
            clReleaseMemObject(kernel_buffer);
            clReleaseMemObject(bias_buffer);
            clReleaseMemObject(out_buffer);
        }
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

                ocl::Kernel kernel(config->kernelName.c_str(), program);
                if (kernel.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
                kernel.set(argIdx++, image_offset);
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(weight));
                kernel.set(argIdx++, kernel_offset);
                if (bias_term_)
                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias));
                else
                    kernel.set(argIdx++, (void *)NULL);
                kernel.set(argIdx++, bias_offset);
                kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
                kernel.set(argIdx++, output_image_offset);
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                kernel.set(argIdx++, (uint16_t)pad_w_);
                kernel.set(argIdx++, (uint16_t)pad_h_);
                if (!kernel.run(3, config->global_work_size,
                                (config->use_null_local) ? NULL : config->local_work_size,
                                false))
                {
                    std::cout << "Basic kernel run failed." << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

template<>
float OCL4DNNConvSpatial<float>::timedConvolve(UMat &bottom, UMat &top,
                                               UMat &weight, UMat &bias,
                                               int32_t numImages, kernelConfig* config)
{
    // warm up.
    bool saved_tuned = tuned_;
    tuned_ = false;
    convolve(bottom, top, weight, bias, numImages, config);
    cv::ocl::Timer timer;
    timer.start();
    bool res = true;;
    dbgPrint(std::cout << "Benchmarking kernel: " << config->kernelName << std::endl);
    tuned_ = true;
    int loop_cnt = 4;
    for (int i = 0; i < loop_cnt; i++) {
        res = convolve(bottom, top, weight, bias, numImages, config);
        if (!res)
            break;
    }
    tuned_ = saved_tuned;
    timer.stop();
    if (!res) {
        config->tested = true;
        config->verified = false;
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
bool OCL4DNNConvSpatial<float>::verifyResult(UMat &bottom,
                                             UMat &top,
                                             UMat &weight,
                                             UMat &bias,
                                             int32_t numImages,
                                             kernelConfig* config,
                                             UMat &verifyTop)
{

    uint32_t verificationFail = 0;

    if (config->verified)
        return true;
    else if (config->tested)
        return false;

    int32_t sz[4] = {numImages, num_output_, output_h_, output_w_};
    top.zeros(4, sz, CV_32FC1);
    bool saved_tuned = tuned_;
    tuned_ = false;
    convolve(bottom, top, weight, bias, numImages, config);
    tuned_ = saved_tuned;

    float *data = (float *)top.getMat(ACCESS_READ).ptr<float>();
    float *verify_data = (float *)verifyTop.getMat(ACCESS_READ).ptr<float>();

    for (int32_t n = 0; n < num_; ++n) {
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
    if (verificationFail == 1)
        return false;
    else
        return true;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::unloadProgram(const std::string& kernelName)
{
    ocl::Program program;
    phash_t::iterator it = phash.find(kernelName);
    if (it != phash.end())
    {
        program = it->second;
        it->second = ocl::Program();
    }
    else
        return;

    ocl::Context ctx = ocl::Context::getDefault();
    ctx.unloadProg(program);
}

template<typename Dtype>
ocl::Program OCL4DNNConvSpatial<Dtype>::compileKernel()
{
    phash_t::iterator it = phash.find(kernel_name_);
    if (it != phash.end())
    {
        return it->second;
    }

    String errmsg;
    ocl::Context ctx = ocl::Context::getDefault();
    std::string options = options_.str();
    CV_Assert(options.size() != 0);
    ocl::Program program = ctx.getProg(src_, options, errmsg);

    phash.insert(std::pair<std::string, ocl::Program>(kernel_name_, program));
    if (!program.ptr())
    {
        std::cout << "Failed to compile kernel: " << kernel_name_
                  << ", buildflags: " << options
                  << ", errmsg: " << errmsg << std::endl;
    }
    return program;
}

template<>
bool OCL4DNNConvSpatial<float>::createGEMMLikeConvKernel(int32_t blockM,
                                                         int32_t blockK,
                                                         int32_t blockN)
{
    int32_t simd_size = blockK;

    int workItemOutput[3] = { blockM, blockK, blockN };
    size_t gx = (size_t)divUp(M_, blockN);
    size_t gy = (size_t)divUp(output_w_ * output_h_, blockM);
    gy = alignSize(gy, simd_size);
    size_t gz = num_;
    size_t global_size[3] = { gx, gy, gz };
    size_t local_size[3] = { 1, static_cast<size_t>(simd_size), 1 };

    kernelType_ = KERNEL_TYPE_GEMM_LIKE;
    blockM_ = blockM;
    blockK_ = blockK;
    blockN_ = blockN;
    setupKernel();

    ocl::Program program = compileKernel();
    if (program.ptr())
    {
        size_t workgroupSize_used;
        ocl::Kernel kernel(kernel_name_.c_str(), program);
        if (kernel.empty())
            return false;

        workgroupSize_used = kernel.preferedWorkGroupSizeMultiple();
        if (workgroupSize_used != simd_size)
        {
            unloadProgram(kernel_name_);
            return false;
        }
        else
        {
            kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &global_size[0], &local_size[0], &workItemOutput[0],
                                                        false, true, false, KERNEL_TYPE_GEMM_LIKE));
            return true;
        }
    }
    else
        return false;
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
        (size_t)divUp(output_width, output_block_width),
        (size_t)divUp(output_height, output_block_height),
        (size_t)num_batches * alignSize(num_output_maps, simd_size) };
    size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };

    kernelType_ = KERNEL_TYPE_INTEL_IDLF;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = simd_size;

    setupKernel();

    ocl::Program program = compileKernel();
    if (program.ptr())
    {
        size_t workgroupSize_used;
        ocl::Kernel kernel(kernel_name_.c_str(), program);
        if (kernel.empty())
            return false;

        workgroupSize_used = kernel.preferedWorkGroupSizeMultiple();
        if (workgroupSize_used != simd_size)
        {
            unloadProgram(kernel_name_);
            return false;
        }
        else
        {
            kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &global_size[0], &local_size[0], &workItemOutput[0],
                                                        false, true, false, KERNEL_TYPE_INTEL_IDLF));
            return true;
        }
    }
    else
        return false;
}

template<>
bool OCL4DNNConvSpatial<float>::tuneLocalSize(UMat &bottom, UMat &top,
                                              UMat &weight, UMat &bias,
                                              kernelConfig* config)
{
    if (config->use_null_local || !config->autoTune)
        return true;

    float fastestTime = std::numeric_limits<float>::infinity();
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

                bool res = true;
                res = convolve(bottom, top, weight, bias, 1, config);

                if (!res)
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
bool OCL4DNNConvSpatial<float>::createConvolutionKernel(int32_t kernelType,
                                                        int32_t blockWidth,
                                                        int32_t blockHeight,
                                                        int32_t blockDepth)
{
    kernelType_ = kernelType;
    options_.str(""); options_.clear(); // clear contents and state flags
    src_ = ocl::ProgramSource();

    if (kernelType == KERNEL_TYPE_INTEL_IDLF)
        return setupIDLF(blockWidth, blockHeight, blockDepth);
    else if (kernelType == KERNEL_TYPE_BASIC)
        return createBasicKernel(blockWidth, blockHeight, blockDepth);
    else if (kernelType == KERNEL_TYPE_GEMM_LIKE)
        return createGEMMLikeConvKernel(blockWidth, blockHeight, blockDepth);
    else
        CV_Assert(0 && "Internal error");
    return false;
}

template<>
void OCL4DNNConvSpatial<float>::generateTunerItems(std::vector< cv::Ptr<tunerParam> > &tunerItems)
{
    if (ocl::Device::getDefault().intelSubgroupsSupport()) {
        /* IDLF kernels are using Intel specific extension which make
           them intel only. */
        // Generates static key_
        int max_compute_units = ocl::Device::getDefault().maxComputeUnits();
        int kernelCnt = 0;
        if (group_ == 1 && ((M_ % 8 == 0) && (M_ % 32 != 24))) {
            tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_GEMM_LIKE, 1, 8, 32));
            tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_GEMM_LIKE, 2, 8, 32));

            if (kernel_w_ < 4 && M_ % 32 == 0)
                tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_GEMM_LIKE, 1, 16, 32));
        }

        for (int simd_size = 8; simd_size <= 16; simd_size += 8) {
            if (simd_size == 8 && !((group_ == 1 || M_ % 8 == 0)))
                continue;
            if (simd_size == 16 && !(group_ == 1 || M_ % 16 == 0))
                continue;
            int width_max, height_max, block_size_max;
            width_max = 14;
            height_max = 8;
            block_size_max = 32;
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
                    int actual_tile_x = kernel_w_ * dilation_w_ + (width - 1) * stride_w_;
                    int tile_x = (actual_tile_x + 3) & ~3;
                    int tile_y = kernel_h_ * dilation_h_ + (height - 1) * stride_h_;
                    if (tile_x > (4 * simd_size))
                        continue;
                    // If actual_tile_x is multiple of 4, we may waste some IO bandwidth.
                    // This could reduce 75% tuning candidates. It has slightly performance
                    // impact for the final tuning result, less than 2% for most cases.
                    if (actual_tile_x % 4 != 0)
                        continue;
                    if ((width * height +
                       (tile_x * tile_y + simd_size - 1)/ simd_size) > block_size_max)
                        continue;
                    int tile_y_stride = (4 * simd_size) / tile_x;

                    if ((tile_y + tile_y_stride - 1) / tile_y_stride < 4) {
                        tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_INTEL_IDLF, width, height, simd_size));
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
}

template<>
void OCL4DNNConvSpatial<float>::useFirstAvailable(UMat &bottom,
                                                  UMat &top,
                                                  UMat &weight,
                                                  UMat &bias,
                                                  int32_t numImages,
                                                  UMat &verifyTop)
{
    std::vector< cv::Ptr<tunerParam> > tunerItems;
    generateTunerItems(tunerItems);
    tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_BASIC, 1, 1, 1));

    for (int i = 0; i < tunerItems.size(); i++) {
        if (createConvolutionKernel(tunerItems[i]->kernelType,
                                    tunerItems[i]->blockWidth,
                                    tunerItems[i]->blockHeight,
                                    tunerItems[i]->blockDepth)) {
            int kernelIdx = kernelQueue.size() - 1;
            if (tuneLocalSize(bottom, top, weight, bias, kernelQueue[kernelIdx]) &&
                verifyResult(bottom, top, weight, bias, numImages, kernelQueue[kernelIdx], verifyTop)) {
                bestKernelConfig = kernelQueue[kernelIdx];
                if (bestKernelConfig->kernelType != KERNEL_TYPE_INTEL_IDLF &&
                    bestKernelConfig->kernelType != KERNEL_TYPE_GEMM_LIKE)
                    if (!swizzled_weights_umat.empty())
                        swizzled_weights_umat.release();

                for (int32_t j = 0; j < kernelIdx; j++) {
                    CV_Assert(phash.find(kernelQueue[j]->kernelName) != phash.end());
                    unloadProgram(kernelQueue[j]->kernelName);
                }
                kernelQueue.clear();
                tuned_ = true;
                break;
            }
        }
    }

    if (tuned_) {
        cv::AutoLock lock(kernelConfigMutex);
        std::stringstream outputKernel;
        outputKernel << bestKernelConfig->workItem_output[0] << " "
                     << bestKernelConfig->workItem_output[1] << " "
                     << bestKernelConfig->workItem_output[2] << " "
                     << bestKernelConfig->kernelType << " "
                     << bestKernelConfig->local_work_size[0] << " "
                     << bestKernelConfig->local_work_size[1] << " "
                     << bestKernelConfig->local_work_size[2] << " "
                     << bestKernelConfig->swizzle_weights << " "
                     << bestKernelConfig->use_null_local << " ";
        kernelConfigMap.insert(std::pair<std::string, std::string>(key_, outputKernel.str()));
    }

    return;
}

template<>
void OCL4DNNConvSpatial<float>::setupConvolution(UMat &bottom,
                                                 UMat &top,
                                                 UMat &weight,
                                                 UMat &bias,
                                                 int32_t numImages,
                                                 UMat &verifyTop)
{
    std::vector< cv::Ptr<tunerParam> > tunerItems;

    generateTunerItems(tunerItems);
    for (int i = 0; i < tunerItems.size(); i++)
        createConvolutionKernel(tunerItems[i]->kernelType,
                                tunerItems[i]->blockWidth,
                                tunerItems[i]->blockHeight,
                                tunerItems[i]->blockDepth);

    for (int32_t x = 0; x < kernelQueue.size(); x++) {
        if (tuneLocalSize(bottom, top, weight, bias, kernelQueue[x])) {
            kernelQueue[x]->executionTime = timedConvolve(bottom, top, weight, bias, numImages,
                                                          kernelQueue[x]);
        } else {
            // skip those kernels without a good local size.
            kernelQueue[x]->verified = false;
            kernelQueue[x]->tested = true;
        }
        #ifdef TEST_ALL_KERNELS
        if (kernelQueue[x]->tested == false) {
            bool verified = verifyResult(bottom, top, weight, bias, numImages, kernelQueue[x], verifyTop);
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
            float fastestTime = std::numeric_limits<float>::infinity();

            for (int32_t x = 0; x < kernelQueue.size(); x++) {
                if (kernelQueue[x]->executionTime < fastestTime &&
                    kernelQueue[x]->tested == false) {
                    fastestKernel = x;
                    fastestTime = kernelQueue[x]->executionTime;
                }
            }
            if (fastestKernel < 0) break;
            // Test fastest kernel
            bool verified = verifyResult(bottom, top, weight, bias, numImages, kernelQueue[fastestKernel], verifyTop);
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
        dbgPrint(std::cout << "Convolution Time:" << kernelQueue[kernel_index_]->executionTime << std::endl);
    } else {
        dbgPrint(std::cout << "fallback to basic kernel" << std::endl);
        options_.str(""); options_.clear(); // clear contents and state flags
        createBasicKernel(1, 1, 1);
        kernel_index_ = kernelQueue.size() - 1;
    }
    this->bestKernelConfig = kernelQueue[kernel_index_];


    if (bestKernelConfig->kernelType != 2 && bestKernelConfig->kernelType != 5)
        if (!swizzled_weights_umat.empty())
            swizzled_weights_umat.release();

    for (int32_t x = 0; x < kernelQueue.size(); x++) {
        if (x != kernel_index_) {
            CV_Assert(phash.find(kernelQueue[x]->kernelName) != phash.end());
            unloadProgram(kernelQueue[x]->kernelName);
        }
    }
    kernelQueue.clear();
    tuned_ = true;
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

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::prepareKernel(UMat &bottom, UMat &top,
                                              UMat &weight, UMat &bias,
                                              int32_t numImages)
{
    std::string previous_key = key_;

    generateKey();
    if (key_.compare(previous_key) == 0 && bestKernelConfig != NULL)
        return;

    if (bestKernelConfig)
    {
        prev_kernel_type_ = bestKernelConfig->kernelType;
        CV_Assert(phash.find(bestKernelConfig->kernelName) != phash.end());
        phash.erase(bestKernelConfig->kernelName);
        bestKernelConfig.release();
    }

    UMat benchData;
    if (auto_tuning_) {
        if (loadTunedConfig())
            return;
        else {
            benchData.create(1, numImages * num_output_ * out_spatial_dim_, CV_32FC1);
            calculateBenchmark(bottom, benchData, weight, bias, numImages);
            return setupConvolution(bottom, top, weight, bias, numImages, benchData);
        }
    } else {
        if (loadCachedConfig())
            return;
        else {
            benchData.create(1, numImages * num_output_ * out_spatial_dim_, CV_32FC1);
            calculateBenchmark(bottom, benchData, weight, bias, numImages);
            useFirstAvailable(bottom, top, weight, bias, numImages, benchData);
        }
    }
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::loadCachedConfig()
{
    cv::AutoLock lock(kernelConfigMutex);
    if (!defaultConfigLoaded)
    {
        for (int i = 0; i < CONFIG_NUM; i++)
        {
            kernelConfigMap.insert(std::pair<std::string, std::string>(default_kernel_config[2 * i], default_kernel_config[2 * i + 1]));
        }
        defaultConfigLoaded = true;
    }

    kernel_hash_t::iterator it = kernelConfigMap.find(key_);
    if (it != kernelConfigMap.end())
    {
        int32_t x, y, z, type, lx, ly, lz;
        bool swizzle, nullLocal;
        std::stringstream cachedKernel(it->second);
        if (cachedKernel)
        {
            cachedKernel >> x;
            cachedKernel >> y;
            cachedKernel >> z;
            cachedKernel >> type;
            cachedKernel >> lx;
            cachedKernel >> ly;
            cachedKernel >> lz;
            cachedKernel >> swizzle;
            cachedKernel >> nullLocal;
            if (setupKernelByConfig(x, y, z, type, lx, ly, lz, swizzle, nullLocal)) {
                tuned_ = true;
                return true;
            }
        }
    }
    return false;
}


template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::setupKernelByConfig(int x, int y, int z, int type,
                                                    int lx, int ly, int lz,
                                                    bool swizzle, bool nullLocal)
{
    if (type == KERNEL_TYPE_INTEL_IDLF)
    {
        if (z == 1)
            z = 16;
        CHECK_EQ(z == 16 || z == 8, true) << "invalid SIMD size" << std::endl;
    }
    kernelQueue.clear();
    createConvolutionKernel(type, x, y, z);
    if (kernelQueue.size() != 1) {
        std::cerr << "Failed setup kernel by config:"
            << " x = " << x
            << " y = " << y
            << " z = " << z
            << " type = " << type
            << std::endl;
        return false;
    }
    bestKernelConfig = kernelQueue[0];
    kernelQueue.clear();
    bestKernelConfig->local_work_size[0] = lx;
    bestKernelConfig->local_work_size[1] = ly;
    bestKernelConfig->local_work_size[2] = lz;
    bestKernelConfig->swizzle_weights = swizzle;
    bestKernelConfig->use_null_local = nullLocal;
    // If kernel type changed to type 2 or 4, we need to reset the swizzled
    // weights pointer to invalidate the previous swizzled weights data.
    if (prev_kernel_type_ != bestKernelConfig->kernelType &&
        (bestKernelConfig->kernelType == KERNEL_TYPE_INTEL_IDLF ||
        bestKernelConfig->kernelType == KERNEL_TYPE_GEMM_LIKE))
    {
        if (!swizzled_weights_umat.empty())
            swizzled_weights_umat.release();
    }
    return true;
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::loadTunedConfig()
{
    int32_t x, y, z, type, lx, ly, lz;
    bool swizzle, nullLocal;

    // Find cached kernel configuration from file
    std::string cacheFile;
    cacheFile = cache_path_.str() + key_;
    std::ifstream cachedKernel(cacheFile.c_str());
    if (cachedKernel)
    {
        cachedKernel >> x;
        cachedKernel >> y;
        cachedKernel >> z;
        cachedKernel >> type;
        cachedKernel >> lx;
        cachedKernel >> ly;
        cachedKernel >> lz;
        cachedKernel >> swizzle;
        cachedKernel >> nullLocal;
        if (setupKernelByConfig(x, y, z, type, lx, ly, lz, swizzle, nullLocal)) {
            tuned_ = true;
            return true;
        }
    }
    return false;
}

template class OCL4DNNConvSpatial<float>;
} // namespace ocl4dnn
}
}
#endif // HAVE_OPENCL
