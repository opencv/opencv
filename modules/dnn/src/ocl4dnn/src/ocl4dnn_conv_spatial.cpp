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

#include <opencv2/core/utils/configuration.private.hpp>

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

    fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_NONE;
    fused_eltwise_ = false;
    power_ = 1.f;
    negative_slope_ = 0;
    min_value_ = 0;
    max_value_ = 0;
    prev_kernel_type_ = -1;
    tuned_ = false;

    // assumption: spatial dimension is 2.
    kernel_h_ = config.kernel.height;
    kernel_w_ = config.kernel.width;
    pad_h_ = config.pad.height;
    pad_w_ = config.pad.width;
    stride_h_ = config.stride.height;
    stride_w_ = config.stride.width;
    dilation_h_ = config.dilation.height;
    dilation_w_ = config.dilation.width;
    M_ = num_output_ / group_;
    height_ = config.in_shape[dims - spatial_dims + 0];
    width_ = config.in_shape[dims - spatial_dims + 1];
    output_h_ = config.out_shape[dims - spatial_dims + 0];
    output_w_ = config.out_shape[dims - spatial_dims + 1];
    bottom_dim_ = channels_ * width_ * height_;
    top_dim_ = num_output_ * output_w_ * output_h_;

    cache_path_ = utils::getConfigurationParameterString("OPENCV_OCL4DNN_CONFIG_PATH", "");
    dwconv_ = (num_output_ == channels_ && channels_ == group_);

    use_cache_path_ = false;
    if (!cache_path_.empty())
    {
#if defined _WIN32
        struct _stat file_stat;
        use_cache_path_ = _stat(cache_path_.c_str(), &file_stat) == 0 &&
                      ((_S_IFDIR & file_stat.st_mode) != 0);
#else
        struct stat file_stat;
        use_cache_path_ = stat(cache_path_.c_str(), &file_stat) == 0 &&
                      S_ISDIR(file_stat.st_mode);
#endif
        if (!use_cache_path_)
        {
            static int warn_ = 0;
            if (!warn_)
            {
                std::cerr
                    << "OpenCV(ocl4dnn): Kernel configuration cache directory doesn't exist: " << cache_path_ << std::endl
                    << std::endl;
                warn_ = true;
            }
        }
    }

    force_auto_tuning_ =
            (use_cache_path_ && !utils::getConfigurationParameterBool("OPENCV_OCL4DNN_DISABLE_AUTO_TUNING", false))
            || utils::getConfigurationParameterBool("OPENCV_OCL4DNN_FORCE_AUTO_TUNING", false);
}

template<typename Dtype>
OCL4DNNConvSpatial<Dtype>::~OCL4DNNConvSpatial()
{
    if (!swizzled_weights_umat.empty()) {
        swizzled_weights_umat.release();
    }
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setFusionDefine(ocl4dnnFusedActiv_t fused_activ, bool fused_eltwise)
{
    if (fused_eltwise)
        addDef("FUSED_CONV_ELTWISE", 1);

    switch (fused_activ) {
        case OCL4DNN_CONV_FUSED_ACTIV_RELU:
            addDef("FUSED_CONV_RELU", 1);
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_PRELU:
            addDef("FUSED_CONV_PRELU", 1);
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_POWER:
            addDef("FUSED_CONV_POWER", 1);
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_TANH:
            addDef("FUSED_CONV_TANH", 1);
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_RELU6:
            addDef("FUSED_CONV_RELU6", 1);
            break;
        default:
            ;
    }
    return;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setFusionArg(ocl4dnnFusedActiv_t fused_activ, bool fused_eltwise, ocl::Kernel &kernel, cl_uint &argIdx)
{
    if (fused_eltwise)
        kernel.set(argIdx++, (cl_mem)bottom_data2_.handle(ACCESS_READ));

    switch (fused_activ) {
        case OCL4DNN_CONV_FUSED_ACTIV_RELU:
            kernel.set(argIdx++, (float)negative_slope_);
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_PRELU:
            kernel.set(argIdx++, (cl_mem)negative_slope_umat_.handle(ACCESS_READ));
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_POWER:
            kernel.set(argIdx++, (float)power_);
            break;
        case OCL4DNN_CONV_FUSED_ACTIV_RELU6:
            kernel.set(argIdx++, (float)min_value_);
            kernel.set(argIdx++, (float)max_value_);
            break;
        default:
            ;
    }
    return;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::collectCommonInformation()
{
    addDef("Dtype", "float");
    addDef("Dtype2", "float2");
    addDef("Dtype4", "float4");
    addDef("Dtype8", "float8");
    addDef("Dtype16", "float16");
    addDef("as_Dtype", "as_float");
    addDef("as_Dtype2", "as_float2");
    addDef("as_Dtype4", "as_float4");
    addDef("as_Dtype8", "as_float8");
}

typedef enum {
    KERNEL_TYPE_INTEL_IDLF = 2,
    KERNEL_TYPE_BASIC = 4,
    KERNEL_TYPE_GEMM_LIKE = 5,
    KERNEL_TYPE_DWCONV = 6
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
        if (clOptionSupport("-cl-no-subgroup-ifp"))
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
        int tile_x = alignSize((output_block_width - 1) * stride_w_ + kernel_w_ * dilation_w_, 4);
        int tile_y = (output_block_height -1) * stride_h_ + kernel_h_ * dilation_h_;
        int tile_y_stride = (4 * simd_size) / tile_x;
        int invec_size = divUp(tile_y, tile_y_stride);

        addDef("SIMD_SIZE", simd_size);
        addDef("filter_qualifier", "__global");
        addDef("OUT_BLOCK_WIDTH", output_block_width);
        addDef("OUT_BLOCK_HEIGHT", output_block_height);
        addDef("LAST_BLOCK_WIDTH", last_block_width);
        addDef("LAST_BLOCK_HEIGHT", last_block_height);
        addDef("INPUT_DEPTH", channels_ / group_);
        addDef("TOTAL_INPUT_DEPTH_SIZE", channels_);
        addDef("TOTAL_OUTPUT_DEPTH", num_output_);
        addDef("NUM_FILTERS", M_);
        addDef("TILE_X", tile_x);
        addDef("TILE_Y", tile_y);
        addDef("TILE_Y_STRIDE", tile_y_stride);
        addDef("INVEC_SIZE", invec_size);
        addDef("ALIGNED_NUM_FILTERS", (int)alignSize(M_, simd_size));
        addDef("OUT_BLOCK_SIZE", (output_block_width*output_block_height));
        addDef("APPLY_BIAS", bias_term_);
        setFusionDefine(fused_activ_, fused_eltwise_);

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
        if (clOptionSupport("-cl-no-subgroup-ifp"))
            options_ << " -cl-no-subgroup-ifp ";

        // defs
        addDef("CHANNELS", channels_ / group_);
        addDef("APPLY_BIAS", bias_term_);
        addDef("OUTPUT_Z", M_);
        addDef("ZPAR", 1);
        setFusionDefine(fused_activ_, fused_eltwise_);

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
        if (clOptionSupport("-cl-no-subgroup-ifp"))
            options_ << " -cl-no-subgroup-ifp ";

        addDef("KERNEL_GEMM_LIKE");
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
        addDef("APPLY_BIAS", bias_term_);
        setFusionDefine(fused_activ_, fused_eltwise_);
        src_ = ocl::dnn::conv_layer_spatial_oclsrc;
    }
    else if (kernelType == KERNEL_TYPE_DWCONV)
    {
        kernelUKey = generateSpecificKey(KERNEL_TYPE_DWCONV, blockM, blockK, blockN);
        kernel_name_ = "DWCONV_";
        kernel_name_ += kernelUKey.c_str();

        options_ << " -cl-fast-relaxed-math ";
        if (clOptionSupport("-cl-no-subgroup-ifp"))
            options_ << " -cl-no-subgroup-ifp ";

        addDef("KERNEL_DWCONV");
        addDef("KERNEL_SIZE", kernel_w_ * kernel_h_);
        addDef("KERNEL_W", kernel_w_);
        addDef("KERNEL_H", kernel_h_);
        addDef("APPLY_BIAS", bias_term_);
        addDef("OUTPUT_Z", num_output_ * num_);
        addDef("CHANNELS", num_output_);
        setFusionDefine(fused_activ_, fused_eltwise_);

        options_ << " -D DWCONV=" << kernel_name_;
        src_ = cv::ocl::dnn::conv_layer_spatial_oclsrc;
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
void OCL4DNNConvSpatial<Dtype>::setBias(bool bias_term)
{
    bias_term_ = bias_term;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setActivReLU(bool fuse_activ, float slope)
{
    if ( fuse_activ )
    {
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_RELU;
        negative_slope_ = slope;
    }
    else
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_NONE;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setActivReLU6(bool fuse_activ, float min, float max)
{
    if ( fuse_activ )
    {
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_RELU6;
        min_value_ = min;
        max_value_ = max;
    }
    else
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_NONE;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setActivPReLU(bool fuse_activ, std::vector<float> &slope)
{
    if ( fuse_activ )
    {
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_PRELU;
        Mat tmpMat = Mat(num_output_, 1, CV_32FC1, (uchar*)&slope[0]);
        tmpMat.copyTo(negative_slope_umat_);
    }
    else
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_NONE;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setActivPower(bool fuse_activ, float power)
{
    if ( fuse_activ )
    {
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_POWER;
        power_ = power;
    }
    else
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_NONE;
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::setActivTanh(bool fuse_activ)
{
    if ( fuse_activ )
    {
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_TANH;
    }
    else
        fused_activ_ = OCL4DNN_CONV_FUSED_ACTIV_NONE;
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::Forward(const UMat& bottom,
                                        const UMat& bottom2,
                                        const UMat& weight,
                                        const UMat& bias,
                                        UMat& top,
                                        int32_t numImages)
{
    num_ = numImages;
    if (!bottom2.empty())
    {
        fused_eltwise_ = true;
        bottom_data2_ = bottom2;
    }
    else
    {
        fused_eltwise_ = false;
    }

    prepareKernel(bottom, top, weight, bias, numImages);
    if (bestKernelConfig.empty())
        return false;
    return convolve(bottom, top, weight, bias, numImages, bestKernelConfig);
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::calculateBenchmark(const UMat &bottom, UMat &verifyTop,
                                                   const UMat &weight, const UMat &bias,
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

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::generateKey()
{
    std::stringstream keyBuilder;
    // FIXME: to support fuse?
    keyBuilder << "k" << kernel_w_ << "x" << kernel_h_ << "_"
               << "cn" << channels_ << "_"
               << "g" << group_ << "_"
               << "s" << stride_w_ << "x" << stride_h_ << "_"
               << "d" << dilation_w_ << "x" << dilation_h_ << "_"
               << "b" << bias_term_ << "_"
               << "in" << TUNING_SIZE(width_) << "x" << TUNING_SIZE(height_) << "_"
               << "p" << pad_w_ << "x" << pad_h_ << "_"
               << "num" << num_ << "_"
               << "M" << M_ << "_"
               << "activ" << fused_activ_ << "_"
               << "eltwise" << fused_eltwise_;


    key_ = ocl::Device::getDefault().vendorName() + "_EU" + cv::format("%d", ocl::Device::getDefault().maxComputeUnits()) + "_" + keyBuilder.str();
    key_sanitized_ = key_;
    for (size_t i = 0; i < key_sanitized_.size(); i++)
    {
        char c = key_sanitized_[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'))
        {
            key_sanitized_[i] = '_';
        }
    }
    // TODO add hash?
    // key_sanitized_ = key_sanitized_ + cv::format("_%08llx", crc64((uchar*)key_.c_str(), key_.size()));
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
bool OCL4DNNConvSpatial<Dtype>::swizzleWeight(const UMat &weight,
                                              int32_t swizzled_factor,
                                              bool interleave)
{
    // Simply skip the weight swizzle if we already got a swizzled_weights_
    // in test phase and not in auto tuning
    // This requires we always call convolve again with the winner configuration
    // during the auto tuning stage.
    if (tuned_ && !swizzled_weights_umat.empty())
        return true;

    if (swizzled_weights_umat.empty())
        swizzled_weights_umat.create(1, (int)alignSize(num_output_, 16) * channels_ *
                                     kernel_h_ * (int)alignSize(kernel_w_, 2), CV_32FC1);

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
bool OCL4DNNConvSpatial<float>::createBasicKernel(int32_t blockWidth,
                                                  int32_t blockHeight, int32_t blockDepth)
{
    kernelType_ = KERNEL_TYPE_BASIC;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = blockDepth;
    setupKernel();

    ocl::Program program = compileKernel();
    if (program.ptr())
    {
        int32_t workItemOutput[3] = { 1, 1, 1 };
        size_t globalSize[3] = { (size_t)output_w_, (size_t)output_h_, (size_t)M_ };
        kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &globalSize[0], (const size_t*)NULL, &workItemOutput[0],
                                                    false, KERNEL_TYPE_BASIC));
        return true;
    }
    else
        return false;
}

template<>
void OCL4DNNConvSpatial<float>::CreateSubBuffer(const UMat& buffer, UMat& sub_buffer,
                                                int32_t offset, int32_t size, bool write_only)
{
    cl_mem sub_mem;
    cl_buffer_region region;
    cl_int err;

    region.origin = offset * sizeof(float);
    region.size = size * sizeof(float);
    sub_mem = clCreateSubBuffer((cl_mem)buffer.handle(ACCESS_READ),
                                write_only ? CL_MEM_WRITE_ONLY : CL_MEM_READ_ONLY,
                                CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err)
    {
        std::cout << "Failed to create sub buffer." << std::endl;
        return;
    }

    int step = sizeof(float), rows = size, cols = 1;
    ocl::convertFromBuffer(sub_mem, step, rows, cols, CV_32FC1, sub_buffer);

    //decrease ocl mem refcount
    clReleaseMemObject(sub_mem);
}

template<>
bool OCL4DNNConvSpatial<float>::convolve(const UMat &bottom, UMat &top,
                                         const UMat &weight, const UMat &bias,
                                         int32_t numImages, kernelConfig* config)
{
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
            setFusionArg(fused_activ_, fused_eltwise_, kernel, argIdx);

            UMat img_buffer;
            if (image_offset)
            {
                CreateSubBuffer(bottom, img_buffer, image_offset,
                                total_bottom_size - image_offset, false);
                if (img_buffer.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(img_buffer));
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
            }

            UMat kernel_buffer;
            if (kernel_offset)
            {
                CreateSubBuffer(swizzled_weights_umat, kernel_buffer, kernel_offset,
                                total_kernel_size - kernel_offset, false);
                if (kernel_buffer.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(kernel_buffer));
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(swizzled_weights_umat));
            }

            UMat bias_buffer;
            if (bias_term_)
            {
                if (bias_offset)
                {
                    CreateSubBuffer(bias, bias_buffer, bias_offset,
                                    total_bias_size - bias_offset, false);
                    if (bias_buffer.empty())
                        return false;

                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias_buffer));
                }
                else
                {
                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias));
                }
            }

            UMat out_buffer;
            if (output_image_offset)
            {
                CreateSubBuffer(top, out_buffer, output_image_offset,
                                total_top_size - output_image_offset, true);
                if (out_buffer.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(out_buffer));
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
            setFusionArg(fused_activ_, fused_eltwise_, kernel, argIdx);

            UMat img_buffer;
            if (image_offset)
            {
                CreateSubBuffer(bottom, img_buffer, image_offset,
                                total_bottom_size - image_offset, false);
                if (img_buffer.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(img_buffer));
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
            }

            UMat kernel_buffer;
            if (kernel_offset)
            {
                CreateSubBuffer(swizzled_weights_umat, kernel_buffer, kernel_offset,
                                total_kernel_size - kernel_offset, false);
                if (kernel_buffer.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(kernel_buffer));
            }
            else
            {
                kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(swizzled_weights_umat));
            }

            UMat bias_buffer;
            if (bias_term_)
            {
                if (bias_offset)
                {
                    CreateSubBuffer(bias, bias_buffer, bias_offset,
                                    total_bias_size - bias_offset, false);
                    if (bias_buffer.empty())
                        return false;

                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias_buffer));
                }
                else
                {
                    kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias));
                }
            }

            UMat out_buffer;
            if (output_image_offset)
            {
                CreateSubBuffer(top, out_buffer, output_image_offset,
                                total_top_size - output_image_offset, true);
                if (out_buffer.empty())
                    return false;

                kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(out_buffer));
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
            size_t gx = divUp(sgemm_n, globalWorkSizeDX);
            size_t gy = divUp(sgemm_m, globalWorkSizeDY);
            gy = alignSize(gy, blockK);
            size_t global_size[3] = { gx, gy, config->global_work_size[2] };

            if (!kernel.run(3, global_size, config->local_work_size, false))
            {
                std::cout << "GEMM like kernel run failed." << std::endl;
                return false;
            }
        }
    } else if (config->kernelType == KERNEL_TYPE_DWCONV) {
        ocl::Kernel kernel(config->kernelName.c_str(), program);
        if (kernel.empty())
            return false;

        cl_uint argIdx = 0;
        setFusionArg(fused_activ_, fused_eltwise_, kernel, argIdx);
        kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bottom));
        kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(weight));
        if (bias_term_)
            kernel.set(argIdx++, ocl::KernelArg::PtrReadOnly(bias));
        kernel.set(argIdx++, ocl::KernelArg::PtrWriteOnly(top));
        kernel.set(argIdx++, (uint16_t)width_);
        kernel.set(argIdx++, (uint16_t)height_);
        kernel.set(argIdx++, (uint16_t)output_w_);
        kernel.set(argIdx++, (uint16_t)output_h_);

        size_t global_size[3];
        global_size[0] = output_w_;
        global_size[1] = output_h_;
        global_size[2] = num_output_ * num_;

        if (!kernel.run(3, global_size, NULL, false))
        {
            std::cout << "DWCONV kernel run failed." << std::endl;
            return false;
        }
    } else {
        for (int32_t n = 0; n < numImages; ++n) {
            for (int32_t g = 0; g < group_; ++g) {
                bias_offset = M_ * g;
                int32_t image_offset = n * bottom_dim_
                    + width_ * height_ * (channels_ / group_) * g;
                int32_t output_image_offset = n * top_dim_
                    + output_w_ * output_h_ * M_ * g;

                int32_t kernel_offset = kernel_h_ * kernel_w_ *
                                       (channels_ / group_) * M_
                                       * g;

                ocl::Kernel kernel(config->kernelName.c_str(), program);
                if (kernel.empty())
                    return false;

                cl_uint argIdx = 0;
                setFusionArg(fused_activ_, fused_eltwise_, kernel, argIdx);
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
float OCL4DNNConvSpatial<float>::timedConvolve(const UMat &bottom, UMat &top,
                                               const UMat &weight, const UMat &bias,
                                               int32_t numImages, kernelConfig* config)
{
    cv::ocl::Queue queue;
    try
    {
        queue = cv::ocl::Queue::getDefault();
    }
    catch (const cv::Exception&)
    {
        static int warn_ = 0;
        if (!warn_)
        {
            std::cout << "OpenCV(ocl4dnn): Can't get OpenCL default queue for auto-tuning." << std::endl;
            warn_ = true;
        }
        return 1e6;
    }

    // warm up.
    bool saved_tuned = tuned_;
    tuned_ = false;
    convolve(bottom, top, weight, bias, numImages, config);

    cv::ocl::Timer timer(queue);
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

    float elapsedTime = timer.durationNS() * 1e-6 / loop_cnt;
    #ifdef dbg
    double out_w = output_w_;
    double out_h = output_h_;
    double out_z = M_;
    double k_w = kernel_w_;
    double k_h = kernel_h_;
    double k_z = channels_;
    double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z)*num_;
    std::cout << "\tEstimated Gflops:" << (totalFlops * 1e-9)
              << std::endl;
    std::cout << "\tEstimated GFLOPS/S: " << ((totalFlops * 1e-9)*(1000.0/elapsedTime))
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
bool OCL4DNNConvSpatial<float>::verifyResult(const UMat &bottom,
                                             UMat &top,
                                             const UMat &weight,
                                             const UMat &bias,
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
            std::cerr << "OpenCV(ocl4dnn): The OpenCL compiler chose a simd size (" << workgroupSize_used << ") that " << std::endl;
            std::cerr << "                 does not equal the size (" << simd_size << ") kernel source required." << std::endl;
            std::cerr << "                 Skip this kernel " << kernel_name_ << std::endl;
            unloadProgram(kernel_name_);
            return false;
        }
        else
        {
            kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &global_size[0], &local_size[0], &workItemOutput[0],
                                                        true, KERNEL_TYPE_GEMM_LIKE));
            return true;
        }
    }
    else
        return false;
}

template<>
bool OCL4DNNConvSpatial<float>::createIDLFKernel(int32_t blockWidth,
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
            std::cerr << "OpenCV(ocl4dnn): The OpenCL compiler chose a simd size (" << workgroupSize_used << ") that " << std::endl;
            std::cerr << "                 does not equal the size (" << simd_size << ") kernel source required." << std::endl;
            std::cerr << "                 Skip this kernel " << kernel_name_ << std::endl;
            unloadProgram(kernel_name_);
            return false;
        }
        else
        {
            kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &global_size[0], &local_size[0], &workItemOutput[0],
                                                        true, KERNEL_TYPE_INTEL_IDLF));
            return true;
        }
    }
    else
        return false;
}

template<>
bool OCL4DNNConvSpatial<float>::createDWConvKernel(int32_t blockWidth,
                                                   int32_t blockHeight,
                                                   int32_t blockDepth)
{
    if (!dwconv_)
        return false;

    int workItemOutput[3] = { 1, 1, 1 };
    size_t local_size[3] = { 1, 1, 1 };
    size_t global_size[3];
    global_size[0] = divUp(output_w_, workItemOutput[0]);
    global_size[1] = divUp(output_h_, workItemOutput[1]);
    global_size[2] = divUp(M_ * num_, workItemOutput[2]);

    kernelType_ = KERNEL_TYPE_DWCONV;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = blockDepth;

    setupKernel();

    ocl::Program program = compileKernel();
    if (program.ptr())
    {
        kernelQueue.push_back(makePtr<kernelConfig>(kernel_name_, &global_size[0], &local_size[0],
                              &workItemOutput[0], false, KERNEL_TYPE_DWCONV));
        return true;
    }
    else
        return false;
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
        return createIDLFKernel(blockWidth, blockHeight, blockDepth);
    else if (kernelType == KERNEL_TYPE_BASIC)
        return createBasicKernel(blockWidth, blockHeight, blockDepth);
    else if (kernelType == KERNEL_TYPE_GEMM_LIKE)
        return createGEMMLikeConvKernel(blockWidth, blockHeight, blockDepth);
    else if (kernelType == KERNEL_TYPE_DWCONV)
        return createDWConvKernel(blockWidth, blockHeight, blockDepth);
    else
        CV_Assert(0 && "Internal error");
    return false;
}

template<>
void OCL4DNNConvSpatial<float>::generate_gemmlike_tuneritems(std::vector< cv::Ptr<tunerParam> > &tunerItems,
                                                             int blockM, int blockK, int blockN)
{
    if (group_ != 1 || ((M_ % 8 != 0) || (M_ % 32 == 24)))
        return;

    if (blockM != 1 && blockM != 2)
        return;

    if (blockN != 32)
        return;

    if (blockK != 8 && blockK != 16)
        return;

    if (blockK == 16)
    {
        if ((blockM == 1 && (kernel_w_ > 4)) || M_ % 32 != 0)
            return;
        if ((blockM == 2) || M_ % 32 != 0)
            return;
    }

    tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_GEMM_LIKE, blockM, blockK, blockN));
}

template<>
void OCL4DNNConvSpatial<float>::generate_idlf_tuneritems(std::vector< cv::Ptr<tunerParam> > &tunerItems,
                                                         int blockM, int blockK, int simd_size)
{
    int max_compute_units = ocl::Device::getDefault().maxComputeUnits();

    if (simd_size != 8 && simd_size != 16)
        return;

    if (simd_size == 8 && !((group_ == 1 || M_ % 8 == 0)))
        return;

    if (simd_size == 16 && !(group_ == 1 || M_ % 16 == 0))
        return;

    int width_max, height_max, block_size_max;
    width_max = 14;
    height_max = 14;
    block_size_max = 32;

    if (blockM > width_max)
        return;
    if (blockK > height_max)
        return;

    if (blockM > output_w_)
        return;
    if (blockK > output_h_)
        return;

    // Only when the work items count is less than the device
    // max work items or the M_ is less than 16, we will tune
    // for simd 8.
    if (simd_size == 8 &&  M_ >= 16 &&
        ((num_ * M_ * output_w_ * output_h_ / static_cast<float>(blockM * blockK)) >=
        max_compute_units * 7 * 16))
        return;

    int actual_tile_x = kernel_w_ * dilation_w_ + (blockM - 1) * stride_w_ ;
    int tile_x = alignSize(actual_tile_x, 4);
    int tile_y = kernel_h_ * dilation_h_ + (blockK - 1) * stride_h_;
    if (tile_x > (4 * simd_size))
        return;

    if ((blockM * blockK + divUp(tile_x * tile_y, simd_size)) > block_size_max)
        return;

    int tile_y_stride = (4 * simd_size) / tile_x;
    int invec_size = divUp(tile_y, tile_y_stride);
    if (invec_size > 4)
        return;

    tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_INTEL_IDLF, blockM, blockK, simd_size));
}

template<>
void OCL4DNNConvSpatial<float>::generate_dwconv_tuneritems(std::vector< cv::Ptr<tunerParam> > &tunerItems,
                                                           int blockM, int blockK, int blockN)
{
    if (!dwconv_)
        return;

    tunerItems.push_back(makePtr<tunerParam>(KERNEL_TYPE_DWCONV, blockM, blockK, blockN));
}

template<>
void OCL4DNNConvSpatial<float>::generateTunerItems(std::vector< cv::Ptr<tunerParam> > &tunerItems)
{
    if (ocl::Device::getDefault().intelSubgroupsSupport())
    {
        // depthwise kernel
        generate_dwconv_tuneritems(tunerItems, 1, 1, 1);
        if (tunerItems.size() > 0 && group_ > 8)
            return;

        // gemm like kernel
        generate_gemmlike_tuneritems(tunerItems, 1, 8, 32);
        generate_gemmlike_tuneritems(tunerItems, 2, 8, 32);
        generate_gemmlike_tuneritems(tunerItems, 1, 16, 32);
        generate_gemmlike_tuneritems(tunerItems, 2, 16, 32);

        // idlf kernel
        for (int simd_size = 8; simd_size <= 16; simd_size += 8)
        {
            int width_max, height_max;
            width_max = 14;
            height_max = 14;
            for (uint32_t width = width_max; width > 0; width--)
            {
                for (uint32_t height = height_max; height > 0; height--)
                {
                    generate_idlf_tuneritems(tunerItems, width, height, simd_size);
                    if (tunerItems.size() >= 8 && height == 2)
                        break;
                }
                if (tunerItems.size() >= 12 && width == 2)
                    break;
            }
        }
    }
}

template<>
void OCL4DNNConvSpatial<float>::useFirstAvailable(const UMat &bottom,
                                                  UMat &top,
                                                  const UMat &weight,
                                                  const UMat &bias,
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
            if (verifyResult(bottom, top, weight, bias, numImages, kernelQueue[kernelIdx], verifyTop)) {
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
}

template<>
void OCL4DNNConvSpatial<float>::cacheTunedConfig()
{
    if (tuned_)
    {
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
}

template<>
void OCL4DNNConvSpatial<float>::setupConvolution(const UMat &bottom,
                                                 UMat &top,
                                                 const UMat &weight,
                                                 const UMat &bias,
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
        kernelQueue[x]->executionTime = timedConvolve(bottom, top, weight, bias, numImages,
                                                      kernelQueue[x]);
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


    if (bestKernelConfig->kernelType != KERNEL_TYPE_INTEL_IDLF && bestKernelConfig->kernelType != KERNEL_TYPE_GEMM_LIKE)
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
    saveTunedConfig();
}

template<typename Dtype>
void OCL4DNNConvSpatial<Dtype>::saveTunedConfig()
{
    CV_Assert(tuned_);
    if (!use_cache_path_ || cache_path_.empty())
        return;

    std::string outputFile;
    outputFile = cache_path_ + "/" + key_sanitized_;
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
void OCL4DNNConvSpatial<Dtype>::prepareKernel(const UMat &bottom, UMat &top,
                                              const UMat &weight, const UMat &bias,
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

    if (loadCachedConfig()) // check in-memory cache
        return;

    if (loadTunedConfig()) // check external storage
        return;

    UMat benchData(1, numImages * top_dim_, CV_32FC1);
    if (force_auto_tuning_)
    {
        calculateBenchmark(bottom, benchData, weight, bias, numImages);
        setupConvolution(bottom, top, weight, bias, numImages, benchData);
    }
    else
    {
        calculateBenchmark(bottom, benchData, weight, bias, numImages);
        useFirstAvailable(bottom, top, weight, bias, numImages, benchData);
    }
    cacheTunedConfig();
}

template<typename Dtype>
bool OCL4DNNConvSpatial<Dtype>::loadCachedConfig()
{
    cv::AutoLock lock(kernelConfigMutex);
    if (!defaultConfigLoaded)
    {
        const size_t numConfigs = sizeof(default_kernel_config_intel)/sizeof(default_kernel_config_intel[0])/2;
        for (size_t i = 0; i < numConfigs; i++)
        {
            std::pair<std::string, std::string> entry(
                    std::string("Intel(R) Corporation_") + default_kernel_config_intel[2 * i],
                    default_kernel_config_intel[2 * i + 1]);
            kernelConfigMap.insert(entry);
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
    if (!use_cache_path_)
    {
        if (cache_path_.empty() && !force_auto_tuning_)
        {
            static int warn_ = 0;
            if (!warn_)
            {
                std::cout << "OpenCV(ocl4dnn): consider to specify kernel configuration cache directory " << std::endl
                          << "                 via OPENCV_OCL4DNN_CONFIG_PATH parameter." << std::endl;
                warn_ = true;
            }
        }
        return false;
    }

    int32_t x, y, z, type, lx, ly, lz;
    bool swizzle, nullLocal;

    // Find cached kernel configuration from file
    std::string cacheFile = cache_path_ + "/" + key_sanitized_;
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
