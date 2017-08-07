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

#ifndef _OPENCV_LIBDNN_HPP_
#define _OPENCV_LIBDNN_HPP_
#include "../../precomp.hpp"
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common.hpp"

namespace cv { namespace dnn { namespace ocl4dnn {
#ifdef HAVE_OPENCL
struct OCL4DNNConvConfig
{
    OCL4DNNConvConfig() : in_shape(3, 1),
                         out_shape(3, 1),
                         kernel(1, 1),
                         pad(0, 0),
                         stride(1, 1),
                         dilation(1, 1)
    {}
    std::vector<int32_t> in_shape;
    std::vector<int32_t> out_shape;
    std::vector<int32_t> kernel;
    std::vector<int32_t> pad;
    std::vector<int32_t> stride;
    std::vector<int32_t> dilation;
    int32_t group = 1;
    bool bias_term = false;
    bool weights_backward = true;
    bool bias_backward = true;
    bool phase_test = true;
};

template<typename Dtype>
class OCL4DNNConvSpatial
{
    public:
        explicit OCL4DNNConvSpatial(OCL4DNNConvConfig config);
        ~OCL4DNNConvSpatial();
        bool Forward(const Dtype* bottom_data, const Dtype* weight,
                     const Dtype* bias,
                     Dtype* top_data, int32_t batch_size);

    private:
        struct kernelConfig
        {
            std::string kernelName;
            float executionTime;
            size_t local_work_size[3];
            size_t global_work_size[3];
            int32_t workItem_output[3];
            bool verified;
            bool autoTune;
            bool tested;
            bool swizzle_weights;
            bool use_null_local;
            int32_t kernelType;

            kernelConfig()
            {}

            kernelConfig(std::string name, size_t* global_size, size_t* local_size,
                         int32_t* workItem,
                         bool tune, bool swizzle, bool null_local,
                         int32_t type = 0)
            {
                kernelName = name;
                for (int32_t x = 0; x < 3; x++)
                {
                    local_work_size[x] = local_size[x];
                    global_work_size[x] = global_size[x];
                    workItem_output[x] = workItem[x];
                }
                autoTune = tune;
                swizzle_weights = swizzle;
                use_null_local = null_local;
                verified = false;
                tested = false;
                kernelType = type;
            }
        };

        template<class T>
        inline void addDef(std::stringstream& ss,  // NOLINT
                           const char* name, T value)
        {
            ss << "#ifdef " << name << std::endl;
            ss << "#undef " << name << std::endl;
            ss << "#endif" << std::endl;
            if (std::is_same<T, float>::value)
            {
                ss << "#define " << name << " (float) " << std::setprecision(32) << value
                   << std::endl;
            }
            else if (std::is_same<T, double>::value)
            {
                ss << "#define " << name << " (double) " << std::setprecision(32) << value
                   << std::endl;
            }
            else
            {
                ss << "#define " << name << " " << value << std::endl;
            }
        }

        template<class T>
        inline void addDef(std::stringstream& ss,  // NOLINT
                            const std::string name, T value)
        {
            addDef(ss, name.c_str(), value);
        }

        void tune(Dtype* top_data,
                  const Dtype* weight,
                  const Dtype* bias,
                  const Dtype* bottom_data,
                  int32_t batch_size);
        void generateKernelSrc();
        uint64 crc64(const uchar* data, size_t size, uint64 crc0 = 0);
        std::string generateHeader();
        std::string generateDefs();
        std::string generateKernels(int32_t kernelType,
                                    int32_t blockM,
                                    int32_t blockK,
                                    int32_t blockN);

        ocl::Program compileKernel();
        typedef std::map<std::string, ocl::Program> phash_t;
        phash_t phash;
        void calculateBenchmark(const Dtype* bottom,
                                const Dtype* w,
                                const Dtype* bias,
                                Dtype* verify_data);

        void setupConvolution(const Dtype *bottom,
                              Dtype *top,
                              const Dtype *verify_blob);
        void createConvolutionKernel(int32_t kernelType,
                                     int32_t blockWidth,
                                     int32_t blockHeight,
                                     int32_t blockDepth);
        bool setupIDLF(int32_t blockWidth,
                       int32_t blockHeight,
                       int32_t blockDepth);
        bool createBasicKernel(int32_t blockWidth,
                               int32_t blockHeight,
                               int32_t blockDepth);
        bool createGEMMLikeConvKernel(int32_t blockWidth,
                                      int32_t blockHeight,
                                      int32_t blockDepth);
        cl_int convolve(const Dtype *bottom,
                        const Dtype *top, int32_t index,
                        int32_t numImages,
                        kernelConfig* config);
        float timedConvolve(const Dtype *bottom,
                            const Dtype *top, int32_t index,
                            int32_t numImages,
                            kernelConfig* config);
        bool verifyResult(const Dtype *bottom,
                          Dtype *top,
                          int32_t index,
                          int32_t numImages,
                          const Dtype *verify_blob,
                          kernelConfig* config);
        bool tuneLocalSize(const Dtype *bottom,
                           const Dtype *top,
                           kernelConfig*);
        void swizzleWeights(const Dtype *bottom,
                            const Dtype *top,
                            int32_t swizzle_factor,
                            bool interleave = false);
        void generateKey();
        std::string generateSpecificKey(int32_t type, int32_t blockWidth,
                                          int32_t blockHeight,
                                          int32_t blockDepth);
        void computeGlobalSize(int32_t batch,
                               int32_t* workItemOutput,
                               size_t* localSizes,
                               size_t* globalSizes);
        bool loadCachedConfig();
        void prepareKernel();
        void setBufferKernelArg(const Dtype *bottom,
                                const Dtype *top,
                                ocl::Kernel *cl_kernel,
                                const cl_uint &argIdx,
                                ocl::Context *ctx,
                                cl_mem buffer, size_t offset,
                                size_t size, bool readOnly,
                                bool preserved);
        void cleanTmpSubBuffers(const Dtype *bottom,
                                const Dtype *top);

        int32_t group_;
        bool bias_term_;
        std::map<std::tuple<cl_mem, size_t, size_t>, cl_mem> subBufferMap;
        std::vector<cl_mem> tmpSubBuffers;
        const Dtype* bottom_data_;
        Dtype* top_data_;
        const Dtype* weight_;
        Dtype* swizzled_weights_;
        const Dtype* bias_;
        int32_t bottom_index_;
        int32_t output_h_;
        int32_t output_w_;
        int32_t kernel_h_;
        int32_t kernel_w_;
        int32_t height_;
        int32_t width_;
        int32_t pad_h_;
        int32_t pad_w_;
        int32_t stride_h_;
        int32_t stride_w_;
        int32_t dilation_h_;
        int32_t dilation_w_;

        /// M_ is the channel dimension of the output for a single group, which is the
        /// leading dimension of the filter matrix.
        int32_t M_;
        /// K_ is the dimension of an unrolled input for a single group, which is the
        /// leading dimension of the data matrix.
        int32_t K_;
        /// N_ is the spatial dimension of the output, the H x W, which are the last
        /// dimensions of the data and filter matrices.
        int32_t N_;

        bool tuned_;
        std::string key_;
        std::string short_key_;
        std::string kernel_name_;
        std::stringstream cache_path_;
        int32_t kernel_index_;
        std::vector<kernelConfig*> kernelQueue;
        kernelConfig* bestKernelConfig;

        int32_t bottom_dim_;
        int32_t top_dim_;
        int32_t num_;
        int32_t channels_;
        int32_t out_spatial_dim_;
        int32_t num_output_;
        int32_t kernel_dim_;

        int32_t kernelType_;
        int32_t blockM_;
        int32_t blockK_;
        int32_t blockN_;
        std::string options_;
        std::string kernel_;
        int32_t prev_kernel_type_;
};

typedef enum {
    LIBDNN_POOLING_METHOD_MAX                 = 0,
    LIBDNN_POOLING_METHOD_AVE                 = 1,
    LIBDNN_POOLING_METHOD_STO                 = 2
} ocl4dnnPoolingMethod_t;

struct OCL4DNNPoolConfig
{
    OCL4DNNPoolConfig() : in_shape(3, 1),
                         out_shape(3, 1),
                         kernel(1, 1),
                         pad(0, 0),
                         stride(1, 1),
                         dilation(1, 1)
    {}
    std::vector<int32_t> in_shape;
    std::vector<int32_t> out_shape;
    std::vector<int32_t> kernel;
    std::vector<int32_t> pad;
    std::vector<int32_t> stride;
    std::vector<int32_t> dilation;

    int32_t channels;
    ocl4dnnPoolingMethod_t pool_method = LIBDNN_POOLING_METHOD_MAX;
    bool global_pooling = false;
};

template<typename Dtype>
class OCL4DNNPool
{
    public:
        explicit OCL4DNNPool(OCL4DNNPoolConfig config);
        ~OCL4DNNPool();
        bool Forward(const Dtype *bottom_data,
                     Dtype *top_data,
                     Dtype *top_mask = NULL);

    private:
        UMat mask_idx_;

        // Pooling parameters
        std::vector<int32_t> pad_;
        std::vector<int32_t> stride_;
        std::vector<int32_t> kernel_shape_;
        std::vector<int32_t> im_in_shape_;
        std::vector<int32_t> im_out_shape_;

        ocl4dnnPoolingMethod_t pool_method_;
        int32_t count_;
        int32_t batch_size_;
        int32_t channels_;
        int32_t kernel_h_;
        int32_t kernel_w_;
        int32_t stride_h_;
        int32_t stride_w_;
        int32_t pad_h_;
        int32_t pad_w_;
        int32_t height_;
        int32_t width_;
        int32_t pooled_height_;
        int32_t pooled_width_;
};

struct OCL4DNNInnerProductConfig
{
    int32_t num_output;
    int32_t M;
    int32_t K;
    bool bias_term;
    bool transpose = false;
    bool phase_test = true;
};

template<typename Dtype>
class OCL4DNNInnerProduct
{
    public:
        explicit OCL4DNNInnerProduct(OCL4DNNInnerProductConfig config);
        ~OCL4DNNInnerProduct();
        bool Forward(const Dtype* bottom_data,
                     const Dtype* weight,
                     const Dtype* bias,
                     Dtype* top_data);
    private:
        OCL4DNNInnerProductConfig config_;
        int32_t axis_;
        int32_t num_output_;
        int32_t M_;
        int32_t N_;
        int32_t K_;
        bool bias_term_;
        bool transpose_;
        bool image_copied_;
        bool phase_test_;
        UMat bias_multiplier_;
        UMat weight_image_;
};

typedef enum {
    LRNParameter_NormRegion_ACROSS_CHANNELS = 0,
    LRNParameter_NormRegion_WITHIN_CHANNEL = 1
} LRNParameter_NormRegion_WITHIN_CHANNEL_t;

struct OCL4DNNLRNConfig
{
    OCL4DNNLRNConfig()
    {}
    std::vector<int32_t> in_shape;
    LRNParameter_NormRegion_WITHIN_CHANNEL_t lrn_type;
    bool phase_test = true;
    int32_t local_size;
    float alpha;
    float beta;
    float k;
    bool norm_by_size;
    int32_t batch_size;
    int32_t channels;
    int32_t height;
    int32_t width;
};

template<typename Dtype>
class OCL4DNNLRN
{
    public:
        explicit OCL4DNNLRN(OCL4DNNLRNConfig config);
        bool Forward(const Dtype* bottom_data, Dtype* top_data);

    private:
        void crossChannelForward(const Dtype* bottom_data, Dtype* top_data);
        LRNParameter_NormRegion_WITHIN_CHANNEL_t lrn_type_;
        bool phase_test_;
        int32_t size_;
        Dtype alpha_;
        Dtype beta_;
        Dtype k_;
        int32_t num_;
        int32_t channels_;
        int32_t height_;
        int32_t width_;
        bool norm_by_size_;
};

struct OCL4DNNSoftmaxConfig
{
    OCL4DNNSoftmaxConfig()
    {}
    std::vector<int32_t> in_shape;
    int32_t axis;
    int32_t channels;
};

template<typename Dtype>
class OCL4DNNSoftmax
{
    public:
        explicit OCL4DNNSoftmax(OCL4DNNSoftmaxConfig config);
        ~OCL4DNNSoftmax();
        bool Forward(const Dtype* bottom_data, Dtype* top_data);

    private:
        int32_t softmax_axis_;
        int32_t inner_num_;
        int32_t outer_num_;
        int32_t channels_;
        int32_t count_;
        bool use_slm_;
        UMat scale_data_;
};
#endif // HAVE_OPENCL
} // namespace ocl4dnn
} // namespace dnn
} // namespce cv
#endif
