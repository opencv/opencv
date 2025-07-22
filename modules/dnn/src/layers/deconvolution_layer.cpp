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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"
#include "../op_webnn.hpp"
#include "../op_cann.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <iostream>
#include <numeric>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
using namespace cv::dnn::ocl4dnn;
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/transpose_convolution.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

#include "cpu_kernels/convolution.hpp"

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    bool fusedWeights, fusedBias;
    std::vector<double> weightsMultipliers;
    int groups;
    BaseConvolutionLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        getConvolutionKernelParams(params, kernel_size, pads_begin, pads_end, strides, dilations,
                                   padMode, adjust_pads, useWinograd);

        numOutput = -1;
        groups = params.get<int>("group", 1);

        if (kernel_size.size() == 2) {
            kernel = Size(kernel_size[1], kernel_size[0]);
            stride = Size(strides[1], strides[0]);
            pad = Size(pads_begin[1], pads_begin[0]);
            dilation = Size(dilations[1], dilations[0]);

            adjustPad.height = adjust_pads[0];
            adjustPad.width = adjust_pads[1];
        }

        for (int i = 0; i < adjust_pads.size(); i++) {
            CV_Assert(adjust_pads[i] < strides[i]);
        }

        fusedWeights = false;
        fusedBias = false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert((inputs.size() > outputs.size() && blobs.empty()) ||
                  (!inputs.empty() && (blobs.size() == 1 || blobs.size() == 2)));
        MatShape weightShape = blobs.empty() ? inputs[1].shape() : blobs[0].shape();
        numOutput = weightShape[0];

        CV_Assert(inputs[0].dims == outputs[0].dims);
        if (weightShape.dims == 3)
        {
            kernel_size.resize(1, kernel_size[0]);
            strides.resize(1, strides[0]);
            dilations.resize(1, dilations[0]);
            pads_begin.resize(1, pads_begin[0]);
            pads_end.resize(1, pads_end[0]);
        }
        CV_Assert(weightShape.dims == kernel_size.size() + 2);
        for (int i = 0; i < kernel_size.size(); i++) {
            CV_Assert(weightShape[i + 2] == kernel_size[i]);
        }

        const Mat &input = inputs[0];
        CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || input.dims == 4 || input.dims == 5) && (input.type() == CV_32F || input.type() == CV_16F));
        for (size_t i = 0; i < outputs.size(); i++)
        {
            CV_Assert(inputs[i].type() == input.type());
            CV_Assert(((input.dims == 3 && kernel_size.size() == 1) || inputs[i].dims == 4 || inputs[i].dims == 5) && inputs[i].size[1] == input.size[1]);
            for (int j = 0; j < inputs[i].dims; j++) {
                CV_Assert(inputs[i].size[j] == input.size[j]);
            }
        }

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(inpShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            pad = Size(pads_begin[1], pads_begin[0]);
        }
        fusedWeights = false;
        fusedBias = false;
    }

    bool hasBias() const
    {
        return blobs.size() >= 2;
    }

    virtual MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const = 0;
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
               (stride.height == 1 && stride.width == 1) &&
               (dilation.height == 1 && dilation.width == 1);
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        if (fusedAdd)   // If the Conv layer has fused Add layer, it cannot fuse other layers.
            return false;

        Ptr<BlankLayer> blank_layer = top.dynamicCast<BlankLayer>();
        if (blank_layer)
            return true;

        Mat w, b;
        top->getScaleShift(w, b);
        if (!w.empty() || !b.empty())
        {
            fuseWeights(w, b);
            fusedWeights = fusedWeights || !w.empty();
            fusedBias = fusedBias || (hasBias() && !w.empty()) || !b.empty();
            return true;
        }
        return false;
    }

    virtual void fuseWeights(const Mat& w_, const Mat& b_) = 0;
};

class DeConvolutionLayerImpl CV_FINAL : public BaseConvolutionLayerImpl
{
public:
    Mat weightsMat, biasesMat;
    UMat umat_weights;
    UMat umat_biases;

    DeConvolutionLayerImpl(const LayerParams& params) : BaseConvolutionLayerImpl(params) {}

    MatShape computeColRowShape(const MatShape &inpShape, const MatShape &outShape) const CV_OVERRIDE
    {
        int dims = inpShape.size();
        int inpD = dims == 5 ? inpShape[2] : 1;
        int inpH = inpShape[dims - 2];
        int inpW = inpShape.back();
        int outCn = outShape[1];
        int outGroupCn = outCn / groups;
        int ksize = outGroupCn * std::accumulate(kernel_size.begin(), kernel_size.end(),
                                                 1, std::multiplies<size_t>());
        return shape(ksize, inpD * inpH * inpW);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_CUDA)
        {
            /* only deconvolution 2d and 3d supported */
            if (kernel_size.size() == 2 || kernel_size.size() == 3)
                return true;

            return false;
        }

#ifdef HAVE_INF_ENGINE
        const int outGroupCn = blobs[0].size[1];  // Weights are in IOHW or IODHW layout
        const int group = numOutput / outGroupCn;

        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
            return group == 1;
        }
#endif  // HAVE_INF_ENGINE

        {
            return backendId == DNN_BACKEND_CUDA || backendId == DNN_BACKEND_OPENCV ||
            (kernel_size.size() == 2 && backendId == DNN_BACKEND_CANN);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() != 0);
        CV_Assert(inputs.size() > 1 || !blobs.empty());
        MatShape weightShape = blobs.empty() ? inputs[1] : blobs[0].shape();

        int outCn = numOutput;
        if (outCn < 0) {
            outCn = weightShape[1]*groups;
        }
        std::vector<int> outShape;
        outShape.push_back(inputs[0][0]);  // batch
        outShape.push_back(outCn);

        int spatialDims = kernel_size.size();
        CV_Assert(spatialDims >= 1 && spatialDims <= 3);

        int expectedDims = inputs[0].size();

        // If the network is effectively 1-D (input shape N x C x W) but the
        // parser has duplicated the kernel/stride/etc. parameters so that
        // `spatialDims == 2`, switch to the dedicated 1-D branch.
        if (expectedDims == 3)
            spatialDims = 1;

        if (spatialDims == 1) {
            CV_Assert(expectedDims == 3 || expectedDims == 4);
        } else {
            CV_Assert(expectedDims == 2 + spatialDims);
        }

        if (padMode.empty())
        {
            for (int i = 0; i < spatialDims; i++) {
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + ((kernel_size[i] - 1) * dilations[i] + 1) - pads_begin[i] - pads_end[i] + adjust_pads[i]);
            }
        }
        else if (padMode == "VALID")
        {
            for (int i = 0; i < spatialDims; i++) {
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + ((kernel_size[i] - 1) * dilations[i] + 1) + adjust_pads[i]);
            }
        }
        else if (padMode == "SAME")
        {
            for (int i = 0; i < spatialDims; i++)
                outShape.push_back(strides[i] * (inputs[0][2 + i] - 1) + 1 + adjust_pads[i]);
        }
        else
            CV_Error(Error::StsError, "Unsupported padding mode " + padMode);
        CV_Assert(outCn % weightShape[1] == 0);

        int inpCn = inputs[0][1];
        CV_Assert(inpCn % groups == 0 && outCn % groups == 0);
        CV_Assert(weightShape[0] == inpCn);

        outputs.resize(1, MatShape(outShape));

        if (!is1x1())
            internals.push_back(computeColRowShape(inputs[0], outputs[0]));

        return false;
    }

    void getTypes(const std::vector<MatType> &inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType> &outputs,
                  std::vector<MatType> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, CV_32F);
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        BaseConvolutionLayerImpl::finalize(inputs_arr, outputs_arr);

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() > 1 || !blobs.empty());

        MatShape weightShape = blobs.empty() ? inputs[1].shape() : blobs[0].shape();
        numOutput = weightShape[1]*groups;

        std::vector<int> inpShape;
        std::vector<int> outShape;
        for (int i = 2; i < inputs[0].dims; i++) {
            inpShape.push_back(inputs[0].size[i]);
            outShape.push_back(outputs[0].size[i]);
        }
        getConvPoolPaddings(outShape, kernel_size, strides, padMode, pads_begin, pads_end);
        if (pads_begin.size() == 2) {
            for (int i = 0; i < pads_begin.size(); i++) {
                if (pads_begin[i] != pads_end[i])
                    CV_Error(Error::StsNotImplemented, "Unsupported asymmetric padding in deconvolution layer");
            }
            pad = Size(pads_begin[1], pads_begin[0]);
        }

        weightsMultipliers.assign(numOutput, 1.0);

        if (weightsMat.empty() && !blobs.empty()) {
            transpose(blobs[0].reshape(1, blobs[0].size[0]), weightsMat);
        }

        if (biasesMat.empty() && blobs.size() >= 2) {
            biasesMat = blobs[1].reshape(1, numOutput);
        }
    }

    void fuseWeights(const Mat& w_, const Mat& b_) CV_OVERRIDE
    {
        Mat w = w_.total() == 1 ? Mat(1, numOutput, CV_32F, Scalar(w_.at<float>(0))) : w_;
        Mat b = b_.total() == 1 ? Mat(1, numOutput, CV_32F, Scalar(b_.at<float>(0))) : b_;

        CV_Assert_N(!weightsMat.empty(),
                     w.empty() || numOutput == w.total(),
                     b.empty() || numOutput == b.total());

        if (!w.empty())
        {
            transpose(blobs[0].reshape(1, blobs[0].size[0]), weightsMat);
            weightsMat = weightsMat.reshape(1, numOutput);
            for (int i = 0; i < numOutput; ++i)
            {
                double wi = w.at<float>(i);
                weightsMultipliers[i] *= wi;
                cv::multiply(weightsMat.row(i), weightsMultipliers[i], weightsMat.row(i));
                biasesMat.at<float>(i) *= wi;
            }
            weightsMat = weightsMat.reshape(1, weightsMat.total() / blobs[0].size[0]);
        }

        if (!b.empty())
        {
            cv::add(biasesMat, b.reshape(1, numOutput), biasesMat);
        }
    }

    class MatMulInvoker : public ParallelLoopBody
    {
    public:
        MatMulInvoker(const Mat& a, const Mat& b, Mat& c, int nstripes)
        {
            a_ = &a;
            b_ = &b;
            c_ = &c;
            nstripes_ = nstripes;
            useAVX = checkHardwareSupport(CPU_AVX);
            useAVX2 = checkHardwareSupport(CPU_AVX2);
            useAVX512 = CV_CPU_HAS_SUPPORT_AVX512_SKX;
            useRVV = checkHardwareSupport(CPU_RVV);
            useLASX = checkHardwareSupport(CPU_LASX);
        }

        void operator()(const Range& range_) const CV_OVERRIDE
        {
            int stripeSize = (int)alignSize((b_->cols + nstripes_ - 1)/nstripes_, 16);
            Range range(range_.start*stripeSize, std::min(range_.end*stripeSize, b_->cols));
            int mmax = a_->rows;
            int nmax = range.end - range.start;
            int kmax = a_->cols;
            int m, n, k;
            const float* aptr = a_->ptr<float>();
            const float* bptr = b_->ptr<float>() + range.start;
            float* cptr = c_->ptr<float>() + range.start;
            size_t astep = a_->step1();
            size_t bstep = b_->step1();
            size_t cstep = c_->step1();

        #if CV_TRY_AVX512_SKX
            if( useAVX512 )
                opt_AVX512_SKX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_AVX2
            if( useAVX2 )
                opt_AVX2::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_AVX
            if( useAVX )
                opt_AVX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
        #if CV_TRY_RVV && CV_RVV
            if( useRVV ) {
                opt_RVV::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            }
            else
        #endif
        #if CV_TRY_LASX
            if( useLASX )
                opt_LASX::fastGEMM( aptr, astep, bptr, bstep, cptr, cstep, mmax, kmax, nmax );
            else
        #endif
            for( m = 0; m < mmax; m += 2 )
            {
                float* dst0 = cptr + cstep*m;
                float* dst1 = cptr + cstep*std::min(m+1, mmax-1);
                const float* aptr0 = aptr + astep*m;
                const float* aptr1 = aptr + astep*std::min(m+1, mmax-1);

                for( n = 0; n < nmax; n++ )
                {
                    dst0[n] = 0.f;
                    dst1[n] = 0.f;
                }

                for( k = 0; k < kmax; k += 4 )
                {
                    float alpha00 = aptr0[k];
                    float alpha01 = aptr1[k];
                    float alpha10 = 0.f, alpha11 = 0.f;
                    float alpha20 = 0.f, alpha21 = 0.f;
                    float alpha30 = 0.f, alpha31 = 0.f;
                    const float* bptr0 = bptr + k*bstep;
                    const float* bptr1 = bptr0;
                    const float* bptr2 = bptr0;
                    const float* bptr3 = bptr0;

                    if( k+1 < kmax )
                    {
                        alpha10 = aptr0[k+1];
                        alpha11 = aptr1[k+1];
                        bptr1 = bptr0 + bstep;
                        if( k+2 < kmax )
                        {
                            alpha20 = aptr0[k+2];
                            alpha21 = aptr1[k+2];
                            bptr2 = bptr1 + bstep;
                            if( k+3 < kmax )
                            {
                                alpha30 = aptr0[k+3];
                                alpha31 = aptr1[k+3];
                                bptr3 = bptr2 + bstep;
                            }
                        }
                    }
                    n = 0;

                #if CV_SIMD128
                    v_float32x4 a00 = v_setall_f32(alpha00);
                    v_float32x4 a01 = v_setall_f32(alpha01);
                    v_float32x4 a10 = v_setall_f32(alpha10);
                    v_float32x4 a11 = v_setall_f32(alpha11);
                    v_float32x4 a20 = v_setall_f32(alpha20);
                    v_float32x4 a21 = v_setall_f32(alpha21);
                    v_float32x4 a30 = v_setall_f32(alpha30);
                    v_float32x4 a31 = v_setall_f32(alpha31);

                    for( ; n <= nmax - 4; n += 4 )
                    {
                        v_float32x4 d0 = v_load(dst0 + n);
                        v_float32x4 d1 = v_load(dst1 + n);
                        v_float32x4 b0 = v_load(bptr0 + n);
                        v_float32x4 b1 = v_load(bptr1 + n);
                        v_float32x4 b2 = v_load(bptr2 + n);
                        v_float32x4 b3 = v_load(bptr3 + n);
                        // TODO try to improve pipeline width
                        d0 = v_fma(b0, a00, d0);
                        d1 = v_fma(b0, a01, d1);
                        d0 = v_fma(b1, a10, d0);
                        d1 = v_fma(b1, a11, d1);
                        d0 = v_fma(b2, a20, d0);
                        d1 = v_fma(b2, a21, d1);
                        d0 = v_fma(b3, a30, d0);
                        d1 = v_fma(b3, a31, d1);
                        v_store(dst0 + n, d0);
                        v_store(dst1 + n, d1);
                    }
                #endif

                    for( ; n < nmax; n++ )
                    {
                        float b0 = bptr0[n];
                        float b1 = bptr1[n];
                        float b2 = bptr2[n];
                        float b3 = bptr3[n];
                        float d0 = dst0[n] + alpha00*b0 + alpha10*b1 + alpha20*b2 + alpha30*b3;
                        float d1 = dst1[n] + alpha01*b0 + alpha11*b1 + alpha21*b2 + alpha31*b3;
                        dst0[n] = d0;
                        dst1[n] = d1;
                    }
                }
            }
        }

        const Mat *a_, *b_;
        Mat* c_;
        int nstripes_;
        bool useAVX;
        bool useAVX2;
        bool useAVX512;
        bool useRVV;
        bool useLASX;
    };

    class Col2ImInvoker : public cv::ParallelLoopBody
    {
    public:
        const float* data_col;
        const float* biasvec;
        int channels;
        std::vector<int> output_shape;  // spatial dimensions only
        std::vector<int> kernel_shape;
        std::vector<int> pads;
        std::vector<int> strides;
        std::vector<int> dilations;
        std::vector<int> input_shape;   // spatial dimensions only
        float* data_im;
        int nstripes;
        bool is1x1;

        Col2ImInvoker()
            : data_col(0), biasvec(0), channels(0), data_im(0),
              nstripes(0), is1x1(0)
        {}

        static void run(const float* data_col,
                        int channels,
                        const std::vector<int>& output_shape,
                        const std::vector<int>& kernel_shape,
                        const std::vector<int>& pads,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& input_shape,
                        float* data_im,
                        const float* biasvec,
                        bool is1x1)
        {
            const int nstripes = getNumThreads();

            Col2ImInvoker t;
            t.data_col = data_col;
            t.data_im = data_im;
            t.channels = channels;
            t.output_shape = output_shape;
            t.kernel_shape = kernel_shape;
            t.pads = pads;
            t.strides = strides;
            t.dilations = dilations;
            t.input_shape = input_shape;
            t.nstripes = nstripes;
            t.is1x1 = is1x1;
            t.biasvec = biasvec;

            parallel_for_(Range(0, nstripes), t, nstripes);
        }

        virtual void operator ()(const Range &r) const CV_OVERRIDE
        {
            const float* data_col_ = data_col;
            float* data_im_ = data_im;
            bool is1x1_ = is1x1;
            const float* biasvec_ = biasvec;

            int ndims = output_shape.size();

            // Calculate total output size
            int total_output_size = channels;
            int input_spatial_size = 1;
            for (int i = 0; i < ndims; i++) {
                total_output_size *= output_shape[i];
                input_spatial_size *= input_shape[i];
            }

            size_t stripeSize = (total_output_size + nstripes - 1) / nstripes;
            size_t startIndex = r.start * stripeSize;
            size_t endIndex = std::min(r.end * stripeSize, (size_t)total_output_size);

            for (size_t index = startIndex; index < endIndex; index++)
            {
                // Convert linear index to multi-dimensional coordinates
                std::vector<int> coords(ndims + 1);  // +1 for channel dimension
                size_t idx = index;

                // Extract spatial coordinates and channel
                for (int i = ndims - 1; i >= 0; i--) {
                    coords[i + 1] = idx % output_shape[i];
                    idx /= output_shape[i];
                }
                coords[0] = idx;  // channel

                float val = 0.0f;

                if( is1x1_ )
                    val = data_im_[index];
                else {
                    std::vector<int> kernel_coords(ndims);
                    std::function<void(int)> iterate_kernel = [&](int dim) {
                        if (dim == ndims) {
                            std::vector<int> input_coords(ndims);
                            bool valid = true;

                            for (int i = 0; i < ndims; i++) {
                                // Apply dilation to kernel coordinates
                                int dilated_kernel_pos = kernel_coords[i] * dilations[i];
                                input_coords[i] = coords[i + 1] + pads[i] - dilated_kernel_pos;
                                if (input_coords[i] < 0 || input_coords[i] % strides[i] != 0) {
                                    valid = false;
                                    break;
                                }
                                input_coords[i] /= strides[i];
                                if (input_coords[i] >= input_shape[i]) {
                                    valid = false;
                                    break;
                                }
                            }

                            if (valid) {
                                // Calculate offset in column matrix
                                int col_offset = coords[0];  // channel
                                for (int i = 0; i < ndims; i++) {
                                    col_offset = col_offset * kernel_shape[i] + kernel_coords[i];
                                }
                                col_offset *= input_spatial_size;

                                // Calculate input position in flattened input
                                int input_pos = 0;
                                for (int i = 0; i < ndims; i++) {
                                    input_pos = input_pos * input_shape[i] + input_coords[i];
                                }

                                val += data_col_[col_offset + input_pos];
                            }
                        } else {
                            for (int k = 0; k < kernel_shape[dim]; k++) {
                                kernel_coords[dim] = k;
                                iterate_kernel(dim + 1);
                            }
                        }
                    };

                    iterate_kernel(0);
                }
                data_im_[index] = val + biasvec_[coords[0]];
            }
        }
    };

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        if (inputs_.depth() == CV_16F)
            return false;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        internals_.getUMatVector(internals);

        int outCn = numOutput;
        int inpCn = inputs[0].size[1];

        if (is1x1())
            return false;

        if (umat_weights.empty() || inputs.size() >= 2) {
            Mat temp;
            if (fusedWeights)
                weightsMat.copyTo(umat_weights);
            else if (!blobs.empty()) {
                transpose(blobs[0].reshape(1, inpCn), temp);
                temp.copyTo(umat_weights);
            }
            else {
                transpose(inputs[1].reshape(1, inpCn), temp);
                temp.copyTo(umat_weights);
            }
        }

        if (umat_biases.empty() || inputs.size() >= 3) {
            if (fusedBias)
                biasesMat.copyTo(umat_biases);
            else if (blobs.size() > 1)
                blobs[1].reshape(1, outCn).copyTo(umat_biases);
            else if (inputs.size() >= 3)
                inputs[2].reshape(1, outCn).copyTo(umat_biases);
            else
                umat_biases = UMat::zeros(outCn, 1, CV_32F);
        }

        String buildopt = format("-DT=%s ", ocl::typeToStr(inputs[0].type()));
        buildopt += format("-DPAD_H=%d -DPAD_W=%d -DKERNEL_H=%d -DKERNEL_W=%d -DSTRIDE_H=%d -DSTRIDE_W=%d ",
                           pad.height, pad.width, kernel.height, kernel.width, stride.height, stride.width);

        //for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int ii = 0;
            int inpGroupCn = inpCn / groups;
            int outGroupCn = outCn / groups;
            const UMat& inp = inputs[ii];
            UMat& out = outputs[ii];
            int numImg = inp.size[0];
            int inpH = inp.size[2], inpW = inp.size[3];
            int outH = out.size[2], outW = out.size[3];

            MatShape inpshape = shape(numImg*inpCn, inpH*inpW);
            MatShape outshape = shape(numImg*outCn, outH*outW);
            UMat convBlob = inputs[ii].reshape(1, inpshape);
            UMat decnBlob = out.reshape(1, outshape);
            int rows = internals[0].rows / groups;

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < groups; g++)
                {
                    UMat colMat = internals[0].rowRange(_Range(g * rows, rows));
                    UMat convMat = convBlob.rowRange(_Range((g + n * groups) * inpGroupCn, inpGroupCn));
                    UMat wghtMat = umat_weights.colRange(_Range(g * inpGroupCn, inpGroupCn));
                    gemm(wghtMat, convMat, 1, noArray(), 0, colMat, 0);
                }

                for (int g = 0; g < groups; g++)
                {
                    int total = outGroupCn * decnBlob.cols;
                    int index = 0;
                    int height_col = inpH;
                    int width_col = inpW;
                    int coeff_h = (1 - stride.height * kernel.width * height_col) * width_col;
                    int coeff_w = (1 - stride.width * height_col * width_col);

                    ocl::Kernel k("col2im", ocl::dnn::col2im_oclsrc, buildopt);
                    k.set(index++, total);
                    k.set(index++, ocl::KernelArg::PtrReadOnly(internals[0]));
                    k.set(index++, (int)(g * rows * internals[0].cols));
                    k.set(index++, outGroupCn);
                    k.set(index++, outH);
                    k.set(index++, outW);
                    k.set(index++, height_col);
                    k.set(index++, width_col);
                    k.set(index++, coeff_h);
                    k.set(index++, coeff_w);
                    k.set(index++, ocl::KernelArg::PtrReadOnly(umat_biases));
                    k.set(index++, (int)(g * outGroupCn * umat_biases.cols));
                    k.set(index++, ocl::KernelArg::PtrWriteOnly(decnBlob));
                    k.set(index++, (int)((g + n * groups) * outGroupCn * decnBlob.cols));

                    size_t global[] = { (size_t)total };
                    bool ret = k.run(1, global, NULL, false);
                    if (!ret)
                        return false;
                }
            }
        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        // For some reason, tests for deconvolution fail;
        // Also, the current implementation is super-inefficient,
        // Just disabled it. Need to rewrite it and then uncomment back these lines
        //CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
        //           forward_ocl(inputs_arr, outputs_arr, internals_arr));

        if (inputs_arr.depth(0) == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        auto kind = outputs_arr.kind();
        std::vector<Mat> inputs, internals;
        inputs_arr.getMatVector(inputs);
        internals_arr.getMatVector(internals);

        int outCn = numOutput;
        int inpCn = inputs[0].size[1];
        bool is1x1flag = is1x1();
        int nstripes = getNumThreads();
        /*CV_Assert(outputs.size() == 1);
        CV_Assert(inputs[0].size[0] == outputs[0].size[0]);
        CV_Assert(outCn == outputs[0].size[1]);*/

        if (weightsMat.empty() || inputs.size() >= 2) {
            Mat inpWeights = !blobs.empty() ? blobs[0] : inputs[1];
            transpose(inpWeights.reshape(1, inpCn), weightsMat);
        }

        if (biasesMat.empty() || inputs.size() >= 3) {
            Mat inpBias = blobs.size() >= 2 ? blobs[1] : inputs.size() >= 3 ? inputs[2] : Mat();
            Mat biasesMat_ = !inpBias.empty() ? inpBias.reshape(1, outCn) : Mat::zeros(outCn, 1, CV_32F);
            biasesMat_.copyTo(biasesMat);
        }

        /*printf("DeConvolution Input: ");
        pprint(std::cout, inputs[0], 0, 3, 100, '[');
        printf("\nDeConvolution Weights: ");
        pprint(std::cout, weightsMat, 0, 3, 100, '[');
        printf("\nDeConvolution Bias: ");
        pprint(std::cout, biasesMat, 0, 3, 100, '[');
        printf("\n");*/

        //for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            int ii = 0;
            int inpGroupCn = inpCn / groups;
            int outGroupCn = outCn / groups;
            const Mat& inp = inputs[ii];
            MatShape outshape = outputs_arr.shape(0);
            CV_Assert(outshape.dims == inp.dims);
            CV_Assert(outshape[0] == inp.size[0]);
            CV_Assert(outshape[1] == outCn);
            Mat out;
            if (kind == _InputArray::STD_VECTOR_MAT) {
                out = outputs_arr.getMat(0);
            }
            else {
                out.create(outshape, inp.type());
            }
            int numImg = inp.size[0];
            int spatialDims = kernel_size.size();

            std::vector<int> inpSpatialShape(spatialDims);
            std::vector<int> outSpatialShape(spatialDims);

            if (spatialDims == 1) {
                // For 1D convolution, OpenCV can represent it as 3D (N, C, W) or 4D (N, C, 1, W)
                if (inp.dims == 3) {
                    // 3D case: (N, C, W)
                    inpSpatialShape[0] = inp.size[2];
                    outSpatialShape[0] = out.size[2];
                } else {
                    // 4D case: (N, C, 1, W)
                    inpSpatialShape[0] = inp.size[3];
                    outSpatialShape[0] = out.size[3];
                }
            } else {
                for (int i = 0; i < spatialDims; i++) {
                    inpSpatialShape[i] = inp.size[2 + i];
                    outSpatialShape[i] = out.size[2 + i];
                }
            }

            Mat convBlob = inputs[ii].reshape(1, numImg*inpCn);
            Mat decnBlob = out.reshape(1, numImg*outCn);

            for (int n = 0; n < numImg; n++)
            {
                for (int g = 0; g < groups; g++)
                {
                    Mat dstMat = decnBlob.rowRange(_Range((g + n * groups) * outGroupCn, outGroupCn));
                    Mat &colMat = is1x1flag ? dstMat : internals[0];

                    Mat convMat = convBlob.rowRange(_Range((g + n * groups) * inpGroupCn, inpGroupCn));
                    Mat wghtMat = weightsMat.colRange(_Range(g * inpGroupCn, inpGroupCn));
                    Mat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));

                    //gemm(wghtMat, convMat, 1, colMat, 0, colMat, 0);
                    MatMulInvoker mminvoker(wghtMat, convMat, colMat, nstripes);
                    parallel_for_(Range(0, nstripes), mminvoker, nstripes);

                    std::vector<int> kernel_shape_int(kernel_size.begin(), kernel_size.end());
                    std::vector<int> pads_int(pads_begin.begin(), pads_begin.end());
                    std::vector<int> strides_int(strides.begin(), strides.end());
                    std::vector<int> dilations_int(dilations.begin(), dilations.end());

                    Col2ImInvoker::run(colMat.ptr<float>(), outGroupCn, outSpatialShape,
                                       kernel_shape_int, pads_int, strides_int, dilations_int, inpSpatialShape,
                                       dstMat.ptr<float>(), curBiasMat.ptr<float>(), is1x1flag);
                }
            }
            if (kind == _InputArray::STD_VECTOR_UMAT) {
                std::vector<UMat>& u_outputs = outputs_arr.getUMatVecRef();
                out.copyTo(u_outputs[0]);
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        CV_Assert(!blobs.empty());
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(inputs.size() == 1);
        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();

        CV_Assert(outputs.size() == 1);
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapper>();
        auto output_shape = output_wrapper->getShape();

        const auto output_feature_maps = numOutput;
        const auto output_feature_maps_per_group = blobs[0].size[1];
        const auto groups = output_feature_maps / output_feature_maps_per_group;

        TransposeConvolutionConfiguration config;

        config.kernel_size.assign(std::begin(kernel_size), std::end(kernel_size));
        config.dilations.assign(std::begin(dilations), std::end(dilations));
        config.strides.assign(std::begin(strides), std::end(strides));

        if (padMode.empty())
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::MANUAL;
            config.pads_begin.assign(std::begin(pads_begin), std::end(pads_begin));
            config.pads_end.assign(std::begin(pads_end), std::end(pads_end));
        }
        else if (padMode == "VALID")
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::VALID;
        }
        else if (padMode == "SAME")
        {
            config.padMode = TransposeConvolutionConfiguration::PaddingMode::SAME;
        }
        else
        {
            CV_Error(Error::StsNotImplemented, padMode + " padding mode not supported by DeconvolutionLayer");
        }

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.output_shape.assign(std::begin(output_shape), std::end(output_shape));
        config.groups = groups;

        CV_Assert(blobs.size() >= 1);
        Mat filtersMat = fusedWeights ? weightsMat.t() : blobs[0];
        Mat biasMat = (hasBias() || fusedBias) ? biasesMat : Mat();
        if (countNonZero(biasMat) == 0)
            biasMat = Mat();

        return make_cuda_node<cuda4dnn::TransposeConvolutionOp>(
            preferableTarget, std::move(context->stream), std::move(context->cudnn_handle), config, filtersMat, biasMat);
    }
#endif

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        CV_Assert(inputs.size() == 1);
        CV_Assert(nodes.size() == 1);

        bool has_bias = hasBias() || fusedBias;

        auto x = inputs[0].dynamicCast<CannBackendWrapper>();
        auto y = outputs[0].dynamicCast<CannBackendWrapper>();
        const auto shape_x = x->host->size; // [N, C, H, W]
        const auto shape_y = y->host->size; // [N, C, H, W]
        const int filter_out_channel = blobs[0].size[0];
        const int groups = shape_x[1] / filter_out_channel;

        // create operator
        auto op = std::make_shared<ge::op::Conv2DTransposeD>(name);

        // set attributes
        op->set_attr_input_size(
            ge::Operator::OpListInt({(int64_t)shape_y[0],
                                     (int64_t)shape_y[1],
                                     (int64_t)shape_y[2],
                                     (int64_t)shape_y[3],})
        );
        op->set_attr_strides(
            ge::Operator::OpListInt({1, 1, (int64_t)strides[0], (int64_t)strides[1]})
        );
        op->set_attr_pads(ge::Operator::OpListInt(
            {(int64_t)pads_begin[1], (int64_t)pads_end[1], (int64_t)pads_begin[0], (int64_t)pads_end[0]}
        ));
        op->set_attr_dilations(ge::Operator::OpListInt(
            {1, 1, (int64_t)dilations[0], (int64_t)dilations[1]}
        ));
        op->set_attr_groups(groups);
        op->set_attr_data_format("NCHW");
        op->set_attr_output_padding(
            ge::Operator::OpListInt({0, 0, (int64_t)adjust_pads[0], (int64_t)adjust_pads[1]}) // adjust_pads: [height, width]
        );

        // set inputs
        // set inputs : x
        auto op_x = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        op->set_input_x_by_name(*op_x, x->name.c_str());
        auto desc_x = x->getTensorDesc();
        op->update_input_desc_x(*desc_x);
        // set inputs : weight
        const Mat& mat_w = blobs[0];
        auto op_const_w = std::make_shared<CannConstOp>(mat_w.data, mat_w.type(), shape(mat_w), cv::format("%s_w", name.c_str()));
        op->set_input_filter(*(op_const_w->getOp()));
        op->update_input_desc_filter(*(op_const_w->getTensorDesc()));
        // set inputs : bias
        if (has_bias)
        {
            int out_channel = blobs[0].size[0];
            const Mat& mat_b = blobs[1];

            std::vector<int> shape_b{out_channel};
            auto op_const_b = std::make_shared<CannConstOp>(mat_b.data, mat_b.type(), shape_b, cv::format("%s_b", name.c_str()));
            op->set_input_bias(*(op_const_b->getOp()));
            op->update_input_desc_bias(*(op_const_b->getTensorDesc()));
        }

        // set outputs
        auto desc_output = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*desc_output);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> > &inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
       CV_Assert(!blobs.empty());
       const int outGroupCn = blobs[0].size[1];
       const int group = numOutput / outGroupCn;
       CV_Assert(group == 1);

       auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
       std::vector<size_t> kernel_shape = getShape<size_t>(blobs[0]);
       auto ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, blobs[0].data);

        if (fusedWeights)
        {
            Mat newWeights;
            transpose(weightsMat, newWeights);
            ieWeights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, kernel_shape, newWeights.data);
        }
        std::vector<size_t> paddings_end;
        if (padMode == "SAME")
        {
            for (int i = 0; i < pads_begin.size(); i++) {
                paddings_end.push_back(kernel_size[i] - pads_begin[i] - 1 - adjust_pads[i]);
            }
            adjust_pads = std::vector<size_t>(pads_begin.size(), 0);
        } else {
            paddings_end = pads_end;
        }
        ov::op::PadType pad_type = padMode == "VALID" ? ov::op::PadType::VALID : ov::op::PadType::EXPLICIT;

        auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                          ieInpNode,
                          ieWeights,
                          ov::Strides(strides),
                          ov::CoordinateDiff(std::vector<std::ptrdiff_t>(pads_begin.begin(), pads_begin.end())),
                          ov::CoordinateDiff(std::vector<std::ptrdiff_t>(paddings_end.begin(), paddings_end.end())),
                          ov::Strides(dilations),
                          pad_type,
                          ov::CoordinateDiff(std::vector<std::ptrdiff_t>(adjust_pads.begin(), adjust_pads.end())));

        if (hasBias() || fusedBias)
        {
            std::vector<size_t> shape(deconv->get_shape().size(), 1);
            shape[1] = numOutput;
            auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape(shape), blobs[1].data);
            auto deconv_bias = std::make_shared<ov::op::v1::Add>(deconv, bias, ov::op::AutoBroadcastType::NUMPY);
            return Ptr<BackendNode>(new InfEngineNgraphNode(deconv_bias));
        }


        return Ptr<BackendNode>(new InfEngineNgraphNode(deconv));
    }
#endif  // HAVE_DNN_NGRAPH

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == outputs.size());

        float flops = 0;
        int outChannels = blobs[0].size[0];
        size_t karea = std::accumulate(kernel_size.begin(), kernel_size.end(),
                                       1, std::multiplies<size_t>());

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += CV_BIG_INT(2)*outChannels*karea*total(inputs[i]);
        }

        return flops;
    }
};

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(const LayerParams &params)
{
    return Ptr<BaseConvolutionLayer>(new DeConvolutionLayerImpl(params));
}

}
}
