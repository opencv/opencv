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
#include "op_halide.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencl_kernels_dnn.hpp"
#include <algorithm>

#ifdef HAVE_OPENCL
using namespace cv::dnn::ocl4dnn;
#endif

namespace cv
{
namespace dnn
{

class LRNLayerImpl : public LRNLayer
{
public:
    LRNLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        type = -1;
        String nrmType = params.get<String>("norm_region", "ACROSS_CHANNELS");
        if (nrmType == "ACROSS_CHANNELS")
            type = LRNLayer::CHANNEL_NRM;
        else if (nrmType == "WITHIN_CHANNEL")
            type = LRNLayer::SPATIAL_NRM;
        else
            CV_Error(Error::StsBadArg, "Unknown region type \"" + nrmType + "\"");

        size = params.get<int>("local_size", 5);
        if (size % 2 != 1 || size <= 0)
            CV_Error(Error::StsBadArg, "LRN layer supports only positive odd values for local_size");

        alpha = params.get<double>("alpha", 1);
        beta = params.get<double>("beta", 0.75);
        bias = params.get<double>("bias", 1);
        normBySize = params.get<bool>("norm_by_size", true);
    }

#ifdef HAVE_OPENCL
    Ptr<OCL4DNNLRN<float> > lrnOp;
#endif

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide();
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        if (lrnOp.empty())
        {
            OCL4DNNLRNConfig config;
            config.lrn_type = type == CHANNEL_NRM ?
                              LRNParameter_NormRegion_ACROSS_CHANNELS :
                              LRNParameter_NormRegion_WITHIN_CHANNEL;

            CHECK_EQ(size % 2, 1)<< "LRN only supports odd values for local_size";
            config.local_size = size;
            config.alpha = alpha;
            config.beta = beta;
            config.k = bias;
            CHECK_EQ(4, inputs[0]->dims) << "Input must have 4 axes, "
                     << "corresponding to (num, channels, height, width)";
            config.batch_size = inputs[0]->size[0];
            config.channels = inputs[0]->size[1];
            config.height = inputs[0]->size[2];
            config.width = inputs[0]->size[3];
            config.norm_by_size = normBySize;

            lrnOp = Ptr<OCL4DNNLRN<float> >(new OCL4DNNLRN<float>(config));
        }

        UMat inpMat, outMat;
        inpMat = inputs[0]->getUMat(ACCESS_READ);
        outMat = outputs[0].getUMat(ACCESS_WRITE);

        if (!lrnOp->Forward(inpMat, outMat))
            return false;

        return true;
    }
#endif

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(inputs.size() == outputs.size());

        CV_OCL_RUN((preferableTarget == DNN_TARGET_OPENCL) &&
                   OCL_PERFORMANCE_CHECK(ocl::Device::getDefault().isIntel()),
                   forward_ocl(inputs, outputs, internals))

        for (int i = 0; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->dims == 4);

            Mat &src = *inputs[i];
            Mat &dst = outputs[i];

            switch (type)
            {
                case CHANNEL_NRM:
                    channelNormalization(src, dst);
                    break;
                case SPATIAL_NRM:
                    spatialNormalization(src, dst);
                    break;
                default:
                    CV_Error(Error::StsNotImplemented, "Unimplemented mode of LRN layer");
                    break;
            }
        }
    }

    class ChannelLRN : public ParallelLoopBody
    {
    public:
        ChannelLRN(const float* src, float* dst, int channels, int ksize,
                   float alpha1, float bias1, float beta1,
                   size_t planeSize, int nsamples, int nstripes)
        {
            src_ = src; dst_ = dst;
            channels_ = channels;
            ksize_ = ksize;
            alpha1_ = alpha1; bias1_ = bias1; beta1_ = beta1;
            planeSize_ = planeSize; nsamples_ = nsamples; nstripes_ = nstripes;
        }

        void operator()(const Range& r) const
        {
            int nsamples = nsamples_, nstripes = nstripes_;
            size_t planeSize = planeSize_, planeSize_n = planeSize * nsamples;
            size_t elemsPerStripe = (planeSize_n + nstripes - 1)/nstripes;
            size_t rstart = r.start*elemsPerStripe;
            size_t rend = r.end == nstripes ? planeSize_n : r.end*elemsPerStripe;
            rstart = std::min(rstart, planeSize_n);
            rend = std::min(rend, planeSize_n);
            float alpha1 = alpha1_, bias1 = bias1_, beta1 = beta1_;
            int k, channels = channels_, ksize = ksize_;

            AutoBuffer<float> buf_((channels + ksize*2 + 4)*2);
            float* acc = (float*)buf_;
            float* buf = acc + channels + ksize + 1;
            for( k = 0; k <= ksize; k++ )
                buf[-k-1] = buf[channels + k] = 0.f;

            for( size_t ofs = rstart; ofs < rend; )
            {
                int sampleIdx = (int)(ofs/planeSize);
                if( sampleIdx >= nsamples )
                    break;
                size_t ofs0 = ofs - sampleIdx*planeSize;
                size_t ofs1 = std::min(planeSize - ofs0, rend - ofs) + ofs;
                const float* src = src_ + sampleIdx*planeSize*channels + ofs0;
                float* dst = dst_ + sampleIdx*planeSize*channels + ofs0;

                for( ; ofs < ofs1; ofs++, src++, dst++ )
                {
                    for( k = 0; k < channels; k++ )
                        buf[k] = src[k*planeSize];
                    float s = 0;
                    for( k = 0; k < ksize; k++ )
                        s += buf[k]*buf[k];
                    for( k = 0; k < channels; k++ )
                    {
                        float x1 = buf[k + ksize];
                        float x0 = buf[k - ksize - 1];
                        s = std::max(s + (x1 + x0)*(x1 - x0), 0.f);
                        acc[k] = (float)(alpha1*s + bias1);
                    }

                    hal::log32f(acc, acc, channels);
                    for( k = 0; k < channels; k++ )
                        acc[k] *= beta1;
                    hal::exp32f(acc, acc, channels);

                    for( k = 0; k < channels; k++ )
                        dst[k*planeSize] = buf[k]*acc[k];
                }
            }
        }

        const float* src_;
        float* dst_;
        float alpha1_, bias1_, beta1_;
        size_t planeSize_;
        int channels_, ksize_, nsamples_, nstripes_;
    };

    void channelNormalization(Mat &srcBlob, Mat &dstBlob)
    {
        int num = srcBlob.size[0];
        int channels = srcBlob.size[1];
        int ksize = (size - 1) / 2;
        int sizeNormFactor = normBySize ? size : 1;
        size_t planeSize = srcBlob.size[2]*srcBlob.size[3];

        int nstripes = std::max(getNumThreads(), 1);

        ChannelLRN clrn(srcBlob.ptr<float>(), dstBlob.ptr<float>(), channels,
                        ksize, alpha/sizeNormFactor, bias, -beta, planeSize, num, nstripes);
        parallel_for_(Range(0, nstripes), clrn, nstripes);
    }

    void sqrBoxFilter_(const Mat &src, Mat &dst)
    {
        Mat srcRawWrapper(src.rows, src.cols, src.type(), src.data, src.step[0]);
        cv::sqrBoxFilter(srcRawWrapper, dst, dst.depth(), Size(size, size), Point(-1, -1), false, BORDER_CONSTANT);
    }

    void spatialNormalization(Mat &srcBlob, Mat &dstBlob)
    {
        int num = srcBlob.size[0];
        int channels = srcBlob.size[1];
        int sizeNormFactor = normBySize ? size*size : 1;

        Mat srcMat = srcBlob;
        Mat dstMat = dstBlob;

        for (int n = 0; n < num; n++)
        {
            for (int cn = 0; cn < channels; cn++)
            {
                Mat src = getPlane(srcMat, n, cn);
                Mat dst = getPlane(dstMat, n, cn);

                sqrBoxFilter_(src, dst);

                dst.convertTo(dst, dst.type(), alpha/sizeNormFactor, bias);
                cv::pow(dst, beta, dst);
                cv::divide(src, dst, dst);
            }
        }
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        float alphaSize = alpha;
        if (normBySize)
            alphaSize /= (type == CHANNEL_NRM ? size : size * size);
        int width, height, channels, numImgs;
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        getCanonicalSize(inputBuffer, &width, &height, &channels, &numImgs);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Func padded_sq(name + "_padded_sq");
        Halide::Func sq("sq");
        sq(x, y, c, n) = inputBuffer(x, y, c, n) * inputBuffer(x, y, c, n);

        Halide::Func bounded =
            Halide::BoundaryConditions::constant_exterior(sq, 0, 0, width,
                                                          0, height,
                                                          0, channels,
                                                          0, numImgs);
        padded_sq(x, y, c, n) = bounded(x, y, c, n);

        Halide::Expr base;
        if (type == CHANNEL_NRM)
        {
            Halide::RDom r((1 - size) / 2, size);
            base = alphaSize * sum(padded_sq(x, y, c + r, n));
        }
        else  // SPATIAL_NRM
        {
            Halide::RDom r((1 - size) / 2, size, (1 - size) / 2, size);
            base = alphaSize * sum(padded_sq(x + r.x, y + r.y, c, n));
        }
        base += static_cast<float>(bias);
        top(x, y, c, n) = inputBuffer(x, y, c, n) / pow(base, beta);
        return Ptr<BackendNode>(new HalideBackendNode({ padded_sq, top }));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

    virtual void applyHalideScheduler(Ptr<BackendNode>& node,
                                      const std::vector<Mat*> &inputs,
                                      const std::vector<Mat> &outputs,
                                      int targetId) const
    {
#ifdef  HAVE_HALIDE
        if (targetId != DNN_TARGET_CPU)
        {
            Layer::applyHalideScheduler(node, inputs, outputs, targetId);
            return;
        }
        int outW, outH, outC, outN;
        getCanonicalSize(outputs[0].size, &outW, &outH, &outC, &outN);

        Halide::Var x("x"), y("y"), c("c"), n("n"), yo("yo"), yi("yi"), tile("tile");
        Halide::Func& top = node.dynamicCast<HalideBackendNode>()->funcs[1];
        Halide::Func& padded_sq = node.dynamicCast<HalideBackendNode>()->funcs[0];

        if (outW < 8 || outH <= 2)
            return;

        top.reorder(x, c, y, n)
           .split(y, yo, yi, 2)
           .fuse(yo, n, tile)
           .parallel(tile)
           .unroll(yi)
           .vectorize(x, 8);
        padded_sq.store_at(top, tile)
                 .compute_at(top, yi);
#endif  // HAVE_HALIDE
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        CV_Assert(inputs.size() > 0);
        long flops = 0;

        for(int i = 0; i < inputs.size(); i++)
        {
            if (type == CHANNEL_NRM)
            {
                int channels = inputs[i][1];
                int ksize = (size - 1) / 2;

                flops += inputs[i][0]*(std::min(ksize, channels)*2*total(inputs[i], 2) + channels*4*total(inputs[i], 2));

                if (ksize < channels)
                {
                    flops += (size + 2*(channels - size))*total(inputs[i], 2);
                }
            }
            else
            {
                flops += total(inputs[i])*(2*size*size + 2);
            }
        }
        return flops;
    }
};

Ptr<LRNLayer> LRNLayer::create(const LayerParams& params)
{
    return Ptr<LRNLayer>(new LRNLayerImpl(params));
}

}
}
