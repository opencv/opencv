// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/

#include "../precomp.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv
{
namespace dnn
{

class BatchNormLayerImpl CV_FINAL : public BatchNormLayer
{
public:
    Mat weights_, bias_;
    UMat umat_weight, umat_bias;

    BatchNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() >= 2);

        hasWeights = params.get<bool>("has_weight", false);
        hasBias = params.get<bool>("has_bias", false);
        useGlobalStats = params.get<bool>("use_global_stats", true);
        if(params.get<bool>("scale_bias", false))
            hasWeights = hasBias = true;
        epsilon = params.get<float>("eps", 1E-5);

        size_t n = blobs[0].total();
        CV_Assert(blobs[1].total() == n &&
                  blobs[0].isContinuous() && blobs[1].isContinuous() &&
                  blobs[0].type() == CV_32F && blobs[1].type() == CV_32F);

        float varMeanScale = 1.f;
        if (!hasWeights && !hasBias && blobs.size() > 2 && useGlobalStats) {
            CV_Assert(blobs.size() == 3, blobs[2].type() == CV_32F);
            varMeanScale = blobs[2].at<float>(0);
            if (varMeanScale != 0)
                varMeanScale = 1/varMeanScale;
        }

        const int biasBlobIndex = blobs.size() - 1;
        const int weightsBlobIndex = biasBlobIndex - hasBias;

        if( hasWeights )
        {
            CV_Assert((size_t)weightsBlobIndex < blobs.size());
            const Mat& w = blobs[weightsBlobIndex];
            CV_Assert(w.isContinuous() && w.type() == CV_32F && w.total() == (size_t)n);
        }

        if( hasBias )
        {
            CV_Assert((size_t)biasBlobIndex < blobs.size());
            const Mat& b = blobs[weightsBlobIndex];
            CV_Assert(b.isContinuous() && b.type() == CV_32F && b.total() == (size_t)n);
        }

        const float* meanData = blobs[0].ptr<float>();
        const float* stdData = blobs[1].ptr<float>();
        const float* weightsData = hasWeights ? blobs[weightsBlobIndex].ptr<float>() : 0;
        const float* biasData = hasBias ? blobs[biasBlobIndex].ptr<float>() : 0;

        weights_.create(1, (int)n, CV_32F);
        bias_.create(1, (int)n, CV_32F);

        float* dstWeightsData = weights_.ptr<float>();
        float* dstBiasData = bias_.ptr<float>();

        for (size_t i = 0; i < n; ++i)
        {
            float w = (hasWeights ? weightsData[i] : 1.0f) / sqrt(stdData[i] * varMeanScale + epsilon);
            dstWeightsData[i] = w;
            dstBiasData[i] = (hasBias ? biasData[i] : 0.0f) - w * meanData[i] * varMeanScale;
        }
    }

    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale = weights_;
        shift = bias_;
    }

    virtual bool tryFuse(Ptr<Layer>& top) CV_OVERRIDE
    {
        Mat w, b;
        top->getScaleShift(w, b);
        if (w.empty() && b.empty())
            return false;

        const int numChannels = weights_.total();
        const int numFusedWeights = w.total();
        const int numFusedBias = b.total();

        if ((numFusedWeights != numChannels && numFusedWeights != 1 && !w.empty()) ||
            (numFusedBias != numChannels && numFusedBias != 1 && !b.empty()))
            return false;

        if (!w.empty())
        {
            w = w.reshape(1, 1);
            if (numFusedWeights == 1)
            {
                multiply(weights_, w.at<float>(0), weights_);
                multiply(bias_, w.at<float>(0), bias_);
            }
            else
            {
                multiply(weights_, w, weights_);
                multiply(bias_, w, bias_);
            }
        }
        if (!b.empty())
        {
            b = b.reshape(1, 1);
            if (numFusedBias == 1)
                add(bias_, b.at<float>(0), bias_);
            else
                add(bias_, b.reshape(1, 1), bias_);
        }
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if (!useGlobalStats && inputs[0][0] != 1)
            CV_Error(Error::StsNotImplemented, "Batch normalization in training mode with batch size > 1");
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_HALIDE && haveHalide() ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE && haveInfEngine();
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inputs_.depth() == CV_16S);
        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        CV_Assert(blobs.size() >= 2);
        CV_Assert(inputs.size() == 1);

        if (use_half && inputs[0].dims == 2)
            return false;

        if (umat_weight.empty())
        {
            umat_weight = weights_.getUMat(ACCESS_READ);
            umat_bias = bias_.getUMat(ACCESS_READ);
        }

        UMat &inpBlob = inputs[0];
        CV_Assert(inpBlob.dims == 2 || inpBlob.dims == 4);
        int groups = inpBlob.size[0];
        int channels = inpBlob.size[1];
        int rows = inpBlob.dims > 2 ? inpBlob.size[2] : 1;
        int cols = inpBlob.dims > 2 ? inpBlob.size[3] : 1;

        String opts = (use_half) ? " -DDtype=half" : " -DDtype=float";
        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            if (inpBlob.dims == 2)
            {
                UMat& src = inputs[ii];
                UMat& dst = outputs[ii];
                multiply(src, weights_, dst);
                add(dst, bias_, dst);
            }
            else
            {
                MatShape s = shape(groups * channels, rows * cols);
                UMat src = inputs[ii].reshape(1, s.size(), &s[0]);
                UMat dst = outputs[ii].reshape(1, s.size(), &s[0]);
                int number = (s[1] % 8 == 0) ? 8 : ((s[1] % 4 == 0) ? 4 : 1);
                String buildopt = format("-DNUM=%d", number) + opts;
                String kname = format("batch_norm%d", number);
                if (number == 1)
                    buildopt += format(" -Dconvert_T=convert_%s", use_half ? "half" : "float");
                else
                    buildopt += format(" -Dconvert_T=convert_%s%d", use_half ? "half" : "float", number);
                ocl::Kernel kernel(kname.c_str(), ocl::dnn::batchnorm_oclsrc, buildopt);
                if (kernel.empty())
                    return false;
                size_t global[] = { (size_t)s[0], (size_t)(s[1] / number) };
                kernel.set(0, ocl::KernelArg::PtrReadOnly(src));
                kernel.set(1, (int)s[0]);
                kernel.set(2, (int)s[1]);
                kernel.set(3, (int)channels);
                kernel.set(4, ocl::KernelArg::PtrReadOnly(umat_weight));
                kernel.set(5, ocl::KernelArg::PtrReadOnly(umat_bias));
                kernel.set(6, ocl::KernelArg::PtrWriteOnly(dst));
                bool ret = kernel.run(2, global, NULL, false);
                if (!ret)
                    return false;
            }
        }
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) &&
                   OCL_PERFORMANCE_CHECK(ocl::Device::getDefault().isIntel()),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(blobs.size() >= 2);
        CV_Assert(inputs.size() == 1);

        Mat &inpBlob = *inputs[0];
        CV_Assert(inpBlob.dims == 2 || inpBlob.dims == 4);
        int rows = inpBlob.dims > 2 ? inpBlob.size[2] : 1;
        int cols = inpBlob.dims > 2 ? inpBlob.size[3] : 1;

        for (size_t ii = 0; ii < outputs.size(); ii++)
        {
            Mat &outBlob = outputs[ii];

            for(int num = 0; num < outBlob.size[0]; num++)
            {
                for (int n = 0; n < outBlob.size[1]; n++)
                {
                    float w = weights_.at<float>(n);
                    float b = bias_.at<float>(n);
                    Mat inpBlobPlane(rows, cols, CV_32F, inpBlob.ptr<float>(num, n));
                    Mat outBlobPlane(rows, cols, CV_32F, outBlob.ptr<float>(num, n));
                    inpBlobPlane.convertTo(outBlobPlane, CV_32F, w, b);
                }
            }
        }
    }

    virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node) CV_OVERRIDE
    {
        switch (node->backendId)
        {
            case DNN_BACKEND_HALIDE:
            {
#ifdef HAVE_HALIDE
                auto base = node.dynamicCast<HalideBackendNode>();
                Halide::Func& input = base->funcs.back();
                Halide::Var x("x"), y("y"), c("c"), n("n");
                Halide::Func top = attachHalide(input(x, y, c, n));
                return Ptr<BackendNode>(new HalideBackendNode(base, top));
#endif  // HAVE_HALIDE
                break;
            }
        }
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> input = halideBuffer(inputs[0]);
        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = attachHalide(input(x, y, c, n));
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_HALIDE
    // attachHalide can work both with Halide::Buffer and Halide::Func. In the
    // second case it will be a fusion.
    Halide::Func attachHalide(const Halide::Expr& input)
    {
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Var x("x"), y("y"), c("c"), n("n");

        const int numChannels = weights_.total();
        auto weights = wrapToHalideBuffer(weights_, {numChannels});
        auto bias = wrapToHalideBuffer(bias_, {numChannels});
        top(x, y, c, n) = input * weights(c) + bias(c);
        return top;
    }
#endif  // HAVE_HALIDE

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "ScaleShift";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::ScaleShiftLayer> ieLayer(new InferenceEngine::ScaleShiftLayer(lp));

        const size_t numChannels = weights_.total();
        ieLayer->_weights = wrapToInfEngineBlob(weights_, {numChannels}, InferenceEngine::Layout::C);
        ieLayer->_biases = wrapToInfEngineBlob(bias_, {numChannels}, InferenceEngine::Layout::C);

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        (void)outputs; // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 3*total(inputs[i]);
        }
        return flops;
    }

private:
    bool useGlobalStats;
};

Ptr<BatchNormLayer> BatchNormLayer::create(const LayerParams& params)
{
    return Ptr<BatchNormLayer>(new BatchNormLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
