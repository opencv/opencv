// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Quantization & Dequantization Layers.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

////////////////////////////// Quantize Layer /////////////////////////////

class QuantizeLinearLayerImpl CV_FINAL : public QuantizeLinearLayer
{
public:
    QuantizeLinearLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", -1);
        scale = !params.blobs.empty() ? params.blobs[0] : Mat();
        zeroPoint = params.blobs.size() > 1 ? params.blobs[1] : Mat::zeros(1, 1, CV_8U);
        outDepth = CV_32F; // should be changed to CV_8S eventually
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(outputs.size() == 1);

        Mat &inpBlob = inputs[0];
        Mat &outBlob = outputs[0];
        bool isContinuous = inpBlob.isContinuous() && outBlob.isContinuous();

        int odepth = outBlob.depth();
        CV_Assert(inpBlob.depth() == CV_32F && (odepth == CV_32F));
        int ndims = inpBlob.dims;

        CV_Assert(ndims >= 2 && ndims <= 4);
        int W = inpBlob.size.p[ndims-1];
        int H = inpBlob.size.p[ndims-2];
        int C = ndims > 2 ? inpBlob.size.p[ndims-3] : 1;
        int N = ndims > 3 ? inpBlob.size.p[ndims-4] : 1;

        CV_Assert(axis == ndims - 3 || axis == -1);

        int stotal = (int)scale.total();
        CV_Assert(!scale.empty() && scale.isContinuous() &&
                  scale.type() == CV_32FC1 &&
                  (stotal == 1 || stotal == C));
        int ztype = zeroPoint.type();
        int ztotal = (int)zeroPoint.total();
        CV_Assert(zeroPoint.empty() ||
                  ((ztype == CV_8U || ztype == CV_8S) &&
                   (ztotal == 1 || ztotal == C)));

        const float* sptr = scale.ptr<float>();
        const uint8_t* zptr = zeroPoint.ptr<uint8_t>();

        if(isContinuous) {
            W *= H;
            H = 1;
        }

        for(int n = 0; n < N; n++)
        {
            for(int c = 0; c < C; c++)
            {
                const float* inptr =
                    ndims == 2 ? inpBlob.ptr<float>() :
                    ndims == 3 ? inpBlob.ptr<float>(c) :
                    inpBlob.ptr<float>(n, c);
                // [TODO] change the output to int8_t
                float* outptr =
                    ndims == 2 ? outBlob.ptr<float>() :
                    ndims == 3 ? outBlob.ptr<float>(c) :
                    outBlob.ptr<float>(n, c);
                //float a = 1.f/sptr[stotal == C ? c : 0];
                float a = sptr[stotal == C ? c : 0];
                int zidx = ztotal == C ? c : 0;
                float b = !zptr ? 0.f :
                        ztype == CV_8U ? (float)zptr[zidx] :
                        (float)((const int8_t*)zptr)[zidx];
                size_t istep = inpBlob.step.p[ndims-2]/sizeof(inptr[0]);
                size_t ostep = outBlob.step.p[ndims-2]/sizeof(outptr[0]);

                for(int y = 0; y < H; y++, inptr += istep, outptr += ostep)
                {
                    for(int x = 0; x < W; x++)
                        outptr[x] = inptr[x]*a + b;
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        long flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 2*total(inputs[i]);
        }
        return flops;
    }

private:
    Mat scale;
    Mat zeroPoint;
    int axis, outDepth;
};


Ptr<QuantizeLinearLayer> QuantizeLinearLayer::create(const LayerParams& params)
{
    return Ptr<QuantizeLinearLayer>(new QuantizeLinearLayerImpl(params));
}

////////////////////////////// Dequantize Layer /////////////////////////////

class DequantizeLinearLayerImpl CV_FINAL : public DequantizeLinearLayer
{
public:
    DequantizeLinearLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", -1);
        scale = !params.blobs.empty() ? params.blobs[0] : Mat();
        zeroPoint = params.blobs.size() > 1 ? params.blobs[1] : Mat::zeros(1, 1, CV_8U);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(outputs.size() == 1);

        Mat &inpBlob = inputs[0];
        Mat &outBlob = outputs[0];
        int idepth = inpBlob.depth();
        int odepth = outBlob.depth();

        bool isContinuous = inpBlob.isContinuous() && outBlob.isContinuous();

        CV_Assert((idepth == CV_8U || idepth == CV_8S || idepth == CV_32F) &&
                  (odepth == CV_32F));
        int ndims = inpBlob.dims;

        CV_Assert(ndims >= 2 && ndims <= 4);
        int W = inpBlob.size.p[ndims-1];
        int H = inpBlob.size.p[ndims-2];
        int C = ndims > 2 ? inpBlob.size.p[ndims-3] : 1;
        int N = ndims > 3 ? inpBlob.size.p[ndims-4] : 1;

        CV_Assert(axis == ndims - 3 || axis == -1);
        int stotal = (int)scale.total();
        CV_Assert(!scale.empty() && scale.isContinuous() &&
                  scale.type() == CV_32FC1 &&
                  (stotal == 1 || stotal == C));
        int ztype = zeroPoint.type();
        int ztotal = (int)zeroPoint.total();
        CV_Assert(zeroPoint.empty() ||
                  ((ztype == CV_8U || ztype == CV_8S) &&
                   (ztotal == 1 || ztotal == C)));

        const float* sptr = scale.ptr<float>();
        const uint8_t* zptr = zeroPoint.ptr<uint8_t>();

        if(isContinuous) {
            W *= H;
            H = 1;
        }

        for(int n = 0; n < N; n++)
        {
            for(int c = 0; c < C; c++)
            {
                float* outptr =
                    ndims == 2 ? outBlob.ptr<float>() :
                    ndims == 3 ? outBlob.ptr<float>(c) :
                    outBlob.ptr<float>(n, c);
                float a = sptr[stotal == C ? c : 0];
                int zidx = ztotal == C ? c : 0;
                float b = !zptr ? 0.f :
                        ztype == CV_8U ? (float)zptr[zidx] :
                        (float)((const int8_t*)zptr)[zidx];
                size_t istep = inpBlob.step.p[ndims-2];
                size_t ostep = outBlob.step.p[ndims-2]/sizeof(outptr[0]);

                if (idepth == CV_8U)
                {
                    const uint8_t* inptr =
                        ndims == 2 ? inpBlob.ptr<uint8_t>() :
                        ndims == 3 ? inpBlob.ptr<uint8_t>(c) :
                        inpBlob.ptr<uint8_t>(n, c);
                    istep /= sizeof(inptr[0]);
                    for(int y = 0; y < H; y++, inptr += istep, outptr += ostep)
                    {
                        for(int x = 0; x < W; x++)
                            outptr[x] = (inptr[x] - b)*a;
                    }
                }
                else if (idepth == CV_8S)
                {
                    const int8_t* inptr =
                        ndims == 2 ? inpBlob.ptr<int8_t>() :
                        ndims == 3 ? inpBlob.ptr<int8_t>(c) :
                        inpBlob.ptr<int8_t>(n, c);
                    istep /= sizeof(inptr[0]);
                    for(int y = 0; y < H; y++, inptr += istep, outptr += ostep)
                    {
                        for(int x = 0; x < W; x++)
                            outptr[x] = (inptr[x] - b)*a;
                    }
                }
                else
                {
                    CV_Assert(idepth == CV_32F);
                    const float* inptr =
                        ndims == 2 ? inpBlob.ptr<float>() :
                        ndims == 3 ? inpBlob.ptr<float>(c) :
                        inpBlob.ptr<float>(n, c);
                    istep /= sizeof(inptr[0]);
                    for(int y = 0; y < H; y++, inptr += istep, outptr += ostep)
                    {
                        for(int x = 0; x < W; x++)
                            outptr[x] = (inptr[x] - b)*a;
                    }
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        long flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 2*total(inputs[i]);
        }
        return flops;
    }

private:
    Mat scale;
    Mat zeroPoint;
    int axis;
};


Ptr<DequantizeLinearLayer> DequantizeLinearLayer::create(const LayerParams& params)
{
    return Ptr<DequantizeLinearLayer>(new DequantizeLinearLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
