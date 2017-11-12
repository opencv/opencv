// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of shift layer, which adds up const values to blob.
*/

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class ShiftLayerImpl : public ShiftLayer
{
public:
    ShiftLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        internals.assign(1, shape(1, total(inputs[0], 2)));
        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    virtual void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(inputs.size() > 0);
        CV_Assert(blobs.size() > 0);

        if(inputs[0]->dims == blobs[0].dims)
        {
            for (size_t ii = 0; ii < outputs.size(); ii++)
            {
                Mat &inpBlob = *inputs[ii];
                Mat &outBlob = outputs[ii];

                outBlob = inpBlob + blobs[0];
            }
        }
        else
        {
            Mat biasOnesMat = internals[0];
            biasOnesMat.setTo(1);
            for (size_t ii = 0; ii < outputs.size(); ii++)
            {
                Mat &inpBlob = *inputs[ii];
                Mat &outBlob = outputs[ii];

                inpBlob.copyTo(outBlob);

                for (int n = 0; n < inpBlob.size[0]; n++)
                {
                    Mat dstMat(inpBlob.size[1], inpBlob.size[2] * inpBlob.size[3],
                               outBlob.type(), outBlob.ptr(n));
                    gemm(blobs[0], biasOnesMat, 1, dstMat, 1, dstMat); //TODO: gemv
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        long flops = 0;

        for(int i= 0; i < inputs.size(); i++)
        {
           flops += total(inputs[i]);
        }

        return flops;
    }
};

Ptr<ShiftLayer> ShiftLayer::create(const LayerParams& params)
{
    return Ptr<ShiftLayer>(new ShiftLayerImpl(params));
}

}
}
