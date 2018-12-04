// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv { namespace dnn {

class ConstLayerImpl CV_FINAL : public ConstLayer
{
public:
    ConstLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.empty());
        outputs.assign(1, shape(blobs[0]));
        return false;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> outputs;
        outs.getUMatVector(outputs);
        if (outs.depth() == CV_16S)
            convertFp16(blobs[0], outputs[0]);
        else
            blobs[0].copyTo(outputs[0]);
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);
        blobs[0].copyTo(outputs[0]);
    }
};

Ptr<Layer> ConstLayer::create(const LayerParams& params)
{
    return Ptr<Layer>(new ConstLayerImpl(params));
}

}}  // namespace cv::dnn
