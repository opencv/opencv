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
#include "../op_inf_engine.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{

class CropLayerImpl CV_FINAL : public CropLayer
{
public:
    CropLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        startAxis = params.get<int>("axis", 2);
        const DictValue *paramOffset = params.ptr("offset");

        if (paramOffset)
        {
            for (int i = 0; i < paramOffset->size(); i++)
                offset.push_back(paramOffset->get<int>(i));
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE && crop_ranges.size() == 4);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);

        MatShape dstShape = inputs[0];
        int start = clamp(startAxis, dstShape);
        for (int i = start; i < dstShape.size(); i++)
        {
            dstShape[i] = inputs[1][i];
        }

        outputs.resize(1, dstShape);

        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(2 == inputs.size());

        const Mat &inpBlob = inputs[0];
        const Mat &inpSzBlob = inputs[1];

        int dims = inpBlob.dims;
        int start_axis = clamp(startAxis, dims);

        std::vector<int> offset_final(dims, 0);
        if (offset.size() == 1)
        {
            for (int i = start_axis; i < dims; i++)
                offset_final[i] = offset[0];
        }
        else if (offset.size() > 1)
        {
            if ((int)offset.size() != dims - start_axis)
                CV_Error(Error::StsBadArg, "number of offset values specified must be "
                                           "equal to the number of dimensions following axis.");

            for (int i = start_axis; i < dims; i++)
                offset_final[i] = offset[i - start_axis];
        }

        crop_ranges.resize(dims);
        for (int i = 0; i < start_axis; i++)
        {
            crop_ranges[i] = Range(0, inpBlob.size[i]);
        }
        for (int i = start_axis; i < dims; i++)
        {
            if (offset_final[i] < 0 || offset_final[i] + inpSzBlob.size[i] > inpBlob.size[i])
                CV_Error(Error::StsBadArg, "invalid crop parameters or blob sizes");

            crop_ranges[i] = Range(offset_final[i], offset_final[i] + inpSzBlob.size[i]);
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat &input = inputs[0];
        input(&crop_ranges[0]).copyTo(outputs[0]);
    }

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "Crop";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::CropLayer> ieLayer(new InferenceEngine::CropLayer(lp));

        CV_Assert(crop_ranges.size() == 4);

#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2018R3)
        for (int i = 0; i < 4; ++i)
        {
            ieLayer->axis.push_back(i);
            ieLayer->offset.push_back(crop_ranges[i].start);
            ieLayer->dim.push_back(crop_ranges[i].end - crop_ranges[i].start);
        }
#else
        ieLayer->axis.push_back(0);  // batch
        ieLayer->offset.push_back(crop_ranges[0].start);
        ieLayer->dim.push_back(crop_ranges[0].end - crop_ranges[0].start);

        ieLayer->axis.push_back(1);  // channels
        ieLayer->offset.push_back(crop_ranges[1].start);
        ieLayer->dim.push_back(crop_ranges[1].end - crop_ranges[1].start);

        ieLayer->axis.push_back(3);  // height
        ieLayer->offset.push_back(crop_ranges[2].start);
        ieLayer->dim.push_back(crop_ranges[2].end - crop_ranges[2].start);

        ieLayer->axis.push_back(2);  // width
        ieLayer->offset.push_back(crop_ranges[3].start);
        ieLayer->dim.push_back(crop_ranges[3].end - crop_ranges[3].start);
#endif
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    std::vector<Range> crop_ranges;
};


Ptr<CropLayer> CropLayer::create(const LayerParams& params)
{
    return Ptr<CropLayer>(new CropLayerImpl(params));
}

}
}
