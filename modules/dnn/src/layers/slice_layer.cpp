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
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class SliceLayerImpl : public SliceLayer
{
public:
    SliceLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);

        if (params.has("slice_point"))
        {
            const DictValue &indicesValue = params.get("slice_point");
            int i, n = indicesValue.size();
            sliceIndices.resize(n);
            for (i = 0; i < n; i++)
                sliceIndices[i] = indicesValue.get<int>(i);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                            const int requiredOutputs,
                            std::vector<MatShape> &outputs,
                            std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() == 1);

        outputs.clear();

        MatShape inpShape = inputs[0];
        int cAxis = clamp(axis, inpShape.size());
        int axisSize = inpShape[cAxis];

        if (sliceIndices.size()) //divide blob with respect to passed parameters
        {
           std::vector<int> outAxisSize;
           int prevSlice = 0;

           for (size_t i = 0; i < sliceIndices.size(); i++)
           {
               if (!(prevSlice < sliceIndices[i] && sliceIndices[i] < axisSize))
                   CV_Error(Error::StsBadArg, "Slice indices should be positive, increased and don't exceed size of sliced dimension");

               outAxisSize.push_back(sliceIndices[i] - prevSlice);
               prevSlice = sliceIndices[i];
            }
            outAxisSize.push_back(axisSize - prevSlice);

            for (size_t i = 0; i < outAxisSize.size(); i++)
            {
               inpShape[cAxis] = outAxisSize[i];
              outputs.push_back(inpShape);
            }
        }
        else //divide blob with respect to count of output blobs
        {
           CV_Assert(requiredOutputs > 0 && axisSize % requiredOutputs == 0);
           int outAxisSize = axisSize / (int)requiredOutputs;

           for (size_t i = 0; i < requiredOutputs; i++)
            {
               inpShape[cAxis] = outAxisSize;
               outputs.push_back(inpShape);
            }
        }

        return false;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        const Mat& inpMat = *inputs[0];
        std::vector<Range> ranges(inpMat.dims, Range::all());
        int cAxis = clamp(axis, inpMat.dims);

        ranges[cAxis].start = 0;
        for (size_t i = 0; i < outputs.size(); i++)
        {
            ranges[cAxis].end = ranges[cAxis].start + outputs[i].size[cAxis];
            inpMat(&ranges[0]).copyTo(outputs[i]);
            ranges[cAxis].start = ranges[cAxis].end;
        }
    }
};

Ptr<SliceLayer> SliceLayer::create(const LayerParams& params)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(params));
}

}
}
