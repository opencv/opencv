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
            CV_Assert(!params.has("begin") && !params.has("size") && !params.has("end"));
            const DictValue &indicesValue = params.get("slice_point");
            sliceRanges.resize(indicesValue.size() + 1,
                               std::vector<Range>(axis + 1, Range::all()));
            int prevSlice = 0;
            for (int i = 0; i < indicesValue.size(); ++i)
            {
                sliceRanges[i][axis].start = prevSlice;
                sliceRanges[i][axis].end = indicesValue.get<int>(i);
                prevSlice = sliceRanges[i][axis].end;
            }
            sliceRanges.back()[axis].start = prevSlice;
        }
        else if (params.has("begin"))
        {
            CV_Assert(params.has("size") ^ params.has("end"));
            const DictValue &begins = params.get("begin");
            const DictValue &sizesOrEnds = params.has("size") ? params.get("size") : params.get("end");
            CV_Assert(begins.size() == sizesOrEnds.size());

            sliceRanges.resize(1);
            sliceRanges[0].resize(begins.size(), Range::all());
            for (int i = 0; i < begins.size(); ++i)
            {
                int start = begins.get<int>(i);
                int sizeOrEnd = sizesOrEnds.get<int>(i);  // It may be negative to reverse indexation.
                CV_Assert(start >= 0);

                sliceRanges[0][i].start = start;
                if (params.has("size"))
                {
                    int size = sizeOrEnd;
                    CV_Assert(size == -1 || size > 0);  // -1 value means range [start, axis_size).
                    sliceRanges[0][i].end = start > 0 ? start + size : -1;  // We'll finalize a negative value later.
                }
                else
                {
                    int end = sizeOrEnd;
                    CV_Assert(end < 0 || end > start);  // End index is excluded.
                    sliceRanges[0][i].end = end;  // We'll finalize a negative value later.
                }
            }
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                            const int requiredOutputs,
                            std::vector<MatShape> &outputs,
                            std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() == 1);
        MatShape inpShape = inputs[0];

        if (!sliceRanges.empty())
        {
            outputs.resize(sliceRanges.size(), inpShape);
            for (int i = 0; i < outputs.size(); ++i)
            {
                CV_Assert(sliceRanges[i].size() <= inpShape.size());
                for (int j = 0; j < sliceRanges[i].size(); ++j)
                {
                    outputs[i][j] = clamp(sliceRanges[i][j], inpShape[j]).size();
                }
            }
        }
        else  // Divide input blob on equal parts by axis.
        {
            CV_Assert(0 <= axis && axis < inpShape.size());
            CV_Assert(requiredOutputs > 0 && inpShape[axis] % requiredOutputs == 0);
            inpShape[axis] /= requiredOutputs;
            outputs.resize(requiredOutputs, inpShape);
        }
        return false;
    }

    void finalize(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() == 1);
        const MatSize& inpShape = inputs[0]->size;

        if (sliceRanges.empty())
        {
            // Divide input blob on equal parts by axis.
            int outAxisSize = inpShape[axis] / outputs.size();
            sliceRanges.resize(outputs.size(),
                               std::vector<Range>(axis + 1, Range::all()));
            int prevSlice = 0;
            for (int i = 0; i < outputs.size(); ++i)
            {
                sliceRanges[i][axis].start = prevSlice;
                sliceRanges[i][axis].end = sliceRanges[i][axis].start + outAxisSize;
                prevSlice = sliceRanges[i][axis].end;
            }
        }
        else
            CV_Assert(outputs.size() == sliceRanges.size());

        for (int i = 0; i < outputs.size(); ++i)
        {
            CV_Assert(sliceRanges[i].size() <= inpShape[-1]);
            // Clamp.
            for (int j = 0; j < sliceRanges[i].size(); ++j)
            {
                sliceRanges[i][j] = clamp(sliceRanges[i][j], inpShape[j]);
            }
            // Fill the rest of ranges.
            for (int j = sliceRanges[i].size(); j < inpShape[-1]; ++j)
            {
                sliceRanges[i].push_back(Range::all());
            }
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        const Mat& inpMat = *inputs[0];
        CV_Assert(outputs.size() == sliceRanges.size());
        for (size_t i = 0; i < outputs.size(); i++)
        {
            inpMat(sliceRanges[i]).copyTo(outputs[i]);
        }
    }
};

Ptr<SliceLayer> SliceLayer::create(const LayerParams& params)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(params));
}

}
}
