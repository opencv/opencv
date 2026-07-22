// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

/*
    Implementation of CumProd, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__CumProd.html

    Opset 26 is covered.
*/
class CumProdLayerImpl CV_FINAL : public CumProdLayer
{
public:
    CumProdLayerImpl(const LayerParams& params)
    {
        axis_raw = params.get<int>("axis", 0);
        exclusive_raw = params.get<int>("exclusive", 0);
        reverse_raw = params.get<int>("reverse", 0);
        setParamsFrom(params);
    }

    bool supportBackend(int backendId) CV_OVERRIDE { return backendId == DNN_BACKEND_OPENCV; }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int,
                         std::vector<MatShape>& outputs, std::vector<MatShape>&) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
        return exclusive_raw == 0;
    }

    void getTypes(const std::vector<MatType>& inputs, const int, const int,
                  std::vector<MatType>& outputs, std::vector<MatType>&) const CV_OVERRIDE
    {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_64F ||
                     inputs[0] == CV_32S || inputs[0] == CV_64S || inputs[0] == CV_16F, "");
        outputs.assign(1, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_CheckTypeEQ(inputs[0].depth(), outputs[0].depth(), "");

        switch (inputs[0].depth())
        {
            case CV_32F: forwardImpl<float>(inputs, outputs); break;
            case CV_32S: forwardImpl<int32_t>(inputs, outputs); break;
            case CV_64S: forwardImpl<int64_t>(inputs, outputs); break;
            case CV_64F: forwardImpl<double>(inputs, outputs); break;
            default: CV_Error(Error::BadDepth, "");
        }
    }

    template <typename T>
    void forwardImpl(const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        const Mat& src_mat = inputs[0];
        const T* src_ptr = src_mat.ptr<T>();

        int axis = inputs.size() > 1 ? parseAxis(inputs[1]) : axis_raw;
        axis = normalize_axis(axis, src_mat.dims);

        Mat& dst_mat = outputs[0];
        T* dst_ptr = dst_mat.ptr<T>();

        const bool exclusive = exclusive_raw == 1;
        const bool reverse = reverse_raw == 1;

        // View data as [outer_size, target_size, inner_size] around the scan axis.
        const size_t outer_size = src_mat.total(0, axis);
        const size_t target_size = src_mat.size[axis];
        const size_t inner_size = src_mat.total(axis + 1);
        const size_t outer_step_length = target_size * inner_size;

        const int target_start = reverse ? (int)target_size - 1 : 0;
        const int target_stop = reverse ? -1 : (int)target_size;
        const int target_delta = reverse ? -1 : 1;
        const int target_step = target_delta * (int)inner_size;
        const int exclusive_delta = exclusive ? target_step : 0;

        for (size_t outer_idx = 0; outer_idx < outer_size; outer_idx++)
        {
            const size_t target_offset = outer_idx * outer_step_length;

            // First element: multiplicative identity when exclusive, else the source value.
            size_t first_inner_offset = target_offset + (size_t)target_start * inner_size;
            if (exclusive)
                for (size_t inner_idx = 0; inner_idx < inner_size; inner_idx++)
                    dst_ptr[first_inner_offset + inner_idx] = (T)1;
            else
                for (size_t inner_idx = 0; inner_idx < inner_size; inner_idx++)
                    dst_ptr[first_inner_offset + inner_idx] = src_ptr[first_inner_offset + inner_idx];

            for (int target_idx = target_start + target_delta; target_idx != target_stop; target_idx += target_delta)
            {
                const size_t inner_offset = target_offset + (size_t)target_idx * inner_size;
                for (size_t inner_idx = 0; inner_idx < inner_size; inner_idx++)
                {
                    dst_ptr[inner_offset + inner_idx] = dst_ptr[inner_offset - target_step + inner_idx] *
                        src_ptr[inner_offset - exclusive_delta + inner_idx];
                }
            }
        }
    }

    int parseAxis(const Mat& axis_mat)
    {
        CV_CheckEQ(axis_mat.total(), 1u, "Axis tensor should contain single value");
        if (axis_mat.type() == CV_32SC1)
            return axis_mat.at<int32_t>(0);
        Mat axis_mat_int;
        axis_mat.convertTo(axis_mat_int, CV_32SC1);
        return axis_mat_int.at<int32_t>(0);
    }

    int axis_raw;
    int exclusive_raw;
    int reverse_raw;
};

Ptr<CumProdLayer> CumProdLayer::create(const LayerParams& params)
{
    return makePtr<CumProdLayerImpl>(params);
}

}} // namespace cv::dnn
