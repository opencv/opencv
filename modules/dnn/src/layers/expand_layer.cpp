// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

class ExpandLayerImpl CV_FINAL : public ExpandLayer
{
public:
    ExpandLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        // shape as param
        CV_CheckTrue(params.has("shape"), "DNN/Expand: shape is required in Expand layer initialization");
        DictValue param_shape = params.get("shape");
        int ndims_shape = param_shape.size();
        CV_CheckGT(ndims_shape, 0, "DNN/Expand: ndims of shape must be > 0");
        target_shape.resize(ndims_shape);
        for (int i = 0; i < ndims_shape; i++) {
            target_shape[i] = param_shape.get<int>(i);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckGE(inputs.size(), static_cast<size_t>(1), "DNN/Expand: one input at least");
        CV_CheckLE(inputs.size(), static_cast<size_t>(2), "DNN/Expand: two input at most");
        CV_CheckFalse(target_shape.empty(), "DNN/Expand: shape must known before memory is set");

        auto& moreDimension = inputs[0].size() > target_shape.size() ? inputs[0] : target_shape;
        auto& lessDimension = inputs[0].size() <= target_shape.size() ? inputs[0] : target_shape;


        /*  Example:
                             i = 3
                               |
            moreDimension: 0 1 2 3 4, assign non-aligned dimensions to output shape
            lessDimension:     0 1 2, when dimension is aligned, check valid dimension (either equal or one of them is 1) and assign bigger one
        */
        MatShape outputShape(moreDimension.size(), 1);
        for (int i = 0; i < moreDimension.size(); i++) {
            int d = moreDimension[i];
            int j = i - (moreDimension.size() - lessDimension.size());
            std::cout << "i = " << i << ", j = " << j << std::endl;
            if (j >= 0) {
                if (d == 1 || lessDimension[j] == 1 || d == lessDimension[j]) {
                    outputShape[i] = std::max(d, lessDimension[j]);
                } else {
                    CV_Error(Error::StsBadSize, cv::format("DNN/Expand: invalid dimension, d (%d) != d (%d)", moreDimension[i], lessDimension[j]));
                }
            } else {
                outputShape[i] = d;
            }
        }
        outputs.assign(1, outputShape);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        const auto &input = inputs[0];
        const auto input_shape = shape(input);

        auto& moreDimension = input_shape.size() > target_shape.size() ? input_shape : target_shape;
        auto& lessDimension = input_shape.size() <= target_shape.size() ? input_shape : target_shape;

        MatShape final_target_shape(moreDimension.size(), 1);
        for (int i = 0; i < moreDimension.size(); i++) {
            int d = moreDimension[i];
            int j = i - (moreDimension.size() - lessDimension.size());
            if (j >= 0) {
                final_target_shape[i] = std::max(lessDimension[j], d);
            } else {
                final_target_shape[i] = d;
            }
        }
        target_shape.clear();
        target_shape = std::move(final_target_shape);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::broadcast(inputs[0], target_shape, outputs[0]);
    }

private:
    MatShape target_shape;
};

Ptr<ExpandLayer> ExpandLayer::create(const LayerParams &params) {
    return makePtr<ExpandLayerImpl>(params);
}

}}  // cv::dnn
