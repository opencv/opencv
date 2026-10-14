// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "../net_impl.hpp"
#include "../op_cann.hpp"
#include "layers_common.hpp"
#include "../dnn_common.hpp"

namespace cv {
namespace dnn {

class Reduce2LayerImpl CV_FINAL : public Reduce2Layer
{
public:
    static const char* reduceTypeToString(ReduceType t)
    {
        switch (t) {
            case ReduceType::MAX: return "MAX";
            case ReduceType::MIN: return "MIN";
            case ReduceType::MEAN: return "MEAN";
            case ReduceType::SUM: return "SUM";
            case ReduceType::L1: return "L1";
            case ReduceType::L2: return "L2";
            case ReduceType::PROD: return "PROD";
            case ReduceType::SUM_SQUARE: return "SUM_SQUARE";
            case ReduceType::LOG_SUM: return "LOG_SUM";
            case ReduceType::LOG_SUM_EXP: return "LOG_SUM_EXP";
        }
        return "UNKNOWN";
    }
    Reduce2LayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        CV_Assert(params.has("reduce"));
        String reduce_type_str = toLowerCase(params.get<String>("reduce"));
        if (reduce_type_str == "max")
            reduce_type = ReduceType::MAX;
        else if (reduce_type_str == "min")
            reduce_type = ReduceType::MIN;
        else if (reduce_type_str == "mean")
            reduce_type = ReduceType::MEAN;
        else if (reduce_type_str == "sum")
            reduce_type = ReduceType::SUM;
        else if (reduce_type_str == "sum_square")
            reduce_type = ReduceType::SUM_SQUARE;
        else if (reduce_type_str == "l1")
            reduce_type = ReduceType::L1;
        else if (reduce_type_str == "l2")
            reduce_type = ReduceType::L2;
        else if (reduce_type_str == "log_sum")
            reduce_type = ReduceType::LOG_SUM;
        else if (reduce_type_str == "log_sum_exp")
            reduce_type = ReduceType::LOG_SUM_EXP;
        else if (reduce_type_str == "prod")
            reduce_type = ReduceType::PROD;
        else
            CV_Error(Error::StsBadArg, "Unknown reduce type\"" + reduce_type_str + "\"");

        keepdims = params.get<bool>("keepdims", true);
        noop_with_empty_axes = params.get<bool>("noop_with_empty_axes", false);

        if (params.has("axes")) {
            auto param_axes = params.get("axes");
            int num_axes = param_axes.size();
            axes.resize(num_axes);
            for (int i = 0; i < num_axes; ++i)
                axes[i] = param_axes.get<int>(i);
        }
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        if (inputs.size() < 2)
            return false;
        Net::Impl* netimpl_ = getNetImpl(this);
        if (!netimpl_)
            return true;
        return !netimpl_->isConstArg(inputs[1]);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inps,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape> &outs,
                         std::vector<MatShape> &/*internals*/) const CV_OVERRIDE
    {
        CV_Assert(!inps.empty());
        outs.resize(1);
        const MatShape& inp0 = inps[0];
        if (inp0.empty()) {
            outs[0] = MatShape();
            return false;
        }

        std::vector<int> axes;
        if (!this->axes.empty()) {
            axes = this->axes;
        } else if (inps.size() >= 2) {
            Net::Impl* netimpl_ = getNetImpl(this);
            if (netimpl_ && netimpl_->isConstArg(inputs[1])) {
                Mat axesTensor = netimpl_->argTensor(inputs[1]);
                tensorToIntVec(axesTensor, axes);
            }
        }

        if (axes.empty()) {
            if (noop_with_empty_axes) {
                outs[0] = inp0;
            } else {
                if (keepdims) {
                    MatShape shape_out = inp0;
                    std::fill(shape_out.begin(), shape_out.end(), 1);
                    outs[0] = shape_out;
                } else {
                    outs[0] = MatShape::scalar();
                }
            }
            return false;
        }

        std::vector<int> norm_axes = axes;
        for (size_t i = 0; i < norm_axes.size(); ++i)
            norm_axes[i] = normalize_axis(norm_axes[i], inp0);

        auto shape_output_ = inp0;
        for (int axis : norm_axes) shape_output_[axis] = -1;
        MatShape shape_output;
        for (size_t i = 0; i < shape_output_.size(); ++i) {
            if (shape_output_[i] == -1) {
                if (keepdims) shape_output.push_back(1);
            } else {
                shape_output.push_back(shape_output_[i]);
            }
        }
        if (shape_output.empty()) shape_output = MatShape::scalar();
        outs[0] = shape_output;
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_64F || inputs[0] == CV_32S || inputs[0] == CV_64S || inputs[0] == CV_16F || inputs[0] == CV_16BF || inputs[0] == CV_8U || inputs[0] == CV_8S || inputs[0] == CV_Bool, "");
        outputs.assign(1, inputs[0]);
    }

    static int toCoreReduceOp(ReduceType t)
    {
        switch (t) {
            case ReduceType::MAX: return REDUCE_MAX;
            case ReduceType::MIN: return REDUCE_MIN;
            case ReduceType::MEAN: return REDUCE_AVG;
            case ReduceType::SUM: return REDUCE_SUM;
            case ReduceType::L1: return REDUCE_L1;
            case ReduceType::L2: return REDUCE_L2;
            case ReduceType::PROD: return REDUCE_PROD;
            case ReduceType::SUM_SQUARE: return REDUCE_SUM2;
            case ReduceType::LOG_SUM: return REDUCE_LOG_SUM;
            case ReduceType::LOG_SUM_EXP: return REDUCE_LOG_SUM_EXP;
        }
        CV_Error(Error::StsBadArg, "DNN/Reduce: Unsupported operation.");
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);

        CV_Assert(!inputs.empty());
        Mat& src = inputs[0];
        std::vector<int> axes;
        if (!this->axes.empty()) {
            axes = this->axes;
        } else if (inputs.size() >= 2) {
            tensorToIntVec(inputs[1], axes);
        }

        MatShape inpShape = shape(src);
        MatShape outShape;
        if (axes.empty()) {
            if (noop_with_empty_axes) {
                outShape = inpShape;
            } else {
                if (keepdims) {
                    outShape.assign(inpShape.size(), 1);
                } else {
                    outShape = MatShape::scalar();
                }
            }
        } else {
            std::vector<int> norm_axes = axes;
            for (size_t i = 0; i < norm_axes.size(); ++i)
                norm_axes[i] = normalize_axis(norm_axes[i], inpShape);
            MatShape tmp = inpShape;
            for (int a : norm_axes) tmp[a] = -1;
            for (size_t i = 0; i < tmp.size(); ++i) {
                if (tmp[i] == -1) {
                    if (keepdims) outShape.push_back(1);
                } else {
                    outShape.push_back(tmp[i]);
                }
            }
            if (outShape.size() == 0) outShape = MatShape{1};
            axes = norm_axes;
        }

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            outputs_arr.getMatVecRef()[0].fit(outShape, src.type());
        } else {
            CV_Assert(kind == _InputArray::STD_VECTOR_UMAT);
            outputs_arr.getUMatVecRef()[0].fit(outShape, src.type());
        }
        outputs_arr.getMatVector(outputs);
        Mat& dst = outputs[0];

        if (axes.empty() && noop_with_empty_axes) {
            src.copyTo(dst);
            return;
        }
        if (src.depth() == CV_Bool)
            CV_Assert(reduce_type == ReduceType::MAX || reduce_type == ReduceType::MIN);

        // reduce into a keepdims view of the output, which has the same layout
        MatShape keepShape = inpShape;
        if (axes.empty())
            std::fill(keepShape.begin(), keepShape.end(), 1);
        for (int a : axes)
            keepShape[a] = 1;
        CV_Assert(dst.isContinuous() && dst.total() == keepShape.total());
        Mat dstKeep(keepShape, dst.type(), dst.data);
        cv::reduce(src, dstKeep, ReduceParams(toCoreReduceOp(reduce_type), axes));
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "reduce_type: \"" << reduceTypeToString(reduce_type) << "\",\n";

        prindent(strm, indent);
        strm << "keepdims: " << (keepdims ? "true" : "false") << ",\n";

        prindent(strm, indent);
        strm << "noop_with_empty_axes: " << (noop_with_empty_axes ? "true" : "false") << ",\n";

        prindent(strm, indent);
        strm << "axes: [";
        for (size_t i = 0; i < axes.size(); ++i) {
            if (i > 0) strm << ", ";
            strm << axes[i];
        }
        strm << "]\n";
        return strm;
    }
};

Ptr<Reduce2Layer> Reduce2Layer::create(const LayerParams& params)
{
    return Ptr<Reduce2Layer>(new Reduce2LayerImpl(params));
}

}} // cv::dnn
