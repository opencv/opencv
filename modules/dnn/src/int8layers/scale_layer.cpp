// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include "../ie_ngraph.hpp"

namespace cv
{
namespace dnn
{

class ScaleLayerInt8Impl CV_FINAL : public ScaleLayerInt8
{
public:
    Mat weights, bias;
    ScaleLayerInt8Impl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasBias = params.get<bool>("bias_term", false);
        axis = params.get<int>("axis", 1);
        hasWeights = false;

        output_sc = params.get<float>("scales");
        output_zp = params.get<int>("zeropoints");

        DictValue inpSc = params.get("input_scales");
        DictValue inpZp = params.get("input_zeropoints");

        for (int i = 0; i < inpSc.size(); i++)
        {
            inp_sc.push_back(inpSc.get<float>(i));
            inp_zp.push_back(inpZp.get<int>(i));
        }
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
        hasWeights = blobs.size() == 2 || (blobs.size() <= 1 && !hasBias);
        CV_Assert((inputs.size() == 2 && blobs.empty()) || blobs.size() == (int)hasWeights + (int)hasBias);

        if (!blobs.empty())
        {
            Mat w = hasWeights ? blobs[0] : Mat::ones(blobs[0].size(), CV_32F);
            Mat b = hasBias ? blobs.back() : Mat::zeros(blobs.back().size(), CV_32F);

            w = w.reshape(1, 1);
            b = b.reshape(1, 1);

            w.convertTo(weights, CV_32F, inp_sc[0]/output_sc);
            addWeighted(b, 1.0/output_sc, weights, -inp_zp[0], output_zp, bias, CV_32F);
        }
        else
        {
            // initialized during forward()
            weights = Mat(); bias = Mat();
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty())
        {
            return activ_int8->blobs.empty();
        }
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat &inpBlob = inputs[0];
        Mat &outBlob = outputs[0];

        if (blobs.empty())
        {
            CV_Assert(inp_sc.size() == 2 && inp_zp.size() == 2);
            Mat inp_dequantized, w, b;
            inputs[1].reshape(1, 1).convertTo(inp_dequantized, CV_32F, inp_sc[1], -(inp_sc[1]*inp_zp[1]));
            w = hasWeights ? inp_dequantized : Mat::ones(inp_dequantized.size(), CV_32F);
            b = hasBias ? inp_dequantized : Mat::zeros(inp_dequantized.size(), CV_32F);

            w.convertTo(weights, CV_32F, inp_sc[0]/output_sc);
            addWeighted(b, 1.0/output_sc, weights, -inp_zp[0], output_zp, bias, CV_32F);
        }

        MatShape inpShape = shape(inpBlob);
        const int numWeights = weights.total();
        CV_Assert(numWeights != 0);
        CV_CheckEQ(weights.total(), bias.total(), "Incompatible weights/bias blobs");

        int endAxis;
        for (endAxis = axis + 1; endAxis <= inpBlob.dims; ++endAxis)
        {
            if (total(inpShape, axis, endAxis) == numWeights)
                break;
        }
        CV_Assert(total(inpShape, axis, endAxis) == numWeights);
        CV_CheckTypeEQ(inpBlob.type(), CV_8SC1, ""); CV_CheckTypeEQ(outBlob.type(), CV_8SC1, "");

        int numSlices = total(inpShape, 0, axis);
        int8_t* inpData = (int8_t*)inpBlob.data;
        int8_t* outData = (int8_t*)outBlob.data;

        if (endAxis != inpBlob.dims)
        {
            float* weightsData = (float*)weights.data;
            float* biasesData = (float*)bias.data;
            int spatialSize = total(inpShape, endAxis);  // spatialSize != 1
            for (int i = 0; i < numSlices; ++i)
            {
                for (int j = 0; j < numWeights; ++j)
                {
                    float w = weightsData[j];
                    float b = biasesData[j];
                    Mat inpSlice(1, spatialSize, CV_8S, inpData);
                    Mat outSlice(1, spatialSize, CV_8S, outData);
                    inpSlice.convertTo(outSlice, CV_8S, w, b);
                    inpData += spatialSize;
                    outData += spatialSize;
                }
            }
        }
        else
        {
            for (int i = 0; i < numSlices; ++i)
            {
                Mat inpSlice(1, numWeights, CV_8S, inpData);
                Mat outSlice(1, numWeights, CV_8S, outData);

                multiply(inpSlice, weights, outSlice, 1.0, CV_8S);
                add(outSlice, bias, outSlice, Mat(), CV_8S);

                inpData += numWeights;
                outData += numWeights;
            }
        }
    }

    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale = (hasWeights && !blobs.empty()) ? blobs[0] : Mat();
        shift = (hasBias && !blobs.empty()) ? blobs.back() : Mat();
    }

    void getScaleZeropoint(float& scale, int& zeropoint) const CV_OVERRIDE
    {
        scale = output_sc;
        zeropoint = output_zp;
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

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        std::vector<ov::Output<ov::Node>> ieInpNodes(nodes.size());
        for (int i = 0; i < nodes.size(); ++i) {
            ieInpNodes[i] = nodes[i].dynamicCast<InfEngineNgraphNode>()->node;
        }

        ieInpNodes[0] = ngraphDequantize(ieInpNodes[0], inp_sc[0], inp_zp[0]);

        CV_Assert(!blobs.empty() || ieInpNodes.size() == 1 + (int)hasWeights + (int)hasBias);

        ov::Output<ov::Node> weights, bias;
        if (blobs.empty()) {
            if (hasWeights)
                weights = ieInpNodes[1];
            if (hasBias)
                bias = ieInpNodes[1 + (int)hasWeights];
        } else {
            std::vector<size_t> shape = ieInpNodes[0].get_shape();
            int cAxis = normalize_axis(axis, shape.size());

            size_t numWeights = blobs[0].total();
            for (int i = 0; i < cAxis; ++i) {
                shape[i] = 1;
            }
            for (int i = cAxis; i < shape.size(); ++i) {
                if (numWeights == 1) {
                    shape[i] = 1;
                }
                numWeights = std::max(numWeights / shape[i], (size_t)1);
            }

            if (hasWeights)
                weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, blobs[0].data);
            if (hasBias)
                bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape, blobs[(int)hasWeights].data);
        }

        ov::Output<ov::Node> res = ieInpNodes[0];
        if (hasWeights) {
            res = std::make_shared<ov::op::v1::Multiply>(res, weights);
        }
        if (hasBias) {
            res = std::make_shared<ov::op::v1::Add>(res, bias);
        }

        res = ngraphQuantize(res, output_sc, output_zp);

        return new InfEngineNgraphNode(res);
    }
#endif  // HAVE_DNN_NGRAPH

private:
    bool hasWeights;
    std::vector<float> inp_sc;
    std::vector<int> inp_zp;
};


Ptr<ScaleLayerInt8> ScaleLayerInt8::create(const LayerParams& params)
{
    return Ptr<ScaleLayerInt8>(new ScaleLayerInt8Impl(params));
}

Ptr<Layer> ShiftLayerInt8::create(const LayerParams& params)
{
    LayerParams scaleParams = params;
    scaleParams.type = "ScaleInt8";
    scaleParams.set("bias_term", true);
    scaleParams.set("axis", 0);
    return Ptr<ScaleLayerInt8>(new ScaleLayerInt8Impl(scaleParams));
}

}  // namespace dnn
}  // namespace cv
