// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include "npy_blob.hpp"

namespace opencv_test { namespace {

// RAII guard: register a layer for the duration of a test scope.
struct ScopedLayerRegistration
{
    std::string type;
    ScopedLayerRegistration(const std::string& t, LayerFactory::Constructor ctor) : type(t)
    {
        LayerFactory::registerLayer(t, ctor);
    }
    ~ScopedLayerRegistration() { LayerFactory::unregisterLayer(type); }
    ScopedLayerRegistration(const ScopedLayerRegistration&) = delete;
    ScopedLayerRegistration& operator=(const ScopedLayerRegistration&) = delete;
};

// y = scale * x + bias, with scale/bias read from node attributes.
class CustomScaleBiasLayer CV_FINAL : public Layer
{
public:
    CustomScaleBiasLayer(const LayerParams& params) : Layer(params)
    {
        scale = params.get<float>("scale", 1.f);
        bias = params.get<float>("bias", 0.f);
    }

    static Ptr<Layer> create(LayerParams& params)
    {
        return makePtr<CustomScaleBiasLayer>(params);
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outShapes,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        outShapes.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inps, outs;
        inputs_arr.getMatVector(inps);
        outputs_arr.getMatVector(outs);
        inps[0].convertTo(outs[0], outs[0].type(), scale, bias);
    }

private:
    float scale, bias;
};

// ONNX node 'MyCustomOp' (default ai.onnx domain) -> registered C++ layer.
TEST(Test_ONNX_Custom_layer, CustomLayer_DefaultDomain)
{
    ScopedLayerRegistration scope("MyCustomOp", CustomScaleBiasLayer::create);

    std::string model = findDataFile("dnn/onnx/models/custom_layer_default_domain.onnx");
    Mat input = blobFromNPY(findDataFile("dnn/onnx/data/input_custom_layer_default_domain.npy"));
    Mat ref = blobFromNPY(findDataFile("dnn/onnx/data/output_custom_layer_default_domain.npy"));

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setInput(input);
    Mat out = net.forward();

    normAssert(ref, out, "custom_layer_default_domain");
}

// Op in a non-default domain; the lookup type is "<domain>.<op>".
TEST(Test_ONNX_Custom_layer, CustomLayer_CustomDomain)
{
    ScopedLayerRegistration scope("my.namespace.MyDomainOp", CustomScaleBiasLayer::create);

    std::string model = findDataFile("dnn/onnx/models/custom_layer_custom_domain.onnx");
    Mat input = blobFromNPY(findDataFile("dnn/onnx/data/input_custom_layer_custom_domain.npy"));
    Mat ref = blobFromNPY(findDataFile("dnn/onnx/data/output_custom_layer_custom_domain.npy"));

    Net net = readNetFromONNX(model);
    ASSERT_FALSE(net.empty());

    net.setInput(input);
    Mat out = net.forward();

    normAssert(ref, out, "custom_layer_custom_domain");
}

}} // namespace
