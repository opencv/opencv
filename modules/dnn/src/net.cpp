// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Net::Net()
    : impl(makePtr<Net::Impl>())
{
}

Net::~Net()
{
}

int Net::addLayer(const String& name, const String& type, const int& dtype, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->addLayer(name, type, dtype, params);
}

int Net::addLayer(const String& name, const String& type, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    return addLayer(name, type, CV_32F, params);
}

int Net::addLayerToPrev(const String& name, const String& type, const int& dtype, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->addLayerToPrev(name, type, dtype, params);
}

int Net::addLayerToPrev(const String& name, const String& type, LayerParams& params)
{
    CV_TRACE_FUNCTION();
    return addLayerToPrev(name, type, CV_32F, params);
}

void Net::connect(int outLayerId, int outNum, int inpLayerId, int inpNum)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    impl->connect(outLayerId, outNum, inpLayerId, inpNum);
}

void Net::connect(String _outPin, String _inPin)
{
    CV_TRACE_FUNCTION();

    CV_Assert(impl);
    LayerPin outPin = impl->getPinByAlias(_outPin);
    LayerPin inpPin = impl->getPinByAlias(_inPin);

    CV_Assert(outPin.valid() && inpPin.valid());

    impl->connect(outPin.lid, outPin.oid, inpPin.lid, inpPin.oid);
}

int Net::registerOutput(const std::string& outputName, int layerId, int outputPort)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->registerOutput(outputName, layerId, outputPort);
}

Mat Net::forward(const String& outputName)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputName);
}

AsyncArray Net::forwardAsync(const String& outputName)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forwardAsync(outputName);
}

void Net::forward(OutputArrayOfArrays outputBlobs, const String& outputName)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputBlobs, outputName);
}

void Net::forward(OutputArrayOfArrays outputBlobs,
        const std::vector<String>& outBlobNames)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputBlobs, outBlobNames);
}

void Net::forward(std::vector<std::vector<Mat>>& outputBlobs,
        const std::vector<String>& outBlobNames)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->forward(outputBlobs, outBlobNames);
}

// FIXIT drop from inference API
Net Net::quantize(InputArrayOfArrays calibData, int inputsDtype, int outputsDtype, bool perChannel)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->quantize(*this, calibData, inputsDtype, outputsDtype, perChannel);
}

// FIXIT drop from inference API
void Net::getInputDetails(std::vector<float>& scales, std::vector<int>& zeropoints) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->getInputDetails(scales, zeropoints);
}

// FIXIT drop from inference API
void Net::getOutputDetails(std::vector<float>& scales, std::vector<int>& zeropoints) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->getOutputDetails(scales, zeropoints);
}

void Net::setPreferableBackend(int backendId)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(backendId);
    CV_Assert(impl);
    return impl->setPreferableBackend(*this, backendId);
}

void Net::setPreferableTarget(int targetId)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG(targetId);
    CV_Assert(impl);
    return impl->setPreferableTarget(targetId);
}

void Net::setInputsNames(const std::vector<String>& inputBlobNames)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->setInputsNames(inputBlobNames);
}

void Net::setInputShape(const String& inputName, const MatShape& shape)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->setInputShape(inputName, shape);
}

void Net::setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(name, "name", name.c_str());
    CV_Assert(impl);
    return impl->setInput(blob, name, scalefactor, mean);
}

Mat Net::getParam(int layer, int numParam) const
{
    CV_Assert(impl);
    return impl->getParam(layer, numParam);
}

void Net::setParam(int layer, int numParam, const Mat& blob)
{
    CV_Assert(impl);
    return impl->setParam(layer, numParam, blob);
}

int Net::getLayerId(const String& layer) const
{
    CV_Assert(impl);
    return impl->getLayerId(layer);
}

String Net::dump()
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    return impl->dump(true);
}

void Net::dumpToFile(const String& path)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    std::ofstream file(path.c_str());
    file << dump();
    file.close();
}

Ptr<Layer> Net::getLayer(int layerId) const
{
    CV_Assert(impl);
    return impl->getLayer(layerId);
}
Ptr<Layer> Net::getLayer(const LayerId& layerId) const
{
    CV_Assert(impl);
    return impl->getLayer(layerId);
}

std::vector<Ptr<Layer>> Net::getLayerInputs(int layerId) const
{
    CV_Assert(impl);
    return impl->getLayerInputs(layerId);
}

std::vector<String> Net::getLayerNames() const
{
    CV_Assert(impl);
    return impl->getLayerNames();
}

bool Net::empty() const
{
    CV_Assert(impl);
    return impl->empty();
}

// FIXIT drop "unconnected" API
std::vector<int> Net::getUnconnectedOutLayers() const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getUnconnectedOutLayers();
}

// FIXIT drop "unconnected" API
std::vector<String> Net::getUnconnectedOutLayersNames() const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getUnconnectedOutLayersNames();
}

void Net::getLayersShapes(const ShapesVec& netInputShapes,
        std::vector<int>& layersIds,
        std::vector<ShapesVec>& inLayersShapes,
        std::vector<ShapesVec>& outLayersShapes) const
{
    CV_Assert(impl);
    return impl->getLayersShapes(netInputShapes, layersIds, inLayersShapes, outLayersShapes);
}

void Net::getLayersShapes(const MatShape& netInputShape,
        std::vector<int>& layerIds,
        std::vector<ShapesVec>& inLayersShapes,
        std::vector<ShapesVec>& outLayersShapes) const
{
    getLayersShapes(ShapesVec(1, netInputShape),
            layerIds, inLayersShapes, outLayersShapes);
}

void Net::getLayerShapes(const MatShape& netInputShape,
        const int layerId,
        ShapesVec& inLayerShapes,
        ShapesVec& outLayerShapes) const
{
    getLayerShapes(ShapesVec(1, netInputShape),
            layerId, inLayerShapes, outLayerShapes);
}

void Net::getLayerShapes(const ShapesVec& netInputShapes,
        const int layerId,
        ShapesVec& inLayerShapes,
        ShapesVec& outLayerShapes) const
{
    CV_Assert(impl);
    LayerShapes shapes;
    impl->getLayerShapes(netInputShapes, layerId, shapes);
    inLayerShapes = shapes.in;
    outLayerShapes = shapes.out;
}

int64 Net::getFLOPS(const std::vector<MatShape>& netInputShapes) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getFLOPS(netInputShapes);
}

int64 Net::getFLOPS(const MatShape& netInputShape) const
{
    return getFLOPS(std::vector<MatShape>(1, netInputShape));
}

int64 Net::getFLOPS(const int layerId,
        const std::vector<MatShape>& netInputShapes) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getFLOPS(layerId, netInputShapes);
}

int64 Net::getFLOPS(const int layerId,
        const MatShape& netInputShape) const
{
    return getFLOPS(layerId, std::vector<MatShape>(1, netInputShape));
}

void Net::getLayerTypes(std::vector<String>& layersTypes) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getLayerTypes(layersTypes);
}

int Net::getLayersCount(const String& layerType) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getLayersCount(layerType);
}

void Net::getMemoryConsumption(const int layerId,
        const std::vector<MatShape>& netInputShapes,
        size_t& weights, size_t& blobs) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getMemoryConsumption(layerId, netInputShapes, weights, blobs);
}

void Net::getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
        size_t& weights, size_t& blobs) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getMemoryConsumption(netInputShapes, weights, blobs);
}

void Net::getMemoryConsumption(const int layerId,
        const MatShape& netInputShape,
        size_t& weights, size_t& blobs) const
{
    getMemoryConsumption(layerId, std::vector<MatShape>(1, netInputShape),
            weights, blobs);
}

void Net::getMemoryConsumption(const MatShape& netInputShape,
        size_t& weights, size_t& blobs) const
{
    getMemoryConsumption(std::vector<MatShape>(1, netInputShape),
            weights, blobs);
}

void Net::getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
        std::vector<int>& layerIds, std::vector<size_t>& weights,
        std::vector<size_t>& blobs) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getMemoryConsumption(netInputShapes, layerIds, weights, blobs);
}

void Net::getMemoryConsumption(const MatShape& netInputShape, std::vector<int>& layerIds,
        std::vector<size_t>& weights, std::vector<size_t>& blobs) const
{
    getMemoryConsumption(std::vector<MatShape>(1, netInputShape), layerIds,
            weights, blobs);
}

// FIXIT return old value or add get method
void Net::enableFusion(bool fusion)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->enableFusion(fusion);
}

void Net::enableWinograd(bool useWinograd)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->enableWinograd(useWinograd);
}

void Net::setHalideScheduler(const String& scheduler)
{
    CV_TRACE_FUNCTION();
    CV_TRACE_ARG_VALUE(scheduler, "scheduler", scheduler.c_str());
    CV_Assert(impl);
    return impl->setHalideScheduler(scheduler);
}

int64 Net::getPerfProfile(std::vector<double>& timings)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getPerfProfile(timings);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
