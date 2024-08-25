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
    setPreferableBackend(DNN_BACKEND_DEFAULT);
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

void Net::dumpToPbtxt(const String& path)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    CV_Assert(!empty());
    std::ofstream file(path.c_str());
    file << impl->dumpToPbtxt(true);
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
        const TypesVec& netInputTypes,
        std::vector<int>& layersIds,
        std::vector<ShapesVec>& inLayersShapes,
        std::vector<ShapesVec>& outLayersShapes) const
{
    CV_Assert(impl);
    return impl->getLayersShapes(netInputShapes, netInputTypes, layersIds, inLayersShapes, outLayersShapes);
}

void Net::getLayersShapes(const MatShape& netInputShape,
        const MatType& netInputType,
        std::vector<int>& layerIds,
        std::vector<ShapesVec>& inLayersShapes,
        std::vector<ShapesVec>& outLayersShapes) const
{
    getLayersShapes(ShapesVec(1, netInputShape),
            TypesVec(1, netInputType),
            layerIds, inLayersShapes, outLayersShapes);
}

void Net::getLayerShapes(const MatShape& netInputShape,
        const MatType& netInputType,
        const int layerId,
        ShapesVec& inLayerShapes,
        ShapesVec& outLayerShapes) const
{
    getLayerShapes(ShapesVec(1, netInputShape), TypesVec(1, netInputType),
            layerId, inLayerShapes, outLayerShapes);
}

void Net::getLayerShapes(const ShapesVec& netInputShapes,
        const TypesVec& netInputTypes,
        const int layerId,
        ShapesVec& inLayerShapes,
        ShapesVec& outLayerShapes) const
{
    CV_Assert(impl);
    LayerShapes shapes;
    impl->getLayerShapes(netInputShapes, netInputTypes, layerId, shapes);
    inLayerShapes = shapes.in;
    outLayerShapes = shapes.out;
}

int64 Net::getFLOPS(const std::vector<MatShape>& netInputShapes, const std::vector<MatType>& netInputTypes) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getFLOPS(netInputShapes, netInputTypes);
}

int64 Net::getFLOPS(const MatShape& netInputShape, const MatType& netInputType) const
{
    return getFLOPS(std::vector<MatShape>(1, netInputShape),
                    std::vector<MatType>(1, netInputType));
}

int64 Net::getFLOPS(const int layerId,
        const std::vector<MatShape>& netInputShapes,
        const std::vector<MatType>& netInputTypes) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getFLOPS(layerId, netInputShapes, netInputTypes);
}

int64 Net::getFLOPS(const int layerId,
        const MatShape& netInputShape,
        const MatType& netInputType) const
{
    return getFLOPS(layerId, std::vector<MatShape>(1, netInputShape),
                    std::vector<MatType>(1, netInputType));
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
        const std::vector<MatType>& netInputTypes,
        size_t& weights, size_t& blobs) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getMemoryConsumption(layerId, netInputShapes, netInputTypes, weights, blobs);
}

void Net::getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
        const std::vector<MatType>& netInputTypes,
        size_t& weights, size_t& blobs) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getMemoryConsumption(netInputShapes, netInputTypes, weights, blobs);
}

void Net::getMemoryConsumption(const int layerId,
        const MatShape& netInputShape,
        const MatType& netInputType,
        size_t& weights, size_t& blobs) const
{
    getMemoryConsumption(layerId, std::vector<MatShape>(1, netInputShape),
        std::vector<MatType>(1, netInputType),
        weights, blobs);
}

void Net::getMemoryConsumption(const MatShape& netInputShape,
        const MatType& netInputType,
        size_t& weights, size_t& blobs) const
{
    getMemoryConsumption(std::vector<MatShape>(1, netInputShape),
            std::vector<MatType>(1, netInputType),
            weights, blobs);
}

void Net::getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
        const std::vector<MatType>& netInputTypes,
        std::vector<int>& layerIds, std::vector<size_t>& weights,
        std::vector<size_t>& blobs) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getMemoryConsumption(netInputShapes, netInputTypes, layerIds, weights, blobs);
}

void Net::getMemoryConsumption(const MatShape& netInputShape, const MatType& netInputType,
        std::vector<int>& layerIds,
        std::vector<size_t>& weights, std::vector<size_t>& blobs) const
{
    getMemoryConsumption(std::vector<MatShape>(1, netInputShape),
            std::vector<MatType>(1, netInputType),
            layerIds, weights, blobs);
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

int64 Net::getPerfProfile(std::vector<double>& timings)
{
    CV_TRACE_FUNCTION();
    CV_Assert(impl);
    return impl->getPerfProfile(timings);
}

bool Net::isConstArg(Arg arg) const
{
    return argKind(arg) == DNN_ARG_CONST;
}

void Net::checkArg(Arg arg) const
{
    CV_Assert(impl);
    impl->checkArg(arg);
}

void Net::checkArgs(const std::vector<Arg>& args) const
{
    CV_Assert(impl);
    impl->checkArgs(args);
}

const ArgInfo& Net::argInfo(Arg arg) const
{
    checkArg(arg);
    return impl->args[arg.idx];
}

std::string Net::argName(Arg arg) const { return argInfo(arg).name; }

ArgKind Net::argKind(Arg arg) const { return argInfo(arg).kind; }

Arg Net::getArg(const std::string& name)
{
    CV_Assert(impl);
    if (!name.empty()) {
        auto it = impl->argnames.find(name);
        if (it != impl->argnames.end()) {
            return Arg((int)it->second);
        }
    }
    return newArg(name, DNN_ARG_TEMP);
}

bool Net::haveArg(const std::string& name) const
{
    CV_Assert(impl);
    return impl->argnames.find(name) != impl->argnames.end();
}

Arg Net::newConstArg(const std::string& name, const Mat& m) const
{
    CV_Assert(impl);
    Arg arg = newArg(name, DNN_ARG_CONST);
    impl->tensors[arg.idx] = m;
    ArgInfo& info = impl->args[arg.idx];
    info.type = m.type();
    info.shape = m.shape();
    return arg;
}

Arg Net::newArg(const std::string& name, ArgKind kind) const
{
    CV_Assert(impl);
    int idx = (int)impl->args.size();

    if (!name.empty()) {
        CV_Assert(impl->argnames.find(name) == impl->argnames.end());
        impl->argnames.insert(std::make_pair(name, idx));
    }

    ArgInfo info;
    info.name = name;
    info.kind = kind;
    impl->args.push_back(info);
    impl->tensors.push_back(Mat());
    impl->bufidxs.push_back(-1);

    return Arg(idx);
}

Ptr<Graph> Net::getMainGraph() const
{
    CV_Assert(impl);
    return impl->mainGraph;
}

std::ostream& Net::dumpArg(std::ostream& strm, Arg arg, int indent,
                           bool comma, bool dump_details) const
{
    CV_Assert(impl);
    return impl->dumpArg(strm, arg, indent, comma, dump_details);
}

std::ostream& Net::dumpDim(std::ostream& strm, int value) const
{
    CV_Assert(impl);
    return impl->dumpDim(strm, value);
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
