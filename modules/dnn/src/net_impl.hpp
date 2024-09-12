// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_SRC_NET_IMPL_HPP__
#define __OPENCV_DNN_SRC_NET_IMPL_HPP__

#include "op_inf_engine.hpp"
#include "ie_ngraph.hpp"
#include "op_vkcom.hpp"
#include "op_cuda.hpp"
#include "op_webnn.hpp"
#include "op_timvx.hpp"
#include "op_cann.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/layer_reg.private.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/logger.hpp>

#include "layer_internals.hpp"  // LayerPin LayerData DataLayer

#include "legacy_backend.hpp"  // wrapMat BlobManager OpenCLBackendWrapper

#include <unordered_map>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::make_pair;
using std::string;

typedef std::unordered_map<std::string, int64_t> NamesHash;

// NB: Implementation is divided between of multiple .cpp files
struct Net::Impl : public detail::NetImplBase
{
    typedef std::map<int, LayerShapes> LayersShapesMap;
    typedef std::map<int, LayerData> MapIdToLayerData;

    virtual ~Impl();
    Impl();
    Impl(const Impl&) = delete;

    // Inheritance support
    Ptr<Net::Impl> basePtr_;

    Ptr<DataLayer> netInputLayer;
    std::vector<LayerPin> blobsToKeep;
    MapIdToLayerData layers;
    std::map<String, int> layerNameToId;
    std::map<std::string, int> outputNameToId;  // use registerOutput() to populate outputs
    BlobManager blobManager;
    int preferableBackend;
    int preferableTarget;
    bool hasDynamicShapes;
    // Map host data to backend specific wrapper.
    std::map<void*, Ptr<BackendWrapper>> backendWrappers;

    int lastLayerId;

    bool netWasAllocated;
    bool netWasQuantized;
    bool fusion;
    bool isAsync;  // FIXIT: drop
    bool useWinograd;
    std::vector<int64> layersTimings;

    ModelFormat modelFormat;
    DataLayout originalLayout;
    int onnx_opset;

    NamesHash argnames;
    NamesHash dimnames;
    NamesHash graphofs;
    size_t totalLayers;
    std::vector<std::string> dimnames_vec;
    std::vector<ArgData> args;
    std::vector<Mat> tensors;
    std::vector<int> bufidxs;
    std::vector<Mat> buffers;
    std::vector<Mat> scratchBufs;
    std::vector<Ptr<Graph> > allgraphs;

    Ptr<Graph> mainGraph;
    int globGraphIdx;

    int accuracy;
    bool enableFP16, haveFP16;
    bool prepared; // need to rerun graph transformations/optimizations
    bool finalizeLayers; // need to initialize each layer
    TracingMode tracingMode;
    ProfilingMode profilingMode;
    std::vector<int64_t> dimvalues;
    std::ostream* dump_strm;
    int dump_indent;

    virtual bool empty() const;
    virtual void setPreferableBackend(Net& net, int backendId);
    virtual void setPreferableTarget(int targetId);

    // FIXIT use inheritance
    virtual Ptr<BackendWrapper> wrap(Mat& host);


    virtual void clear();


    virtual void validateBackendAndTarget();

    void setUpNet(const std::vector<LayerPin>& blobsToKeep_ = std::vector<LayerPin>());


    virtual Ptr<Layer> createLayerInstance(const LayerData& ld) const
    {
        return LayerFactory::createLayerInstance(ld.type, const_cast<LayerParams&>(ld.params));
    }
    Ptr<Layer> getLayerInstance(LayerData& ld) const
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(type, "type", ld.type.c_str());

        if (ld.layerInstance)
            return ld.layerInstance;

        ld.layerInstance = createLayerInstance(ld);
        if (!ld.layerInstance && basePtr_)
        {
            ld.layerInstance = basePtr_->createLayerInstance(ld);
            CV_LOG_IF_DEBUG(NULL, ld.layerInstance, "Created layer \"" + ld.name + "\" of type \"" + ld.type + "\" from upstream layers registry");
        }
        if (!ld.layerInstance)
        {
            CV_Error(Error::StsError, "Can't create layer \"" + ld.name + "\" of type \"" + ld.type + "\"");
        }

        return ld.layerInstance;
    }

    Ptr<Layer> getLayer(int layerId) const;
    Ptr<Layer> getLayer(const LayerId& layerId) const;

    int getLayerId(const String& layerName) const;

    int getLayerId(int id) const;

    int getLayerId(DictValue& layerDesc) const;

    String getLayerName(int id) const;

    LayerData& getLayerData(int id) const;

    LayerData& getLayerData(const String& layerName) const;

    LayerData& getLayerData(const DictValue& layerDesc) const;

    static void addLayerInput(LayerData& ld, int inNum, LayerPin from);

    int resolvePinOutputName(LayerData& ld, const String& outName) const;

    LayerPin getPinByAlias(const String& layerName) const;

    std::vector<LayerPin> getLayerOutPins(const String& layerName) const;

    // FIXIT remove dtype
    int addLayer(const String& name, const String& type, const int& dtype, LayerParams& params);

    int addLayerToPrev(const String& name, const String& type, const int& dtype, LayerParams& params);


    void connect(int outLayerId, int outNum, int inLayerId, int inNum);

    int registerOutput(const std::string& outputName, int layerId, int outputPort);

    // FIXIT drop "unconnected" API
    std::vector<int> getUnconnectedOutLayers() const;
    std::vector<String> getUnconnectedOutLayersNames() /*const*/;


    void setInputsNames(const std::vector<String>& inputBlobNames);
    void setInputShape(const String& inputName, const MatShape& shape);
    virtual void setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean);
    Mat getParam(int layer, int numParam) const;
    void setParam(int layer, int numParam, const Mat& blob);
    std::vector<Ptr<Layer>> getLayerInputs(int layerId) const;
    std::vector<String> getLayerNames() const;


    // TODO drop?
    void getLayerTypes(std::vector<String>& layersTypes) const;
    int getLayersCount(const String& layerType) const;


    virtual void initBackend(const std::vector<LayerPin>& blobsToKeep_);

#ifdef HAVE_WEBNN
    void addWebnnOutputs(LayerData& ld);
    void initWebnnBackend(const std::vector<LayerPin>& blobsToKeep_);
#endif

#ifdef HAVE_VULKAN
    Ptr<vkcom::Context> context;
    void initVkComBackend();
#endif

#ifdef HAVE_TIMVX
    // Create timVxInfo for reserve tvGraphList.
    TimVXInfo timVxInfo = TimVXInfo();
    void tvUpdateConfictMap(int graphIndex, LayerData& ld, std::vector<std::vector<int> >& graphConflictMap);
    void tvConvertToOutputNode(const LayerData& ld, Ptr<TimVXBackendWrapper>& targetWrap);
    void initTimVXBackend();
#endif

#ifdef HAVE_CUDA
    struct CudaInfo_t
    {
        CudaInfo_t(cuda4dnn::csl::CSLContext ctxt, cuda4dnn::csl::Stream d2h_stream_)
            : context(std::move(ctxt))
            , d2h_stream(std::move(d2h_stream_))
        {}
        cuda4dnn::csl::CSLContext context;
        cuda4dnn::csl::Stream d2h_stream;
        cuda4dnn::csl::Workspace workspace;
    };

    std::unique_ptr<CudaInfo_t> cudaInfo;

    void initCUDABackend(const std::vector<LayerPin>& blobsToKeep_);
#endif

    void allocateLayer(int lid, const LayersShapesMap& layersShapes);

    // TODO add getter
    void enableFusion(bool fusion_);

    virtual void fuseLayers(const std::vector<LayerPin>& blobsToKeep_);
    void enableWinograd(bool useWinograd_);

    void allocateLayers(const std::vector<LayerPin>& blobsToKeep_);

    virtual void forwardLayer(LayerData& ld);

    void forwardToLayer(LayerData& ld, bool clearFlags = true);

    Mat forward(const String& outputName);
    AsyncArray forwardAsync(const String& outputName);
    void forward(OutputArrayOfArrays outputBlobs, const String& outputName);
    void forward(OutputArrayOfArrays outputBlobs,
            const std::vector<String>& outBlobNames);
    void forward(std::vector<std::vector<Mat>>& outputBlobs,
            const std::vector<String>& outBlobNames);


    void getLayerShapesRecursively(int id, LayersShapesMap& inOutShapes);

    void getLayersShapes(
            const ShapesVec& netInputShapes,
            const TypesVec& netInputTypes,
            std::vector<int>& layersIds,
            std::vector<ShapesVec>& inLayersShapes,
            std::vector<ShapesVec>& outLayersShapes) /*const*/;

    void getLayersShapes(const ShapesVec& netInputShapes,
            const TypesVec& netInputTypes,
            LayersShapesMap& inOutShapes);

    void getLayerShapes(const ShapesVec& netInputShapes,
            const TypesVec& netInputTypes,
            const int layerId,
            LayerShapes& shapes);

    void updateLayersShapes();

    int64 getFLOPS(const std::vector<MatShape>& netInputShapes,
                   const std::vector<MatType>& netInputTypes) /*const*/;
    int64 getFLOPS(
            const int layerId,
            const std::vector<MatShape>& netInputShapes,
            const std::vector<MatType>& netInputTypes) /*const*/;

    void getMemoryConsumption(
            const int layerId,
            const std::vector<MatShape>& netInputShapes,
            const std::vector<MatType>& netInputTypes,
            size_t& weights, size_t& blobs) /*const*/;
    void getMemoryConsumption(
            const std::vector<MatShape>& netInputShapes,
            const std::vector<MatType>& netInputTypes,
            size_t& weights, size_t& blobs) /*const*/;
    void getMemoryConsumption(
            const std::vector<MatShape>& netInputShapes,
            const std::vector<MatType>& netInputTypes,
            std::vector<int>& layerIds, std::vector<size_t>& weights,
            std::vector<size_t>& blobs) /*const*/;
    int64 getPerfProfile(std::vector<double>& timings) const;

    // TODO drop
    LayerPin getLatestLayerPin(const std::vector<LayerPin>& pins) const;

    Mat getBlob(const LayerPin& pin) const;

    Mat getBlob(String outputName) const;

    virtual AsyncArray getBlobAsync(const LayerPin& pin);

    AsyncArray getBlobAsync(String outputName);

    string dump(bool forceAllocation = false) const;
    string dumpToPbtxt(bool forceAllocation = false) const;

    void dumpNetworkToFile() const;

    ///////////////////////////// the new engine ////////////////////////////

    // Create a new graph/subgraph, mode 2: we construct the graph manually.
    // First, we create empty graph with certain input Args (they may or may not have names).
    // once the graph is constructed, we set the graph outputs using Graph::setOutputs().
    // When it's the first created graph, it automatically becomes the main model graph.
    Ptr<Graph> newGraph(const std::string& name,
                        const std::vector<Arg>& inputs,
                        bool isMainGraph);

    const ArgData& argData(Arg arg) const;
    const std::string& argName(Arg arg) const;
    ArgKind argKind(Arg arg) const;

    // if the name is empty, always creates a new argument;
    // if it's not empty, returns argument with the specific name if it already exists,
    // otherwise creates new argument with the specified name
    Arg getArg(const std::string& name);
    bool haveArg(const std::string& name) const;

    Arg newConstArg(const std::string& name, const Mat& m);
    Arg newConstScalarArg(const std::string& name, int type, const void* value);
    Arg newArg(const std::string& name, ArgKind kind);
    bool isConstArg(Arg arg) const;
    bool isTempArg(Arg arg) const;
    Mat& argTensor(Arg arg) const;
    MatSize argSize(Arg arg) const;
    int argType(Arg arg) const;
    void checkArg(Arg arg) const;
    void checkArgs(const std::vector<Arg>& args) const;

    int findDim(const std::string& name, bool insert=false);

    void prepareForInference();

    // pre-allocates memory for output tensors.
    // if useBufferPool==true, the method uses 'buffers'
    // for outputs (according to bufidxs)
    // instead of allocating fresh outputs
    void allocateLayerOutputs(const Ptr<Layer>& layer,
                              const std::vector<int>& inpTypes,
                              const std::vector<MatShape>& inpShapes,
                              std::vector<int>& outTypes,
                              std::vector<MatShape>& outShapes,
                              std::vector<Mat>& outputs, // [TODO] replace with something else to cover other backends
                              std::vector<int>& tempTypes,
                              std::vector<MatShape>& tempShapes,
                              std::vector<Mat>& temps, // [TODO] ditto
                              std::vector<Mat>& globalTemps,
                              bool useBufferPool
                              );

    // set input of the model before running it
    void setMainGraphInput(InputArray blob, const std::string& name);
    // set input in some graph, the main one or a subgraph
    void setGraphInput(Ptr<Graph>& graph, size_t idx, const Mat& m);
    // run graph or subgraph.
    void forwardGraph(Ptr<Graph>& graph, InputArrayOfArrays inputs, OutputArrayOfArrays outputs, bool isMainGraph);
    // run the whole model
    void forwardMainGraph(InputArrayOfArrays inputs, OutputArrayOfArrays outputs);
    // run the whole model, convenience wrapper
    Mat forwardWithSingleOutput(const std::string& outname);
    // run the whole model, convenience wrapper
    void forwardWithMultipleOutputs(OutputArrayOfArrays outputBlobs,
                                    const std::vector<std::string>& outBlobNames);
    // try infer shapes; if some layers produce tensors with dynamic shapes, shape inference is impossible
    void tryInferShapes(const std::vector<MatShape>& suggestedInpShapes,
                        const std::vector<MatType>& suggestedInpTypes,
                        LayerShapes& shapes,
                        std::vector<MatShape>& shapeCache,
                        std::vector<MatType>& typeCache) const;
    bool tryInferGraphShapes(const Ptr<Graph>& graph,
                             std::vector<MatShape>& shapeCache,
                             std::vector<MatType>& typeCache) const;

    // helper function for useCounts()
    void updateUseCounts(const Ptr<Graph>& graph, std::vector<int>& usecounts) const;
    // computes how many times each argument is used, i.e. on output usecounts.size() == args.size()
    void useCounts(std::vector<int>& usecounts) const;

    int updateGraphOfs(const Ptr<Graph>& graph, int currofs, bool ismain);

    // deals with numeric and symblic shape values.
    void checkAndUpdateDim(const Ptr<Graph>& graph, const Layer* layer, Arg inp, int j, int64_t value);

    // dump information about certain input or output argument of an operation
    void traceArg(std::ostream& strm_, const char* prefix, size_t i, Arg arg, bool dumpdata);
    std::ostream& dumpArg(std::ostream& strm, Arg arg, int indent,
                          bool comma, bool dump_details) const;
    std::ostream& dumpDim(std::ostream& strm, int value) const;
    std::ostream& dumpTypeShape(std::ostream& strm, int type, const MatShape& shape) const;
    std::ostream& dump(std::ostream& strm);

    // infers all types
    void inferTypes();
    // infers all shapes
    void inferShapes(bool symbolic);
    // sets certain buffer index for each intermediate argument (Arg)
    void assignBuffers();
    //void useBlockLayout();
    void fuse();
    void constFold();
    void constArgs();

};  // Net::Impl

Net readNetFromONNX2(const String&);
Net readNetFromONNX2(const char*, size_t);
Net readNetFromONNX2(const std::vector<uchar>&);

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // __OPENCV_DNN_SRC_NET_IMPL_HPP__
