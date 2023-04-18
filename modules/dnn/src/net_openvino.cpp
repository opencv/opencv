// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "net_impl.hpp"

#include "backend.hpp"
#include "factory.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_INF_ENGINE

// TODO: use "string" target specifier
class NetImplOpenVINO CV_FINAL : public Net::Impl
{
public:
    typedef Net::Impl Base;

    // this default constructor is used with OpenVINO native loader
    // TODO: dedicated Impl?
    NetImplOpenVINO()
        : Net::Impl()
    {
        preferableBackend = DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    // constructor to derive execution implementation from the loaded network
    explicit NetImplOpenVINO(const Ptr<Net::Impl>& basePtr)
        : Net::Impl()
    {
        basePtr_ = basePtr;
        init();
    }

    void init()
    {
        CV_TRACE_FUNCTION();
        CV_Assert(basePtr_);
        Net::Impl& base = *basePtr_;
        CV_Assert(!base.netWasAllocated);
        CV_Assert(!base.netWasQuantized);
        netInputLayer = base.netInputLayer;
        blobsToKeep = base.blobsToKeep;
        layers = base.layers;
        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
        {
            LayerData& ld = it->second;
            ld.resetAllocation();
        }
        layerNameToId = base.layerNameToId;
        outputNameToId = base.outputNameToId;
        //blobManager = base.blobManager;
        preferableBackend = DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;  //base.preferableBackend;
        preferableTarget = base.preferableTarget;
        hasDynamicShapes = base.hasDynamicShapes;
        CV_Assert(base.backendWrappers.empty());  //backendWrappers = base.backendWrappers;
        lastLayerId = base.lastLayerId;
        netWasAllocated = base.netWasAllocated;
        netWasQuantized = base.netWasQuantized;
        fusion = base.fusion;
    }


    //bool isAsync;  // FIXIT: drop


    bool empty() const override
    {
        return Base::empty();
    }
    void setPreferableBackend(Net& net, int backendId) override
    {
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return;  // no-op
        if (!basePtr_)
            CV_Error(Error::StsError, "DNN: Can't switch backend of network created by OpenVINO native loader");
        Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
        impl_ptr_ref = basePtr_;
        basePtr_->setPreferableBackend(net, backendId);
    }

    void setPreferableTarget(int targetId) override
    {
        if (preferableTarget != targetId)
        {
            preferableTarget = targetId;
            clear();
        }
    }

    Ptr<BackendWrapper> wrap(Mat& host) override
    {
        return Ptr<BackendWrapper>(new NgraphBackendWrapper(preferableTarget, host));
    }


    void clear() override
    {
        Base::clear();
    }

    void validateBackendAndTarget() override
    {
        CV_TRACE_FUNCTION();

        CV_Assert(preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);
        CV_Check((int)preferableTarget,
              preferableTarget == DNN_TARGET_CPU ||
              preferableTarget == DNN_TARGET_OPENCL ||
              preferableTarget == DNN_TARGET_OPENCL_FP16 ||
              preferableTarget == DNN_TARGET_MYRIAD ||
              preferableTarget == DNN_TARGET_HDDL ||
              preferableTarget == DNN_TARGET_FPGA,
              "Unknown OpenVINO target"
        );
    }

    Ptr<Layer> createLayerInstance(const LayerData& ld) const override
    {
        // try to create layer instance from backend-specific pool (e.g., plugin)
        Ptr<Layer> instance = LayerFactory::createLayerInstance(ld.type, const_cast<LayerParams&>(ld.params));
        if (!instance)
            instance = Base::createLayerInstance(ld);
        return instance;
    }

    void addNgraphOutputs(LayerData& ld);

    void initBackend(const std::vector<LayerPin>& blobsToKeep_) override;

    void fuseLayers(const std::vector<LayerPin>& blobsToKeep_) override;

    void forwardLayer(LayerData& ld) override;

    AsyncArray getBlobAsync(const LayerPin& pin) override;

    //string dump(bool forceAllocation = false) const override;

    static
    Net createNetworkFromModelOptimizer(InferenceEngine::CNNNetwork& ieNet);

};  // NetImplOpenVINO


void NetImplOpenVINO::forwardLayer(LayerData& ld)
{
    CV_TRACE_FUNCTION();

    Ptr<Layer> layer = ld.layerInstance;

    if (!ld.skip)
    {
        auto it = ld.backendNodes.find(preferableBackend);
        if (ld.id == 0 ||  // input layer
            it == ld.backendNodes.end()  // non-supported layer or its mode
        )
        {
            return Base::forwardLayer(ld);
        }

        CV_Assert(it != ld.backendNodes.end());
        const Ptr<BackendNode>& node = it->second;
        CV_Assert(!node.empty());
        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        CV_Assert(!ieNode.empty());
        CV_Assert(ieNode->net);

        TickMeter tm;
        tm.start();

        ieNode->net->forward(ld.outputBlobsWrappers, isAsync);

        tm.stop();
        int64 t = tm.getTimeTicks();
        layersTimings[ld.id] = (t > 0) ? t : 1;  // zero for skipped layers only
    }
    else
    {
        layersTimings[ld.id] = 0;
    }

    ld.flag = 1;
}

AsyncArray NetImplOpenVINO::getBlobAsync(const LayerPin& pin)
{
    CV_TRACE_FUNCTION();
    if (!pin.valid())
        CV_Error(Error::StsObjectNotFound, "Requested blob not found");

    LayerData& ld = layers[pin.lid];
    if ((size_t)pin.oid >= ld.outputBlobs.size())
    {
        CV_Error(Error::StsOutOfRange, format("Layer \"%s\" produce only %d outputs, "
                                              "the #%d was requested",
                                               ld.name.c_str(), (int)ld.outputBlobs.size(), (int)pin.oid));
    }
    if (preferableTarget != DNN_TARGET_CPU)
    {
        CV_Assert(!ld.outputBlobsWrappers.empty() && !ld.outputBlobsWrappers[pin.oid].empty());
        // Transfer data to CPU if it's require.
        ld.outputBlobsWrappers[pin.oid]->copyToHost();
    }
    CV_Assert(preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);

    Ptr<NgraphBackendWrapper> wrapper = ld.outputBlobsWrappers[pin.oid].dynamicCast<NgraphBackendWrapper>();
    return std::move(wrapper->futureMat);
}


/** mark input pins as outputs from other subnetworks
 * FIXIT must be done by DNN engine not ngraph.
 */
void NetImplOpenVINO::addNgraphOutputs(LayerData& ld)
{
    CV_TRACE_FUNCTION();

    CV_LOG_DEBUG(NULL, "DNN/IE: layer of new subnet: " << ld.name << "@" << ld.type);

    Ptr<InfEngineNgraphNet> layerNet;
    auto it = ld.backendNodes.find(preferableBackend);
    if (it != ld.backendNodes.end())
    {
        Ptr<BackendNode> node = it->second;
        if (!node.empty())
        {
            Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
            CV_Assert(!ieNode.empty());
            CV_Assert(!ieNode->net.empty());
            layerNet = ieNode->net;
        }
    }

    for (int i = 0; i < ld.inputBlobsId.size(); ++i)
    {
        LayerData& inpLd = layers[ld.inputBlobsId[i].lid];
        Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
        if (!inpNode.empty())
        {
            Ptr<InfEngineNgraphNode> ieInpNode = inpNode.dynamicCast<InfEngineNgraphNode>();
            CV_Assert(!ieInpNode.empty());
            CV_Assert(!ieInpNode->net.empty());
            if (layerNet != ieInpNode->net)
            {
                CV_LOG_DEBUG(NULL, "DNN/IE: pin output between subnets: " << ieInpNode->node->get_friendly_name());
                ieInpNode->net->addOutput(ieInpNode);
            }
        }
    }
}

void NetImplOpenVINO::initBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_CheckEQ(preferableBackend, DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, "");

    Ptr<InfEngineNgraphNet> net;

    for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); ++it)
    {
        const LayerData& ld = it->second;
        if (ld.id == 0)
        {
            CV_Assert((netInputLayer->outNames.empty() && ld.outputBlobsWrappers.size() == 1) ||
                      (netInputLayer->outNames.size() == ld.outputBlobsWrappers.size()));
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                std::string outputName = netInputLayer->outNames.empty() ? ld.name : netInputLayer->outNames[i];
                outputName = ld.outputBlobsWrappers.size() > 1 ? (outputName + "." + std::to_string(i)) : outputName;
                ld.outputBlobsWrappers[i].dynamicCast<NgraphBackendWrapper>()->name = outputName;
            }
        }
        else
        {
            for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
            {
                std::string outputName = ld.outputBlobsWrappers.size() > 1 ? (ld.name + "." + std::to_string(i)) : ld.name;
                ld.outputBlobsWrappers[i].dynamicCast<NgraphBackendWrapper>()->name = outputName;
            }
        }
    }

    if (!basePtr_)  // model is loaded by OpenVINO
    {
        Ptr<BackendNode> node = layers[lastLayerId].backendNodes[preferableBackend];
        CV_Assert(!node.empty());

        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        CV_Assert(!ieNode.empty());

        CV_Assert(ieNode->net);
        InfEngineNgraphNet& ienet = *ieNode->net;
        ienet.reset();

        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
        {
            LayerData& ld = it->second;
            if (ld.id == 0)
            {
                for (int i = 0; i < ld.inputBlobsWrappers.size(); ++i)
                {
                    ld.inputBlobsWrappers[i].dynamicCast<NgraphBackendWrapper>()->name = netInputLayer->outNames[i];
                }
            }
            ienet.addBlobs(ld.inputBlobsWrappers);
            ienet.addBlobs(ld.outputBlobsWrappers);
            ld.skip = true;
        }
        layers[lastLayerId].skip = false;
        ienet.init((Target)preferableTarget);
        return;
    }

    bool supportsCPUFallback = !isArmComputePlugin() && (preferableTarget == DNN_TARGET_CPU ||
                               openvino::checkTarget(DNN_TARGET_CPU));

    // Build Inference Engine networks from sets of layers that support this
    // backend. Split a whole model on several Inference Engine networks if
    // some of layers are not implemented.
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData& ld = it->second;

        CV_LOG_DEBUG(NULL, "DNN/IE: processing layer " << ld.name << "@" << ld.type << " (" << ld.id << ") ...");

        if (ld.id == 0 && ld.skip)
        {
            CV_LOG_DEBUG(NULL, "DNN/IE:    SKIP!");
            continue;
        }

        bool fused = ld.skip;
        Ptr<Layer> layer = ld.layerInstance;
        if (!fused && !layer->supportBackend(preferableBackend))
        {
            CV_LOG_DEBUG(NULL, "DNN/IE:    NOT supported!");
            bool customizable = ld.id != 0 && supportsCPUFallback;

            // TODO: there is a bug in Myriad plugin with custom layers shape infer.
            if (preferableTarget == DNN_TARGET_MYRIAD || preferableTarget == DNN_TARGET_HDDL)
            {
                for (int i = 0; customizable && i < ld.inputBlobs.size(); ++i)
                {
                    customizable = ld.inputBlobs[i]->size[0] == 1;
                }
            }

            // TODO: fix these workarounds
            if (preferableTarget == DNN_TARGET_MYRIAD ||
                preferableTarget == DNN_TARGET_HDDL ||
                preferableTarget == DNN_TARGET_OPENCL ||
                preferableTarget == DNN_TARGET_OPENCL_FP16)
                customizable &= ld.type != "Concat";

            if (preferableTarget == DNN_TARGET_OPENCL ||
                preferableTarget == DNN_TARGET_OPENCL_FP16)
                customizable &= ld.type != "Power";

            if (preferableTarget == DNN_TARGET_OPENCL)
                customizable &= ld.type != "Eltwise";

            if (!customizable)
            {
                CV_LOG_DEBUG(NULL, "DNN/IE:    NOT customizable!");
                addNgraphOutputs(ld);
                net = Ptr<InfEngineNgraphNet>();
                layer->preferableTarget = DNN_TARGET_CPU;

                for (int i = 0; i < ld.inputBlobsId.size(); ++i)
                {
                    LayerData& inpLd = layers[ld.inputBlobsId[i].lid];
                    Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
                    if (!inpNode.empty())
                    {
                        Ptr<InfEngineNgraphNode> ieNode = inpNode.dynamicCast<InfEngineNgraphNode>();
                        CV_Assert(!ieNode.empty());
                        ieNode->net->addOutput(ieNode);
                    }
                }
                continue;
            }
        }
        ld.skip = true;  // Initially skip all Inference Engine supported layers.

        // Create a new network if one of inputs from different Inference Engine graph.
        std::vector<Ptr<BackendNode>> inputNodes;
        for (int i = 0; i < ld.inputBlobsId.size(); ++i)
        {
            // Layer_Test_ROIPooling.Accuracy has 2 inputs inpLD = 0, 0 -> has 4 inputNodes (input, rois, input, rois)
            if (inputNodes.size() == ld.inputBlobsId.size())
            {
                break;
            }
            LayerData& inpLd = layers[ld.inputBlobsId[i].lid];
            Ptr<BackendNode> inpNode = inpLd.backendNodes[preferableBackend];
            if (!inpNode.empty())
            {
                Ptr<InfEngineNgraphNode> ieInpNode = inpNode.dynamicCast<InfEngineNgraphNode>();
                CV_Assert(!ieInpNode.empty());
                CV_Assert(!ieInpNode->net.empty());
                if (ieInpNode->net == net && !fused)
                {
                    inputNodes.push_back(inpNode);
                    continue;
                }
            }

            if (net.empty())
            {
                net = Ptr<InfEngineNgraphNet>(new InfEngineNgraphNet(*this));
            }

            if (!fused)
            {
                std::vector<std::string> inputNames;
                std::vector<cv::Mat> inputs;

                auto curr_pos = inpLd.consumers.begin();
                auto compare = [&ld](const LayerPin& lp) { return lp.lid == ld.id; };
                auto cons = curr_pos;
                while ((cons = std::find_if(curr_pos, inpLd.consumers.end(), compare)) !=
                        inpLd.consumers.end()) {
                    int cons_inp = cons->oid;
                    Ptr<NgraphBackendWrapper> inpWrapper = inpLd.outputBlobsWrappers[cons_inp].
                                                                 dynamicCast<NgraphBackendWrapper>();
                    CV_Assert(!inpWrapper.empty());
                    auto iter = std::find(inputNames.begin(), inputNames.end(),
                            inpWrapper->name);
                    if (iter == inputNames.end())
                    {
                        inputNames.push_back(inpWrapper->name);
                        inputs.push_back(inpLd.outputBlobs[cons_inp]);
                    }
                    curr_pos = cons + 1;
                }

                auto inps = net->setInputs(inputs, inputNames);
                for (auto& inp : inps)
                {
                    inputNodes.emplace_back(Ptr<BackendNode>(new InfEngineNgraphNode(inp)));
                }
            }
        }

        Ptr<BackendNode> node;
        if (!net.empty())
        {
            if (fused)
            {
                bool inPlace = ld.inputBlobsId.size() == 1 && ld.outputBlobs.size() == 1 &&
                               ld.inputBlobs[0]->data == ld.outputBlobs[0].data;
                CV_Assert(inPlace);
                node = layers[ld.inputBlobsId[0].lid].backendNodes[preferableBackend];
                ld.inputBlobsWrappers = layers[ld.inputBlobsId[0].lid].inputBlobsWrappers;
            }
        }
        else
        {
            net = Ptr<InfEngineNgraphNet>(new InfEngineNgraphNet(*this));
        }

        if (!fused)
        {
            CV_Assert(ld.inputBlobsId.size() == inputNodes.size());
            for (int i = 0; i < ld.inputBlobsId.size(); ++i)
            {
                int lid = ld.inputBlobsId[i].lid;
                int oid = ld.inputBlobsId[i].oid;
                if (oid == 0 || lid == 0)
                    continue;

                auto ieInpNode = inputNodes[i].dynamicCast<InfEngineNgraphNode>();
                const auto& ngraph_input_node = ieInpNode->node;
                CV_LOG_DEBUG(NULL, "DNN/IE: bind output port " << lid << ":" << oid << " (" << ngraph_input_node->get_friendly_name() << ":" << ngraph_input_node->get_type_info().name << ")");

                // Handle parameters from other subnets. Output port is not used in this case
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
                if ((ngraph::op::is_parameter(ngraph_input_node) || ngraph::op::is_constant(ngraph_input_node)) &&
#else
                if ((ngraph_input_node->is_parameter() || ngraph_input_node->is_constant()) &&
#endif

                        ngraph_input_node->get_output_size() == 1)
                {
                    inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(ngraph_input_node));
                    continue;
                }
                CV_CheckLT((size_t)oid, ngraph_input_node->get_output_size(), "");
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_4)
                // FIXIT refactor ".initNgraph()" API to use Output<Node>
                // WA: use Concat to emulate Identity operation with requested output port
                auto oid_node = std::make_shared<ngraph::op::Concat>(ngraph::OutputVector { ngraph_input_node->output(oid) }, 0);
                inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(oid_node));
#elif INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2020_3)
                inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(ieInpNode->node->get_output_as_single_output_node(oid)));
#else
                inputNodes[i] = Ptr<BackendNode>(new InfEngineNgraphNode(ieInpNode->node->get_output_as_single_output_node(oid, false)));
#endif
            }

            if (layer->supportBackend(preferableBackend))
            {
                CV_LOG_DEBUG(NULL, "DNN/IE: wrap layer " << ld.name << "@" << ld.type << " - outputs: " << ld.outputBlobsWrappers.size());
                node = layer->initNgraph(ld.inputBlobsWrappers, inputNodes);
#if 0  // FIXIT doesn't work with multiple outputs (set name is applied to the same node)
                for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                {
                    InferenceEngine::DataPtr dataPtr = ngraphDataNode(ld.outputBlobsWrappers[i]);
                    node.dynamicCast<InfEngineNgraphNode>()->setName(dataPtr->getName());
                }
#else
                node.dynamicCast<InfEngineNgraphNode>()->setName(layer->name);
#endif
            }
            else
            {
                CV_LOG_DEBUG(NULL, "DNN/IE: layer is not supported: " << ld.name << "@" << ld.type);
                node = Ptr<BackendNode>(new InfEngineNgraphNode(inputNodes,
                        ld.layerInstance, ld.inputBlobs, ld.outputBlobs, ld.internals));
            }
        }
        else if (node.empty())
        {
            CV_LOG_DEBUG(NULL, "DNN/IE: node.empty() bypass...");
            continue;
        }

        ld.backendNodes[preferableBackend] = node;

        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        CV_Assert(!ieNode.empty());
        ieNode->net = net;

        for (const auto& pin : blobsToKeep_)
        {
            if (pin.lid == ld.id)
            {
                ieNode->net->addOutput(ieNode);
                break;
            }
        }
        ieNode->net->setNodePtr(&ieNode->node);

        net->addBlobs(ld.inputBlobsWrappers);
        net->addBlobs(ld.outputBlobsWrappers);
        addNgraphOutputs(ld);
    }

    // Initialize all networks.
    for (MapIdToLayerData::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it)
    {
        LayerData& ld = it->second;
        auto iter = ld.backendNodes.find(preferableBackend);
        if (iter == ld.backendNodes.end())
            continue;

        Ptr<BackendNode>& node = iter->second;
        if (node.empty())
            continue;

        Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
        if (ieNode.empty())
            continue;

        CV_Assert(!ieNode->net.empty());

        if (!ieNode->net->isInitialized())
        {
            ieNode->net->addOutput(ieNode);
            ieNode->net->createNet((Target)preferableTarget);
            ld.skip = false;
        }
    }
}


#if 0
#define printf_(args) printf args
#else
#define printf_(args)
#endif

void NetImplOpenVINO::fuseLayers(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();

    if(!fusion)
       return;

    CV_Check((int)preferableBackend, preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH, "");

#if 0  // FIXIT mode without fusion is broken due to unsupported layers and handling of "custom" nodes
    return;
#endif

    // scan through all the layers. If there is convolution layer followed by the activation layer,
    // we try to embed this activation into the convolution and disable separate execution of the activation

    // FIXIT replace by layersToKeep to avoid hacks like "LayerPin(lid, 0)"
    std::set<LayerPin> pinsToKeep(blobsToKeep_.begin(),
                                  blobsToKeep_.end());
    for (MapIdToLayerData::const_iterator it = layers.begin(); it != layers.end(); it++)
    {
        int lid = it->first;
        LayerData& ld = layers[lid];
        if (ld.skip)
        {
            printf_(("skipped %s: %s\n", ld.layerInstance->name.c_str(), ld.layerInstance->type.c_str()));
            continue;
        }
        printf_(("analyzing %s: %s\n", ld.layerInstance->name.c_str(), ld.layerInstance->type.c_str()));

        // the optimization #1. try to fuse batch norm, scaling and/or activation layers
        // with the current layer if they follow it. Normally, the are fused with the convolution layer,
        // but some of them (like activation) may be fused with fully-connected, elemwise (+) and
        // some other layers.
        Ptr<Layer>& currLayer = ld.layerInstance;
        if (ld.consumers.size() == 1 && pinsToKeep.count(LayerPin(lid, 0)) == 0)
        {
            LayerData* nextData = &layers[ld.consumers[0].lid];
            LayerPin lpNext(ld.consumers[0].lid, 0);
            while (nextData)
            {
                if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && pinsToKeep.count(lpNext) != 0)
                {
                    CV_LOG_DEBUG(NULL, "DNN/IE: skip fusing with 'output' node: " << nextData->name << "@" << nextData->type);
                    break;
                }

                /* we use `tryFuse` member of convolution layer to fuse eltwise later
                 * it's not intended to be fused here; hence, we stop when we encounter eltwise
                 */
                Ptr<Layer> nextLayer = nextData->layerInstance;
                if (currLayer->tryFuse(nextLayer))
                {
                    printf_(("\tfused with %s\n", nextLayer->name.c_str()));
                    nextData->skip = true;
                    ld.outputBlobs = layers[lpNext.lid].outputBlobs;
                    ld.outputBlobsWrappers = layers[lpNext.lid].outputBlobsWrappers;
                    if (nextData->consumers.size() == 1)
                    {
                        int nextLayerId = nextData->consumers[0].lid;
                        nextData = &layers[nextLayerId];
                        lpNext = LayerPin(nextLayerId, 0);
                    }
                    else
                    {
                        nextData = 0;
                        break;
                    }
                }
                else
                    break;
            }
        }
    }
}



void switchToOpenVINOBackend(Net& net)
{
    CV_TRACE_FUNCTION();
    Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
    CV_Assert(impl_ptr_ref);
    CV_LOG_INFO(NULL, "DNN: switching to OpenVINO backend... (networkID=" << impl_ptr_ref->networkId << ")");
    Ptr<NetImplOpenVINO> openvino_impl_ptr = makePtr<NetImplOpenVINO>(impl_ptr_ref);
    impl_ptr_ref = openvino_impl_ptr;
}


/*static*/
Net NetImplOpenVINO::createNetworkFromModelOptimizer(InferenceEngine::CNNNetwork& ieNet)
{
    CV_TRACE_FUNCTION();

    CV_TRACE_REGION("register_inputs");

    auto ngraphFunction = ieNet.getFunction();
    CV_Assert(ngraphFunction);

    std::vector<String> inputsNames;
    std::vector<MatShape> inp_shapes;
    for (auto& it : ngraphFunction->get_parameters())
    {
        inputsNames.push_back(it->get_friendly_name());
        std::vector<size_t> dims = it->get_shape();
        inp_shapes.push_back(std::vector<int>(dims.begin(), dims.end()));
    }
    // nGraph models produce output "Result" layers which have "/sink_port" suffix in their names.
    // Their inputs are actual model outputs and we change friendly name to it.
    // By this workaround, we produce similar outputs names comparing to ieNet.getOutputsInfo()
    for (int i = 0; i < ngraphFunction->get_output_size(); ++i) {
        auto res = ngraphFunction->output(i);
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
        const std::string& name = res.get_any_name();
#else
        auto out = res.get_node()->input(0).get_source_output();
        std::string name = out.get_node()->get_friendly_name();
        if (out.get_node()->get_output_size() > 1)
            name += "." + std::to_string(out.get_index());
#endif
        if (res.get_node()->get_friendly_name() != name)
            res.get_node()->set_friendly_name(name);
    }

    Net cvNet;
    Ptr<NetImplOpenVINO> openvino_impl_ptr = makePtr<NetImplOpenVINO>();
    NetImplOpenVINO& openvino_impl = *openvino_impl_ptr;
    accessor::DnnNetAccessor::getImplPtrRef(cvNet) = openvino_impl_ptr;

    cvNet.setInputsNames(inputsNames);

    // set empty input to determine input shapes
    for (int inp_id = 0; inp_id < inputsNames.size(); ++inp_id)
    {
        cvNet.setInputShape(inputsNames[inp_id], inp_shapes[inp_id]);
    }

    CV_TRACE_REGION_NEXT("backendNode");

    Ptr<BackendNode> backendNode;
    {
        auto fake_node = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape {});
        Ptr<InfEngineNgraphNode> backendNodeNGraph(new InfEngineNgraphNode(fake_node));
        backendNodeNGraph->net = Ptr<InfEngineNgraphNet>(new InfEngineNgraphNet(openvino_impl, ieNet));
        backendNode = backendNodeNGraph;
    }

    CV_TRACE_REGION_NEXT("register_outputs");

    std::vector<std::shared_ptr<ngraph::Node>> ngraphOperations = ngraphFunction->get_ops();

    for (auto& it : ngraphFunction->get_results())
    {
        CV_TRACE_REGION("output");
        const auto& outputName = it->get_friendly_name();

        LayerParams lp;
        int lid = cvNet.addLayer(outputName, "", lp);

        LayerData& ld = openvino_impl.layers[lid];

        {
            Ptr<Layer> cvLayer(new NgraphBackendLayer(ieNet));
            cvLayer->name = outputName;
            cvLayer->type = "_unknown_";

            auto process_layer = [&](const std::string& name) -> bool
            {
                CV_TRACE_REGION("ngraph_function");
                for (const auto& op : ngraphOperations)
                {
                    CV_Assert(op);
                    if (op->get_friendly_name() == name)
                    {
                        const std::string typeName = op->get_type_info().name;
                        cvLayer->type = typeName;
                        return true;
                    }
                }
                return false;
            };

            bool found = process_layer(outputName);
            if (!found)
            {
                auto pos = outputName.rfind('.');  // cut port number: ".0"
                if (pos != std::string::npos)
                {
                    std::string layerName = outputName.substr(0, pos);
                    found = process_layer(layerName);
                }
            }
            if (!found)
                CV_LOG_WARNING(NULL, "DNN/IE: Can't determine output layer type: '" << outputName << "'");

            ld.layerInstance = cvLayer;
            ld.backendNodes[DNN_BACKEND_INFERENCE_ENGINE_NGRAPH] = backendNode;
        }

        for (int i = 0; i < inputsNames.size(); ++i)
            cvNet.connect(0, i, lid, i);
    }

    CV_TRACE_REGION_NEXT("finalize");

    cvNet.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE_NGRAPH);

    return cvNet;
}


static
Net openvino_readNetwork(const String& modelPath, const String& binPath)
{
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    InferenceEngine::Core& ie = getCore("");
    InferenceEngine::CNNNetwork ieNet;
    try
    {
        ieNet = ie.ReadNetwork(modelPath, binPath);
    }
    catch (const std::exception& e)
    {
        CV_Error(Error::StsError, std::string("DNN: OpenVINO failed to read model '") + modelPath + "': " + e.what());
    }

    return NetImplOpenVINO::createNetworkFromModelOptimizer(ieNet);
}


static
Net openvino_readNetwork(
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize
)
{
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    InferenceEngine::Core& ie = getCore("");

    std::string model; model.assign((char*)bufferModelConfigPtr, bufferModelConfigSize);

    InferenceEngine::CNNNetwork ieNet;
    try
    {
#if INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2022_1)
        ov::Tensor weights_blob(ov::element::u8, {bufferWeightsSize}, (void*)bufferWeightsPtr);
        ieNet = ie.read_model(model, weights_blob);
#else
        InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, { bufferWeightsSize }, InferenceEngine::Layout::C);
        InferenceEngine::Blob::CPtr weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, (uint8_t*)bufferWeightsPtr, bufferWeightsSize);

        ieNet = ie.ReadNetwork(model, weights_blob);
#endif
    }
    catch (const std::exception& e)
    {
        CV_Error(Error::StsError, std::string("DNN: OpenVINO failed to read model: ") + e.what());
    }

    return NetImplOpenVINO::createNetworkFromModelOptimizer(ieNet);
}

#endif  // HAVE_INF_ENGINE

Net Net::readFromModelOptimizer(const String& xml, const String& bin)
{
    CV_TRACE_FUNCTION();
#if defined(HAVE_INF_ENGINE)
    return openvino_readNetwork(xml, bin);
#elif defined(ENABLE_PLUGINS)
    auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
    return networkBackend.readNetwork(std::string(), xml, bin);
#else
    CV_UNUSED(xml); CV_UNUSED(bin);
    CV_Error(Error::StsError, "Build OpenCV with Inference Engine to enable loading models from Model Optimizer.");
#endif
}

Net Net::readFromModelOptimizer(const std::vector<uchar>& bufferModelConfig, const std::vector<uchar>& bufferWeights)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!bufferModelConfig.empty());
    CV_Assert(!bufferWeights.empty());
    return readFromModelOptimizer(bufferModelConfig.data(), bufferModelConfig.size(),
                                           bufferWeights.data(), bufferWeights.size());
}

Net Net::readFromModelOptimizer(
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize
)
{
    CV_TRACE_FUNCTION();
#if defined(HAVE_INF_ENGINE)
    return openvino_readNetwork(bufferModelConfigPtr, bufferModelConfigSize, bufferWeightsPtr, bufferWeightsSize);
#elif defined(ENABLE_PLUGINS)
    auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
    return networkBackend.readNetwork(std::string(), bufferModelConfigPtr, bufferModelConfigSize, bufferWeightsPtr, bufferWeightsSize);
#else
    CV_UNUSED(bufferModelConfigPtr); CV_UNUSED(bufferWeightsPtr);
    CV_UNUSED(bufferModelConfigSize); CV_UNUSED(bufferModelConfigSize);
    CV_Error(Error::StsError, "Build OpenCV with Inference Engine to enable loading models from Model Optimizer.");
#endif
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn



#ifdef BUILD_PLUGIN

#define ABI_VERSION 0
#define API_VERSION 0
#include "plugin_api.hpp"


namespace cv { namespace dnn_backend {

using namespace cv::dnn;

class NetworkBackendOpenVINO : public NetworkBackend
{
public:
    void switchBackend(Net& net) CV_OVERRIDE
    {
        cv::dnn::switchToOpenVINOBackend(net);
    }
    Net readNetwork(const std::string& loaderID, const std::string& model, const std::string& config) CV_OVERRIDE
    {
        if (!loaderID.empty())  // only auto ("") is supported
        {
            CV_Error(Error::StsError, "DNN/OpenVINO: unsupported network loader ID: " + loaderID);
        }
        return openvino_readNetwork(model, config);
    }
    Net readNetwork(
        const std::string& loaderID,
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize
    ) CV_OVERRIDE
    {
        if (!loaderID.empty())  // only auto ("") is supported
        {
            CV_Error(Error::StsError, "DNN/OpenVINO: unsupported network loader ID: " + loaderID);
        }
        return openvino_readNetwork(bufferModelConfigPtr, bufferModelConfigSize, bufferWeightsPtr, bufferWeightsSize);
    }
    bool checkTarget(Target target) CV_OVERRIDE
    {
        return openvino::checkTarget(target);
    }
};

static
std::shared_ptr<NetworkBackendOpenVINO>& getInstanceNetworkBackendOpenVINO()
{
    static std::shared_ptr<NetworkBackendOpenVINO> g_instance = std::make_shared<NetworkBackendOpenVINO>();
    return g_instance;
}


}}  // namespace


static
CvResult cv_getInstanceNetworkBackend(CV_OUT CvPluginDNNNetworkBackend* handle) CV_NOEXCEPT
{
    try
    {
        if (!handle)
            return CV_ERROR_FAIL;
        *handle = cv::dnn_backend::getInstanceNetworkBackendOpenVINO().get();
        return CV_ERROR_OK;
    }
    catch (...)
    {
        return CV_ERROR_FAIL;
    }
}

static const OpenCV_DNN_Plugin_API plugin_api =
{
    {
        sizeof(OpenCV_DNN_Plugin_API), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "OpenVINO OpenCV DNN plugin (" CVAUX_STR(INF_ENGINE_RELEASE) ")"
    },
    {
        /*  1*/cv_getInstanceNetworkBackend
    }
};

const OpenCV_DNN_Plugin_API* CV_API_CALL opencv_dnn_plugin_init_v0(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == ABI_VERSION && requested_api_version <= API_VERSION)
        return &plugin_api;
    return NULL;
}

#endif  // BUILD_PLUGIN
