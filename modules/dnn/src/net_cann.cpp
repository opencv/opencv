// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/core/utils/logger.hpp>

#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_CANN

class NetImplCann CV_FINAL : public Net::Impl
{
public:
    typedef Net::Impl Base;

    bool netWasConverted;
    

    explicit NetImplCann(const Ptr<Net::Impl>& basePtr)
        : Net::Impl()
    {
        CV_LOG_INFO(NULL, "Initializing NetImplCann");
        basePtr_ = basePtr;
        netWasConverted = false;

        init();

        CV_LOG_INFO(NULL, "Finished initializing NetImplCann");
    }

    void init()
    {
        CV_TRACE_FUNCTION();
        CV_Assert(basePtr_);
        Net::Impl& base = *basePtr_;
        CV_Assert(!base.netWasAllocated);
        CV_Assert(!base.netWasQuantized); // does not support quantized net for now
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
        preferableBackend = DNN_BACKEND_CANN;
        preferableTarget = DNN_TARGET_NPU; // force using NPU
        hasDynamicShapes = base.hasDynamicShapes;
        CV_Assert(base.backendWrappers.empty());  //backendWrappers = base.backendWrappers;
        lastLayerId = base.lastLayerId;
        netWasAllocated = base.netWasAllocated;
        netWasQuantized = base.netWasQuantized;
        fusion = base.fusion;
    }

    bool empty() const override
    {
        return Base::empty();
    }

    void setPreferableBackend(Net& net, int backendId) override
    {
        if (backendId == preferableBackend)
            return;  // no-op
        else
            CV_Error(Error::StsError, "DNN: Can't switch backend from CANN to other");
        Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
        impl_ptr_ref = basePtr_;
        basePtr_->setPreferableBackend(net, backendId);
    }

    void setPreferableTarget(int targetId) override
    {
        if (targetId != preferableTarget)
        {
            CV_Error(Error::StsError, "DNN: Can't switch target from NPU to other");
        }
    }

    Ptr<BackendWrapper> wrap(Mat& host) override
    {
        return Ptr<BackendWrapper>(new CannBackendWrapper(host));
    }

    // void fuseLayers(const std::vector<LayerPin>& blobsToKeep_); // fusion is done in the CANN graph engine

    void initBackend(const std::vector<LayerPin>& blobsToKeep_) override;

    void forwardLayer(LayerData& ld) override;
};

// TODO: rebuild cann_net if network was changed.
void NetImplCann::initBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_CheckEQ(preferableBackend, DNN_BACKEND_CANN, "");

    if (netWasConverted && netWasAllocated) // in case requested output is changed
        return;

    // convert layers to CANN operators,
    // collect graph input and output operators,
    // collect and input and output wrappers
    int firstOutputLayerId = -1;
    std::vector<Ptr<BackendNode> > netInputNodes;
    std::vector<ge::Operator> graphInputOps, graphOutputOps;
    std::vector<Ptr<BackendWrapper>> graphInputWrappers, graphOutputWrappers;
    CV_LOG_INFO(NULL, "DNN/CANN: converting layers to CANN operators");
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData& ld = it->second;
        auto layer = ld.layerInstance;

        ld.skip = true; // skip all cann operators

        if (ld.id == 0)
        {
            for (int i = 0; i < ld.outputBlobsWrappers.size(); i++)
            {
                std::string inputName = netInputLayer->outNames.empty() ? cv::format("%s_%d", ld.name.c_str(), i) : netInputLayer->outNames[i];
                auto inputOp = std::make_shared<ge::op::Data>(inputName);

                // retrieve tensor description
                auto wrapper = ld.outputBlobsWrappers[i];
                graphInputWrappers.push_back(wrapper);
                auto cannWrapper = wrapper.dynamicCast<CannBackendWrapper>();
                CV_Assert(!cannWrapper.empty());

                inputOp->update_input_desc_x(*(cannWrapper->desc_));
                inputOp->update_output_desc_y(*(cannWrapper->desc_));

                graphInputOps.push_back(*inputOp);
                netInputNodes.push_back(Ptr<BackendNode>(new CannBackendNode(inputOp)));
            }
        }
        else
        {
            if (layer->supportBackend(preferableBackend))
            {
                std::vector<Ptr<BackendNode> > layerInputNodes;
                for (int i = 0; i < ld.inputBlobsId.size(); i++)
                {
                    int layerInputLid = ld.inputBlobsId[i].lid;
                    int layerInputOid = ld.inputBlobsId[i].oid;
                    if (layerInputLid == 0)
                    {
                        layerInputNodes.push_back(netInputNodes[layerInputOid]);
                    }
                    else // here we do not consider an op with multiple outputs
                    {
                        layerInputNodes.push_back(layers[layerInputLid].backendNodes[preferableBackend]);
                    }
                }

                CV_LOG_INFO(NULL, "DNN/CANN: converting layer " << ld.name << "@" << ld.type << "@" << ld.id << " to CANN operator");
                auto backendNode = layer->initCann(ld.inputBlobsWrappers, ld.id, layerInputNodes);

                // collect outputs
                bool isOutputNode = ld.consumers.size() == 0 ? true : false;
                if (isOutputNode)
                {
                    if (firstOutputLayerId < 0)
                        firstOutputLayerId = ld.id;
                    auto cannNode = backendNode.dynamicCast<CannBackendNode>();
                    graphOutputOps.push_back(*(cannNode->getOp()));
                    // assume cann graph outputs and dnn net outputs have the same order
                    for (int i = 0; i < ld.outputBlobsWrappers.size(); ++i)
                    {
                        graphOutputWrappers.push_back(ld.outputBlobsWrappers[i]);
                    }
                }

                ld.backendNodes[preferableBackend] = backendNode;
            }
            else
            {
                CV_LOG_INFO(NULL, "DNN/CANN: layer (name=" << ld.name << ", type=" << ld.type << ") is not supported by CANN backend. Going back to CPU backend");
            }
        }
    }
    CV_LOG_INFO(NULL, "DNN/CANN: done converting layers to CANN operators");

    // build graph from collected graph inputs and outputs
    CV_LOG_INFO(NULL, "DNN/CANN: building ge::Graph");
    std::string graphName = cv::format("graph_%d", 0);
    std::shared_ptr<ge::Graph> graph = std::make_shared<ge::Graph>(graphName.c_str());
    (void)graph->SetInputs(graphInputOps);
    (void)graph->SetOutputs(graphOutputOps);
    CV_LOG_INFO(NULL, "DNN/CANN: done building ge::Graph");

    // convert ge::Graph to OM buffer
    CV_LOG_INFO(NULL, "DNN/CANN: converting ge::Graph to OM buffer");
    std::shared_ptr<CannNet> net = std::shared_ptr<CannNet>(new CannNet());
    net = std::shared_ptr<CannNet>(new CannNet());
    net->buildFromGraph(graph);
    net->loadToDevice();
    net->bindInputWrappers(graphInputWrappers);
    net->bindOutputWrappers(graphOutputWrappers);
    CV_LOG_INFO(NULL, "DNN/CANN: done building ge::Graph to OM buffer");

    // keep net in the first output node and mark the node runnable
    auto& ld = layers[firstOutputLayerId];
    auto cannNode = ld.backendNodes[preferableBackend].dynamicCast<CannBackendNode>();
    cannNode->net = net;
    ld.skip = false;

    netWasConverted = true;
}

void NetImplCann::forwardLayer(LayerData& ld)
{
    CV_TRACE_FUNCTION();

    auto layer = ld.layerInstance;

    if (!ld.skip)
    {
        auto it = ld.backendNodes.find(preferableBackend);
        if (ld.id == 0 || it == ld.backendNodes.end()) // input layer
        {
            return Base::forwardLayer(ld);
        }

        CV_Assert(it != ld.backendNodes.end());
        const Ptr<BackendNode>& node = it->second;
        CV_Assert(!node.empty());
        auto cannNode = node.dynamicCast<CannBackendNode>();
        CV_Assert(!cannNode.empty());
        CV_Assert(cannNode->net);

        TickMeter tm;
        tm.start();

        cannNode->net->forward();

        tm.stop();
        int64_t t = tm.getTimeTicks();
        layersTimings[ld.id] = (t > 0) ? t : 1;
    }
    else
    {
        layersTimings[ld.id] = 0;
    }

    ld.flag = 1;
}

void switchToCannBackend(Net& net)
{
    CV_TRACE_FUNCTION();
    Ptr<Net::Impl>& impl_ptr_ref = accessor::DnnNetAccessor::getImplPtrRef(net);
    CV_Assert(impl_ptr_ref);
    CV_LOG_INFO(NULL, "DNN: switching to CANN backend... (networkID=" << impl_ptr_ref->networkId << ")");
    Ptr<NetImplCann> impl_ptr_cann = makePtr<NetImplCann>(impl_ptr_ref);
    impl_ptr_ref = impl_ptr_cann;
}

#endif // HAVE_CANN

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
