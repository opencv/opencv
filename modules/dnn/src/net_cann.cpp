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

    // MapIdToLayerData subNets;
    std::shared_ptr<CannNet> cann_net{nullptr};

    explicit NetImplCann(const Ptr<Net::Impl>& basePtr)
        : Net::Impl()
    {
        CV_LOG_INFO(NULL, "Initializing NetImplCann");
        basePtr_ = basePtr;

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

        cann_net = std::make_shared<CannNet>();
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

    // void setInput(InputArray blob, const String& name, double scalefactor, const Scalar& mean) override
    // {
    //     Mat blob_ = blob.getMat();
    //     Mat input_;
    //     blob_.convertTo(input_, CV_32F, scalefactor, -mean[0] * scalefactor);

    //     cann_net->setInput(input_, name);
    // }

    //    1. call initBackend
    //    2. call setInput from the netInputLayer
    //    3. forward
    Mat forward(const String& outputName) override
    {
        String layerName = outputName;

        if (layerName.empty())
        {
            std::vector<String> layerNames = getLayerNames(); // collects names of all layers on the go
            CV_Assert(!layerNames.empty());
            layerName = layerNames.back();
        }

        std::vector<LayerPin> pins(1, getPinByAlias(layerName));
        setUpNet(pins); // calls initBackend

        // set input
        auto it = layers.find(0);
        const auto& ld_input = it->second;
        const auto n_input = ld_input.outputBlobsWrappers.size();
        for (int i = 0; i < n_input; i++)
        {
            const String& i_input_name = netInputLayer->outNames[i];
            Mat i_input_mat;
            netInputLayer->inputsData[i].convertTo(i_input_mat, CV_32F);
            cann_net->setInput(i_input_mat, i_input_name);
        }

        cann_net->forward();

        Mat output;
        cann_net->fetchOutput(output, outputName);
        return output;
    }

    // void forward(OutputArrayOfArrays outputBlobs, const String& outputName) override
    // {
    //     cann_net->forward();

    //     if (outputBlobs.isMat())
    //     {
    //         Mat output;
    //         cann_net->fetchOutput(output, outputName);
    //         outputBlobs.assign(output);
    //     }
    //     else if (outputBlobs.isMatVector())
    //     {
    //         int output_num = cann_net->getOutputNum();
    //         std::vector<Mat> matVec;
    //         for (int i = 0; i < output_num; i++)
    //         {
    //             Mat output_i;
    //             cann_net->fetchOutput(output_i, i);
    //             matVec.push_back(output_i);
    //         }
    //         outputBlobs.create(output_num, 1, CV_32F, -1);
    //         outputBlobs.assign(matVec);
    //     }
    //     else
    //         CV_Error(Error::StsNotImplemented, "Content of outputBlobs should be Mat or std::vector<Mat>");
    // }

    // void forward(OutputArrayOfArrays outputBlobs,
    //              const std::vector<String>& outBlobNames) override
    // {
    //     cann_net->forward();

    //     std::vector<Mat> matVec;
    //     for (size_t i = 0; i < outBlobNames.size(); i++)
    //     {
    //         Mat output_i;
    //         cann_net->fetchOutput(output_i, outBlobNames[i]);
    //         matVec.push_back(output_i);
    //     }
    //     outputBlobs.create((int)outBlobNames.size(), 1, CV_32F, -1);
    //     outputBlobs.assign(matVec);
    // }

    // void forward(std::vector<std::vector<Mat>>& outputBlobs,
    //              const std::vector<String>& outBlobNames) override
    // {
    //     // FIXIT: what does this API mean?
    //     CV_Error(Error::StsNotImplemented, "Not supported");
    // }

    // void fuseLayers(const std::vector<LayerPin>& blobsToKeep_); // consider to fuse the the extra flatten

    void initBackend(const std::vector<LayerPin>& blobsToKeep_) override;
};

// TODO: rebuild cann_net if network was changed.
void NetImplCann::initBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    CV_CheckEQ(preferableBackend, DNN_BACKEND_CANN, "");

    if (cann_net != nullptr && !cann_net->empty())
        return;

    // add inputs to the graph and connect operators
    std::vector<Ptr<BackendNode> > inputs;
    std::vector<ge::Operator> graph_inputs;
    CV_LOG_INFO(NULL, "build graphs");
    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData& ld = it->second;
        auto layer = ld.layerInstance;

        ld.skip = true; // skip all cann operators

        if (ld.id == 0)
        {
            // inputs.resize(ld.outputBlobsWrappers.size());
            for (int i = 0; i < ld.outputBlobsWrappers.size(); i++)
            {
                std::string input_i_name = netInputLayer->outNames.empty() ? cv::format("%s_%d", ld.name.c_str(), i) : netInputLayer->outNames[i];
                auto input_i = std::make_shared<ge::op::Data>(input_i_name);

                // retrieve tensor description
                auto p = ld.outputBlobsWrappers[i].dynamicCast<CannBackendWrapper>();
                CV_Assert(!p.empty());

                input_i->update_input_desc_x(*(p->desc_));
                input_i->update_output_desc_y(*(p->desc_));

                graph_inputs.push_back(*input_i);
                inputs.push_back(Ptr<BackendNode>(new CannBackendNode(input_i)));
            }
        }
        else
        {
            std::vector<Ptr<BackendNode> > input_nodes;
            for (int i = 0; i < ld.inputBlobsId.size(); i++)
            {
                int input_node_id = ld.inputBlobsId[i].lid;
                int input_node_oid = ld.inputBlobsId[i].oid;
                if (input_node_id == 0)
                {
                    input_nodes.push_back(inputs[input_node_oid]);
                }
                else
                {
                    LayerData& input_node_ld = layers[input_node_id];
                    input_nodes.push_back(input_node_ld.backendNodes[preferableBackend]);
                }
            }
            if (layer->supportBackend(preferableBackend))
            {
                CV_LOG_INFO(NULL, "DNN/CANN: converting layer " << ld.name << "@" << ld.type << "@" << ld.id << " to CANN operator");
                ld.backendNodes[preferableBackend] = layer->initCann(ld.inputBlobsWrappers, ld.id, input_nodes);
            }
            else
            {
                CV_LOG_INFO(NULL, "DNN/CANN: layer " << ld.name << "@" << ld.type << "is not supported by CANN backend");
            }
        }
    }

    // collect outputs
    std::vector<int> outputs_lid;
    std::vector<ge::Operator> graph_outputs;
    std::vector<std::string> graph_output_names;
    CV_LOG_INFO(NULL, "Collect outputs");
    for (MapIdToLayerData::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it)
    {
        LayerData& ld = it->second;

        if (ld.consumers.size() == 0) // outputs
        {
            CV_LOG_INFO(NULL, "DNN/CANN: collecting output on layer " << ld.name << "@" << ld.type);
            outputs_lid.push_back(ld.id);
            graph_output_names.push_back(ld.name);
            auto node = ld.backendNodes[preferableBackend].dynamicCast<CannBackendNode>();
            graph_outputs.push_back(*(node->getOp()));
        }
        else
            continue;
    }

    // build graph and keep it in the subgraph
    CV_LOG_INFO(NULL, "build ge::Graph");
    ge::Graph graph("graph");
    graph.SetInputs(graph_inputs).SetOutputs(graph_outputs);
    cann_net->buildFromGraph(graph);
    cann_net->loadToDevice();
    cann_net->setOutputNames(graph_output_names);
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
