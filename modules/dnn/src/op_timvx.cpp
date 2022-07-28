// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021, Shenzhen Institute of Artificial Intelligence and
// Robotics for Society, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_timvx.hpp"
#include "net_impl.hpp"

namespace cv
{
namespace dnn
{
#ifdef HAVE_TIMVX

CV__DNN_INLINE_NS_BEGIN

// update all comsumer
void Net::Impl::tvUpdateConfictMap(int graphIndex, LayerData& ld, std::vector<std::vector<int> >& graphConflictMap)
{
    if (ld.consumers.empty())
        return;
    for (int i = 0; i < ld.consumers.size(); i++)
    {
        LayerData &consumerld = layers[ld.consumers[i].lid];
        std::vector<int>::iterator it = std::find(graphConflictMap[ld.consumers[i].lid].begin(),
                                                    graphConflictMap[ld.consumers[i].lid].end(), graphIndex);

        if (it == graphConflictMap[ld.consumers[i].lid].end())
        {
            graphConflictMap[ld.consumers[i].lid].push_back(graphIndex);
            tvUpdateConfictMap(graphIndex, consumerld, graphConflictMap);
        }
        else
            continue;
    }
}

// Convert TRANSIENT to OUTPUT
void Net::Impl::tvConvertToOutputNode(const LayerData& ld, Ptr<TimVXBackendWrapper>& targetWrap)
{
    // find right layer.
    for (auto& inputLayerId : ld.inputLayersId)
    {
        LayerData &inputld = layers[inputLayerId];
        auto itWrap = std::find(inputld.outputBlobsWrappers.begin(),
                                inputld.outputBlobsWrappers.end(), targetWrap);
        if (itWrap != inputld.outputBlobsWrappers.end())
        {
            auto outputWrap = (*itWrap).dynamicCast<TimVXBackendWrapper>();
            if (!outputWrap->isTensor())
                continue;

            auto inputNode = inputld.backendNodes[DNN_BACKEND_TIMVX].dynamicCast<TimVXBackendNode>();
            if (!inputNode->isLast && inputNode->opIndex != -1)
            {
                CV_Assert(outputWrap->getTensorAttr() == tim::vx::TRANSIENT);
                // set last
                inputNode->isLast = true;

                auto shapeType = getShapeTypeFromMat(outputWrap->getMat());
                auto outQuant = outputWrap->getTensorQuantization();

                outputWrap->setTensorShape(shapeType);
                outputWrap->createTensor(inputNode->tvGraph->graph,
                                            tim::vx::TensorAttribute::OUTPUT, outQuant);
                int outIndex = inputNode->tvGraph->addWrapper(outputWrap);
                inputNode->outputIndexList.clear();
                inputNode->outputIndexList.push_back(outIndex);
            }
        }
    }
}

void Net::Impl::initTimVXBackend()
{
    CV_TRACE_FUNCTION();
    CV_Assert(preferableBackend == DNN_BACKEND_TIMVX);

    // Build TimVX Graph from sets of layers that support this TimVX backend.
    // Split a whole model on several TimVX Graph if some of layers are not implemented by TimVX backend.
    if (!haveTimVX())
        return;

    // Allocate graphConflictMap
    if (timVxInfo.graphConflictMap.empty())
        timVxInfo.graphConflictMap.resize(layers.size());

    auto it = layers.begin();
    bool isLast = false; // If the node is the last node in current tvGraph.

    for (; it != layers.end(); it++)
    {
        isLast = false;
        LayerData &ld = it->second;
        if(ld.skip)
            continue;
        Ptr<Layer> layer = ld.layerInstance;
        if (!layer->supportBackend(preferableBackend))
        {
            continue;
        }

        // If layer consumers are more than one, set isLast true.
        // For now, TimVX backend divides multiple branchs into multiple tvGraph.
        if (ld.consumers.size() == 0)
        {
            isLast = true;
        }
        else if(ld.consumers.size() == 1)
        {
            LayerData* consumerld = &layers[ld.consumers[0].lid];

            while (consumerld)
            {
                if (consumerld->skip)
                {
                    if (consumerld->consumers.size() == 1)
                    {
                        int nextLayerId = consumerld->consumers[0].lid;
                        consumerld = &layers[nextLayerId];
                    }
                    else
                    {
                        isLast = true;
                        break;
                    }
                }
                else
                {
                    break;
                }
            }
            Ptr<Layer>& consumerLayer = consumerld->layerInstance;

            if (!isLast && !consumerLayer->supportBackend(preferableBackend))
            {
                isLast = true;
            }
        }
        else
        {
            // If there are is multiple input, and only one of them is supported.
            int tvSupportNum = 0;
            for (int i = 0; i<ld.consumers.size(); i++)
            {
                LayerData* consumerld = &layers[ld.consumers[0].lid];

                while (consumerld)
                {
                    if (consumerld->skip)
                    {
                        if (consumerld->consumers.size() == 1)
                        {
                            int nextLayerId = consumerld->consumers[0].lid;
                            consumerld = &layers[nextLayerId];
                        }
                        else
                        {
                            isLast = true;
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                Ptr<Layer>& consumerLayer = consumerld->layerInstance;

                if (consumerLayer->supportBackend(preferableBackend))
                {
                    tvSupportNum++;
                }
            }

            if (tvSupportNum != 1)
                isLast = true;
        }

        int graphIndex = -1;
        bool needRecorrect = !timVxInfo.findGraphIndex(ld.inputBlobsWrappers, graphIndex);

        if (graphIndex != -1 && !needRecorrect)
        {
            needRecorrect = timVxInfo.isConflict(ld.id, graphIndex);
        }

        // Recorrect the input layer.
        if (needRecorrect)
        {
            // set all inputLayers' as last layer, and convert TRANSIENT to output.
            for (int i = 0; i < ld.inputBlobsWrappers.size(); i++)
            {
                auto inputWrap = ld.inputBlobsWrappers[i];
                auto tvInputWrap = inputWrap.dynamicCast<TimVXBackendWrapper>();
                if (!tvInputWrap->isTensor())
                    continue;

                auto attr = tvInputWrap->getTensorAttr();
                if (attr == tim::vx::TensorAttribute::OUTPUT)
                {
                    continue;
                }
                else if (attr == tim::vx::TensorAttribute::INPUT)
                {
                    Mat matTmp = tvInputWrap->getMat();
                    tvInputWrap = Ptr<TimVXBackendWrapper>(new TimVXBackendWrapper(matTmp));

                }
                else if (attr == tim::vx::TensorAttribute::TRANSIENT)
                {
                    tvConvertToOutputNode(ld, tvInputWrap);
                    // updateConflictMap
                    tvUpdateConfictMap(graphIndex, ld, timVxInfo.graphConflictMap);
                }
            }
            graphIndex = -1;
        }

        if (graphIndex == -1)
        {
            graphIndex = timVxInfo.createGraph();
        }
        timVxInfo.setTmpGraphIndex(graphIndex);

        ld.backendNodes[DNN_BACKEND_TIMVX] =
                layer->initTimVX(&timVxInfo, ld.inputBlobsWrappers, ld.outputBlobsWrappers, isLast);

        // post process, create last node correctly.
        if (isLast && ld.backendNodes[DNN_BACKEND_TIMVX])
        {
            auto tmpNode = ld.backendNodes[DNN_BACKEND_TIMVX].dynamicCast<TimVXBackendNode>();
            tmpNode->isLast = true;
            // update graphConflictMap
            tvUpdateConfictMap(graphIndex, ld, timVxInfo.graphConflictMap);
        }

        // post process for failing to create timvx Node.
        if (!ld.backendNodes[DNN_BACKEND_TIMVX])
        {
            for (int i = 0; i < ld.inputBlobsWrappers.size(); i++)
            {
                auto inputWrap = ld.inputBlobsWrappers[i];
                auto tvInputWrap = inputWrap.dynamicCast<TimVXBackendWrapper>();
                if (!tvInputWrap->isTensor())
                    continue;

                auto attr = tvInputWrap->getTensorAttr();
                if (attr == tim::vx::TensorAttribute::TRANSIENT)
                {
                    tvConvertToOutputNode(ld, tvInputWrap);
                }
            }
        }
    }

    // Op Binding
    it = layers.begin();
    Ptr<TimVXBackendNode> node;
    std::vector<Ptr<TimVXGraph> > tmpGrapList;
    for (; it != layers.end(); it++)
    {
        LayerData &ld = it->second;

        if (ld.backendNodes[DNN_BACKEND_TIMVX])
            node = ld.backendNodes[DNN_BACKEND_TIMVX].dynamicCast<TimVXBackendNode>();
        else
            continue;

        // Binding tvTensor and tvOp
        if (node->opIndex >= 0)
            node->opBinding();
    }
}

CV__DNN_INLINE_NS_END

// from CPU to NPU
bool copyToTensor(std::shared_ptr<tim::vx::Tensor> &dst, const Mat &src)
{
    CV_Assert(src.isContinuous() && (src.type() == CV_8S || src.type() == CV_32F));
    if (dst->CopyDataToTensor(src.data, src.total()))
    {
        return true;
    }
    else
        return false;
}

// from NPU to CPU
bool copyToMat(const Mat &dst, std::shared_ptr<tim::vx::Tensor> &src)
{
    CV_Assert(dst.isContinuous() && (dst.type() == CV_8S || dst.type() == CV_32F));
    if (src->CopyDataFromTensor(dst.data))
    {
        return true;
    }
    else
        return false;
}

tvActivationType getTimVXActType(String & actString)
{
    if (actString == "ReLUInt8") return tvActReLU;
    if (actString == "ReLU6Int8") return tvActReLU6;
    if (actString == "TanHInt8") return tvActTanH;
    if (actString == "SwishInt8") return tvActSwish;
    if (actString == "MishInt8") return tvActMish;
    if (actString == "SigmoidInt8") return tvActSigmoid;
    if (actString == "ELUInt8") return tvActELU;

    return tvActNotSupported;
}

tim::vx::ShapeType getShapeTypeFromMat(const Mat& mat, bool ifConst)
{
    /* Convert Mat shape to TimVX Tensor shape.
    DataLayout in TimVX is WHCN, while NCHW in OpenCV.
    So we do vector reverse.
    */
    CV_Assert(!mat.empty());
    tim::vx::ShapeType tvInputShape;
    auto matShape = shape(mat);
    tvInputShape.assign(matShape.begin(), matShape.end());

    if ( matShape.size() > 1 )  // TODO: check when we need reverse the shape vector.
    {
        if (ifConst && tvInputShape.size() == 2 && tvInputShape[1] == 1)
        {   // if bias vector, shape [n, 1] to [n].
            tvInputShape.resize(1);
        }
        else
            std::reverse(tvInputShape.begin(), tvInputShape.end());
    }
    return tvInputShape;
}

bool getQuantType(const std::vector<float>& scales, int numOutput)
{
    CV_Assert(!scales.empty());
    if (numOutput == -1)
    {
        numOutput = scales.size();
    }
    bool tvSymmetric = false;

    for (int i =1; i < numOutput; i++)
    {
        if (std::abs(scales[0] - scales[i]) > std::numeric_limits<float>::epsilon())
        {
            tvSymmetric = true;
            break;
        }
    }

    return tvSymmetric;
}

// convert mat Depth to tensorDataType
tim::vx::DataType dataTypeConvert(int matDepth)
{
    tim::vx::DataType tensorDataType;
    switch(matDepth)
    {
        case CV_8U:
        {
            tensorDataType = tim::vx::DataType::UINT8;
            break;
        }
        case CV_8S:
        {
            tensorDataType = tim::vx::DataType::INT8;
            break;
        }
        case CV_16U:
        {
            tensorDataType = tim::vx::DataType::UINT16;
            break;
        }
        case CV_16S:
        {
            tensorDataType = tim::vx::DataType::INT16;
            break;
        }
        case CV_32S:
        {
            tensorDataType = tim::vx::DataType::INT32;
            break;
        }
        case CV_32F:
        {
            tensorDataType = tim::vx::DataType::FLOAT32;
            break;
        }
        case CV_16F:
        {
            tensorDataType = tim::vx::DataType::FLOAT16;
            break;
        }
        default:
        {
            tensorDataType = tim::vx::DataType::UNKNOWN;
            break;
        }
    }
    return tensorDataType;
}

std::vector<Ptr<TimVXBackendWrapper> > getWrappers(const std::vector<int> wrappersIndex,
                                                          Ptr<TimVXGraph> tvGraph)
{
    std::vector<Ptr<TimVXBackendWrapper> > wrappers;
    for (int i = 0; i<wrappersIndex.size(); i++)
    {
        auto wrapper = tvGraph->getWrapper(wrappersIndex[i]);
        if (wrapper)
            wrappers.push_back(wrapper);
    }

    return wrappers;
}

// *********************** TimVXGraph ********************
TimVXGraph::TimVXGraph()
{
    // new TimVX Graph
    context = tim::vx::Context::Create();
    graph = context->CreateGraph();
    isCompiled = false;
}

TimVXGraph::~TimVXGraph()
{

    // release opList
    for (auto& tensor: tensorList)
        tensor.reset();

    // release tensorList
    for (auto& op: opList)
        op.reset();

    // release graph
    graph.reset();

    // release context
    context.reset();
}

std::shared_ptr<tim::vx::Operation> TimVXGraph::getOp(const int opIndex)
{
    CV_Assert(0 <= opIndex && !opList.empty() && opIndex < opList.size());
    return opList[opIndex];
}

int TimVXGraph::addWrapper(Ptr<TimVXBackendWrapper>& tensorWrapper)
{
    CV_Assert(tensorWrapper->isTensor());
    tim::vx::TensorAttribute tensorAttr = tensorWrapper->getTensorAttr();

    wrapperList.push_back(tensorWrapper);
    tensorList.push_back(tensorWrapper->getTensor());
    int wrapperIndex = wrapperList.size() -1;

    if (tensorAttr == tim::vx::TensorAttribute::INPUT)
    {
        inputWrappersIndex.push_back(wrapperIndex);
    }

    if (tensorAttr == tim::vx::TensorAttribute::OUTPUT)
    {
        outputWrappersIndex.push_back(wrapperIndex);
    }

    return wrapperIndex;
}

Ptr<TimVXBackendWrapper> TimVXGraph::getWrapper(int wrapperIndex)
{
    CV_Assert(wrapperIndex>=0 && wrapperIndex < wrapperList.size());
    return wrapperList[wrapperIndex];
}

int TimVXGraph::addOp(const std::shared_ptr<tim::vx::Operation>& op)
{
    CV_Assert(op);
    opList.emplace_back(op);
    return opList.size()-1;
}

int TimVXGraph::getTensorIndex(const std::shared_ptr<tim::vx::Tensor>& tensor)
{
    auto it = find(tensorList.begin(), tensorList.end(), tensor);
    if (it != tensorList.end())
        return it - tensorList.begin();
    else
        return -1;
}

void TimVXGraph::forward()
{
    CV_Assert(!inputWrappersIndex.empty() && !outputWrappersIndex.empty());

    // Every TimVXGraph Instance only compiles once.
    if (!this->isCompiled)
    {
        if (!graph->Compile())
            CV_Error(cv::Error::StsBadArg, " Fail to compile TimVX graph!");
        this->isCompiled = true;
    }

    if (!graph->Run())
        CV_Error(cv::Error::StsBadArg, " Fail to run TimVX graph!");
}

// *********************** TimVXBackendNode ********************
TimVXBackendNode::TimVXBackendNode(const Ptr<TimVXGraph>& tvGraph_): BackendNode(DNN_BACKEND_TIMVX)
{
    opIndex = -1;
    tvGraph = tvGraph_;
    isLast = false;
}

TimVXBackendNode::TimVXBackendNode(const Ptr<TimVXGraph>& tvGraph_,
                                   const std::shared_ptr<tim::vx::Operation>& op_): BackendNode(DNN_BACKEND_TIMVX)
{
    tvGraph = tvGraph_;
    opIndex = tvGraph->addOp(op_);
    isLast = false;
}

TimVXBackendNode::TimVXBackendNode(const Ptr<TimVXGraph>& tvGraph_, std::shared_ptr<tim::vx::Operation>& op_,
                                   std::vector<int>& inputsIndex, std::vector<int>& outpusIndex)
                                   :BackendNode(DNN_BACKEND_TIMVX)
{
    tvGraph = tvGraph_;
    opIndex = tvGraph->addOp(op_);
    isLast = false;

    if (!inputsIndex.empty())
        inputIndexList.assign(inputsIndex.begin(), inputsIndex.end());

    if (!outpusIndex.empty())
        outputIndexList.assign(outpusIndex.begin(), outpusIndex.end());
}

bool TimVXBackendNode::opBinding()
{
    if (!tvGraph || tvGraph->isCompiled || opIndex == -1)
        return false;

    std::shared_ptr<tim::vx::Operation> op = tvGraph->getOp(opIndex);

    if (!inputIndexList.empty())
    {
        std::vector<Ptr<TimVXBackendWrapper> > inputsWrapper = getWrappers(inputIndexList, tvGraph);
        // Binding input Tensor.
        for (auto& warpper: inputsWrapper)
        {
            op->BindInput(warpper->getTensor());
        }
    }

    if (!outputIndexList.empty())
    {
        std::vector<Ptr<TimVXBackendWrapper> > outputsWrapper = getWrappers(outputIndexList, tvGraph);
        for (auto& warpper: outputsWrapper)
        {
            op->BindOutput(warpper->getTensor());
        }
    }
    return true;
}

void TimVXBackendNode::setInputTensor()
{
    if (!tvGraph || opIndex == -1)
        return;

    if (!inputIndexList.empty())
    {
        std::vector<Ptr<TimVXBackendWrapper> > inputsWrapper = getWrappers(inputIndexList, tvGraph);

        // Binding input Tensor.
        for (auto& warpper: inputsWrapper)
        {
            if (warpper->getTensorAttr() == tim::vx::TensorAttribute::INPUT)
            {
                warpper->setHostDirty();
                warpper->copyToDevice();
            }
        }
    }
}

// *********************** TimVXBackendWrapper ********************
// Default Constructor
TimVXBackendWrapper::TimVXBackendWrapper() : BackendWrapper(DNN_BACKEND_TIMVX, DNN_TARGET_NPU)
{
    isTensor_ = false;
    deviceDirty = false;
    hostDirty = false;
    tensorType = tim::vx::DataType::UNKNOWN;
    tensorShape = {};
    tensorIndex = -1;
    tensorAttr = tim::vx::TensorAttribute::CONSTANT;
}

TimVXBackendWrapper::TimVXBackendWrapper(Mat& m) : BackendWrapper(DNN_BACKEND_TIMVX,
                                                                  DNN_TARGET_NPU)
{
    host = m;
    isTensor_ = false;
    deviceDirty = false;
    hostDirty = true;
    tensorType = dataTypeConvert(m.type());
    tensorShape = {};
    tensorIndex = -1;
    tensorAttr = tim::vx::TensorAttribute::CONSTANT;

    // TODO: unsupported data by TimVX should run convert function first.
    CV_Assert(tensorType != tim::vx::DataType::UNKNOWN);
}

TimVXBackendWrapper::TimVXBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
    :BackendWrapper(DNN_BACKEND_TIMVX, DNN_TARGET_NPU)
{
    Ptr<TimVXBackendWrapper> base = baseBuffer.dynamicCast<TimVXBackendWrapper>();
    CV_Assert(!base.empty());
    tensor = base->tensor;
    isTensor_ = base->isTensor_;
    tensorIndex = base->tensorIndex;
    tensorType = base->tensorType;
    tensorAttr = base->tensorAttr;
    tensorShape = base->tensorShape;
    deviceDirty = base->deviceDirty;
    hostDirty = base->hostDirty;
    host = m;
}

TimVXBackendWrapper::TimVXBackendWrapper(std::shared_ptr<tim::vx::Tensor>& tensor_)
    :BackendWrapper(DNN_BACKEND_TIMVX, DNN_TARGET_NPU)
{
    tensor = tensor_;
    isTensor_ = true;
    deviceDirty = true;
    hostDirty = false;
    tensorType = tensor_->GetDataType(); // getTensor DataType.
    tensorAttr = tensor_->GetSpec().attr_;  // getTensor Attribution.
    tensorShape = tensor_->GetShape();
    tensorIndex = -1;
}

void TimVXBackendWrapper::setTensorShape(const tim::vx::ShapeType & matShape)
{
    CV_Assert(!matShape.empty());
    tensorShape.assign(matShape.begin(), matShape.end());
}

int TimVXBackendWrapper::getTensorIndex()
{
    CV_Assert(isTensor_);
    return tensorIndex;
}

tim::vx::TensorAttribute TimVXBackendWrapper::getTensorAttr()
{
    CV_Assert(isTensor_);
    return tensorAttr;
}

// Create tensor
void TimVXBackendWrapper::createTensor(std::shared_ptr<tim::vx::Graph>& graph,
    tim::vx::TensorAttribute tensorAttribute)
{
    Ptr<tim::vx::Quantization> epmtyQuant = nullptr;
    return this->createTensor(graph, tensorAttribute, epmtyQuant);
}

// Create tensor
void TimVXBackendWrapper::createTensor(std::shared_ptr<tim::vx::Graph>& graph,
    tim::vx::TensorAttribute tensorAttribute, Ptr<tim::vx::Quantization>& tvQuant)
{
    CV_Assert(graph);
    tim::vx::TensorSpec tensorSpec;

    if (tensorAttribute == tim::vx::INPUT)
    {
        CV_Assert(!host.empty());
        tensorShape = getShapeTypeFromMat(host);
    }
    else if (tensorAttribute == tim::vx::OUTPUT)
    {
        CV_Assert(!tensorShape.empty() && !host.empty());
        tensorShape = getShapeTypeFromMat(host);
    }
    else if (tensorAttribute == tim::vx::CONSTANT)
    {
        if (!host.empty())
            tensorShape = getShapeTypeFromMat(host, true);
    }
    else
    {
        if (!host.empty())
            tensorShape = getShapeTypeFromMat(host);
    }

    // Tensor shape
    if (tvQuant)
    {
        tensorSpec = tim::vx::TensorSpec(tensorType, tensorShape, tensorAttribute, *tvQuant);
    }
    else
    {
        tensorSpec = tim::vx::TensorSpec(tensorType, tensorShape, tensorAttribute);
    }

    if (!host.empty() && tensorAttribute != tim::vx::INPUT && tensorAttribute != tim::vx::OUTPUT && tensorAttribute != tim::vx::TRANSIENT)
    {
        tensor = graph->CreateTensor(tensorSpec, (void *)(host.data));
    }
    else
    {
        tensor = graph->CreateTensor(tensorSpec);
    }
    isTensor_ = true;

    // set Attribution
    tensorAttr = tensorAttribute;
}

Ptr<tim::vx::Quantization> TimVXBackendWrapper::getTensorQuantization()
{
    CV_Assert(isTensor_ && tensor);
    auto quantize = tensor->GetQuantization();
    return makePtr<tim::vx::Quantization>(quantize);
}

std::shared_ptr<tim::vx::Tensor> TimVXBackendWrapper::getTensor()
{
     CV_Assert(isTensor_);
     return tensor;
}

Mat TimVXBackendWrapper::getMat()
{
    if (host.empty())
        return {};
    return host;
}


bool TimVXBackendWrapper::isTensor()
{
    return isTensor_;
}

void TimVXBackendWrapper::copyToHost()
{
    if (deviceDirty && !host.empty())
    {
        copyToMat(host, tensor);
        deviceDirty = false;
    }
}

void TimVXBackendWrapper::setHostDirty()
{
    hostDirty = true;
}

void TimVXBackendWrapper::setDeviceDirty()
{
    deviceDirty = true;
}

void TimVXBackendWrapper::copyToDevice()
{
    if (isTensor_ && hostDirty && !host.empty())
    {
        copyToTensor(tensor, host);
        hostDirty = false;
    }
}

// *********************** TimVXInfo ********************
TimVXInfo::TimVXInfo()
{
    graphIndex = -1;
}

TimVXInfo::~TimVXInfo()
{}

int TimVXInfo::createGraph()
{
    Ptr<TimVXGraph> tmpGraph =  Ptr<TimVXGraph>(new TimVXGraph());
    this->tvGraphList.push_back(tmpGraph);
    return this->tvGraphList.size() - 1;
}

bool TimVXInfo::findGraphIndex(const std::vector<Ptr<BackendWrapper> >  &inputsWrapper, int& graphIndex)
{
    graphIndex = -1;
    int wrapperSize = inputsWrapper.size();
    int graphSize = tvGraphList.size();

    if (wrapperSize != 0 && graphSize == 0)
    {
        return true;
    }

    int tensorIndex = -1;
    Ptr<TimVXBackendWrapper> wrapper;
    Ptr<TimVXGraph> tvGraph;

    for (int i = 0; i < graphSize; i++)
    {
        tvGraph = tvGraphList[i];
        for (int j = 0; j < wrapperSize; j++ )
        {
            wrapper = inputsWrapper[j].dynamicCast<TimVXBackendWrapper>();

            if (!wrapper->isTensor()) // Skip wrapper without Tensor.
                continue;

            tensorIndex = tvGraph->getTensorIndex(wrapper->getTensor());
            if (tensorIndex != -1 && wrapper->getTensorAttr() == tim::vx::TensorAttribute::TRANSIENT)
            {
                if (graphIndex == -1)
                    graphIndex = i;
                else if (graphIndex != i)  // if inputs of the same inputWrapper are from differen tvGraph.
                {
                    graphIndex = -1;
                    return false;
                }
            }
        }
    }
    return true;
}

void TimVXInfo::setTmpGraphIndex(int graphIndex)
{
    this->graphIndex = graphIndex;
}

int TimVXInfo::getTmpGraphIndex()
{
    int res = -1;
    if (graphIndex != -1)
    {
        res = graphIndex;
        graphIndex = -1;
    }
    return res;
}

bool TimVXInfo::isConflict(int layerId, int graphIndex)
{
    if (graphConflictMap[layerId].empty())
        return false;

    std::vector<int>::iterator it = std::find(graphConflictMap[layerId].begin(),
                                              graphConflictMap[layerId].end(), graphIndex);
    if (it != graphConflictMap[layerId].end())
        return true;
    else
        return false;
}

Ptr<TimVXGraph> TimVXInfo::getGraph()
{
    int index = getTmpGraphIndex();
    if (0 <= index && index < tvGraphList.size())
        return tvGraphList[index];
    else
        return {};
}

#endif

void forwardTimVX(std::vector<Ptr<BackendWrapper> >& outputs, const Ptr<BackendNode>& node_)
{
#ifdef HAVE_TIMVX
    CV_Assert(!node_.empty());
    Ptr<TimVXBackendNode> node = node_.dynamicCast<TimVXBackendNode>();

    if (node)
    {
        // set input
        node->setInputTensor();

        // graph Forward
        if (node->isLast)
        {
            node->tvGraph->forward();
        }
    }
    else
        return;

    // set ouput
    Ptr<TimVXBackendWrapper> outWarpper;
    for (int i = 0; i < outputs.size(); i++)
    {
        outWarpper = outputs[i].dynamicCast<TimVXBackendWrapper>();
        if (outWarpper->isTensor() && outWarpper->getTensorAttr() == tim::vx::TensorAttribute::OUTPUT)
        {
            outWarpper->setDeviceDirty();
            outWarpper->copyToHost();
        }
    }
#endif
}

bool haveTimVX()
{
#ifdef HAVE_TIMVX
    return true;
#else
    return false;
#endif
}
} // namespace dnn
} // namespace cv