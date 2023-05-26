// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

#ifdef HAVE_CUDA
#include "cuda4dnn/primitives/eltwise.hpp"  // required by fuseLayers
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


void Net::Impl::enableFusion(bool fusion_)
{
    if (fusion != fusion_)
    {
        fusion = fusion_;
        clear();
    }
}


#if 0
#define printf_(args) printf args
#else
#define printf_(args)
#endif


void Net::Impl::fuseLayers(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();

    if(!fusion || (preferableBackend != DNN_BACKEND_OPENCV &&
                    preferableBackend != DNN_BACKEND_CUDA &&
                    preferableBackend != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH &&
                    preferableBackend != DNN_BACKEND_TIMVX &&
                    preferableBackend != DNN_BACKEND_VKCOM))
       return;

#if 0  // FIXIT mode without fusion is broken due to unsupported layers and handling of "custom" nodes
    if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
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
#ifdef HAVE_INF_ENGINE
                if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && pinsToKeep.count(lpNext) != 0)
                {
                    CV_LOG_DEBUG(NULL, "DNN/IE: skip fusing with 'output' node: " << nextData->name << "@" << nextData->type);
                    break;
                }
#endif
                /* we use `tryFuse` member of convolution layer to fuse eltwise/naryEltwise later
                 * it's not intended to be fused here; hence, we stop when we encounter eltwise
                 */
                if (preferableBackend == DNN_BACKEND_CUDA && ld.type == "Convolution" &&
                        (nextData->type == "Eltwise" || nextData->type == "NaryEltwise"))
                    break;
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

            if (preferableBackend != DNN_BACKEND_OPENCV && preferableBackend != DNN_BACKEND_CUDA
                && preferableBackend != DNN_BACKEND_VKCOM)
                continue;  // Go to the next layer.

            // TODO: OpenCL target support more fusion styles.
            if ( preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget) &&
                 (!cv::ocl::useOpenCL() || (ld.layerInstance->type != "Convolution" &&
                 ld.layerInstance->type != "MVN" && ld.layerInstance->type != "Pooling" &&
                 ld.layerInstance->type != "Concat")) )
                continue;

            if (preferableBackend == DNN_BACKEND_CUDA && IS_DNN_CUDA_TARGET(preferableTarget)
                && ld.layerInstance->type != "Convolution"
                && ld.layerInstance->type != "Concat")
                continue;

            while (nextData)
            {
                // For now, OpenCL target support fusion with activation of ReLU/ChannelsPReLU/Power/Tanh
                if (IS_DNN_OPENCL_TARGET(preferableTarget) &&
                    nextData->type != "ReLU" &&
                    nextData->type != "ChannelsPReLU" &&
                    nextData->type != "ReLU6" &&
                    nextData->type != "TanH" &&
                    nextData->type != "Power")
                    break;

                Ptr<ActivationLayer> nextActivLayer = nextData->layerInstance.dynamicCast<ActivationLayer>();
                if (nextActivLayer.empty())
                    break;

                // For now, Vulkan target support fusion with activation of ReLU/ReLU6
                if (IS_DNN_VULKAN_TARGET(preferableTarget))
                {
                    if (nextData->type == "ReLU")
                    {
                        Ptr<ReLULayer> nextReLULayer = nextData->layerInstance.dynamicCast<ReLULayer>();
                        CV_Assert(nextReLULayer);
                        if (nextReLULayer->negativeSlope != 0.0f)
                            break; // Skip LeakyReLU
                    }
                    else if (nextData->type == "ReLU6")
                    {
                        Ptr<ReLU6Layer> nextReLU6Layer = nextData->layerInstance.dynamicCast<ReLU6Layer>();
                        CV_Assert(nextReLU6Layer);

                        if( fabs(nextReLU6Layer->minValue) > FLT_EPSILON || fabs(nextReLU6Layer->maxValue - 6.0f) > FLT_EPSILON)
                            break; // Skip ReLU6 if the minValue != 0 or maxValue != 6.
                    }
                    else
                        break;
                }

                if (currLayer->setActivation(nextActivLayer))
                {
                    printf_(("\tfused with %s\n", nextActivLayer->name.c_str()));
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

            // CPU: fuse Convolution 2D layer followed by Add + activation.
            while (nextData && (IS_DNN_CPU_TARGET(preferableTarget)) && ld.layerInstance->type == "Convolution")
            {
                // Note that we can only deal with conv + Add + activ here.
                // To avoid the order like: conv + activ + add, if we found the conv has been fused with activ, we break.
                Ptr<ConvolutionLayer> convLayer = ld.layerInstance.dynamicCast<ConvolutionLayer>();

                // Only Convolution layer without fusion Activation supports this fusion, other-wise, we skip.
                if (convLayer->fusedActivation)
                    break;

                // For now, there are currently two layers in OpenCV that run the Add operator.
                Ptr<NaryEltwiseLayer> nextNaryEltwiseLayer = nextData->layerInstance.dynamicCast<NaryEltwiseLayer>();
                Ptr<EltwiseLayer> nextEltwiseLayer = nextData->layerInstance.dynamicCast<EltwiseLayer>();
                if (nextNaryEltwiseLayer.empty() && nextEltwiseLayer.empty())
                    break;

                if (nextData->inputBlobsId.size() != 2)
                    break;

                if (!nextData->params.has("operation") || toLowerCase(nextData->params.get<String>("operation")) != "add")
                {
                    CV_LOG_DEBUG(NULL, "DNN/CPU: fusion with NaryEltwise or Eltwise Layer operation is not supported: "
                        << nextData->params.get<String>("operation"));
                    break;
                }

                // This optimization is for cases like
                // some_layer                      conv
                //   |                              |
                //   +-- eltwise or (naryEltwise) --+
                //               |
                //             activ
                // This way all the element-wise computations
                // (i.e. some_layer+conv) would be done at [conv] layer.
                // So we need to replace [conv]'s output blob to [eltwise]'s one
                // considering that [activ] is an in-place layer.
                // Also we need to move all the consumers' references.
                // To prevent memory collisions (i.e. when input of
                // [conv] and output of [eltwise or naryEltwise] is the same blob)
                // we allocate a new blob.
                {
                    LayerData *naryOrEltwiseData = nextData;

                    // Eltwise or NaryEltwise layer has two inputs. We need to determine which
                    // is a base convolution layer and which could be used as it's bias.
                    LayerData* biasLayerData = 0;
                    for (int i = 0; i < 2; ++i)
                    {
                        LayerData *downLayerData = &layers[naryOrEltwiseData->inputBlobsId[i].lid];
                        CV_Assert(downLayerData);
                        // If the current downLayerData is skip, it means it is fused into the parent node.
                        while (downLayerData->skip)
                        {
                            if (downLayerData->inputBlobsId.size() == 1)
                                downLayerData = &layers[downLayerData->inputBlobsId[0].lid];
                            else
                            {
                                downLayerData = 0;
                                break;
                            }
                        }

                        if (downLayerData && ld.id == downLayerData->id)
                        {
                            biasLayerData = &layers[naryOrEltwiseData->inputBlobsId[1 - i].lid];
                            break;
                        }
                    }

                    // We check if biasLayerData is expected layer.
                    if (!biasLayerData)
                        break;

                    // We check if the bias output shape and the ld output shape are the same.
                    MatShape biasOutShape = shape(biasLayerData->outputBlobs[0]);
                    MatShape ldOutShape = shape(ld.outputBlobs[0]);
                    if (biasOutShape != ldOutShape)
                        break;

                    CV_Assert(biasLayerData);
                    {
                        // fuse naryEltwise layer
                        // bias must already be computed to fuse => bias layer must appear before convolution
                        if (biasLayerData->id < ld.id && biasLayerData->consumers.size() == 1)
                        {
                            // conv + naryEltwise.
                            CV_Assert_N(biasLayerData->outputBlobs.size() == 1, ld.inputBlobs.size() == 1);
                            CV_Assert_N(biasLayerData->outputBlobsWrappers.size() == 1, ld.inputBlobsWrappers.size() == 1);

                            printf_(("\tfused with %s\n", nextNaryEltwiseLayer->name.c_str()));
                            naryOrEltwiseData->skip = true;


                            CV_Assert_N(ld.outputBlobs.size() == 1, ld.outputBlobsWrappers.size() == 1);
                            // Note: Here's a trick. We set the output of conv as the output of biasLayer.
                            ld.outputBlobs[0] = ld.outputBlobs[0].clone();
                            ld.outputBlobsWrappers[0] = wrap(ld.outputBlobs[0]);

                            // Recursively modifies the output data of biasLayerData and its parent.
                            std::vector<LayerData*> skipDataList;
                            skipDataList.push_back(biasLayerData);

                            while (!skipDataList.empty())
                            {
                                LayerData* skipData = skipDataList.back();
                                skipDataList.pop_back();

                                CV_Assert(skipData->outputBlobs.size() == 1);
                                skipData->outputBlobs[0] = ld.outputBlobs[0];
                                skipData->outputBlobsWrappers[0] = ld.outputBlobsWrappers[0];
                                if (skipData->skip)
                                {
                                    for (auto& inputLayerId : skipData->inputLayersId)
                                    {
                                        LayerData* inputld = &layers[inputLayerId];

                                        if (inputld && inputld->outputBlobs.size() == 1)
                                            skipDataList.push_back(inputld);
                                    }
                                }
                            }

                            naryOrEltwiseData->outputBlobs = ld.outputBlobs;
                            naryOrEltwiseData->outputBlobsWrappers = ld.outputBlobsWrappers;

                            // set the fusedAdd flag in [Conv];
                            convLayer->fusedAdd = true;
                            LayerData* finalData = naryOrEltwiseData;
                            /* After fused Conv + naryEltwise or eltwise, we can fuse activation if:
                             * => activation layer that follows is the only consumer of eltwise output
                             * => activation layer does not process multiple inputs
                             * => we do not require to keep the output of eltwise
                             */
                            if (naryOrEltwiseData->consumers.size() == 1)
                            {
                                Ptr<ActivationLayer> nextFusabeleActivLayer;
                                LayerData* nextAct = &layers[naryOrEltwiseData->consumers[0].lid];

                                if (nextData->outputBlobs.size() == 1)
                                    nextFusabeleActivLayer = nextAct->layerInstance.dynamicCast<ActivationLayer>();

                                if (!nextFusabeleActivLayer.empty())
                                {
                                    convLayer->setActivation(nextFusabeleActivLayer);
                                    nextAct->skip = true;

                                    nextAct->outputBlobs = ld.outputBlobs;
                                    nextAct->outputBlobsWrappers = ld.outputBlobsWrappers;
                                }
                            }

                            // Move references of finalData (eltwise or activation) layer consumers to the newly allocated blob.
                            for (int i = 0; i < finalData->consumers.size(); ++i)
                            {
                                LayerData& consumer = layers[finalData->consumers[i].lid];
                                for (int j = 0; j < consumer.inputBlobsId.size(); ++j)
                                {
                                    if (consumer.inputBlobsId[j].lid == finalData->id)
                                    {
                                        consumer.inputBlobs[j] = &ld.outputBlobs[0];
                                        consumer.inputBlobsWrappers[j] = ld.outputBlobsWrappers[0];
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            }

            // OpenCL: fuse convolution layer followed by eltwise + relu
            // CUDA: fuse convolution layer followed by eltwise/naryEltwise (and optional activation)
            while (nextData &&
                (IS_DNN_OPENCL_TARGET(preferableTarget) || IS_DNN_CUDA_TARGET(preferableTarget)) &&
                ld.layerInstance->type == "Convolution"
            )  // semantic of 'if'
            {
                Ptr<EltwiseLayer> nextEltwiseLayer = nextData->layerInstance.dynamicCast<EltwiseLayer>();
                Ptr<NaryEltwiseLayer> nextNaryEltwiseLayer = nextData->layerInstance.dynamicCast<NaryEltwiseLayer>();
                if (nextEltwiseLayer.empty() && nextNaryEltwiseLayer.empty())
                    break;

                // TODO: fused the Conv+NaryEltwise on OpenCL backend. At present, we can only support it at CUDA backend.
                if (IS_DNN_OPENCL_TARGET(preferableTarget) && nextNaryEltwiseLayer)
                    break;

#ifdef HAVE_CUDA
                // CUDA backend supports fusion with eltwise sum (without variable channels)
                if (IS_DNN_CUDA_TARGET(preferableTarget) && (!nextEltwiseLayer.empty() || !nextNaryEltwiseLayer.empty()))
                {
                    // we create a temporary backend node for eltwise layer to obtain the eltwise configuration
                    cuda4dnn::csl::CSLContext context; // assume that initCUDA and EltwiseOp do not use the context during init

                    if (!nextData->layerInstance->supportBackend(DNN_BACKEND_CUDA))
                        break;

                    const auto node = nextData->layerInstance->initCUDA(&context, nextData->inputBlobsWrappers, nextData->outputBlobsWrappers);
                    auto eltwiseNode = node.dynamicCast<cuda4dnn::EltwiseOpBase>();

                    // broadcasting not supported in fused ops
                    auto required_shape = shape(nextData->outputBlobs[0]);
                    for (int i = 0; i < nextData->inputBlobs.size(); i++)
                    {
                        if (shape(*nextData->inputBlobs[i]) != required_shape)
                        {
                            eltwiseNode.reset();
                            break;
                        }
                    }

                    // CUDA backend uses EltwiseOp when all operands have the same number of channels; otherwise, ShortcutOp is used.
                    // Hence, a successful cast to EltwiseOp implies that the number of channels is same in all operand tensors.
                    if (eltwiseNode.empty() || eltwiseNode->op != cuda4dnn::EltwiseOpType::SUM || !eltwiseNode->coeffs.empty())
                        break;
                }
#endif

                if (IS_DNN_OPENCL_TARGET(preferableTarget) && pinsToKeep.count(lpNext) != 0)
                    break;
                if (nextData->inputBlobsId.size() != 2)
                    break;

                if (IS_DNN_OPENCL_TARGET(preferableTarget))
                {
                    if (!nextData->params.has("operation") || toLowerCase(nextData->params.get<String>("operation")) == "sum")
                    {
                        if (nextData->params.has("coeff"))
                        {
                            DictValue paramCoeff = nextData->params.get("coeff");
                            int n = paramCoeff.size();
                            bool isCoeffOneOne = (n == 2);
                            for (int i = 0; isCoeffOneOne && i < n; i++)
                            {
                                float c = paramCoeff.get<float>(i);
                                isCoeffOneOne &= (c == 1.0f);
                            }
                            if (!isCoeffOneOne)
                            {
                                CV_LOG_DEBUG(NULL, "DNN/OpenCL: fusion of 'Sum' without coeffs (or {1.0, 1.0}) is supported only");
                                break;
                            }
                        }
                    }
                    else
                    {
                        CV_LOG_DEBUG(NULL, "DNN/OpenCL: fusion with eltwise operation is not supported: " << nextData->params.get<String>("operation"));
                        break;
                    }
                }

                {
                    LayerData *eltwiseData = nextData;

                    // Eltwise/NaryEltwise layer has two inputs. We need to determine which
                    // is a base convolution layer and which could be used as it's bias.
                    LayerData* biasLayerData = 0;
                    for (int i = 0; i < 2; ++i)
                    {
                        LayerData *downLayerData = &layers[eltwiseData->inputBlobsId[i].lid];
                        CV_Assert(downLayerData);
                        while (downLayerData->skip)
                        {
                            if (downLayerData->inputBlobsId.size() == 1)
                                downLayerData = &layers[downLayerData->inputBlobsId[0].lid];
                            else
                            {
                                downLayerData = 0;
                                break;
                            }
                        }
                        if (downLayerData && ld.id == downLayerData->id)
                        {
                            biasLayerData = &layers[eltwiseData->inputBlobsId[1 - i].lid];
                            break;
                        }
                    }
                    CV_Assert(biasLayerData);
                    {
                        // fuse eltwise + activation layer
                        // bias must already be computed to fuse => bias layer must appear before convolution
                        if (biasLayerData->id < ld.id)
                        {
                            /* we can fuse activation if:
                             * => activation layer that follows is the only consumer of eltwise output
                             * => activation layer does not process multiple inputs
                             * => we do not require to keep the output of eltwise
                             */
                            Ptr<ActivationLayer> nextFusabeleActivLayer;
                            if (eltwiseData->consumers.size() == 1 && pinsToKeep.count(lpNext) == 0)
                            {
                                nextData = &layers[eltwiseData->consumers[0].lid];
                                lpNext = LayerPin(eltwiseData->consumers[0].lid, 0);
                                CV_Assert(nextData);
                                if (nextData->outputBlobs.size() == 1)
                                    nextFusabeleActivLayer = nextData->layerInstance.dynamicCast<ActivationLayer>();
                            }
                            else
                            {
                                // OCL backend cannot fuse in this case but the CUDA backend can continue with just eltwise
                                nextData = 0;
                            }

                            // the requirements of OCV OpenCL backend and CUDA backend are different
                            // we need to check them separately; hence, the fuse variables
                            bool fuse_eltwise = false, fuse_activation = false;

                            Ptr<PowerLayer> activ_power;
                            if (IS_DNN_OPENCL_TARGET(preferableTarget) && !nextFusabeleActivLayer.empty() &&
                                nextData &&
                                (!nextData->type.compare("ReLU") ||
                                 !nextData->type.compare("ChannelsPReLU") ||
                                 (!nextData->type.compare("Power") && (activ_power = nextFusabeleActivLayer.dynamicCast<PowerLayer>()) && activ_power->scale == 1.0f)
                                ) &&
                                currLayer->setActivation(nextFusabeleActivLayer))
                            {
                                fuse_eltwise = true;
                                fuse_activation = true;
                            }

                            if (IS_DNN_CUDA_TARGET(preferableTarget))
                            {
                                /* supported fusion options:
                                 * => convolution + eltwise
                                 * => activation(convolution) + eltwise
                                 *    > convolution + activation would have been fused already; we have to fuse eltwise
                                 * => activation(convolution + eltwise)
                                 *    > fuse eltwise and then activation
                                 */
                                Ptr<Layer> layer = nullptr;
                                if (nextNaryEltwiseLayer)
                                    layer = nextNaryEltwiseLayer.staticCast<Layer>();
                                else if (nextEltwiseLayer)
                                    layer = nextEltwiseLayer.staticCast<Layer>();
                                else
                                    CV_Error(Error::StsError, "Both nextNaryEltwiseLayer and nextEltwiseLayer are empty!");

                                if (currLayer->tryFuse(layer))
                                {
                                    fuse_eltwise = true; /* eltwise was successfully fused */
                                    if (!nextFusabeleActivLayer.empty() && nextData)
                                    {
                                        if ((!nextData->type.compare("ReLU") ||
                                             !nextData->type.compare("ReLU6") ||
                                             !nextData->type.compare("Power") ||
                                             !nextData->type.compare("TanH") ||
                                             !nextData->type.compare("Sigmoid") ||
                                             !nextData->type.compare("Swish") ||
                                             !nextData->type.compare("Mish")) &&
                                            currLayer->setActivation(nextFusabeleActivLayer))
                                        {
                                            // activation was fused
                                            fuse_activation = true;
                                        }
                                    }
                                }
                            }

                            CV_Assert(!fuse_activation || fuse_eltwise); /* cannot fuse activation without eltwise */
                            if(fuse_eltwise && fuse_activation)
                            {
                                CV_Assert(nextData);
                                CV_Assert_N(biasLayerData->outputBlobsWrappers.size() == 1, ld.inputBlobsWrappers.size() == 1);
                                ld.inputBlobsWrappers.push_back(biasLayerData->outputBlobsWrappers[0]);

                                if (nextEltwiseLayer)
                                    printf_(("\tfused with %s\n", nextEltwiseLayer->name.c_str()));
                                else if (nextNaryEltwiseLayer)
                                    printf_(("\tfused with %s\n", nextEltwiseLayer->name.c_str()));
                                else
                                    CV_Error(Error::StsError, "Both nextNaryEltwiseLayer and nextEltwiseLayer are empty!");

                                printf_(("\tfused with %s\n", nextFusabeleActivLayer->name.c_str()));
                                eltwiseData->skip = true;
                                nextData->skip = true;
                                // This optimization for cases like
                                // some_layer   conv
                                //   |             |
                                //   +-- eltwise --+
                                //          |
                                //        activ
                                // This way all the element-wise computations
                                // (i.e. some_layer+conv or some_layer*conv)
                                // would be done at [conv] layer. So we need to
                                // replace [conv]'s output blob to [eltwise]'s one
                                // considering that [activ] is an in-place layer.
                                // Also we need to move all the consumers' references.
                                // To prevent memory collisions (i.e. when input of
                                // [conv] and output of [eltwise] is the same blob)
                                // we allocate a new blob.
                                CV_Assert_N(ld.outputBlobs.size() == 1, ld.outputBlobsWrappers.size() == 1);
                                ld.outputBlobs[0] = ld.outputBlobs[0].clone();
                                ld.outputBlobsWrappers[0] = wrap(ld.outputBlobs[0]);

                                eltwiseData->outputBlobs = ld.outputBlobs;
                                nextData->outputBlobs = ld.outputBlobs;
                                eltwiseData->outputBlobsWrappers = ld.outputBlobsWrappers;
                                nextData->outputBlobsWrappers = ld.outputBlobsWrappers;

                                // Move references of [activ] layer consumers to the newly allocated blob.
                                for (int i = 0; i < nextData->consumers.size(); ++i)
                                {
                                    LayerData& consumer = layers[nextData->consumers[i].lid];
                                    for (int j = 0; j < consumer.inputBlobsId.size(); ++j)
                                    {
                                        if (consumer.inputBlobsId[j].lid == lpNext.lid)
                                        {
                                            consumer.inputBlobs[j] = &ld.outputBlobs[0];
                                            consumer.inputBlobsWrappers[j] = ld.outputBlobsWrappers[0];
                                            break;
                                        }
                                    }
                                }
                            }
                            else if (fuse_eltwise) // conv + eltwise/naryEltwise (note: conv could have fused activations before eltwise)
                            {
                                CV_Assert(IS_DNN_CUDA_TARGET(preferableTarget));
                                CV_Assert_N(biasLayerData->outputBlobsWrappers.size() == 1, ld.inputBlobsWrappers.size() == 1);
                                ld.inputBlobsWrappers.push_back(biasLayerData->outputBlobsWrappers[0]);

                                if (nextEltwiseLayer)
                                    printf_(("\tfused with %s\n", nextEltwiseLayer->name.c_str()));
                                else if (nextNaryEltwiseLayer)
                                    printf_(("\tfused with %s\n", nextEltwiseLayer->name.c_str()));
                                else
                                    CV_Error(Error::StsError, "Both nextNaryEltwiseLayer and nextEltwiseLayer are empty!");

                                eltwiseData->skip = true;
                                // This optimization is for cases like
                                // some_layer   conv (maybe fused with activ)
                                //   |             |
                                //   +-- eltwise --+
                                //
                                // This way all the element-wise computations
                                // (i.e. some_layer+conv or some_layer*conv)
                                // would be done at [conv] layer. So we need to
                                // replace [conv]'s output blob to [eltwise]'s one.
                                // Also, we need to move all the consumers' references.
                                // To prevent memory collisions (i.e. when input of
                                // [conv] and output of [eltwise] is the same blob)
                                // we allocate a new blob.
                                CV_Assert_N(ld.outputBlobs.size() == 1, ld.outputBlobsWrappers.size() == 1);
                                ld.outputBlobs[0] = ld.outputBlobs[0].clone();
                                ld.outputBlobsWrappers[0] = wrap(ld.outputBlobs[0]);

                                eltwiseData->outputBlobs = ld.outputBlobs;
                                eltwiseData->outputBlobsWrappers = ld.outputBlobsWrappers;

                                // Move references of [eltwise] layer consumers to the newly allocated blob.
                                for (int i = 0; i < eltwiseData->consumers.size(); ++i)
                                {
                                    LayerData& consumer = layers[eltwiseData->consumers[i].lid];
                                    for (int j = 0; j < consumer.inputBlobsId.size(); ++j)
                                    {
                                        if (consumer.inputBlobsId[j].lid == eltwiseData->id)
                                        {
                                            consumer.inputBlobs[j] = &ld.outputBlobs[0];
                                            consumer.inputBlobsWrappers[j] = ld.outputBlobsWrappers[0];
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                break;
            }
        }

        if (preferableBackend != DNN_BACKEND_OPENCV && preferableBackend != DNN_BACKEND_CUDA)
            continue;  // Go to the next layer.

        // the optimization #2. if there is concat layer that concatenates channels
        // from the inputs together (i.e. axis == 1) then we make the inputs of
        // the concat layer to write to the concatenation output buffer
        // (and so we eliminate the concatenation layer, because the channels
        // are concatenated implicitly).
        Ptr<ConcatLayer> concatLayer = ld.layerInstance.dynamicCast<ConcatLayer>();
        if( !concatLayer.empty() && !concatLayer->padding && ld.outputBlobs.size() == 1 )
        {
            Mat& output = ld.outputBlobs[0];
            UMat umat_output;
#ifdef HAVE_OPENCL
            if (!ld.outputBlobsWrappers.empty() &&
                (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget)))
            {
                size_t i, ninputs = ld.inputBlobsId.size();
                bool conv_layer = true;
                for( i = 0; i < ninputs; i++ )
                {
                    LayerPin pin = ld.inputBlobsId[i];
                    LayerData* inp_i_data = &layers[pin.lid];
                    while(inp_i_data->skip &&
                          inp_i_data->inputBlobsId.size() == 1 &&
                          inp_i_data->consumers.size() == 1)
                    {
                        pin = inp_i_data->inputBlobsId[0];
                        inp_i_data = &layers[pin.lid];
                    }
                    conv_layer = conv_layer && (getLayerInstance(*inp_i_data)->type == "Convolution");
                }
                if (!conv_layer)
                    continue;
                std::vector<UMat> umat_outputBlobs;
                umat_outputBlobs = OpenCLBackendWrapper::getUMatVector(ld.outputBlobsWrappers);
                umat_output = umat_outputBlobs[0];
            }
#endif

            // TODO: in general, this optimization can always be done, but
            // many layers currently check that the input/output blobs are
            // continuous arrays. Unfortunately, this is not true when
            // the concatenation optimization is applied with batch_size > 1.
            // so, for now, we only apply this optimization in the most popular
            // case batch_size == 1.
            int axis = normalize_axis(concatLayer->axis, output.dims);
            if( output.total(0, axis) == 1 )
            {
                size_t i, ninputs = ld.inputBlobsId.size();
                std::vector<LayerPin> realinputs(ninputs);
                for( i = 0; i < ninputs; i++ )
                {
                    LayerPin pin = ld.inputBlobsId[i];
                    LayerData* inp_i_data = &layers[pin.lid];
                    while(inp_i_data->skip &&
                          inp_i_data->inputBlobsId.size() == 1 &&
                          inp_i_data->consumers.size() == 1)
                    {
                        pin = inp_i_data->inputBlobsId[0];
                        inp_i_data = &layers[pin.lid];
                    }
                    printf_(("\treal input for %s is %s\n",
                           layers[ld.inputBlobsId[i].lid].getLayerInstance()->name.c_str(),
                           inp_i_data->getLayerInstance()->name.c_str()));

                    if(inp_i_data->skip || inp_i_data->consumers.size() != 1)
                        break;
#ifdef HAVE_CUDA
                    if (preferableBackend == DNN_BACKEND_CUDA &&
                        (inp_i_data->layerInstance->supportBackend(DNN_BACKEND_CUDA) == false ||
                         (inp_i_data->layerInstance->type != "Convolution" &&
                          inp_i_data->layerInstance->type != "Pooling" &&
                          inp_i_data->layerInstance->type != "Resize"  &&
                          inp_i_data->layerInstance->type != "Flatten" &&
                          inp_i_data->layerInstance->type != "Permute" &&
                          inp_i_data->layerInstance->type != "Reorg" &&
                          inp_i_data->layerInstance->type != "Eltwise" &&
                          inp_i_data->layerInstance->type != "NaryEltwise" &&
                          inp_i_data->layerInstance.dynamicCast<ActivationLayer>().empty())))
                    {
                        break;
                    }
#endif
                    realinputs[i] = pin;
                }

                if( i >= ninputs )
                {
                    // Allocate new memory to prevent collisions during memory
                    // reusing (see https://github.com/opencv/opencv/pull/10456).
                    output = output.clone();
#ifdef HAVE_OPENCL
                    if (preferableBackend == DNN_BACKEND_OPENCV &&
                        IS_DNN_OPENCL_TARGET(preferableTarget))
                    {
                        std::vector<UMat> umats(1);
                        umat_output = umat_output.clone();
                        umats[0] = umat_output;
                        OpenCLBackendWrapper::update(ld.outputBlobsWrappers, umats);
                    }
#endif

#ifdef HAVE_CUDA
                    if (preferableBackend == DNN_BACKEND_CUDA)
                        ld.outputBlobsWrappers[0] = wrap(output);
#endif
                    std::vector<Range> chrange(output.dims, Range::all());
                    int ofs = 0;
                    for( i = 0; i < ninputs; i++ )
                    {
                        LayerPin pin = realinputs[i];
                        LayerData* inp_i_data = &layers[pin.lid];
                        int channels_i = ld.inputBlobs[i]->size[axis];
                        chrange[axis] = Range(ofs, ofs + channels_i);
                        printf_(("\toutput %s(%d) to channels (%d, %d)\n", inp_i_data->layerInstance->name.c_str(),
                               pin.oid, ofs, ofs + channels_i));
                        ofs += channels_i;
                        Mat output_slice = output(chrange);
                        Mat& curr_output = inp_i_data->outputBlobs[pin.oid];
                        CV_Assert(output_slice.isContinuous() && output_slice.size == curr_output.size);
                        Mat* oldPtr = &curr_output;
                        curr_output = output_slice;
#ifdef HAVE_OPENCL
                        if (preferableBackend == DNN_BACKEND_OPENCV && IS_DNN_OPENCL_TARGET(preferableTarget))
                        {
                            std::vector<UMat> umats(inp_i_data->outputBlobsWrappers.size());
                            umats[pin.oid] = umat_output(chrange);
                            OpenCLBackendWrapper::update(inp_i_data->outputBlobsWrappers, umats);
                        }
#endif
#ifdef HAVE_CUDA
                        if (preferableBackend == DNN_BACKEND_CUDA)
                        {
                            auto cuda_wrapper = wrap(output).dynamicCast<CUDABackendWrapper>();
                            auto offset = chrange[axis].start * output_slice.total(axis + 1, output.dims);
                            auto new_shape = shape(output_slice);
                            cuda_wrapper->update(new_shape, offset);
                            inp_i_data->outputBlobsWrappers[pin.oid] = cuda_wrapper.staticCast<BackendWrapper>();
                        }
#endif
                        // Layers that refer old input Mat will refer to the
                        // new data but the same Mat object.
                        CV_Assert_N(curr_output.data == output_slice.data, oldPtr == &curr_output);
                    }

#ifdef HAVE_CUDA
                    if (preferableBackend == DNN_BACKEND_CUDA)
                    {
                        for (int i = 0; i < ld.consumers.size(); i++)
                        {
                            LayerData& consumer = layers[ld.consumers[i].lid];
                            for (int j = 0; j < consumer.inputBlobsId.size(); j++)
                            {
                                if (consumer.inputBlobsId[j].lid == ld.id)
                                {
                                    CV_Assert(consumer.inputBlobs[j]->data == ld.outputBlobs[0].data);
                                    consumer.inputBlobsWrappers[j] = ld.outputBlobsWrappers[0];
                                    break;
                                }
                            }
                        }
                    }
#endif
                    ld.skip = true;
                    printf_(("\toptimized out Concat layer %s\n", concatLayer->name.c_str()));
                }
            }
        }
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
