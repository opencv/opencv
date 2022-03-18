// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


// FIXIT drop from inference API
static
void getQuantizationParams(const Mat& src, std::vector<float>& scales, std::vector<int>& zeropoints)
{
    const int qmin = -128; // INT8_MIN
    const int qmax = 127;  // INT8_MAX

    double rmin, rmax, sc, zp;
    cv::minMaxIdx(src, &rmin, &rmax);

    // 0 must be present in the range [rmin, rmax]
    rmin = std::min(rmin, 0.0);
    rmax = std::max(rmax, 0.0);

    sc = (rmax == rmin) ? 1.0 : (rmax - rmin)/(qmax - qmin);
    zp = qmin - (rmin/sc);

    scales.push_back((float)sc);
    zeropoints.push_back((int)std::round(zp));
}

// FIXIT drop from inference API
Net Net::Impl::quantize(InputArrayOfArrays calibData, int inputsDtype, int outputsDtype)
{
    // Net can be quantized only once.
    if (netWasQuantized)
        CV_Error(Error::StsBadArg, "Cannot quantize a quantized net");

    CV_CheckType(inputsDtype, inputsDtype == CV_32F || inputsDtype == CV_8S, "Input depth should be CV_32F or CV_8S");
    CV_CheckType(outputsDtype, outputsDtype == CV_32F || outputsDtype == CV_8S, "Output depth should be CV_32F or CV_8S");

    bool originalFusion = fusion;
    int prefBackend = preferableBackend;
    int prefTarget = preferableTarget;

    // Disable fusions and use CPU backend to quantize net
    setPreferableBackend(DNN_BACKEND_OPENCV);
    setPreferableTarget(DNN_TARGET_CPU);
    enableFusion(false);

    if (calibData.isMat())
    {
        setInput(calibData.getMat(), /*name=*/"", /*scalefactor=*/1.0, /*mean=*/Scalar());
    }
    else if (calibData.isMatVector())
    {
        std::vector<Mat> calibDataVec;
        calibData.getMatVector(calibDataVec);

        std::vector<String> inpNames = netInputLayer->outNames;
        CV_CheckEQ(calibDataVec.size(), inpNames.size(), "Calibration data size should be equal to number of inputs");
        for (int i = 0; i < calibDataVec.size(); i++)
            setInput(calibDataVec[i], inpNames[i], /*scalefactor=*/1.0, /*mean=*/Scalar());
    }

    std::vector<String> outNames = getUnconnectedOutLayersNames();
    std::vector<LayerPin> pins;
    for (int i = 0; i < outNames.size(); i++)
        pins.push_back(getPinByAlias(outNames[i]));
    setUpNet(pins);

    // Compute scales and zeropoints for all the layers
    std::vector<std::vector<float> > scales;
    std::vector<std::vector<int> > zeropoints;
    for (Impl::MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
    {
        LayerData& ld = it->second;
        if (!ld.skip)
        {
            Ptr<Layer> layer = ld.layerInstance;
            std::vector<Mat> inps(ld.inputBlobs.size());
            for (int i = 0; i < ld.inputBlobs.size(); ++i)
                inps[i] = *ld.inputBlobs[i];
            layer->forward(inps, ld.outputBlobs, ld.internals);
        }

        std::vector<float> sc;
        std::vector<int> zp;
        if (ld.type == "TanH")
        {
            sc.push_back(1.f/128);
            zp.push_back(0);
        }
        else if (ld.type == "Sigmoid" || ld.type == "Softmax" || ld.type == "SoftMax")
        {
            if (ld.params.get<bool>("log_softmax", false))
            {
                sc.push_back(16.f/256);
                zp.push_back(127);
            }
            else
            {
                sc.push_back(1.f/256);
                zp.push_back(-128);
            }
        }
        else if (ld.type == "Split" || ld.type == "Slice" || ld.type == "Crop")
        {
            std::vector<float> inp_sc; std::vector<int> inp_zp;
            getQuantizationParams(*ld.inputBlobs[0], inp_sc, inp_zp);
            sc.assign(ld.outputBlobs.size(), inp_sc[0]);
            zp.assign(ld.outputBlobs.size(), inp_zp[0]);
        }
        else
        {
            for (int i = 0; i < ld.outputBlobs.size(); i++)
                getQuantizationParams(ld.outputBlobs[i], sc, zp);
        }
        scales.push_back(sc);
        zeropoints.push_back(zp);
    }

    // For some layers, the input and output scales/zeropoints must be equal so that rescaling of inputs
    // is not needed during quantized inference. We start from the last layer and modify the layer's input scales/zeropoints
    // TODO : Need a different approach. Current solution fails when 2 such layers have the same input layer
    for (Impl::MapIdToLayerData::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it)
    {
        LayerData& ld = it->second;
        // Layers with multiple outputs. Number of outputs is equal to number of inputs
        if (ld.type == "Blank" || ld.type == "Dropout" || ld.type == "Identity" || ld.type == "Silence" ||
            ld.type == "Flatten" || ld.type == "Padding" || ld.type == "Permute" || ld.type == "Reshape" ||
            ld.type == "ReLU6" || ld.type == "Reorg" || ld.type == "ShuffleChannel" || ld.type == "Resize" ||
           (ld.type == "ReLU" && !ld.params.get<float>("negative_slope", 0.f)) || /* ReLU with negative slope 0 */
           (ld.type == "Reduce" && (toLowerCase(ld.params.get<String>("reduce")) == "max" ||
            toLowerCase(ld.params.get<String>("reduce")) == "min")))
        {
            for (int i = 0; i < ld.outputBlobs.size(); i++)
            {
                LayerPin &pin = ld.inputBlobsId[i];
                scales[pin.lid][pin.oid] = scales[ld.id][i];
                zeropoints[pin.lid][pin.oid] = zeropoints[ld.id][i];
            }
        }
        // Layers with multiple inputs and single output.
        else if ((ld.type == "Pooling" && toLowerCase(ld.params.get<String>("pool", "max")) == "max") /* Max Pooling */ ||
                 (ld.type == "Eltwise" && toLowerCase(ld.params.get<String>("operation", "sum")) == "max") /* Elementwise max */ ||
                  ld.type == "Concat")
        {
            for (int i = 0; i < ld.inputBlobsId.size(); i++)
            {
                LayerPin &pin = ld.inputBlobsId[i];
                scales[pin.lid][pin.oid] = scales[ld.id][0];
                zeropoints[pin.lid][pin.oid] = zeropoints[ld.id][0];
            }
        }
    }

    // Create a new Net and add quantized layers to it.
    Net dstNet_;
    Net::Impl& dstNet = *(dstNet_.impl);
    dstNet.netWasQuantized = true;
    dstNet.setInputsNames(netInputLayer->outNames);
    dstNet.setPreferableBackend(prefBackend);
    dstNet.setPreferableTarget(prefTarget);
    dstNet.enableFusion(originalFusion);

    for (Impl::MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
    {
        LayerData ld = it->second;
        if (ld.id == 0)
        {
            LayerData &quantInpLd = dstNet.layers[0];
            quantInpLd.dtype = inputsDtype;
            quantInpLd.params.set("scales", DictValue::arrayReal(scales[0].data(), scales[0].size()));
            quantInpLd.params.set("zeropoints", DictValue::arrayInt(zeropoints[0].data(), zeropoints[0].size()));
            continue;
        }

        std::vector<LayerPin> inpPins = ld.inputBlobsId;
        // Fill input and output scales/zeropoints for the layer
        std::vector<std::vector<float> > inp_out_sc(2);
        std::vector<std::vector<int> > inp_out_zp(2);
        for (int i = 0; i < inpPins.size(); i++)
        {
            LayerPin &pin = inpPins[i];
            inp_out_sc[0].push_back(scales[pin.lid][pin.oid]);
            inp_out_zp[0].push_back(zeropoints[pin.lid][pin.oid]);
        }
        inp_out_sc[1] = scales[ld.id];
        inp_out_zp[1] = zeropoints[ld.id];

        // Quantize layer
        Ptr<Layer> layer = ld.layerInstance;
        if (layer->tryQuantize(inp_out_sc, inp_out_zp, ld.params))
        {
            ld.type += "Int8";
            ld.dtype = CV_8S;
        }
        ld.params.set("scales", DictValue::arrayReal(inp_out_sc[1].data(), inp_out_sc[1].size()));
        ld.params.set("zeropoints", DictValue::arrayInt(inp_out_zp[1].data(), inp_out_zp[1].size()));

        // Check and add quantize/dequantize node before layer
        for (int i = 0; i < inpPins.size(); i++)
        {
            LayerPin &pin = inpPins[i];
            LayerData &inpLd = dstNet.getLayerData(getLayerName(pin.lid));
            pin.lid = inpLd.id;
            if (inpLd.dtype != ld.dtype)
            {
                String layerName = (inpLd.dtype == CV_32F && ld.dtype == CV_8S) ? cv::format("quantize/%s/%d", inpLd.name.c_str(), pin.oid)
                                                                                : cv::format("dequantize/%s/%d", inpLd.name.c_str(), pin.oid);
                // Check if quantize/dequantize node for the input layer already exists
                if (dstNet.getLayerId(layerName) >= 0)
                {
                    pin.lid = dstNet.getLayerId(layerName);
                    pin.oid = 0;
                }
                else
                {
                    LayerParams lp;
                    lp.set("scales", inp_out_sc[0][i]);
                    lp.set("zeropoints", inp_out_zp[0][i]);
                    lp.name = layerName;
                    lp.type = (inpLd.dtype == CV_32F && ld.dtype == CV_8S) ? "Quantize" : "Dequantize";
                    int newLid = dstNet.addLayer(lp.name, lp.type, ld.dtype, lp);
                    dstNet.connect(pin.lid, pin.oid, newLid, 0);
                    pin.lid = newLid; pin.oid = 0;
                }
            }
        }

        // Add quantized layer to Net and connect to its inputs.
        int newLid = dstNet.addLayer(ld.name, ld.type, ld.dtype, ld.params);
        for( int i = 0; i < inpPins.size(); i++ )
            dstNet.connect(inpPins[i].lid, inpPins[i].oid, newLid, i);

        // If the layer is a output layer, add quantize/dequantize node after it based on output's data type.
        if (ld.requiredOutputs.size() == 0 && ld.dtype != outputsDtype)
        {
            LayerParams lp;
            lp.set("scales", inp_out_sc[1][0]);
            lp.set("zeropoints", inp_out_zp[1][0]);
            lp.name = ((ld.dtype == CV_32F && outputsDtype == CV_8S) ? "quantize/" : "dequantize/") + ld.name;
            lp.type = (ld.dtype == CV_32F && outputsDtype == CV_8S) ? "Quantize" : "Dequantize";
            dstNet.addLayerToPrev(lp.name, lp.type, outputsDtype, lp);
        }
    }
    // Restore FP32 Net's backend, target and fusion
    setPreferableBackend(prefBackend);
    setPreferableTarget(prefTarget);
    enableFusion(originalFusion);
    return dstNet_;
}

// FIXIT drop from inference API
void Net::Impl::getInputDetails(std::vector<float>& scales, std::vector<int>& zeropoints) /*const*/
{
    if (!netWasQuantized)
        CV_Error(Error::StsBadFunc, "Net isn't quantized");

    LayerParams &lp = layers[0].params;
    DictValue sc = lp.get("scales");
    DictValue zp = lp.get("zeropoints");

    for (int i = 0; i < sc.size(); i++)
    {
        scales.push_back(sc.get<float>(i));
        zeropoints.push_back(zp.get<int>(i));
    }
}

// FIXIT drop from inference API
void Net::Impl::getOutputDetails(std::vector<float>& scales, std::vector<int>& zeropoints) /*const*/
{
    if (!netWasQuantized)
        CV_Error(Error::StsBadFunc, "Net isn't quantized");

    std::vector<int> outLayerIds = getUnconnectedOutLayers();
    for (auto &lid : outLayerIds)
    {
        LayerParams &lp = layers[lid].params;
        DictValue sc = lp.get("scales");
        DictValue zp = lp.get("zeropoints");

        for (int i = 0; i < sc.size(); i++)
        {
            scales.push_back(sc.get<float>(i));
            zeropoints.push_back(zp.get<int>(i));
        }
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
