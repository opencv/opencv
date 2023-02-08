// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_FLATBUFFERS
#include "schema_generated.h"
#endif

#include "builtin_op_data.h"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_FLATBUFFERS

using namespace opencv_tflite;

// This values are used to indicate layer output's data layout where it's possible.
// Approach is similar to TensorFlow importer but TFLite models do not have explicit
// layout field "data_format". So we consider that all 4D inputs are in NHWC data layout.
enum DataLayout
{
    DATA_LAYOUT_NHWC,
    DATA_LAYOUT_NCHW,
    DATA_LAYOUT_NDHWC,
    DATA_LAYOUT_UNKNOWN,
    DATA_LAYOUT_PLANAR  // 2-dimensional outputs (matmul, flatten, reshape to 2d)
};

class TFLiteImporter {
public:
    TFLiteImporter(Net& net, const char* modelBuffer, size_t bufSize);

private:
    const opencv_tflite::Model* model;
    const flatbuffers::Vector<flatbuffers::Offset<opencv_tflite::Tensor> >* modelTensors;
    std::map<int, Mat> allTensors;
    Net& dstNet;

    // This is a vector of pairs (layerId, outputId) where we iterate over
    // indices from TFLite notation and get created OpenCV layers.
    std::map<int, std::pair<int, int> > layerIds;

    // Tracking of layouts for layers outputs.
    std::vector<DataLayout> layouts;

    void populateNet();

    // Wrap TFLite Tensor to OpenCV Mat without data copying
    Mat parseTensor(const Tensor& tensor);

    typedef void (TFLiteImporter::*TFLiteImporterNodeParser)(const Operator&, const std::string&, LayerParams&);
    typedef std::map<std::string, TFLiteImporterNodeParser> DispatchMap;

    const DispatchMap dispatch;
    static DispatchMap buildDispatchMap();

    void parseConvolution(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseDWConvolution(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parsePadding(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseEltwise(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parsePooling(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parsePoolingWithArgmax(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseUnpooling(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseReshape(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseConcat(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseResize(const Operator& op, const std::string& opcode, LayerParams& layerParams);
    void parseDeconvolution(const Operator& op, const std::string& opcode, LayerParams& layerParams);

    int addPermuteLayer(const std::vector<int>& order, const std::string& permName, const std::pair<int, int>& inpId);
};

Mat TFLiteImporter::parseTensor(const Tensor& tensor) {
    CV_Assert(tensor.shape());
    std::vector<int> shape(tensor.shape()->begin(), tensor.shape()->end());
    int bufferIdx = tensor.buffer();
    CV_Assert(bufferIdx != 0);  // 0th buffer is a no-data buffer
    const Buffer* buffer = model->buffers()->Get(bufferIdx);
    CV_Assert(buffer && buffer->data());
    const void* data = buffer->data()->data();

    int dtype = -1;
    switch (tensor.type()) {
    case TensorType_FLOAT32:
        dtype = CV_32F;
        break;
    case TensorType_INT32:
        dtype = CV_32S;
        break;
    case TensorType_FLOAT16:
        dtype = CV_16S;
        break;
    default:
        CV_Error(Error::StsNotImplemented, format("Parse tensor with type %s", EnumNameTensorType(tensor.type())));
    }
    return Mat(shape, dtype, const_cast<void*>(data));
}

TFLiteImporter::TFLiteImporter(Net& dstNet, const char* modelBuffer, size_t bufSize)
    : dstNet(dstNet), dispatch(buildDispatchMap())
{
    flatbuffers::Verifier verifier((const uint8_t*)modelBuffer, bufSize);
    if (!VerifyModelBuffer(verifier)) {
        CV_Error(Error::StsError, "TFLite model is incorrect");
    }

    model = GetModel(modelBuffer);
    CV_Assert(model && model->subgraphs() && model->buffers());
    CV_CheckEQ(model->subgraphs()->size(), 1, "");

    modelTensors = model->subgraphs()->Get(0)->tensors();
    CV_Assert(modelTensors);
    for (int i = 0; i < modelTensors->size(); ++i) {
        const Tensor* tensor = modelTensors->Get(i);
        CV_Assert(tensor);
        if (tensor->buffer() != 0) {
            allTensors[i] = parseTensor(*tensor);
        }
    }

    populateNet();
}

DataLayout estimateLayout(const Tensor* t) {
    CV_Assert(t && t->shape());
    switch (t->shape()->size()) {
    case 5: return DATA_LAYOUT_NDHWC;
    case 4: return DATA_LAYOUT_NHWC;
    case 2: return DATA_LAYOUT_PLANAR;
    default: return DATA_LAYOUT_UNKNOWN;
    }
}

void TFLiteImporter::populateNet() {
    const SubGraph* subgraph = model->subgraphs()->Get(0);
    const auto* opCodes = model->operator_codes();
    CV_Assert(subgraph && subgraph->inputs() && subgraph->operators());
    CV_Assert(opCodes);

    layouts.resize(modelTensors->size(), DATA_LAYOUT_UNKNOWN);
    for (int i = 0; i < subgraph->inputs()->size(); ++i) {
        int idx = subgraph->inputs()->Get(i);
        layerIds[idx] = std::make_pair(0, i);
        layouts[idx] = estimateLayout(modelTensors->Get(idx));
    }
    for (const auto op : *subgraph->operators()) {
        CV_Assert(op && op->inputs() && op->outputs());
        int idx = op->opcode_index();

        LayerParams layerParams;
        layerParams.name = modelTensors->Get(op->outputs()->Get(0))->name()->str();

        std::string type = EnumNameBuiltinOperator(BuiltinOperator(opCodes->Get(idx)->deprecated_builtin_code()));
        if (type == "CUSTOM") {
            type = opCodes->Get(idx)->custom_code()->str();
        }

        try
        {
            if (type == "DEQUANTIZE") {
                // Convert from FP16 to FP32
                Mat data = allTensors[op->inputs()->Get(0)];
                Mat dataFP32;
                convertFp16(data, dataFP32);
                allTensors[op->outputs()->Get(0)] = dataFP32;
                continue;
            }

            DispatchMap::const_iterator iter = dispatch.find(type);
            if (iter != dispatch.end()) {
                CALL_MEMBER_FN(*this, iter->second)(*op, type, layerParams);
            }
            else {
                CV_Error(Error::StsNotImplemented, "Unsupported layer type " + type);
            }

            // Collect input blobs
            std::vector<int> layerInputs;
            std::vector<DataLayout> inpLayouts;
            for (int idx : *op->inputs()) {
                if (layerIds.find(idx) != layerIds.end()) {
                    layerInputs.push_back(idx);
                    inpLayouts.push_back(layouts[idx]);
                    continue;  // Output from a different layer
                }

                Mat blob = allTensors[idx];
                layerParams.blobs.push_back(blob.u ? blob : blob.clone());  // some tensors are owned by OpenCV
            }

            int layerId = dstNet.addLayer(layerParams.name, layerParams.type, layerParams);

            // Connect layer to inputs
            int i = 0;
            for (int idx : layerInputs) {
                auto it = layerIds.find(idx);
                dstNet.connect(it->second.first, it->second.second, layerId, i++);
            }

            // Predict output layout. Some layer-specific parsers may set them explicitly.
            // Otherwise, propagate input layout.
            if (layouts[op->outputs()->Get(0)] == DATA_LAYOUT_UNKNOWN) {
                DataLayout predictedLayout = DATA_LAYOUT_UNKNOWN;
                for (auto layout : inpLayouts) {
                    if (layout != DATA_LAYOUT_UNKNOWN) {
                        if (predictedLayout == DATA_LAYOUT_UNKNOWN)
                            predictedLayout = layout;
                        else if (predictedLayout != layout) {
                            predictedLayout = DATA_LAYOUT_UNKNOWN;
                            break;
                        }
                    }
                }
                layouts[op->outputs()->Get(0)] = predictedLayout;
            }

            // Register outputs
            i = 0;
            for (int idx : *op->outputs()) {
                layerIds[idx] = std::make_pair(layerId, i++);
            }
        }
        catch (const cv::Exception& e)
        {
            CV_UNUSED(e);
            if (DNN_DIAGNOSTICS_RUN)
            {
                CV_LOG_ERROR(NULL, "DNN/TFLite: Problem during import of layer " << layerParams.name.c_str() << " with type " << type.c_str());
                continue;
            }
            throw;
        }
    }
}

TFLiteImporter::DispatchMap TFLiteImporter::buildDispatchMap()
{
    static DispatchMap dispatch;
    if (!dispatch.empty())
        return dispatch;

    dispatch["CONV_2D"] = &TFLiteImporter::parseConvolution;
    dispatch["DEPTHWISE_CONV_2D"] = &TFLiteImporter::parseDWConvolution;
    dispatch["RELU"] = dispatch["ADD"] = dispatch["MUL"] = dispatch["PRELU"] =
        dispatch["HARD_SWISH"] = dispatch["LOGISTIC"] = &TFLiteImporter::parseEltwise;
    dispatch["MAX_POOL_2D"] = dispatch["AVERAGE_POOL_2D"] = &TFLiteImporter::parsePooling;
    dispatch["MaxPoolingWithArgmax2D"] = &TFLiteImporter::parsePoolingWithArgmax;
    dispatch["MaxUnpooling2D"] = &TFLiteImporter::parseUnpooling;
    dispatch["PAD"] = &TFLiteImporter::parsePadding;
    dispatch["RESHAPE"] = &TFLiteImporter::parseReshape;
    dispatch["CONCATENATION"] = &TFLiteImporter::parseConcat;
    dispatch["RESIZE_BILINEAR"] = &TFLiteImporter::parseResize;
    dispatch["Convolution2DTransposeBias"] = &TFLiteImporter::parseDeconvolution;
    return dispatch;
}

void TFLiteImporter::parseConvolution(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Convolution";

    auto options = reinterpret_cast<const Conv2DOptions*>(op.builtin_options());
    layerParams.set("pad_mode", EnumNamePadding(options->padding()));
    layerParams.set("stride_w", options->stride_w());
    layerParams.set("stride_h", options->stride_h());
    layerParams.set("dilation_w", options->dilation_w_factor());
    layerParams.set("dilation_h", options->dilation_h_factor());

    // Get filter size
    int filterIdx = op.inputs()->Get(1);
    Mat filter = allTensors[filterIdx];
    int oc = filter.size[0];
    int kh = filter.size[1];
    int kw = filter.size[2];
    int ic = filter.size[3];
    layerParams.set("kernel_w", kw);
    layerParams.set("kernel_h", kh);
    layerParams.set("num_output", oc);

    // Reorder filter data from OHWI to OIHW and change shape correspondingly.
    filter = allTensors[filterIdx] = filter.reshape(1, {oc, ic, kh, kw});

    CV_CheckTypeEQ(filter.type(), CV_32F, "");
    Mat filterCopy = filter.clone();
    float* data = filterCopy.ptr<float>();
    float* dstData = filter.ptr<float>();

    int total = oc * ic * kh * kw;
    for (int i_oc = 0; i_oc < oc; i_oc++) {
        for (int i_ic = 0; i_ic < ic; i_ic++) {
            for (int i_h = 0; i_h < kh; i_h++) {
                for (int i_w = 0; i_w < kw; i_w++) {
                    int dst_i = kw * (kh * (ic * i_oc + i_ic) + i_h) + i_w;
                    int src_i = ic * (kw * (kh * i_oc + i_h) + i_w) + i_ic;
                    CV_CheckLT(dst_i, total, "");
                    CV_CheckLT(src_i, total, "");
                    dstData[dst_i] = data[src_i];
                }
            }
        }
    }
}

void TFLiteImporter::parseDWConvolution(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Convolution";

    auto options = reinterpret_cast<const DepthwiseConv2DOptions*>(op.builtin_options());
    layerParams.set("pad_mode", EnumNamePadding(options->padding()));
    layerParams.set("stride_w", options->stride_w());
    layerParams.set("stride_h", options->stride_h());
    layerParams.set("dilation_w", options->dilation_w_factor());
    layerParams.set("dilation_h", options->dilation_h_factor());

    int filterIdx = op.inputs()->Get(1);
    Mat filter = allTensors[filterIdx];
    int kh = filter.size[1];
    int kw = filter.size[2];
    int oc = filter.size[3];
    layerParams.set("kernel_w", kw);
    layerParams.set("kernel_h", kh);
    layerParams.set("num_output", oc);
    layerParams.set("group", oc);

    filter = allTensors[filterIdx] = filter.reshape(1, {oc, 1, kh, kw});
    cv::transpose(filter.reshape(1, kh * kw).clone(), filter.reshape(1, oc));
}

void TFLiteImporter::parsePadding(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Padding";
    Mat paddings = allTensors[op.inputs()->Get(1)];

    CV_CheckTypeEQ(paddings.type(), CV_32S, "");
    //  N    H    W    C
    // 0 1  2 3  4 5  6 7
    std::swap(paddings.at<int32_t>(2), paddings.at<int32_t>(6));
    std::swap(paddings.at<int32_t>(3), paddings.at<int32_t>(7));
    //  N    C    W    H
    // 0 1  2 3  4 5  6 7
    std::swap(paddings.at<int32_t>(4), paddings.at<int32_t>(6));
    std::swap(paddings.at<int32_t>(5), paddings.at<int32_t>(7));
    //  N    C    H    W
    // 0 1  2 3  4 5  6 7

    layerParams.set("paddings", DictValue::arrayInt<int32_t*>((int32_t*)paddings.data, paddings.total()));
}

void TFLiteImporter::parseEltwise(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    if (opcode == "PRELU") {
        layerParams.type = "PReLU";
    } else if (opcode == "RELU") {
        layerParams.type = "ReLU";
    } else if (opcode == "ADD") {
        layerParams.type = "Eltwise";
        layerParams.set("operation", "sum");
    } else if (opcode == "MUL") {
        layerParams.type = "Eltwise";
        layerParams.set("operation", "prod");
    } else if (opcode == "HARD_SWISH") {
        layerParams.type = "HardSwish";
    } else if (opcode == "LOGISTIC") {
        layerParams.type = "Sigmoid";
    } else {
        CV_Error(Error::StsNotImplemented, "Unknown eltwise operator opcode: " + opcode);
    }
}

void TFLiteImporter::parsePooling(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Pooling";

    auto options = reinterpret_cast<const Pool2DOptions*>(op.builtin_options());
    layerParams.set("pad_mode", EnumNamePadding(options->padding()));
    layerParams.set("stride_w", options->stride_w());
    layerParams.set("stride_h", options->stride_h());
    layerParams.set("kernel_w", options->filter_width());
    layerParams.set("kernel_h", options->filter_height());
    if (opcode == "MAX_POOL_2D")
        layerParams.set("pool", "max");
    else if (opcode == "AVERAGE_POOL_2D")
        layerParams.set("pool", "ave");
    else
        CV_Error(Error::StsNotImplemented, "Pool type selection for " + opcode);
}

void TFLiteImporter::parsePoolingWithArgmax(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Pooling";

    CV_CheckLE(op.custom_options()->size(), sizeof(TfLitePoolParams), "");
    const auto* params = reinterpret_cast<const TfLitePoolParams*>(op.custom_options()->Data());
    if (params->padding != kTfLitePaddingUnknown)
        layerParams.set("pad_mode", params->padding == kTfLitePaddingSame ? "SAME" : "VALID");
    layerParams.set("stride_w", params->stride_width);
    layerParams.set("stride_h", params->stride_height);
    layerParams.set("kernel_w", params->filter_width);
    layerParams.set("kernel_h", params->filter_height);
    layerParams.set("pool", "max");
}

void TFLiteImporter::parseUnpooling(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "MaxUnpool";

    CV_CheckLE(op.custom_options()->size(), sizeof(TfLitePoolParams), "");
    const auto* params = reinterpret_cast<const TfLitePoolParams*>(op.custom_options()->Data());
    layerParams.set("pool_stride_w", params->stride_width);
    layerParams.set("pool_stride_h", params->stride_height);
    layerParams.set("pool_k_w", params->filter_width);
    layerParams.set("pool_k_h", params->filter_height);
    layerParams.set("pool_pad_w", 0);
    layerParams.set("pool_pad_h", 0);
}

void TFLiteImporter::parseReshape(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    DataLayout inpLayout = layouts[op.inputs()->Get(0)];

    if (inpLayout == DATA_LAYOUT_NHWC) {
        // Permute to NCHW
        int permId = addPermuteLayer({0, 2, 3, 1}, layerParams.name + "/permute", layerIds[op.inputs()->Get(0)]);  // NCHW -> NHWC
        layerIds[op.inputs()->Get(0)] = std::make_pair(permId, 0);
        layouts[op.outputs()->Get(0)] = DATA_LAYOUT_NCHW;
    }

    layerParams.type = "Reshape";
    auto options = reinterpret_cast<const ReshapeOptions*>(op.builtin_options());
    std::vector<int> shape(options->new_shape()->begin(), options->new_shape()->end());
    // std::swap(shape[1], shape[2]);
    layerParams.set("dim", DictValue::arrayInt<int*>(shape.data(), shape.size()));
}

void TFLiteImporter::parseConcat(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Concat";
    auto options = reinterpret_cast<const ConcatenationOptions*>(op.builtin_options());
    int axis = options->axis();

    DataLayout inpLayout = layouts[op.inputs()->Get(0)];
    if (inpLayout == DATA_LAYOUT_NHWC) {
        // OpenCV works in NCHW data layout. So change the axis correspondingly.
        CV_Assert(-4 < axis && axis < 4);
        int remap[] = {0, 2, 3, 1};
        axis = axis > 0 ? axis : 4 + axis;
        axis = remap[axis];
    }
    layerParams.set("axis", axis);
}

void TFLiteImporter::parseResize(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Resize";

    auto options = reinterpret_cast<const ResizeBilinearOptions*>(op.builtin_options());

    layerParams.set("interpolation", "bilinear");
    layerParams.set("align_corners", options->align_corners());
    layerParams.set("half_pixel_centers", options->half_pixel_centers());

    Mat shape = allTensors[op.inputs()->Get(1)].reshape(1, 1);
    layerParams.set("height", shape.at<int>(0, 0));
    layerParams.set("width", shape.at<int>(0, 1));
}

int TFLiteImporter::addPermuteLayer(const std::vector<int>& order, const std::string& permName,
                                    const std::pair<int, int>& inpId)
{
    LayerParams permLP;
    permLP.set("order", DictValue::arrayInt<const int*>(order.data(), order.size()));
    int permId = dstNet.addLayer(permName, "Permute", permLP);
    dstNet.connect(inpId.first, inpId.second, permId, 0);
    return permId;
}

void TFLiteImporter::parseDeconvolution(const Operator& op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Deconvolution";

    CV_CheckLE(op.custom_options()->size(), sizeof(TfLiteTransposeConvParams), "");
    const auto* params = reinterpret_cast<const TfLiteTransposeConvParams*>(op.custom_options()->Data());
    if (params->padding != kTfLitePaddingUnknown)
        layerParams.set("pad_mode", params->padding == kTfLitePaddingSame ? "SAME" : "VALID");
    layerParams.set("stride_w", params->stride_width);
    layerParams.set("stride_h", params->stride_height);

    // Get filter size
    int filterIdx = op.inputs()->Get(1);
    Mat filter = allTensors[filterIdx];
    int oc = filter.size[0];
    int kh = filter.size[1];
    int kw = filter.size[2];
    int ic = filter.size[3];
    layerParams.set("kernel_w", kw);
    layerParams.set("kernel_h", kh);
    layerParams.set("num_output", oc);

    // Add adjust padding similar to TensorFlow (see tf_importer)
    const auto* outShape = modelTensors->Get(op.outputs()->Get(0))->shape();
    const int outH = outShape->Get(1);
    const int outW = outShape->Get(2);
    if (params->padding == kTfLitePaddingSame)
    {
        layerParams.set("adj_w", (outW - 1) % params->stride_width);
        layerParams.set("adj_h", (outH - 1) % params->stride_height);
    }
    else if (params->padding == kTfLitePaddingValid)
    {
        layerParams.set("adj_w", (outW - kw) % params->stride_width);
        layerParams.set("adj_h", (outH - kh) % params->stride_height);
    }

    // Reorder filter data from OHWI to IOHW and change shape correspondingly.
    filter = allTensors[filterIdx] = filter.reshape(1, {ic, oc, kh, kw});

    CV_CheckTypeEQ(filter.type(), CV_32F, "");
    Mat filterCopy = filter.clone();
    float* data = filterCopy.ptr<float>();
    float* dstData = filter.ptr<float>();

    int total = oc * ic * kh * kw;
    for (int i_oc = 0; i_oc < oc; i_oc++) {
        for (int i_ic = 0; i_ic < ic; i_ic++) {
            for (int i_h = 0; i_h < kh; i_h++) {
                for (int i_w = 0; i_w < kw; i_w++) {
                    int dst_i = kw * (kh * (oc * i_ic + i_oc) + i_h) + i_w;
                    int src_i = ic * (kw * (kh * i_oc + i_h) + i_w) + i_ic;
                    CV_Assert(dst_i < total);
                    CV_Assert(src_i < total);
                    dstData[dst_i] = data[src_i];
                }
            }
        }
    }
}

Net readNetFromTFLite(const String &modelPath) {
    Net net;

    std::vector<char> content;

    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(modelPath, mode);
    CV_Assert(ifs.is_open());

    ifs.seekg(0, std::ios::end);
    const size_t sz = ifs.tellg();
    CV_Assert(sz > 0);
    content.resize(sz);
    ifs.seekg(0, std::ios::beg);

    ifs.read(content.data(), sz);

    TFLiteImporter(net, content.data(), content.size());
    return net;
}

Net readNetFromTFLite(const std::vector<uchar>& bufferModel) {
    return readNetFromTFLite((const char*)bufferModel.data(), bufferModel.size());
}

Net readNetFromTFLite(const char *bufferModel, size_t bufSize) {
    Net net;
    TFLiteImporter(net, bufferModel, bufSize);
    return net;
}

#else

Net readNetFromTFLite(const String &) {
    CV_Error(Error::StsError, "Build OpenCV with FlatBuffers to import TFLite models");
}

Net readNetFromTFLite(const std::vector<uchar>&) {
    CV_Error(Error::StsError, "Build OpenCV with FlatBuffers to import TFLite models");
}

Net readNetFromTFLite(const char *, size_t) {
    CV_Error(Error::StsError, "Build OpenCV with FlatBuffers to import TFLite models");
}

#endif // HAVE_FLATBUFFERS

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
