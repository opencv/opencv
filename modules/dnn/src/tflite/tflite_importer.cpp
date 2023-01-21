// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "schema_generated.h"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using namespace tflite;

// void parseOperator(Operator* op, LayerParams params) {
//     std::string type = 
// }


class TFLiteImporter {
public:
    TFLiteImporter(Net& net, const std::string& modelPath);

private:
    const tflite::Model* model;
    std::map<int, Mat> allTensors;

    void populateNet(Net& net);

    // Wrap TFLite Tensor to OpenCV Mat without data copying
    Mat parseTensor(const Tensor* tensor);

    typedef void (TFLiteImporter::*TFLiteImporterNodeParser)(const Operator*, const std::string&, LayerParams&);
    typedef std::map<std::string, TFLiteImporterNodeParser> DispatchMap;

    const DispatchMap dispatch;
    static DispatchMap buildDispatchMap();

    void parseConvolution(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parseDWConvolution(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parsePadding(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parseEltwise(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parsePooling(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parseReshape(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parseConcat(const Operator* op, const std::string& opcode, LayerParams& layerParams);
    void parseResize(const Operator* op, const std::string& opcode, LayerParams& layerParams);
};

Mat TFLiteImporter::parseTensor(const Tensor* tensor) {
    std::vector<int> shape(tensor->shape()->begin(), tensor->shape()->end());
    int bufferIdx = tensor->buffer();
    CV_Assert(bufferIdx != 0);  // 0th buffer is a no-data buffer
    const void* data = model->buffers()->Get(bufferIdx)->data()->data();

    int dtype = -1;
    switch (tensor->type()) {
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
        CV_Error(Error::StsNotImplemented, format("Parse tensor with type %s", EnumNameTensorType(tensor->type())));
    }
    return Mat(shape, dtype, const_cast<void*>(data));
}

TFLiteImporter::TFLiteImporter(Net& dstNet, const std::string& modelPath)
    : dispatch(buildDispatchMap())
{
    std::vector<char> content;

    const std::ios::openmode mode = std::ios::in | std::ios::binary;
    std::ifstream ifs(modelPath, mode);

    ifs.seekg(0, std::ios::end);
    const size_t sz = ifs.tellg();
    content.resize(sz);
    ifs.seekg(0, std::ios::beg);

    ifs.read((char*)content.data(), sz);

    model = GetModel(content.data());

    CV_CheckEQ(model->subgraphs()->size(), 1, "");

    auto tensors = model->subgraphs()->Get(0)->tensors();
    for (int i = 0; i < tensors->size(); ++i) {
        const Tensor* tensor = tensors->Get(i);
        if (tensor->buffer() != 0) {
            allTensors[i] = parseTensor(tensor);
        }
    }

    populateNet(dstNet);
}

void TFLiteImporter::populateNet(Net& dstNet) {
    // This is a vector of pairs (layerId, outputId) where we iterate over
    // indices from TFLite notation and get created OpenCV layers.
    std::map<int, std::pair<int, int> > layerIds;
    layerIds[0] = std::make_pair(0, 0);

    const SubGraph* subgraph = model->subgraphs()->Get(0);
    for (const auto op : *subgraph->operators()) {
        int idx = op->opcode_index();

        LayerParams layerParams;
        layerParams.name = subgraph->tensors()->Get(op->outputs()->Get(0))->name()->str();

        std::string type = EnumNameBuiltinOperator(BuiltinOperator(model->operator_codes()->Get(idx)->deprecated_builtin_code()));
        if (type == "CUSTOM")
            break;

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
            CALL_MEMBER_FN(*this, iter->second)(op, type, layerParams);
        }
        else {
            CV_Error(Error::StsNotImplemented, "Cannot import layer of type " + type);
        }

        // Collect input blobs
        std::vector<int> layerInputs;
        for (int idx : *op->inputs()) {
            if (layerIds.find(idx) != layerIds.end()) {
                layerInputs.push_back(idx);
                continue;  // Output from a different layer
            }

            Mat blob = allTensors[idx];
            layerParams.blobs.push_back(blob.clone());
        }

        int layerId = dstNet.addLayer(layerParams.name, layerParams.type, layerParams);

        // Connect layer to inputs
        int i = 0;
        for (int idx : layerInputs) {
            auto it = layerIds.find(idx);
            // std::cout << "connect " << idx << " " << layerId << " " << i << " " << it->second.first << " " << it->second.second << std::endl;
            dstNet.connect(it->second.first, it->second.second, layerId, i++);
        }

        // Register outputs
        i = 0;
        for (int idx : *op->outputs()) {
            layerIds[idx] = std::make_pair(layerId, i++);
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
    dispatch["PAD"] = &TFLiteImporter::parsePadding;
    dispatch["RESHAPE"] = &TFLiteImporter::parseReshape;
    dispatch["CONCATENATION"] = &TFLiteImporter::parseConcat;
    dispatch["RESIZE_BILINEAR"] = &TFLiteImporter::parseResize;
    return dispatch;
}

void TFLiteImporter::parseConvolution(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Convolution";

    auto options = reinterpret_cast<const Conv2DOptions*>(op->builtin_options());
    layerParams.set("pad_mode", EnumNamePadding(options->padding()));
    layerParams.set("stride_w", options->stride_w());
    layerParams.set("stride_h", options->stride_h());
    layerParams.set("dilation_w", options->dilation_w_factor());
    layerParams.set("dilation_h", options->dilation_h_factor());

    // Get filter size
    int filterIdx = op->inputs()->Get(1);
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

    CV_CheckEQ(filter.type(), CV_32F, "Float weights expected");
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
                    CV_Assert(dst_i < total);
                    CV_Assert(src_i < total);
                    dstData[dst_i] = data[src_i];
                }
            }
        }
    }
}

void TFLiteImporter::parseDWConvolution(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Convolution";

    auto options = reinterpret_cast<const DepthwiseConv2DOptions*>(op->builtin_options());
    layerParams.set("pad_mode", EnumNamePadding(options->padding()));
    layerParams.set("stride_w", options->stride_w());
    layerParams.set("stride_h", options->stride_h());
    layerParams.set("dilation_w", options->dilation_w_factor());
    layerParams.set("dilation_h", options->dilation_h_factor());

    int filterIdx = op->inputs()->Get(1);
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

void TFLiteImporter::parsePadding(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Padding";
    Mat paddings = allTensors[op->inputs()->Get(1)];

    CV_CheckEQ(paddings.type(), CV_32S, "");
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

void TFLiteImporter::parseEltwise(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
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

void TFLiteImporter::parsePooling(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Pooling";

    auto options = reinterpret_cast<const Pool2DOptions*>(op->builtin_options());
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

void TFLiteImporter::parseReshape(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Reshape";
    auto options = reinterpret_cast<const ReshapeOptions*>(op->builtin_options());
    std::vector<int> shape(options->new_shape()->begin(), options->new_shape()->end());
    std::swap(shape[1], shape[2]);
    layerParams.set("dim", DictValue::arrayInt<int*>(shape.data(), shape.size()));
}

void TFLiteImporter::parseConcat(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Concat";
    auto options = reinterpret_cast<const ConcatenationOptions*>(op->builtin_options());
    // options->axis()
    layerParams.set("axis", -1);
}

void TFLiteImporter::parseResize(const Operator* op, const std::string& opcode, LayerParams& layerParams) {
    layerParams.type = "Resize";

    auto options = reinterpret_cast<const ResizeBilinearOptions*>(op->builtin_options());

    layerParams.set("interpolation", "bilinear");
    layerParams.set("align_corners", options->align_corners());
    layerParams.set("half_pixel_centers", options->half_pixel_centers());

    Mat shape = allTensors[op->inputs()->Get(1)].reshape(1, 1);
    layerParams.set("height", shape.at<int>(0, 0));
    layerParams.set("width", shape.at<int>(0, 1));
}

Net readNetFromTFLite(const String &modelPath)
{
    Net net;
    TFLiteImporter(net, modelPath);
    return net;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn