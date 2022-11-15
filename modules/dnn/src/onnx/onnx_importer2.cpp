// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/dnn/layer_reg.private.hpp>
#include <opencv2/core/utils/fp_control_utils.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <string>

#if defined _MSC_VER && _MSC_VER < 1910/*MSVS 2017*/
#pragma warning(push)
#pragma warning(disable: 4503)  // decorated name length exceeded, name was truncated
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "../engine/engine.hpp"
#include "opencv-onnx-pb-c.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

typedef Net2::Impl NetImpl;

static void onnxParseError(const string& ctx, const string& msg)
{
    throw std::runtime_error((ctx.empty() ? " " + ctx : "") + ": " + msg);
}

#define OnnxAssert(ctx, expr) if (!!(expr)) ; else onnxParseError(ctx, "assertion '" #expr "' is invalid")

static string onnxConcatCtx(const string& ctx, const string& subctx)
{
    return ctx.empty() ? subctx : ctx + ", " + subctx;
}

class OnnxImporter2
{
public:
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;
    OnnxImporter2(Net2& net, const char* fileName);
    OnnxImporter2(Net2& net, const char* buffer, size_t bufsize);

protected:
    void init(Net2& net);
    bool parse(const char* fileName);
    bool parse(const char* buffer, size_t bufsize);
    void parseGraph(const OpenCVOnnx__GraphProto* proto,
                    Graph& graph, bool subgraph);
    void updateLayerParams(const string& ctx,
                           const OpenCVOnnx__AttributeProto* attr_proto,
                           LayerParams& params);
    typedef void (OnnxImporter2::*NodeParser)
        (const string& ctx, const OpenCVOnnx__NodeProto*,
         LayerParams&, vector<int>&, vector<int>&);

    NetImpl* netimpl;
    typedef std::unordered_map<string, NodeParser> DispatchMap;
    typedef std::unordered_map<string, DispatchMap> DomainDispatchMap;
    DomainDispatchMap alldispatch;
    string filename;
    string defaultOnnxDomain = "ai.onnx";
    string frameworkName;
    std::set<string> unsupportedOps;

    void parseIf(const string&, const OpenCVOnnx__NodeProto*,
                 LayerParams&, vector<int>&, vector<int>&,
                 const vector<string>& subgraphNames, vector<PGraph>& subgraphs);
    void parseLoop(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&,
                   const vector<string>& subgraphNames, vector<PGraph>& subgraphs);
    void parseScan(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&,
                   const vector<string>& subgraphNames, vector<PGraph>& subgraphs);

    void parseArg(const string&, const OpenCVOnnx__NodeProto*,
                  LayerParams&, vector<int>&, vector<int>&);
    void parseBatchNormalization(const string&, const OpenCVOnnx__NodeProto*,
                                 LayerParams&, vector<int>&, vector<int>&);
    void parseCast(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&);
    void parseClip(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&);
    void parseConcat(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseConstant(const string&, const OpenCVOnnx__NodeProto*,
                       LayerParams&, vector<int>&, vector<int>&);
    void parseConstantOfShape(const string&, const OpenCVOnnx__NodeProto*,
                              LayerParams&, vector<int>&, vector<int>&);
    void parseConv(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&);
    void parseConvTranspose(const string&, const OpenCVOnnx__NodeProto*,
                            LayerParams&, vector<int>&, vector<int>&);
    void parseCumSum(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseDepthToSpace(const string&, const OpenCVOnnx__NodeProto*,
                           LayerParams&, vector<int>&, vector<int>&);
    void parseDetectionOutput(const string&, const OpenCVOnnx__NodeProto*,
                              LayerParams&, vector<int>&, vector<int>&);
    void parseElemwiseBinary(const string&, const OpenCVOnnx__NodeProto*,
                             LayerParams&, vector<int>&, vector<int>&);
    void parseElemwiseNary(const string&, const OpenCVOnnx__NodeProto*,
                           LayerParams&, vector<int>&, vector<int>&);
    void parseElemwiseUnary(const string&, const OpenCVOnnx__NodeProto*,
                            LayerParams&, vector<int>&, vector<int>&);
    void parseExpand(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseFlatten(const string&, const OpenCVOnnx__NodeProto*,
                      LayerParams&, vector<int>&, vector<int>&);
    void parseGather(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseGemm(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&);
    void parseGlobalPool(const string&, const OpenCVOnnx__NodeProto*,
                         LayerParams&, vector<int>&, vector<int>&);
    void parseGRU(const string&, const OpenCVOnnx__NodeProto*,
                  LayerParams&, vector<int>&, vector<int>&);
    void parseImageScaler(const string&, const OpenCVOnnx__NodeProto*,
                          LayerParams&, vector<int>&, vector<int>&);
    void parseInstanceNormalization(const string&, const OpenCVOnnx__NodeProto*,
                                    LayerParams&, vector<int>&, vector<int>&);
    void parseLeakyRelu(const string&, const OpenCVOnnx__NodeProto*,
                        LayerParams&, vector<int>&, vector<int>&);
    void parseLRN(const string&, const OpenCVOnnx__NodeProto*,
                  LayerParams&, vector<int>&, vector<int>&);
    void parseLSTM(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&);
    void parseMatMul(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseMaxPool(const string&, const OpenCVOnnx__NodeProto*,
                      LayerParams&, vector<int>&, vector<int>&);
    void parseMaxUnpool(const string&, const OpenCVOnnx__NodeProto*,
                        LayerParams&, vector<int>&, vector<int>&);
    void parsePad(const string&, const OpenCVOnnx__NodeProto*,
                  LayerParams&, vector<int>&, vector<int>&);
    void parsePooling(const string&, const OpenCVOnnx__NodeProto*,
                      LayerParams&, vector<int>&, vector<int>&);
    void parsePRelu(const string&, const OpenCVOnnx__NodeProto*,
                    LayerParams&, vector<int>&, vector<int>&);
    void parseReduce(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseRelu(const string&, const OpenCVOnnx__NodeProto*,
                   LayerParams&, vector<int>&, vector<int>&);
    void parseResize(const string&, const OpenCVOnnx__NodeProto*,
                     LayerParams&, vector<int>&, vector<int>&);
    void parseReshape(const string&, const OpenCVOnnx__NodeProto*,
                      LayerParams&, vector<int>&, vector<int>&);
    void parseShape(const string&, const OpenCVOnnx__NodeProto*,
                    LayerParams&, vector<int>&, vector<int>&);
    void parseSlice(const string&, const OpenCVOnnx__NodeProto*,
                    LayerParams&, vector<int>&, vector<int>&);
    void parseSoftMax(const string&, const OpenCVOnnx__NodeProto*,
                      LayerParams&, vector<int>&, vector<int>&);
    void parseSplit(const string&, const OpenCVOnnx__NodeProto*,
                    LayerParams&, vector<int>&, vector<int>&);
    void parseSqueeze(const string&, const OpenCVOnnx__NodeProto*,
                      LayerParams&, vector<int>&, vector<int>&);
    void parseTranspose(const string&, const OpenCVOnnx__NodeProto*,
                        LayerParams&, vector<int>&, vector<int>&);
    void parseUnsqueeze(const string&, const OpenCVOnnx__NodeProto*,
                        LayerParams&, vector<int>&, vector<int>&);
    void parseUpsample(const string&, const OpenCVOnnx__NodeProto*,
                       LayerParams&, vector<int>&, vector<int>&);

    // Domain: com.microsoft
    // URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
    void parseDequantizeLinear(const string&, const OpenCVOnnx__NodeProto*,
                               LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearAveragePool(const string&, const OpenCVOnnx__NodeProto*,
                                 LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearConcat(const string&, const OpenCVOnnx__NodeProto*,
                            LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearConv(const string&, const OpenCVOnnx__NodeProto*,
                          LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearElemwiseBinary(const string&, const OpenCVOnnx__NodeProto*,
                                    LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearGlobalAveragePool(const string&, const OpenCVOnnx__NodeProto*,
                                       LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearLeakyRelu(const string&, const OpenCVOnnx__NodeProto*,
                               LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearMatMul(const string&, const OpenCVOnnx__NodeProto*,
                            LayerParams&, vector<int>&, vector<int>&);
    void parseQLinearSigmoid(const string&, const OpenCVOnnx__NodeProto*,
                             LayerParams&, vector<int>&, vector<int>&);
    void parseQuantizeLinear(const string&, const OpenCVOnnx__NodeProto*,
                             LayerParams&, vector<int>&, vector<int>&);
};

struct OnnxTensor
{
    string name;
    Tensor t;
};

static int onnxDatatypeToDepth(const string& ctx, int datatype)
{
    int typ =
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED ? -1 :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT8 ? CV_8U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT8 ? CV_8S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT16 ? CV_16U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT16 ? CV_16S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT32 ? CV_32S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT32 ? CV_32S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT64 ? CV_32S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT64 ? CV_32S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__BOOL ? CV_8U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT ? CV_32F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE ? CV_64F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 ? CV_16F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16 ? CV_32F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64 ? CV_32FC2 :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128 ? CV_64FC2 : -2;
    if (typ < -1)
        onnxParseError(ctx, format("unsupported data_type %d", datatype));
    return typ;
}

template<typename fromT, typename toT>
static void onnxParseArray(NetImpl* netimpl, const string& ctx,
                           fromT** arr, size_t nelems, vector<toT>& result);

template<typename fromT, typename toT> struct OnnxParseElem
{
    OnnxParseElem(NetImpl*, const string&) {}
    toT parse(const fromT* proto) const { return static_cast<toT>(*proto); }
};

template<> struct OnnxParseElem<OpenCVOnnx__OperatorSetIdProto, OnnxOpSet>
{
    OnnxParseElem(NetImpl*, const string&) {}
    OnnxOpSet parse(const OpenCVOnnx__OperatorSetIdProto* proto) {
        return std::make_pair(proto->version, proto->domain);
    }
};

template<> struct OnnxParseElem<OpenCVOnnx__TensorShapeProto__Dimension, TensorDim>
{
    OnnxParseElem(NetImpl*, const string& ctx_) : ctx(ctx_), idx(-1) {}
    TensorDim parse(const OpenCVOnnx__TensorShapeProto__Dimension* proto) {
        idx++;
        if (proto->value_case == OPENCV_ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM)
            return TensorDim(proto->dim_param);
        if (proto->value_case != OPENCV_ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
            onnxParseError(ctx, format("unknown type of dimension #%d", idx));
        return TensorDim(proto->dim_value);
    }
    string ctx;
    int idx;
};

template<> struct OnnxParseElem<OpenCVOnnx__ValueInfoProto, ArgInfo>
{
    OnnxParseElem(NetImpl* netimpl_, const string& ctx_) : netimpl(netimpl_), ctx(ctx_) {}
    ArgInfo parse(const OpenCVOnnx__ValueInfoProto* proto) {
        ArgInfo arginfo;
        arginfo.name = proto->name;
        string subctx = onnxConcatCtx(ctx,
                format("parsing value info '%s'", proto->name));
        if (proto->type->value_case == OPENCV_ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE) {
            arginfo.typ = onnxDatatypeToDepth(subctx, proto->type->tensor_type->elem_type);
            const OpenCVOnnx__TensorShapeProto* shape = proto->type->tensor_type->shape;
            if (shape)
                onnxParseArray(netimpl, subctx, shape->dim, shape->n_dim, arginfo.shape);
        } else if (proto->type->value_case == OPENCV_ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE) {
            onnxParseError(subctx, "sequences are not supported");
        } else {
            onnxParseError(subctx, format("unsupported value info tag %d",
                           (int)proto->type->value_case));
        }
        return arginfo;
    }
    NetImpl* netimpl;
    string ctx;
};

static int64_t unpack_int64(uint8_t* p)
{
    uint64_t x = p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) | ((uint64_t)p[3]<<24) |
        ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) | ((uint64_t)p[6]<<48) | ((uint64_t)p[7]<<56);
    return (int64_t)(x);
}

static Tensor onnxParseTensor(NetImpl* netimpl, const string& ctx,
                              const OpenCVOnnx__TensorProto* tensor_proto,
                              Mat* m)
{
    Tensor tensor;
    int i, n_dims = (int)tensor_proto->n_dims;
    size_t total = 1, elemsize;
    TensorShape shape;
    uchar* data;
    shape.ndims = n_dims;

    for (i = 0; i < n_dims; i++) {
        int64_t size_i = (int64_t)tensor_proto->dims[i];
        shape.shape[i] = size_i;
        total *= size_i;
    }
    shape.layout = n_dims <= 2 ? DNN_LAYOUT_ND : DNN_LAYOUT_NCHW;
    int typ = onnxDatatypeToDepth(ctx, tensor_proto->data_type);
    if (typ < 0)
        onnxParseError(ctx, format("type of tensor '%s' is invalid (=%d)",
                        tensor_proto->name, tensor_proto->data_type));
    elemsize = CV_ELEM_SIZE(typ);
    if (m) {
        int mshape[CV_MAX_DIM];
        OnnxAssert(ctx, n_dims <= CV_MAX_DIM);
        for (i = 0; i < n_dims; i++) mshape[i] = (int)shape.shape[i];
        for (; i < 2; i++) mshape[i] = 1;
        m->create(std::max(n_dims, 2), mshape, typ);
        tensor.set(shape, typ, m->data, false);
        data = m->data;
    } else {
        tensor.fit(shape, typ);
        data = (uchar*)tensor.data();
    }

    if (elemsize == 1 && tensor_proto->raw_data.len == total) {
        memcpy(data, tensor_proto->raw_data.data, total*elemsize);
    } else if (elemsize == 1 && tensor_proto->n_int32_data == total) {
        for(int j = 0; j < total; j++)
            data[j] = (uchar)(tensor_proto->int32_data[j]);
    } else if (elemsize == 4 && tensor_proto->n_float_data == total) {
        memcpy(data, tensor_proto->float_data, total*elemsize);
    } else if (elemsize == 4 &&
               tensor_proto->data_type == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 &&
               tensor_proto->n_int32_data == total) {
        float* dst = (float*)data;
        for(size_t j = 0; j < total; j++) {
            ushort bits = (ushort)(tensor_proto->int32_data[j]);
            dst[j] = (float)float16_t::fromBits(bits);
        }
    } else if (elemsize == 4 && tensor_proto->raw_data.len == total*4) {
        uint32_t* dst = (uint32_t*)data;
        for(size_t j = 0; j < total; j++) {
            uint8_t* p = tensor_proto->raw_data.data + j*4;
            dst[j] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }
    } else if (elemsize == 4 && tensor_proto->n_int64_data == total) {
        int32_t* dst = (int32_t*)data;
        for(size_t j = 0; j < total; j++) {
            int64_t v = tensor_proto->int64_data[j];
            dst[j] = (int32_t)(v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : (int)v);
        }
    } else if (elemsize == 4 && tensor_proto->raw_data.len == total*8) {
        int32_t* dst = (int32_t*)data;
        for(size_t j = 0; j < total; j++) {
            uint8_t* p = tensor_proto->raw_data.data + j*8;
            int64_t v = unpack_int64(p);
            dst[j] = (int32_t)(v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : (int)v);
        }
    } else {
        onnxParseError(ctx, format("unsupported tensor data_type %d", (int)tensor_proto->data_type));
    }
    return tensor;
}

template<> struct OnnxParseElem<OpenCVOnnx__TensorProto, OnnxTensor>
{
    OnnxParseElem(NetImpl* netimpl_, const string& ctx_) : netimpl(netimpl_), ctx(ctx_) {}
    OnnxTensor parse(const OpenCVOnnx__TensorProto* proto) {
        OnnxTensor tensor;
        tensor.name = proto->name;
        tensor.t = onnxParseTensor(netimpl, ctx, proto, 0);
        return tensor;
    }
    NetImpl* netimpl;
    string ctx;
};

template<typename fromT, typename toT>
static void onnxParseArray(NetImpl* netimpl, const string& ctx,
                           fromT** arr, size_t nelems, vector<toT>& result)
{
    OnnxParseElem<fromT, toT> elemParser(netimpl, ctx);
    result.reserve(nelems);
    for (size_t i = 0; i < nelems; i++)
        result.push_back(elemParser.parse(arr[i]));
}

void OnnxImporter2::updateLayerParams(const string& ctx,
                                      const OpenCVOnnx__AttributeProto* attr_proto,
                                      LayerParams& params)
{
    string name = attr_proto->name;
    OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
    if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT)
        params.set(name, attr_proto->f);
    else if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT)
        params.set(name, attr_proto->i);
    else if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING)
        params.set(name, String((const char*)attr_proto->s.data));
    else if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS)
        params.set(name, DictValue::arrayInt(attr_proto->ints, (int)attr_proto->n_ints));
    else if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS)
        params.set(name, DictValue::arrayReal(attr_proto->floats, (int)attr_proto->n_floats));
    else if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS) {
        std::vector<String> strings(attr_proto->n_strings);
        for (size_t i = 0; i < attr_proto->n_strings; i++)
            strings[i] = String((const char*)attr_proto->strings[i].data);
        params.set(name, DictValue::arrayString(strings.begin(), (int)attr_proto->n_strings));
    } else if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR) {
        Mat m;
        Tensor t = onnxParseTensor(netimpl, ctx, attr_proto->t, &m);
        params.set(name, DictValue(m, (int)t.shape.ndims));
    } else {
        onnxParseError(ctx, format("unrecognized/unsupported attribute type %d", (int)tag));
    }
}

void OnnxImporter2::parseGraph(const OpenCVOnnx__GraphProto* proto,
                               Graph& graph, bool subgraph)
{
    graph.name = proto->name;
    string ctx = subgraph ? string() : "parsing subgraph '" + graph.name + "'";
    vector<ArgInfo> inputs, outputs, values;
    vector<OnnxTensor> initializers;
    vector<int> node_inputs, node_outputs;
    vector<PGraph> subgraphs;
    vector<string> subgraph_names;

    onnxParseArray(netimpl, ctx, proto->input, proto->n_input, inputs);
    onnxParseArray(netimpl, ctx, proto->output, proto->n_output, outputs);
    onnxParseArray(netimpl, ctx, proto->value_info, proto->n_value_info, values);
    onnxParseArray(netimpl, ctx, proto->initializer, proto->n_initializer, initializers);

    for (const OnnxTensor& t: initializers)
        netimpl->addConstTensor(t.name, t.t);

    for (int k = 0; k < 3; k++) {
        int argkind = k == 0 ? DNN_ARG_INPUT : k == 1 ? DNN_ARG_OUTPUT : DNN_ARG_TEMP;
        if (subgraph) argkind = DNN_ARG_TEMP;
        const vector<ArgInfo>* graph_args = k == 0 ? &inputs : k == 1 ? &outputs : &values;
        for (const ArgInfo& arginfo: *graph_args) {
            int argidx = netimpl->addArg(argkind, arginfo);
            if (k == 0)
                graph.inputs.push_back(argidx);
            else if (k == 1)
                graph.outputs.push_back(argidx);
        }
    }

    for (size_t i = 0; i < proto->n_node; i++) {
        OpenCVOnnx__NodeProto* node_proto = proto->node[i];
        string node_ctx = onnxConcatCtx(ctx, format("when parsing '%s' (%s)",
                                        node_proto->name, node_proto->op_type));
        node_inputs.clear();
        node_outputs.clear();
        subgraph_names.clear();
        subgraphs.clear();
        LayerParams params;
        params.name = node_proto->name;
        params.type = node_proto->op_type;
        PLayer layer;

        for (size_t j = 0; j < node_proto->n_input; j++) {
            int argidx = netimpl->findArg(node_proto->input[j]);
            if (argidx < 0) {
                onnxParseError(node_ctx, format("cannot find input '%s'", node_proto->input[j]));
            }
            node_inputs.push_back(argidx);
        }
        for (size_t j = 0; j < node_proto->n_output; j++) {
            int argidx = netimpl->findOutputArg(node_proto->output[j]);
            node_outputs.push_back(argidx);
        }
        for (size_t j = 0; j < node_proto->n_attribute; j++) {
            OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
            if (attr_proto->type == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH) {
                PGraph pg = makePtr<Graph>();
                parseGraph(attr_proto->g, *pg.get(), true);
                subgraph_names.push_back(attr_proto->name);
                subgraphs.push_back(pg);
            } else {
                updateLayerParams(node_ctx, attr_proto, params);
            }
        }

        /*if (strcmp(node_proto->op_type, "If") == 0 && node_proto->domain == defaultOnnxDomain) {
            parseIf(node_ctx, node_proto, params, node_inputs, node_outputs, subgraph_names, subgraphs);
        } else if (strcmp(node_proto->op_type, "Loop") == 0 && node_proto->domain == defaultOnnxDomain) {
            parseLoop(node_ctx, node_proto, params, node_inputs, node_outputs, subgraph_names, subgraphs);
        } else if (strcmp(node_proto->op_type, "Scan") == 0 && node_proto->domain == defaultOnnxDomain) {
            parseScan(node_ctx, node_proto, params, node_inputs, node_outputs, subgraph_names, subgraphs);
        } else*/ if (!subgraphs.empty()) {
            onnxParseError(node_ctx, "no subgraphs is expected in this node");
        } else {
            auto domain_it = alldispatch.find(node_proto->domain);
            if (domain_it != alldispatch.end()) {
                auto& dispatch = domain_it->second;
                auto it = dispatch.find(node_proto->op_type);
                if (it != dispatch.end()) {
                    (this->*(it->second))(node_ctx, node_proto, params, node_inputs, node_outputs);
                    if (strcmp(node_proto->op_type, "Constant") == 0)
                        continue;
                }
            }
        }
        layer = LayerFactory::createLayerInstance(params.type, params);
        if (layer.empty()) {
            string name = (node_proto->domain == string("") ||
                           node_proto->domain == defaultOnnxDomain ? string() :
                           string(node_proto->domain) + ".") + string(node_proto->op_type);
            unsupportedOps.insert(name);
        }
        Node node(layer, node_inputs, node_outputs, subgraphs);
        graph.prog.push_back(node);
        //printf("pushed %snode '%s'\n", (layer.empty() ? "dummy " : ""),
        //  (layer.empty() ? params.name.c_str() : layer->name.c_str()));
    }
}

OnnxImporter2::OnnxImporter2(Net2& net, const char *fileName)
{
    init(net);
    parse(fileName);
}

OnnxImporter2::OnnxImporter2(Net2& net, const char* buffer, size_t bufsize)
{
    init(net);
    parse(buffer, bufsize);
}

bool OnnxImporter2::parse(const char* filename_)
{
    size_t fsize, freaded;
    vector<char> buf;

    filename = filename_;
    FILE* f = fopen(filename_, "rb");

    if (!f) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: cannot open file " << filename_);
        return false;
    }
    fseek(f, 0, SEEK_END);
    fsize = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(fsize+256);
    freaded = fread(&buf[0], 1, fsize, f);
    if (freaded != fsize) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: cannot read file " << filename_);
        return false;
    }
    return parse(&buf[0], freaded);
}

bool OnnxImporter2::parse(const char* buffer, size_t datasize)
{
    bool ok = true;
    if (datasize == 0) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: the file/buffer is empty");
        return false;
    }

    {
        const uint8_t* data = (const uint8_t*)buffer;
        OpenCVOnnx__ModelProto* model = opencv_onnx__model_proto__unpack(0, datasize, data);
        if(model == 0) {
            CV_LOG_DEBUG(NULL, "DNN/Onnx: could not parse the model" << (filename.empty() ? " " + filename : ""));
            return false;
        }

        netimpl->modelFormat = DNN_MODEL_ONNX;
        netimpl->defaultLayout = DNN_LAYOUT_NCHW;
        netimpl->onnxInfo.IRVersion = model->ir_version;
        netimpl->onnxInfo.producer = model->producer_name;
        netimpl->onnxInfo.domain = model->domain;
        netimpl->onnxInfo.docString = model->doc_string;
        onnxParseArray(netimpl, string(), model->opset_import, model->n_opset_import, netimpl->onnxInfo.opsets);
        if (model->producer_name)
            frameworkName = model->producer_name;
        try {
            parseGraph(model->graph, netimpl->graph, false);
        } catch (const std::exception& exn) {
            string filectx = filename.empty() ? "" : " (" + filename + ") ";
            CV_LOG_WARNING(NULL, "DNN/Onnx: parse error" + filectx + exn.what());
            ok = false;
        }
        opencv_onnx__model_proto__free_unpacked(model, 0);
    }

    if (!unsupportedOps.empty()) {
        std::stringstream msg;
        bool plural = unsupportedOps.size() > 1, first = true;
        msg << string("DNN/Onnx: the operation") + (plural ? "s " : " ");
        for (const string& opname: unsupportedOps) {
            msg << (first ? "'" : ", '") + opname + "'";
            first = false;
        }
        msg << (plural ? " are not supported" : " is not supported");
        CV_LOG_ERROR(NULL, msg.str());
        ok = false;
    }
    if (!ok) netimpl->clear();
    return ok;
}

void OnnxImporter2::init(Net2& net)
{
    netimpl = net.impl();
    DispatchMap dispatch, msdispatch;

    dispatch["Equal"] = dispatch["Greater"] = dispatch["Less"] = dispatch["Pow"] = dispatch["Add"] =
            dispatch["Sub"] = dispatch["Mul"] = dispatch["Div"] = &OnnxImporter2::parseElemwiseBinary;
    //dispatch["ArgMax"] = dispatch["ArgMin"] = &OnnxImporter2::parseArg;
    dispatch["AveragePool"] = &OnnxImporter2::parsePooling;
    dispatch["BatchNormalization"] = &OnnxImporter2::parseBatchNormalization;
    dispatch["Cast"] = &OnnxImporter2::parseCast;
    dispatch["Clip"] = &OnnxImporter2::parseClip;
    dispatch["Concat"] = &OnnxImporter2::parseConcat;
    dispatch["Constant"] = &OnnxImporter2::parseConstant;
    dispatch["ConstantFill"] = dispatch["ConstantOfShape"] = &OnnxImporter2::parseConstantOfShape;
    dispatch["Conv"] = &OnnxImporter2::parseConv;
    dispatch["ConvTranspose"] = &OnnxImporter2::parseConvTranspose;
    dispatch["CumSum"] = &OnnxImporter2::parseCumSum;
    //dispatch["DetectionOutput"] = &OnnxImporter2::parseDetectionOutput;
    dispatch["Expand"] = &OnnxImporter2::parseExpand;
    dispatch["Flatten"] = &OnnxImporter2::parseFlatten;
    dispatch["Gather"] = &OnnxImporter2::parseGather;
    dispatch["Gemm"] = &OnnxImporter2::parseGemm;
    dispatch["GlobalAveragePool"] = dispatch["GlobalMaxPool"] = &OnnxImporter2::parseGlobalPool;
    //dispatch["GRU"] = &OnnxImporter2::parseGRU;
    //dispatch["ImageScaler"] = &OnnxImporter2::parseImageScaler;
    //dispatch["InstanceNormalization"] = &OnnxImporter2::parseInstanceNormalization;
    dispatch["LeakyRelu"] = &OnnxImporter2::parseLeakyRelu;
    dispatch["LRN"] = &OnnxImporter2::parseLRN;
    //dispatch["LSTM"] = &OnnxImporter2::parseLSTM;
    dispatch["MatMul"] = &OnnxImporter2::parseMatMul;
    dispatch["Max"] = dispatch["Min"] = dispatch["Sum"] = &OnnxImporter2::parseElemwiseNary;
    dispatch["MaxPool"] = &OnnxImporter2::parsePooling;
    //dispatch["MaxUnpool"] = &OnnxImporter2::parseMaxUnpool;
    //dispatch["Pad"] = &OnnxImporter2::parsePad;
    dispatch["PRelu"] = &OnnxImporter2::parsePRelu;
    dispatch["ReduceMax"] = dispatch["ReduceMin"] = dispatch["ReduceMean"] = dispatch["ReduceSum"] = dispatch["ReduceMax"] =
    dispatch["ReduceMin"] = dispatch["ReduceSumSquare"] = dispatch["ReduceProd"] = dispatch["ReduceL1"] =
    dispatch["ReduceL2"] = dispatch["ReduceLogSum"] = dispatch["ReduceLogSumExp"] = &OnnxImporter2::parseReduce;
    dispatch["Relu"] = &OnnxImporter2::parseRelu;
    dispatch["Reshape"] = &OnnxImporter2::parseReshape;
    dispatch["Resize"] = &OnnxImporter2::parseResize;
    dispatch["Shape"] = &OnnxImporter2::parseShape;
    dispatch["Slice"] = &OnnxImporter2::parseSlice;
    dispatch["SoftMax"] = dispatch["LogSoftmax"] = &OnnxImporter2::parseSoftMax;
    dispatch["SpaceToDepth"] = dispatch["DepthToSpace"] = &OnnxImporter2::parseDepthToSpace;
    dispatch["Split"] = &OnnxImporter2::parseSplit;
    dispatch["Squeeze"] = &OnnxImporter2::parseSqueeze;
    dispatch["Transpose"] = &OnnxImporter2::parseTranspose;
    //dispatch["Upsample"] = &OnnxImporter2::parseUpsample;
    dispatch["Unsqueeze"] = &OnnxImporter2::parseUnsqueeze;

    vector<string> simpleLayers{
        "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh",
        "Ceil", "Celu", "Cos", "Cosh", "Dropout", "Elu", "Erf",
        "Exp", "Floor", "HardSigmoid", "HardSwish", "Identity",
        "Log", "Neg", "Round", "Reciprocal", "Selu",
        "Sign", "Sigmoid", "Sin", "Sinh", "Softmax", "Softplus",
        "Softsign", "Shrink", "Sqrt", "Tan", "Tanh", "ThresholdedRelu"};
    for (const auto& name : simpleLayers)
        dispatch[name] = &OnnxImporter2::parseElemwiseUnary;

    // ai.onnx: opset 10+
    dispatch["DequantizeLinear"] = &OnnxImporter2::parseDequantizeLinear;
    dispatch["QLinearConv"] = &OnnxImporter2::parseQLinearConv;
    dispatch["QLinearMatMul"] = &OnnxImporter2::parseQLinearMatMul;
    dispatch["QuantizeLinear"] = &OnnxImporter2::parseQuantizeLinear;
    alldispatch[defaultOnnxDomain] = dispatch;
    alldispatch[""] = dispatch;

    msdispatch["QLinearAdd"] = &OnnxImporter2::parseQLinearElemwiseBinary;
    msdispatch["QLinearAveragePool"] = &OnnxImporter2::parsePooling;
    msdispatch["QLinearGlobalAveragePool"] = &OnnxImporter2::parseQLinearGlobalAveragePool;
    msdispatch["QLinearConcat"] = &OnnxImporter2::parseQLinearConcat;
    msdispatch["QLinearLeakyRelu"] = &OnnxImporter2::parseQLinearLeakyRelu;
    msdispatch["QLinearSigmoid"] = &OnnxImporter2::parseQLinearSigmoid;
    alldispatch["com.microsoft"] = msdispatch;
}

static inline void replaceLayerParam(LayerParams& layerParams, const String& oldKey, const String& newKey)
{
    if (layerParams.has(oldKey)) {
        layerParams.set(newKey, layerParams.get(oldKey));
        layerParams.erase(oldKey);
    }
}

/*LayerParams OnnxImporter2::getLayerParams(const opencv_onnx::NodeProto& node_proto)
{
    LayerParams lp;
    for(int i = 0; i < node_proto.attribute_size(); i++)
    {
        opencv_onnx::AttributeProto attribute_proto = node_proto.attribute(i);
        string attribute_name = attribute_proto.name();

        try
        {
            if(attribute_name == "kernel_shape")
            {
                OnnxAssert(ctx, attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("kernel_size", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "strides")
            {
                OnnxAssert(ctx, attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("stride", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "pads")
            {
                if (node_proto.op_type() == "Pad")
                {
                    // Padding layer.
                    // Paddings are in order begin0, begin1, .. beginN, end0, end1, ..., endN.
                    // We need to shuffle it to begin0, end0, begin1, end1, ...
                    OnnxAssert(ctx, attribute_proto.ints_size() % 2 == 0);
                    const int dims = attribute_proto.ints_size() / 2;
                    vector<int32_t> paddings;
                    paddings.reserve(attribute_proto.ints_size());
                    for (int i = 0; i < dims; ++i)
                    {
                        paddings.push_back(attribute_proto.ints(i));
                        paddings.push_back(attribute_proto.ints(dims + i));
                    }
                    lp.set("paddings", DictValue::arrayInt(&paddings[0], paddings.size()));
                }
                else
                {
                    // Convolution or pooling.
                    OnnxAssert(ctx, attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 4 || attribute_proto.ints_size() == 6);
                    lp.set("pad", parse(attribute_proto.ints()));
                }
            }
            else if(attribute_name == "auto_pad")
            {
                if (attribute_proto.s() == "SAME_UPPER" || attribute_proto.s() == "SAME_LOWER") {
                    lp.set("pad_mode",  "SAME");
                }
                else if (attribute_proto.s() == "VALID") {
                    lp.set("pad_mode", "VALID");
                }
            }
            else if(attribute_name == "dilations")
            {
                OnnxAssert(ctx, attribute_proto.ints_size() == 1 || attribute_proto.ints_size() == 2 || attribute_proto.ints_size() == 3);
                lp.set("dilation", parse(attribute_proto.ints()));
            }
            else if(attribute_name == "activations" && node_proto.op_type() == "LSTM")
            {
                lp.set(attribute_name, parseStr(attribute_proto.strings()));
            }
            else if (attribute_proto.has_i())
            {
                ::google::protobuf::int64 src = attribute_proto.i();
                if (src < std::numeric_limits<int32_t>::min() || src > std::numeric_limits<int32_t>::max())
                    CV_Error(Error::StsOutOfRange, "Input is out of OpenCV 32S range");
                else
                    lp.set(attribute_name, saturate_cast<int32_t>(src));
            }
            else if (attribute_proto.has_f())
            {
                lp.set(attribute_name, attribute_proto.f());
            }
            else if (attribute_proto.has_s())
            {
                lp.set(attribute_name, attribute_proto.s());
            }
            else if (attribute_proto.floats_size() > 0)
            {
                lp.set(attribute_name, DictValue::arrayReal(
                    attribute_proto.floats().data(), attribute_proto.floats_size()));
            }
            else if (attribute_proto.ints_size() > 0)
            {
                lp.set(attribute_name, parse(attribute_proto.ints()));
            }
            else if (attribute_proto.has_t())
            {
                opencv_onnx::TensorProto tensor = attribute_proto.t();
                Mat blob = getMatFromTensor(tensor);
                lp.blobs.push_back(blob);
                lp.set("original_dims_of_mat", tensor.dims_size());
            }
            else if (attribute_proto.has_g())
            {
                CV_Error(Error::StsNotImplemented, cv::format("DNN/Onnx/Attribute[%s]: 'Graph' is not supported", attribute_name.c_str()));
            }
            else if (attribute_proto.graphs_size() > 0)
            {
                CV_Error(Error::StsNotImplemented,
                        cv::format("DNN/Onnx/Attribute[%s]: 'Graphs' (%d) in attributes is not supported",
                                attribute_name.c_str(), attribute_proto.graphs_size())
                );
            }
            else if (attribute_proto.strings_size() > 0)
            {
                string msg = cv::format("DNN/Onnx/Attribute[%s]: 'Strings' (%d) are not supported",
                        attribute_name.c_str(), attribute_proto.strings_size());
                CV_LOG_ERROR(NULL, msg);
                for (int i = 0; i < attribute_proto.strings_size(); i++)
                {
                    CV_LOG_ERROR(NULL, "    Attribute[" << attribute_name << "].string(" << i << ") = '" << attribute_proto.strings(i) << "'");
                }
                CV_Error(Error::StsNotImplemented, msg);
            }
            else if (attribute_proto.tensors_size() > 0)
            {
                CV_Error(Error::StsNotImplemented,
                        cv::format("DNN/Onnx/Attribute[%s]: 'Tensors' (%d) in attributes are not supported",
                                attribute_name.c_str(), attribute_proto.tensors_size())
                );
            }
            else
            {
                CV_Error(Error::StsNotImplemented, cv::format("DNN/Onnx/Attribute[%s]: unsupported attribute format", attribute_name.c_str()));
            }
        }
        catch (const cv::Exception& e)
        {
            CV_UNUSED(e);
            if (DNN_DIAGNOSTICS_RUN)
            {
                CV_LOG_ERROR(NULL, "DNN/Onnx: Potential problem with processing attributes for node " << node_proto.name() << " Attribute " << attribute_name.c_str()
                );
                continue;
            }
            throw;
        }
    }
    return lp;
}*/

void OnnxImporter2::parseBatchNormalization(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                            LayerParams& layerParams,
                                            vector<int>& inputs, vector<int>& outputs)
{
    size_t noutputs = outputs.size();
    OnnxAssert(ctx, noutputs == 1 || noutputs == 3);
    if (inputs.size() != 5)
        onnxParseError(ctx, "Expected input, scale, bias, mean and var");

    layerParams.type = "BatchNorm";
    replaceLayerParam(layerParams, "epsilon", "eps");
    replaceLayerParam(layerParams, "spatial", "use_global_stats");
}

void OnnxImporter2::parseCast(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                              LayerParams& layerParams,
                              vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    int datatype = layerParams.get<int>("to");
    int typ = onnxDatatypeToDepth(ctx, datatype);
    layerParams.set("to", typ);
}

void OnnxImporter2::parseClip(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                              LayerParams& layerParams,
                              vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "ReLU6";
}

void OnnxImporter2::parseConcat(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 1);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("axis"));
}


void OnnxImporter2::parseConstant(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                  LayerParams& layerParams,
                                  vector<int>& inputs, vector<int>& outputs)
{
    Tensor t;
    TensorShape scalar_shape;
    OnnxAssert(ctx, inputs.size() == 0);
    OnnxAssert(ctx, outputs.size() == 1);
    if (layerParams.has("value")) {
        DictValue v = layerParams.get("value");
        int ndims = v.getDims();
        Mat m = v.getMat();
        t = Tensor(m, ndims, true);
    } else if (layerParams.has("value_int")) {
        int v = saturate_cast<int>(layerParams.get<int64>("value_int"));
        t = Tensor(scalar_shape, CV_32S, &v, true);
    } else if (layerParams.has("value_float")) {
        float v = (float)layerParams.get<double>("value_float");
        t = Tensor(scalar_shape, CV_32F, &v, true);
    } else if (layerParams.has("value_ints")) {
        vector<int> v = layerParams.get<vector<int> >("value_ints");
        TensorShape shape;
        shape.ndims = 1;
        shape.shape[0] = (int64_t)v.size();
        t = v.empty() ? Tensor() : Tensor(shape, CV_32S, &v[0], true);
    } else if (layerParams.has("value_floats")) {
        vector<float> v = layerParams.get<vector<float> >("value_floats");
        TensorShape shape;
        shape.ndims = 1;
        shape.shape[0] = (int64_t)v.size();
        t = v.empty() ? Tensor() : Tensor(shape, CV_32F, &v[0], true);
    } else {
        onnxParseError(ctx, "invalid/unsupported constant type");
    }
    netimpl->addConstTensor("", t, outputs[0]);
}

void OnnxImporter2::parseConstantOfShape(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                         LayerParams& layerParams,
                                         vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
}

void OnnxImporter2::parseConv(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                              LayerParams& layerParams,
                              vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    if (layerParams.has("kernel_shape"))
        replaceLayerParam(layerParams, "kernel_shape", "kernel_size");
    layerParams.type = "Convolution";
}

void OnnxImporter2::parseConvTranspose(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                       LayerParams& layerParams,
                                       vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "Deconvolution";
}

void OnnxImporter2::parseCumSum(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "CumSum";
}

void OnnxImporter2::parseDepthToSpace(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                      LayerParams& layerParams,
                                      vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("blocksize"));
}

/*void OnnxImporter2::parseDetectionOutput(const string& ctx,
                                         const OpenCVOnnx__NodeProto* node_proto,
                                         LayerParams& layerParams,
                                         vector<int>& inputs, vector<int>& outputs)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    CV_CheckEQ(inputs.size(), 3, "");
    if (constBlobs.find(node_proto.input(2)) != constBlobs.end())
    {
        Mat priors = getBlob(node_proto, 2);

        LayerParams constParams;
        constParams.name = layerParams.name + "/priors";
        constParams.type = "Const";
        constParams.blobs.push_back(priors);

        opencv_onnx::NodeProto priorsProto;
        priorsProto.add_output(constParams.name);
        addLayer(constParams, priorsProto);

        node_proto.set_input(2, constParams.name);
    }
    addLayer(layerParams, node_proto);
}*/

// "Equal" "Greater" "Less" "Pow" "Add" "Sub" "Mul" "Div" ...
void OnnxImporter2::parseElemwiseBinary(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                        LayerParams& layerParams,
                                        vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    layerParams.set("op", toUpperCase(layerParams.type));
    layerParams.type = "NaryEltwise";
}

// "Sum" "Min" "Max"
void OnnxImporter2::parseElemwiseNary(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                      LayerParams& layerParams,
                                      vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 2 && outputs.size() == 1);
    layerParams.set("op", toUpperCase(layerParams.type));
    layerParams.type = "NaryEltwise";
}

void OnnxImporter2::parseElemwiseUnary(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                       LayerParams& layerParams,
                                       vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.set("op", layerParams.type);
    layerParams.type = "Elemwise";
}

void OnnxImporter2::parseExpand(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
}

void OnnxImporter2::parseFlatten(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                 LayerParams& layerParams,
                                 vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
}

void OnnxImporter2::parseGather(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2);
    OnnxAssert(ctx, outputs.size() == 1);
}

// A * B + C = Y, we require that the dimension of A is [m, k], and the dimension of B is [n, k].
// And the dim of output Y is [m, n]
void OnnxImporter2::parseGemm(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                              LayerParams& layerParams,
                              vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    layerParams.type = "InnerProduct";
}

void OnnxImporter2::parseGlobalPool(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                    LayerParams& layerParams,
                                    vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.type = "Pooling";
    string pool;
    if (strcmp(node_proto->op_type, "GlobalMaxPool") == 0)
        pool = "MAX";
    else if (strcmp(node_proto->op_type, "GlobalAveragePool") == 0)
        pool = "AVE";
    else
        onnxParseError(ctx, format("Unsupported pooling operation '%s'", node_proto->op_type));

    layerParams.set("global_pooling", true);
    layerParams.set("pool", pool);
}


/*void OnnxImporter2::parseGRU(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                             LayerParams& layerParams,
                             vector<int>& inputs, vector<int>& outputs)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    const string output_name = node_proto.output(0);
    LayerParams gruParams = layerParams;
    gruParams.name += "/gru";

    // https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#
    OnnxAssert(ctx, inputs.size() == 6);
    Mat Wx = getBlob(node_proto, 1);
    Mat Wh = getBlob(node_proto, 2);
    Mat b = getBlob(node_proto, 3);
    Mat h0 = getBlob(node_proto, 5);

    Wx = Wx.reshape(1, Wx.size[0] * Wx.size[1]);
    Wh = Wh.reshape(1, Wh.size[0] * Wh.size[1]);
    h0 = h0.reshape(1, h0.size[0] * h0.size[1]);
    b = b.reshape(1, b.size[0]);

    gruParams.blobs.resize(4);
    gruParams.blobs[0] = Wh;
    gruParams.blobs[1] = Wx;
    gruParams.blobs[2] = b;
    gruParams.blobs[3] = h0;
    gruParams.set("bidirectional", gruParams.get<string>("direction", "") == "bidirectional");

    node_proto.set_output(0, gruParams.name);  // set different name so output shapes will be registered on that name
    addLayer(gruParams, node_proto);

    MatShape gruShape = outShapes[node_proto.output(0)];

    // Add fake 1 as it is done in Onnx
    gruShape.insert(gruShape.begin() + 1, 1);

    layerParams.type = "Reshape";
    layerParams.set("dim", DictValue::arrayInt(&gruShape[0], gruShape.size()));
    node_proto.set_input(0, gruParams.name);  // redirect input to GRU
    node_proto.set_output(0, output_name);  // keep origin GRU's name
    addLayer(layerParams, node_proto);
}

void OnnxImporter2::parseImageScaler(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                     LayerParams& layerParams,
                                     vector<int>& inputs, vector<int>& outputs)
{
    const float scale = layerParams.has("scale") ? layerParams.get<float>("scale") : 1.0f;
    layerParams.erase("scale");

    if (layerParams.has("bias"))
    {
        layerParams.type = "Scale";
        layerParams.blobs.push_back(
                Mat(Size(1,  layerParams.get("bias").size()), CV_32FC1, scale));

        layerParams.set("bias_term", true);
        Mat bias(1, layerParams.get("bias").size(), CV_32FC1);
        for (int j = 0; j < bias.total(); j++) {
            bias.at<float>(0, j) = layerParams.get("bias").getRealValue(j);
        }
        layerParams.blobs.push_back(bias);
        layerParams.erase("bias");
    }
    else {
        layerParams.set("scale", scale);
        layerParams.type = "Power";
    }
    addLayer(layerParams, node_proto);
}


void OnnxImporter2::parseInstanceNormalization(const string& ctx,
                                               const OpenCVOnnx__NodeProto* node_proto,
                                               LayerParams& layerParams,
                                               vector<int>& inputs, vector<int>& outputs)
{
    opencv_onnx::NodeProto node_proto = node_proto_;
    if (inputs.size() != 3)
        CV_Error(Error::StsNotImplemented,
                 "Expected input, scale, bias");

    layerParams.blobs.resize(4);
    layerParams.blobs[2] = getBlob(node_proto, 1);  // weightData
    layerParams.blobs[3] = getBlob(node_proto, 2);  // biasData
    layerParams.set("has_bias", true);
    layerParams.set("has_weight", true);

    // Get number of channels in input
    int size = layerParams.blobs[2].total();
    layerParams.blobs[0] = Mat::zeros(size, 1, CV_32F); // mean
    layerParams.blobs[1] = Mat::ones(size, 1, CV_32F); // std

    LayerParams mvnParams;
    mvnParams.name = layerParams.name + "/MVN";
    mvnParams.type = "MVN";
    mvnParams.set("eps", layerParams.get<float>("epsilon"));
    layerParams.erase("epsilon");

    //Create MVN layer
    int id = dstNet.addLayer(mvnParams.name, mvnParams.type, mvnParams);
    //Connect to input
    IterLayerId_t layerId = layer_id.find(node_proto.input(0));
    OnnxAssert(ctx, layerId != layer_id.end());
    dstNet.connect(layerId->second.layerId, layerId->second.outputId, id, 0);
    //Add shape
    layer_id.insert(std::make_pair(mvnParams.name, LayerInfo(id, 0)));
    outShapes[mvnParams.name] = outShapes[node_proto.input(0)];

    //Replace Batch Norm's input to MVN
    node_proto.set_input(0, mvnParams.name);
    layerParams.type = "BatchNorm";
    addLayer(layerParams, node_proto);
}*/

void OnnxImporter2::parseLeakyRelu(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                   LayerParams& layerParams,
                                   vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 2);
    layerParams.type = "ReLU";
    layerParams.set("negative_slope", layerParams.get<float>("alpha", 0.01));
}

void OnnxImporter2::parseLRN(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                             LayerParams& layerParams,
                             vector<int>& inputs, vector<int>& outputs)
{
    replaceLayerParam(layerParams, "size", "local_size");
}

/*void OnnxImporter2::parseLSTM(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                              LayerParams& layerParams,
                              vector<int>& inputs, vector<int>& outputs)
{
    opencv_onnx::NodeProto lstm_proto = node_proto_;
    layerParams.name += "/lstm";

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM
    OnnxAssert(ctx, lstm_proto.input_size() >= 3);
    for (size_t i = 1; i < 3; ++i)
    {
        const string& name = lstm_proto.input(i);
        OnnxAssert(ctx, !name.empty() && constBlobs.count(name) == 1);
    }

    IterShape_t shapeIt = outShapes.find(lstm_proto.input(0));
    OnnxAssert(ctx, shapeIt != outShapes.end());
    const MatShape x_shape = shapeIt->second;

    const int seq_length = x_shape[0];
    const int batch_size = x_shape[1];
    const int input_size = x_shape[2];
    const int hidden_size = layerParams.get<int>("hidden_size");
    const int num_directions = constBlobs[lstm_proto.input(1)].size[0];

    int w_size[] = {num_directions, 4*hidden_size, input_size};
    lstm_extractConsts(layerParams, lstm_proto, 1, w_size, sizeof(w_size) / sizeof(w_size[0])); // W

    int r_size[] =  {num_directions, 4*hidden_size, hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 2, r_size, sizeof(r_size) / sizeof(r_size[0])); // R

    int b_size[] = {num_directions, 8*hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 3, b_size, sizeof(b_size) / sizeof(b_size[0])); // B

    if (4 < lstm_proto.input_size() && !lstm_proto.input(4).empty())
    {
        Mat blob = getBlob(lstm_proto, 4);
        OnnxAssert(ctx, blob.total() == batch_size);
        for (MatIterator_<int32_t> it = blob.begin<int32_t>(); it != blob.end<int32_t>(); ++it)
        {
            OnnxAssert(ctx, *it == seq_length);
        }
    }

    int h_size[] = {num_directions, batch_size, hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 5, h_size, sizeof(h_size) / sizeof(h_size[0])); // initial_h

    int c_size[] = {num_directions, batch_size, hidden_size};
    lstm_extractConsts(layerParams, lstm_proto, 6, c_size, sizeof(c_size) / sizeof(c_size[0])); // initial_c

    if (lstm_proto.input_size() > 7 && !lstm_proto.input(7).empty())
    {
        layerParams.set("use_peephole", true);
        int p_size[] = {num_directions, 3 * hidden_size};
        lstm_extractConsts(layerParams, lstm_proto, 7, p_size, sizeof(p_size) / sizeof(p_size[0])); // P
    }

    transformBlobs(layerParams.blobs);

    layerParams.set("is_onnx", true);
    layerParams.set("reverse", layerParams.get<string>("direction", "") == "reverse");
    layerParams.set("bidirectional", layerParams.get<string>("direction", "") == "bidirectional");

    bool need_yc = lstm_proto.output_size() > 2 && !lstm_proto.output(2).empty();
    bool need_yh = lstm_proto.output_size() > 1 && !lstm_proto.output(1).empty();
    bool need_y = lstm_proto.output_size() > 0 && !lstm_proto.output(0).empty();

    const string y_name = need_y ? lstm_proto.output(0) : "";
    const string yh_name = need_yh ? lstm_proto.output(1) : "";
    const string yc_name = need_yc ? lstm_proto.output(2) : "";

    layerParams.set("produce_cell_output", need_yc);

    lstm_proto.clear_output();
    if (need_y || need_yh)
    {
        // give random names to LSTMLayer's outputs because every output needs postprocessing
        lstm_proto.add_output(cv::format("%s_y", layerParams.name.c_str()));
    }
    if (need_yc)
    {
        lstm_proto.add_output(yc_name);
    }

    addLayer(layerParams, lstm_proto);

    string y_output = lstm_fix_dims(layerParams, lstm_proto, batch_size, num_directions, hidden_size, need_y,
                                         y_name, 0);
    if (need_yh)
    {
        lstm_add_transform(num_directions, batch_size, hidden_size, 0, y_output, yh_name);
    }
}*/


void OnnxImporter2::parseMatMul(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    layerParams.type = "Gemm";
}

/*
void OnnxImporter2::parseMaxUnpool(const string& ctx,
                                const OpenCVOnnx__NodeProto* node_proto, LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.type = "MaxUnpool";
}

void OnnxImporter2::parsePad(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                             LayerParams& layerParams,
                             vector<int>& inputs, vector<int>& outputs)
{
    int depth = layerParams.get<int>("depth", CV_32F);
    layerParams.type = (depth == CV_8S) ? "PaddingInt8" : "Padding";
    replaceLayerParam(layerParams, "mode", "type");
    if (inputs.size() == 3 || inputs.size() == 2)
    {
        // Paddings are in order begin0, begin1, .. beginN, end0, end1, ..., endN.
        // We need to shuffle it to begin0, end0, begin1, end1, ...
        Mat paddings = getBlob(node_proto, 1).reshape(1, 2);
        paddings = paddings.t();
        layerParams.set("paddings", DictValue::arrayInt(paddings.ptr<int>(), paddings.total()));

        if (inputs.size() == 3)
        {
            Mat value = getBlob(node_proto, 2);
            float padValue = (depth == CV_8S) ? (float)value.ptr<int8_t>()[0] : value.ptr<float>()[0];
            layerParams.set("value", padValue);
        }
    }
    addLayer(layerParams, node_proto);
}*/

void OnnxImporter2::parsePooling(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                 LayerParams& layerParams,
                                 vector<int>& inputs, vector<int>& outputs)
{
    bool quantized = strcmp(node_proto->op_type, "QLinearAveragePool") == 0;
    string pool_type = strcmp(node_proto->op_type, "AveragePool") == 0 ? "AVE" :
                       strcmp(node_proto->op_type, "MaxPool") == 0 ? "MAX" :
                       strcmp(node_proto->op_type, "QLinearAveragePool") == 0 ? "AVE" : "";
    layerParams.type = "Pooling";
    layerParams.set("pool", pool_type);
    replaceLayerParam(layerParams, "kernel_shape", "kernel_size");
    // auto_pad attribute is deprecated and uses ceil
    if (layerParams.has("pad_mode"))
        layerParams.set("ceil_mode", true);
    else if (!layerParams.has("ceil_mode"))
        layerParams.set("ceil_mode", false);
    layerParams.set("ave_pool_padded_area", frameworkName == "pytorch");
    size_t ninputs = inputs.size(), noutputs = outputs.size();
    if (quantized) {
        OnnxAssert(ctx, ninputs == 4 || ninputs == 5);
    } else {
        OnnxAssert(ctx, ninputs == 1);
    }
    OnnxAssert(ctx, noutputs == 1);
}

void OnnxImporter2::parsePRelu(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                               LayerParams& layerParams,
                               vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    layerParams.type = "PReLU";
}

void OnnxImporter2::parseReduce(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    string op_type = node_proto->op_type;
    string reduceType;

    if (op_type == "ReduceMax")
        reduceType = "MAX";
    else if (op_type == "ReduceMin")
        reduceType = "MIN";
    else if (op_type == "ReduceSum")
        reduceType = "SUM";
    else if (op_type == "ReduceSumSquare")
        reduceType = "SUM_SQUARE";
    else if (op_type == "ReduceProd")
        reduceType = "PROD";
    else if (op_type == "ReduceL1")
        reduceType = "L1";
    else if (op_type == "ReduceL2")
        reduceType = "L2";
    else if (op_type == "ReduceLogSum")
        reduceType = "LOG_SUM";
    else if (op_type == "ReduceLogSumExp")
        reduceType = "LOG_SUM_EXP";
    else if (op_type == "ReduceMean")
        reduceType = "AVE";
    else
        onnxParseError(ctx, format("unsupported reduce operation '%s'", node_proto->op_type));

    layerParams.type = "Reduce";
    layerParams.set("reduce", reduceType);
}

void OnnxImporter2::parseRelu(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                              LayerParams& layerParams,
                              vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.type = "ReLU";
}

void OnnxImporter2::parseReshape(const string& ctx,
                                 const OpenCVOnnx__NodeProto* node_proto,
                                 LayerParams& layerParams,
                                 vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || (ninputs == 1 && layerParams.has("shape")));
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseResize(const string& ctx,
                                const OpenCVOnnx__NodeProto* node_proto,
                                LayerParams& layerParams,
                                vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, 1 <= ninputs && ninputs <= 4);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseShape(const string& ctx,
                               const OpenCVOnnx__NodeProto* node_proto,
                               LayerParams& layerParams,
                               vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseSlice(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                               LayerParams& layerParams, vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, 3 <= ninputs && ninputs <= 5);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseSoftMax(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                 LayerParams& layerParams,
                                 vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.set("log_softmax", layerParams.type == "LogSoftmax");
    layerParams.type = "Softmax";
}

void OnnxImporter2::parseSplit(const string& ctx,
                               const OpenCVOnnx__NodeProto* node_proto,
                               LayerParams& layerParams,
                               vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() >= 1);
    if (layerParams.has("num_split")) {
        int num_split = layerParams.get<int>("num_split");
        OnnxAssert(ctx, num_split == outputs.size());
    }
}

void OnnxImporter2::parseSqueeze(const string& ctx,
                                 const OpenCVOnnx__NodeProto* node_proto,
                                 LayerParams& layerParams,
                                 vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || (ninputs == 1 && layerParams.has("axes")));
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseTranspose(const string& ctx,
                                   const OpenCVOnnx__NodeProto* node_proto,
                                   LayerParams& layerParams,
                                   vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("perm"));
}

/*void OnnxImporter2::parseUpsample(const string& ctx, const OpenCVOnnx__NodeProto* node_proto,
                                  LayerParams& layerParams,
                                  vector<int>& inputs, vector<int>& outputs)
{
    //fused from Resize Subgraph
    if (layerParams.has("coordinate_transformation_mode"))
    {
        string interp_mode = layerParams.get<string>("coordinate_transformation_mode");
        CV_Assert_N(interp_mode != "tf_crop_and_resize", interp_mode != "tf_half_pixel_for_nn");

        layerParams.set("align_corners", interp_mode == "align_corners");
        if (layerParams.get<string>("mode") == "linear")
        {
            layerParams.set("mode", interp_mode == "pytorch_half_pixel" ?
                                    "opencv_linear" : "bilinear");
        }
    }
    if (layerParams.get<string>("mode") == "linear" && framework_name == "pytorch")
        layerParams.set("mode", "opencv_linear");

    layerParams.type = "Resize";
    if (layerParams.has("scales"))
    {
        // Pytorch layer
        DictValue scales = layerParams.get("scales");
        OnnxAssert(ctx, scales.size() == 4);
        layerParams.set("zoom_factor_y", scales.getIntValue(2));
        layerParams.set("zoom_factor_x", scales.getIntValue(3));
    }
    else if (layerParams.has("height_scale") && layerParams.has("width_scale"))
    {
        // Caffe2 layer
        replaceLayerParam(layerParams, "height_scale", "zoom_factor_y");
        replaceLayerParam(layerParams, "width_scale", "zoom_factor_x");
    }
    else
    {
        // scales as input
        const string& input1 = node_proto.input(1);
        if (constBlobs.find(input1) != constBlobs.end())
        {
            Mat scales = getBlob(input1);
            OnnxAssert(ctx, scales.total() == 4);
            layerParams.set("zoom_factor_y", scales.at<float>(2));
            layerParams.set("zoom_factor_x", scales.at<float>(3));
        }
    }
    replaceLayerParam(layerParams, "mode", "interpolation");
    addLayer(layerParams, node_proto);
}*/

void OnnxImporter2::parseUnsqueeze(const string& ctx,
                                   const OpenCVOnnx__NodeProto* node_proto,
                                   LayerParams& layerParams,
                                   vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || (ninputs == 1 && layerParams.has("axes")));
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseDequantizeLinear(const string& ctx,
                                          const OpenCVOnnx__NodeProto* node_proto,
                                          LayerParams& layerParams,
                                          vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQLinearConcat(const string& ctx,
                                       const OpenCVOnnx__NodeProto* node_proto,
                                       LayerParams& layerParams,
                                       vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 3);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("axis"));
}

void OnnxImporter2::parseQLinearConv(const string& ctx,
                                     const OpenCVOnnx__NodeProto* node_proto,
                                     LayerParams& layerParams,
                                     vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 8 || ninputs == 9);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQLinearElemwiseBinary(const string& ctx,
                                               const OpenCVOnnx__NodeProto* node_proto,
                                               LayerParams& layerParams,
                                               vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    string op = layerParams.type == "QLinearAdd" ? "ADD" :
                layerParams.type == "QlinearMul" ? "MUL" : "";
    if (op.empty())
        onnxParseError(ctx, format("unrecognized quantized binary operation '%s'", node_proto->op_type));
    OnnxAssert(ctx, ninputs == 7 || ninputs == 8);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "QElemwise";
    layerParams.set("op", op);
}

void OnnxImporter2::parseQLinearGlobalAveragePool(const string& ctx,
                                       const OpenCVOnnx__NodeProto* node_proto,
                                       LayerParams& layerParams,
                                       vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 5);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("channels_last"));
}

void OnnxImporter2::parseQLinearLeakyRelu(const string& ctx,
                                          const OpenCVOnnx__NodeProto* node_proto,
                                          LayerParams& layerParams,
                                          vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 4 || ninputs == 5);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("alpha"));
}

void OnnxImporter2::parseQLinearMatMul(const string& ctx,
                                       const OpenCVOnnx__NodeProto* node_proto,
                                       LayerParams& layerParams,
                                       vector<int>& inputs, vector<int>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 8);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQLinearSigmoid(const string& ctx,
                                        const OpenCVOnnx__NodeProto* node_proto,
                                        LayerParams& layerParams,
                                        vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 4 || ninputs == 5);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQuantizeLinear(const string& ctx,
                                        const OpenCVOnnx__NodeProto* node_proto,
                                        LayerParams& layerParams,
                                        vector<int>& inputs, vector<int>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
}

Net2 readNetFromONNX2(const String& onnxFile)
{
    Net2 net;
    OnnxImporter2 importer(net, onnxFile.c_str());
    return net;
}

CV__DNN_INLINE_NS_END
}} // namespace
