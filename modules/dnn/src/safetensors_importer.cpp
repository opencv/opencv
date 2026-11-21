// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Safetensors + graph.json importer for OpenCV DNN.
//
// Reads network topology from a graph JSON file and trained weights from a
// safetensors file, constructs an in-memory ONNX ModelProto, and forwards it
// to the standard ONNX importer.  This avoids the need for an intermediate
// .onnx file on disk and removes the runtime dependency on the Python `onnx`
// package (graph_to_onnx.py).
//
// Graph JSON schema (as produced by the export tool):
//   {
//     "model_name": "name",                                  // optional
//     "inputs":  [ { "name": "x", "shape": [1, 3, "H", "W"], // int or dynamic dim name
//                   "dtype": "float32" }, ... ],              // interface dtype, see below
//     "outputs": [ { "name": "y", "shape": [1, 2], "dtype": "int64" }, ... ],
//     "nodes":   [ { "name": "conv.0", "op_type": "Conv",
//                    "inputs":  ["x", "conv.weight", "conv.bias"],
//                    "outputs": ["y"],
//                    "attrs":   { "strides": [1, 1], "group": 1, ... } }, ... ]
//   }
// Tensor names in "nodes" refer either to graph inputs/outputs, to other
// nodes' outputs, or to initializer names in the safetensors file. Node
// attribute values follow ONNX attribute types (int/float/string or arrays).
// The interface dtype vocabulary is: float32, float64, float16, bfloat16,
// int8, int16, int32, int64, uint8, uint16, uint32, uint64 and bool
// (defaulting to float32 when "dtype" is omitted).

#include "precomp.hpp"
#include "net_impl.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <fstream>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <locale>
#include <codecvt>
#endif

#ifdef HAVE_PROTOBUF

// ===== Read entire file into memory (cross-platform) =====
// On Windows, std::ifstream cannot open UTF-8 encoded non-ASCII paths.
// Convert UTF-8 path to UTF-16 wstring and use MSVC's wchar_t* ifstream extension.
static bool readFileToBytes(const cv::String& path, std::vector<uchar>& out)
{
#if defined(_WIN32)
    // std::wstring_convert is deprecated in C++17 but still usable
    std::wstring wpath;
    try
    {
        typedef std::codecvt_utf8_utf16<wchar_t> cvt_t;
        std::wstring_convert<cvt_t, wchar_t> converter;
        wpath = converter.from_bytes(path);
    }
    catch (const std::exception&)
    {
        // Fallback: narrow copy
        wpath = std::wstring(path.begin(), path.end());
    }
    std::ifstream f(wpath.c_str(), std::ios::binary);
#else
    std::ifstream f(path.c_str(), std::ios::binary);
#endif
    if (!f.is_open())
    {
        CV_LOG_ERROR(NULL, "DNN/Safetensors: cannot open file: " << path);
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    f.seekg(0);
    out.resize((size_t)sz);
    if (sz > 0)
        f.read(reinterpret_cast<char*>(out.data()), sz);
    return (bool)f;
}

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

// ===== Safetensors file format =====
//   8 bytes : uint64 little-endian  -> header length N
//   N bytes : JSON header describing every tensor
//   rest    : raw tensor data (concatenated)

struct SafetensorsTensorInfo {
    std::string dtype;                       // "F32", "I64", "I32", ...
    std::vector<int64_t> shape;
    uint64_t data_start = 0;                 // offset into data buffer
    uint64_t data_end = 0;
};

// Parse safetensors header JSON using cv::FileStorage (memory mode).
// Returns false on failure.
static bool parseSafetensorsHeader(const std::string& header_json,
                                    std::map<std::string, SafetensorsTensorInfo>& out)
{
    FileStorage fs(header_json.c_str(), FileStorage::READ | FileStorage::MEMORY);
    if (!fs.isOpened())
    {
        CV_LOG_ERROR(NULL, "DNN/Safetensors: failed to parse header JSON");
        return false;
    }
    FileNode root = fs.root();
    for (FileNodeIterator it = root.begin(); it != root.end(); ++it)
    {
        FileNode t = *it;
        std::string name = t.name();
        if (name == "__metadata__")
            continue;
        SafetensorsTensorInfo info;
        info.dtype = (std::string)t["dtype"];
        FileNode shape_node = t["shape"];
        for (FileNodeIterator s = shape_node.begin(); s != shape_node.end(); ++s)
            info.shape.push_back((int64_t)(*s));
        FileNode off = t["data_offsets"];
        info.data_start = (uint64_t)(int64_t)off[0];
        info.data_end   = (uint64_t)(int64_t)off[1];
        out[name] = info;
    }
    return true;
}

// ===== dtype string -> element size in bytes (-1 for unknown) =====
static int64_t dtypeElementSize(const std::string& dtype)
{
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "I8"  || dtype == "U8"  || dtype == "BOOL") return 1;
    return -1;
}

static bool isIntegerDtype(const std::string& dtype)
{
    return dtype == "I64" || dtype == "I32" || dtype == "I16" || dtype == "I8" ||
           dtype == "U64" || dtype == "U32" || dtype == "U16" || dtype == "U8";
}

static bool parseSafetensorsFile(const String& path,
                                 std::map<std::string, SafetensorsTensorInfo>& out_info,
                                 std::vector<uchar>& out_data,
                                 size_t& out_data_offset)
{
    // Read entire file via readFileToBytes (handles UTF-8 non-ASCII paths on Windows)
    std::vector<uchar> file_bytes;
    if (!readFileToBytes(path, file_bytes))
    {
        CV_LOG_ERROR(NULL, "DNN/Safetensors: cannot open file: " << path);
        return false;
    }
    if (file_bytes.size() < 8)
    {
        CV_LOG_ERROR(NULL, "DNN/Safetensors: file too small: " << path);
        return false;
    }

    // Parse 8-byte header length (little-endian uint64)
    uint64_t header_len = 0;
    memcpy(&header_len, file_bytes.data(), 8);
    if (header_len == 0 || header_len > 256ull * 1024 * 1024)
    {
        CV_LOG_ERROR(NULL, "DNN/Safetensors: invalid header length: " << header_len);
        return false;
    }

    size_t data_offset = 8 + (size_t)header_len;
    if (file_bytes.size() < data_offset)
    {
        CV_LOG_ERROR(NULL, "DNN/Safetensors: file truncated, cannot read header");
        return false;
    }

    // Parse header JSON
    std::string header_json((const char*)(file_bytes.data() + 8),
                            (size_t)header_len);
    if (!parseSafetensorsHeader(header_json, out_info))
        return false;

    // Validate every tensor: known dtype, data range fits into the data
    // section, and the declared shape matches the data span exactly.
    size_t data_size = file_bytes.size() - data_offset;
    for (const auto& kv : out_info)
    {
        const SafetensorsTensorInfo& t = kv.second;
        if (t.data_end > (uint64_t)data_size || t.data_start > t.data_end)
        {
            CV_LOG_ERROR(NULL, "DNN/Safetensors: tensor '" << kv.first
                         << "' data range [" << t.data_start << ", " << t.data_end
                         << ") exceeds data section size " << data_size);
            return false;
        }
        const int64_t elem_size = dtypeElementSize(t.dtype);
        if (elem_size < 0)
        {
            CV_LOG_ERROR(NULL, "DNN/Safetensors: tensor '" << kv.first
                         << "' has unsupported dtype '" << t.dtype << "'");
            return false;
        }
        uint64_t numel = 1;
        bool bad_shape = false;
        for (int64_t d : t.shape)
        {
            if (d < 0 || (d != 0 && numel > (((uint64_t)1) << 60) / (uint64_t)d))
            {
                bad_shape = true;
                break;
            }
            numel *= (uint64_t)d;
        }
        if (bad_shape || numel * (uint64_t)elem_size != (uint64_t)(t.data_end - t.data_start))
        {
            CV_LOG_ERROR(NULL, "DNN/Safetensors: tensor '" << kv.first
                         << "' declared shape does not match its data span ("
                         << (t.data_end - t.data_start) << " bytes, dtype "
                         << t.dtype << ")");
            return false;
        }
    }

    // Take ownership of the file buffer without copying the data section:
    // out_data holds the whole file, out_data_offset points to the first tensor byte.
    out_data = std::move(file_bytes);
    out_data_offset = data_offset;

    CV_LOG_INFO(NULL, "DNN/Safetensors: parsed " << out_info.size()
                << " tensors, data size: " << data_size << " bytes");
    return true;
}

// ===== Identify tensors that must be INT64 =====
// safetensors stores all tensors as float32, but some tensors are actually
// integer metadata (Slice starts/ends, Reshape shape, Concat axis, ...).
// These must be converted to int64 when building the ONNX ModelProto.
// Positions follow the ONNX spec: only inputs that are *always* integer
// tensors are listed here (never float inputs like Resize roi/scales,
// RoiAlign rois, Pad constant_value, Scatter updates or OneHot values).
// Transpose/Concat are kept for legacy exporters that pass perm/axis as
// graph inputs instead of attributes.
static const std::map<std::string, std::set<int>>& getInt64Positions()
{
    static const std::map<std::string, std::set<int>> positions = {
        {"Slice",           {1, 2, 3, 4}},   // starts, ends, axes, steps
        {"Reshape",         {1}},            // shape
        {"Split",           {1}},            // split
        {"Gather",          {1}},            // indices
        {"GatherElements",  {1}},            // indices
        {"GatherND",        {1}},            // indices
        {"Squeeze",         {1}},            // axes
        {"Unsqueeze",       {1}},            // axes
        {"Tile",            {1}},            // repeats
        {"Transpose",       {1}},            // legacy: perm as input
        {"Concat",          {1}},            // legacy: axis as input
        {"Expand",          {1}},            // shape
        {"ConstantOfShape", {0}},            // input shape
        {"TopK",            {1}},            // k
        {"Pad",             {1, 3}},         // pads, axes (opset 18)
        {"Resize",          {3}},            // sizes
        {"RoiAlign",        {2}},            // batch_indices
        {"Trilu",           {1}},            // k
        {"Scatter",         {1}},            // indices
        {"ScatterElements", {1}},            // indices
        {"ScatterND",       {1}},            // indices
        {"CenterCropPad",   {1}},            // shape
        {"ReverseSequence", {1}},            // seq_lens
        {"AffineGrid",      {1}},            // size
        {"MaxUnpool",       {1}},            // indices
        {"OneHot",          {0, 1}},         // indices, depth
        {"CumSum",          {1}},            // axis
        {"ReduceSum",       {1}},            // axes (opset 13+)
        {"ReduceMean",      {1}},
        {"ReduceMax",       {1}},
        {"ReduceMin",       {1}},
        {"ReduceProd",      {1}},
        {"ReduceL1",        {1}},
        {"ReduceL2",        {1}},
        {"ReduceLogSum",    {1}},
        {"ReduceLogSumExp", {1}},
        {"ReduceSumSquare", {1}},
    };
    return positions;
}

// ===== dtype string -> ONNX TensorProto data type =====
// Returns 0 for unknown dtype strings (the caller must raise an error).
static int dtypeStringToOnnx(const std::string& dtype)
{
    if (dtype == "F32")  return opencv_onnx::TensorProto_DataType_FLOAT;
    if (dtype == "F64")  return opencv_onnx::TensorProto_DataType_DOUBLE;
    if (dtype == "I64")  return opencv_onnx::TensorProto_DataType_INT64;
    if (dtype == "I32")  return opencv_onnx::TensorProto_DataType_INT32;
    if (dtype == "I16")  return opencv_onnx::TensorProto_DataType_INT16;
    if (dtype == "I8")   return opencv_onnx::TensorProto_DataType_INT8;
    if (dtype == "U8")   return opencv_onnx::TensorProto_DataType_UINT8;
    if (dtype == "U16")  return opencv_onnx::TensorProto_DataType_UINT16;
    if (dtype == "U32")  return opencv_onnx::TensorProto_DataType_UINT32;
    if (dtype == "U64")  return opencv_onnx::TensorProto_DataType_UINT64;
    if (dtype == "BOOL") return opencv_onnx::TensorProto_DataType_BOOL;
    if (dtype == "BF16") return opencv_onnx::TensorProto_DataType_BFLOAT16;
    if (dtype == "F16")  return opencv_onnx::TensorProto_DataType_FLOAT16;
    return 0;  // unknown dtype
}

// ===== graph.json interface dtype string -> ONNX TensorProto data type =====
// The graph interface vocabulary (as written by the export tool) uses full
// type names, unlike the safetensors dtype strings ("F32" etc.).
// Returns 0 for unknown dtype strings (the caller must raise an error).
static int graphDtypeStringToOnnx(const std::string& dtype)
{
    if (dtype == "float32")  return opencv_onnx::TensorProto_DataType_FLOAT;
    if (dtype == "float64")  return opencv_onnx::TensorProto_DataType_DOUBLE;
    if (dtype == "float16")  return opencv_onnx::TensorProto_DataType_FLOAT16;
    if (dtype == "bfloat16") return opencv_onnx::TensorProto_DataType_BFLOAT16;
    if (dtype == "int64")   return opencv_onnx::TensorProto_DataType_INT64;
    if (dtype == "int32")   return opencv_onnx::TensorProto_DataType_INT32;
    if (dtype == "int16")   return opencv_onnx::TensorProto_DataType_INT16;
    if (dtype == "int8")    return opencv_onnx::TensorProto_DataType_INT8;
    if (dtype == "uint64")  return opencv_onnx::TensorProto_DataType_UINT64;
    if (dtype == "uint32")  return opencv_onnx::TensorProto_DataType_UINT32;
    if (dtype == "uint16")  return opencv_onnx::TensorProto_DataType_UINT16;
    if (dtype == "uint8")   return opencv_onnx::TensorProto_DataType_UINT8;
    if (dtype == "bool")    return opencv_onnx::TensorProto_DataType_BOOL;
    return 0;  // unknown dtype
}

// Resolve the interface dtype of a graph input/output entry.
// Falls back to FLOAT when the entry omits "dtype" (the only interface type
// the original export pipelines produced), raises on unknown dtype strings.
static int resolveInterfaceDtype(const std::string& tensorName, const FileNode& dtypeNode)
{
    if (dtypeNode.empty())
        return opencv_onnx::TensorProto_DataType_FLOAT;
    const std::string dtype = (std::string)dtypeNode;
    const int onnx_dtype = graphDtypeStringToOnnx(dtype);
    if (onnx_dtype == 0)
    {
        CV_Error_(Error::StsError,
                  ("DNN/Safetensors: interface tensor '%s' has unsupported dtype '%s'",
                   tensorName.c_str(), dtype.c_str()));
    }
    return onnx_dtype;
}

// ===== Set ONNX attribute from a cv::FileNode =====
static void setAttribute(opencv_onnx::AttributeProto* attr,
                          const std::string& name, const FileNode& val)
{
    attr->set_name(name);
    if (val.isInt())
    {
        attr->set_type(opencv_onnx::AttributeProto::INT);
        attr->set_i((int64_t)val);
    }
    else if (val.isReal())
    {
        attr->set_type(opencv_onnx::AttributeProto::FLOAT);
        attr->set_f((float)(double)val);
    }
    else if (val.isString())
    {
        attr->set_type(opencv_onnx::AttributeProto::STRING);
        attr->set_s((std::string)val);
    }
    else if (val.isSeq())
    {
        bool all_int = true;
        for (FileNodeIterator s = val.begin(); s != val.end(); ++s)
        {
            if (!(*s).isInt()) { all_int = false; break; }
        }
        if (all_int)
        {
            attr->set_type(opencv_onnx::AttributeProto::INTS);
            for (FileNodeIterator s = val.begin(); s != val.end(); ++s)
                attr->add_ints((int64_t)(*s));
        }
        else
        {
            attr->set_type(opencv_onnx::AttributeProto::FLOATS);
            for (FileNodeIterator s = val.begin(); s != val.end(); ++s)
            {
                if ((*s).isInt())
                    attr->add_floats((float)(int64_t)(*s));
                else
                    attr->add_floats((float)(double)(*s));
            }
        }
    }
}

// ===== Build ONNX ModelProto from graph.json + safetensors =====
Net readNetFromSafetensors(const String& graphJsonPath,
                            const String& weightsPath,
                            int engine)
{
    CV_LOG_INFO(NULL, "DNN/Safetensors: loading " << graphJsonPath
                << " + " << weightsPath);

    // 1. Parse safetensors weights file
    std::map<std::string, SafetensorsTensorInfo> st_info;
    std::vector<uchar> st_data;
    size_t st_data_offset = 0;
    if (!parseSafetensorsFile(weightsPath, st_info, st_data, st_data_offset))
        CV_Error(Error::StsError, "DNN/Safetensors: failed to parse weights: " + weightsPath);

    // 2. Parse graph.json (topology) - read into memory via readFileToBytes,
    //    then parse with FileStorage MEMORY mode to handle UTF-8 non-ASCII paths on Windows
    std::vector<uchar> graph_bytes;
    if (!readFileToBytes(graphJsonPath, graph_bytes))
        CV_Error(Error::StsError, "DNN/Safetensors: failed to open graph JSON: " + graphJsonPath);
    std::string graph_json_str((const char*)graph_bytes.data(), graph_bytes.size());
    FileStorage fs(graph_json_str.c_str(), FileStorage::READ | FileStorage::MEMORY);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "DNN/Safetensors: failed to parse graph JSON: " + graphJsonPath);

    // 3. Build ONNX ModelProto
    opencv_onnx::ModelProto model;
    model.set_ir_version(8);
    auto* opset = model.add_opset_import();
    opset->set_domain("");
    opset->set_version(17);
    auto* graph = model.mutable_graph();

    // graph name
    FileNode name_node = fs["model_name"];
    if (!name_node.empty())
        graph->set_name((std::string)name_node);

    // 4. Inputs
    FileNode inputs = fs["inputs"];
    for (FileNodeIterator it = inputs.begin(); it != inputs.end(); ++it)
    {
        FileNode inp = *it;
        const std::string in_name = (std::string)inp["name"];
        auto* vi = graph->add_input();
        vi->set_name(in_name);
        auto* ttype = vi->mutable_type()->mutable_tensor_type();
        ttype->set_elem_type(resolveInterfaceDtype(in_name, inp["dtype"]));
        auto* shape = ttype->mutable_shape();
        FileNode shape_node = inp["shape"];
        for (FileNodeIterator s = shape_node.begin(); s != shape_node.end(); ++s)
        {
            auto* dim = shape->add_dim();
            FileNode d = *s;
            if (d.isInt())
            {
                int64_t v = (int64_t)d;
                if (v > 0) dim->set_dim_value(v);
            }
            else if (d.isString())
                dim->set_dim_param((std::string)d);
        }
    }

    // 5. Outputs
    FileNode outputs = fs["outputs"];
    for (FileNodeIterator it = outputs.begin(); it != outputs.end(); ++it)
    {
        FileNode out = *it;
        const std::string out_name = (std::string)out["name"];
        auto* vi = graph->add_output();
        vi->set_name(out_name);
        auto* ttype = vi->mutable_type()->mutable_tensor_type();
        ttype->set_elem_type(resolveInterfaceDtype(out_name, out["dtype"]));
        auto* shape = ttype->mutable_shape();
        FileNode shape_node = out["shape"];
        for (FileNodeIterator s = shape_node.begin(); s != shape_node.end(); ++s)
        {
            auto* dim = shape->add_dim();
            FileNode d = *s;
            if (d.isInt())
            {
                int64_t v = (int64_t)d;
                if (v > 0) dim->set_dim_value(v);
            }
        }
    }

    // 6. Identify tensors that must be INT64 (metadata tensors stored as F32)
    std::set<std::string> int64_names;
    FileNode nodes = fs["nodes"];
    const auto& int64_positions = getInt64Positions();
    for (FileNodeIterator it = nodes.begin(); it != nodes.end(); ++it)
    {
        FileNode node = *it;
        std::string op_type = (std::string)node["op_type"];
        auto pos_it = int64_positions.find(op_type);
        if (pos_it == int64_positions.end())
            continue;
        FileNode node_inputs = node["inputs"];
        int idx = 0;
        for (FileNodeIterator i = node_inputs.begin(); i != node_inputs.end(); ++i, ++idx)
        {
            if (pos_it->second.count(idx))
            {
                std::string inp_name = (std::string)(*i);
                if (st_info.count(inp_name))
                    int64_names.insert(inp_name);
            }
        }
    }

    // 7. Initializers (weights from safetensors)
    for (const auto& kv : st_info)
    {
        const std::string& name = kv.first;
        const SafetensorsTensorInfo& info = kv.second;
        auto* init = graph->add_initializer();
        init->set_name(name);
        for (int64_t dim : info.shape)
            init->add_dims(dim);

        bool needs_int64 = int64_names.count(name) > 0;
        size_t data_size = (size_t)(info.data_end - info.data_start);
        const uchar* raw_ptr = st_data.data() + st_data_offset + info.data_start;

        if (needs_int64 && (info.dtype == "F32" || info.dtype == "F64"))
        {
            // float -> int64 conversion: safetensors stores metadata tensors
            // (Slice starts/ends, Reshape shape, ...) as float32/float64, but
            // they are actually integers and ONNX expects int64 data.  Reject
            // anything that is not an exactly representable integer instead
            // of silently truncating (out-of-range casts would be UB, and
            // non-integral values would be lossy).
            const size_t elem_size = (info.dtype == "F32") ? sizeof(float) : sizeof(double);
            const size_t num_elements = data_size / elem_size;
            std::vector<int64_t> int64_buf(num_elements);
            for (size_t i = 0; i < num_elements; ++i)
            {
                const double v = (info.dtype == "F32")
                    ? (double)reinterpret_cast<const float*>(raw_ptr)[i]
                    : reinterpret_cast<const double*>(raw_ptr)[i];
                if (!std::isfinite(v) || v != std::trunc(v) ||
                    v < -9223372036854775808.0 || v >= 9223372036854775808.0)
                {
                    CV_Error_(Error::StsError,
                              ("DNN/Safetensors: tensor '%s' element %lld is not an exactly representable integer: %g",
                               name.c_str(), (long long)i, v));
                }
                int64_buf[i] = static_cast<int64_t>(v);
            }
            std::string raw;
            raw.resize(num_elements * sizeof(int64_t));
            if (!int64_buf.empty())
                memcpy(&raw[0], int64_buf.data(), raw.size());
            init->set_data_type(opencv_onnx::TensorProto_DataType_INT64);
            init->set_raw_data(raw);
        }
        else if (needs_int64 && !isIntegerDtype(info.dtype))
        {
            CV_Error_(Error::StsError,
                      ("DNN/Safetensors: tensor '%s' is used as an integer input but has dtype '%s'",
                       name.c_str(), info.dtype.c_str()));
        }
        else
        {
            const int onnx_dtype = dtypeStringToOnnx(info.dtype);
            if (onnx_dtype == 0)
                CV_Error_(Error::StsError,
                          ("DNN/Safetensors: tensor '%s' has unsupported dtype '%s'",
                           name.c_str(), info.dtype.c_str()));
            init->set_data_type(onnx_dtype);
            std::string raw;
            raw.resize(data_size);
            if (data_size > 0)
                memcpy(&raw[0], raw_ptr, data_size);
            init->set_raw_data(raw);
        }
    }

    // 8. Nodes (topology)
    for (FileNodeIterator it = nodes.begin(); it != nodes.end(); ++it)
    {
        FileNode node = *it;
        auto* np = graph->add_node();
        np->set_op_type((std::string)node["op_type"]);
        FileNode nn = node["name"];
        if (!nn.empty())
            np->set_name((std::string)nn);

        FileNode node_inputs = node["inputs"];
        for (FileNodeIterator i = node_inputs.begin(); i != node_inputs.end(); ++i)
            np->add_input((std::string)(*i));

        FileNode node_outputs = node["outputs"];
        for (FileNodeIterator o = node_outputs.begin(); o != node_outputs.end(); ++o)
            np->add_output((std::string)(*o));

        FileNode attrs = node["attrs"];
        for (FileNodeIterator a = attrs.begin(); a != attrs.end(); ++a)
        {
            FileNode attr_node = *a;
            setAttribute(np->add_attribute(), attr_node.name(), attr_node);
        }
    }

    // 9. Forward the in-memory ModelProto directly to the ONNX importer,
    //    skipping the serialize/parse round-trip (two large copies).
    //    Engine resolution mirrors readNetFromONNX(buffer, size, engine):
    //    ENGINE_ORT does not accept in-memory models.
    if (engine == ENGINE_AUTO)
        engine = ENGINE_OPENCV;
    if (engine != ENGINE_OPENCV && engine != ENGINE_ORT)
    {
        CV_LOG_WARNING(NULL, "DNN/Safetensors: only ENGINE_AUTO, ENGINE_OPENCV and ENGINE_ORT are supported; "
                             "falling back to ENGINE_OPENCV.");
        engine = ENGINE_OPENCV;
    }
    if (engine == ENGINE_ORT)
    {
#ifdef HAVE_ONNXRUNTIME
        CV_Error(Error::StsNotImplemented, "DNN/Safetensors: loading from in-memory model is not supported by ENGINE_ORT");
#else
        CV_LOG_WARNING(NULL, "DNN/Safetensors: OpenCV was built without ONNX Runtime (WITH_ONNXRUNTIME=OFF). Falling back to ENGINE_OPENCV.");
#endif
    }

    CV_LOG_INFO(NULL, "DNN/Safetensors: built ONNX model, forwarding to ONNX importer");

    return readNetFromONNX2(std::move(model));
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#else  // !HAVE_PROTOBUF

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Net readNetFromSafetensors(const String&, const String&, int)
{
    CV_Error(Error::StsError, "DNN/Safetensors: OpenCV was built without Protobuf (HAVE_PROTOBUF=0)");
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif  // HAVE_PROTOBUF
