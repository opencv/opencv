#include "../precomp.hpp"
#include "../net_impl.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/layer_reg.private.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <opencv2/core/utils/configuration.private.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <limits>
#include <set>
#include <string>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN



// =====================
// GGML Type Enumeration
// =====================
enum ggml_type : uint32_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

// ==============================
// GGUF Metadata Value Type Enum
// ==============================
enum gguf_metadata_value_type : uint32_t {
    GGUF_METADATA_VALUE_TYPE_UINT8    = 0,
    GGUF_METADATA_VALUE_TYPE_INT8     = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16   = 2,
    GGUF_METADATA_VALUE_TYPE_INT16    = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32   = 4,
    GGUF_METADATA_VALUE_TYPE_INT32    = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32  = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL     = 7,
    GGUF_METADATA_VALUE_TYPE_STRING   = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY    = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64   = 10,
    GGUF_METADATA_VALUE_TYPE_INT64    = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64  = 12,
};

// ===============
// GGUF Structures
// ===============
struct gguf_string_t {
    uint64_t len;
    std::string data;
};

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values
        gguf_metadata_value_t*array;
    } array;
};

struct gguf_metadata_kv_t {
    gguf_string_t key;
    gguf_metadata_value_type value_type;
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    std::vector<gguf_metadata_kv_t> metadata_kv;
};

struct gguf_tensor_info_t {
    gguf_string_t name;
    uint32_t n_dimensions;
    std::vector<uint64_t> dimensions;
    ggml_type type;
    uint64_t offset;
};

struct gguf_file_t {
    gguf_header_t header;
    std::vector<gguf_tensor_info_t> tensor_infos;
    uint8_t _padding;
    uint8_t tensor_data[]; // flexible array member (pointer interpreted)
};

struct MetadataValueNode{
    virtual ~MetadataValueNode() = default;
};

struct MetadataKeyValueNode
{
    std::string key;
    std::unique_ptr<MetadataValueNode> value;
};

template<typename T>
struct MetadataSingleValueNode : public MetadataValueNode
{
    T value;    
     // Add a constructor that initializes 'value'
     explicit MetadataSingleValueNode(const T &v) : value(v) {}
    
     // Optionally, also add a move constructor if needed:
     explicit MetadataSingleValueNode(T &&v) : value(std::move(v)) {}
};

struct MetadataArrayNode : public MetadataValueNode
{
    std::vector<std::unique_ptr<MetadataValueNode>> array;
};

std::string parseGGUFString(const uint8_t* buffer, size_t offset)
{
    std::string result;
    uint64_t len = *reinterpret_cast<const uint64_t*>(buffer + offset);
    result.assign(reinterpret_cast<const char*>(buffer + offset + sizeof(uint64_t)), len);
    return result;
}

std::unique_ptr<MetadataValueNode> parseMetadataSingleValueNode(const uint8_t* buffer, size_t& offset, uint32_t type)
{
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8: {
            uint8_t val = *(buffer + offset);
            offset += sizeof(uint8_t);
            return std::make_unique<MetadataSingleValueNode<uint8_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_INT8: {
            int8_t val = *reinterpret_cast<const int8_t*>(buffer + offset);
            offset += sizeof(int8_t);
            return std::make_unique<MetadataSingleValueNode<int8_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT16: {
            uint16_t val = *reinterpret_cast<const uint16_t*>(buffer + offset);
            offset += sizeof(uint16_t);
            return std::make_unique<MetadataSingleValueNode<uint16_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_INT16: {
            int16_t val = *reinterpret_cast<const int16_t*>(buffer + offset);
            offset += sizeof(int16_t);
            return std::make_unique<MetadataSingleValueNode<int16_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT32: {
            uint32_t val = *reinterpret_cast<const uint32_t*>(buffer + offset);
            offset += sizeof(uint32_t);
            return std::make_unique<MetadataSingleValueNode<uint32_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_INT32: {
            int32_t val = *reinterpret_cast<const int32_t*>(buffer + offset);
            offset += sizeof(int32_t);
            return std::make_unique<MetadataSingleValueNode<int32_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64: {
            uint64_t val = *reinterpret_cast<const uint64_t*>(buffer + offset);
            offset += sizeof(uint64_t);
            return std::make_unique<MetadataSingleValueNode<uint64_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_INT64: {
            int64_t val = *reinterpret_cast<const int64_t*>(buffer + offset);
            offset += sizeof(int64_t);
            return std::make_unique<MetadataSingleValueNode<int64_t>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
            float val = *reinterpret_cast<const float*>(buffer + offset);
            offset += sizeof(float);
            return std::make_unique<MetadataSingleValueNode<float>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
            double val = *reinterpret_cast<const double*>(buffer + offset);
            offset += sizeof(double);
            return std::make_unique<MetadataSingleValueNode<double>>(val);
        }
        case GGUF_METADATA_VALUE_TYPE_BOOL: {
            uint8_t val = *(buffer + offset);
            offset += sizeof(uint8_t);
            bool boolVal = (val != 0);
            return std::make_unique<MetadataSingleValueNode<bool>>(boolVal);
        }
        case GGUF_METADATA_VALUE_TYPE_STRING: {
            std::string str = parseGGUFString(buffer, offset);
            return std::make_unique<MetadataSingleValueNode<std::string>>(str);
        }
        default:
            throw std::runtime_error("Unknown metadata value type ID: " + std::to_string(type));
    }
}

std::unique_ptr<MetadataValueNode> parseMetadataValueNode(const uint8_t* buffer, size_t& offset)
{
    uint32_t type = *reinterpret_cast<const uint32_t*>(buffer + offset);
    offset += sizeof(uint32_t);


    if (type != GGUF_METADATA_VALUE_TYPE_ARRAY) {
        return parseMetadataSingleValueNode(buffer, offset, type);
    } 

    uint32_t elementType = *reinterpret_cast<const uint32_t*>(buffer + offset);
    offset += sizeof(uint32_t);

    uint64_t count = *reinterpret_cast<const uint64_t*>(buffer + offset);
    offset += sizeof(uint64_t);
    
    // do not handle arrays of arrays for now
    MetadataArrayNode arrayNode;
    for (uint64_t i = 0; i < count; ++i) {
        arrayNode.array.push_back(parseMetadataSingleValueNode(buffer, offset,elementType));
    }
    return std::make_unique<MetadataArrayNode>(std::move(arrayNode));
}


struct TensorMetadata {
    std::string name;
    MatShape dims;
    size_t data_offset;
    //std::vector<uint8_t> data;

    size_t size() const;
};

size_t TensorMetadata::size() const {
    return dims.total();
}

std::unique_ptr<TensorMetadata> parseTensorMetaData(const uint8_t* buffer, size_t& offset)
{
    auto tensor = std::make_unique<TensorMetadata>();
    auto name = parseGGUFString(buffer, offset);

    tensor->name = name;

    uint32_t ndims = *reinterpret_cast<const uint32_t*>(buffer + offset);
    offset += sizeof(uint32_t);

    std::vector<int> shape(ndims);

    // @TODO: Potential bug here, we are assuming that the shape is always int32_t
    for (uint64_t i = 0; i < ndims; ++i) {
        shape[i] = *reinterpret_cast<const int32_t*>(buffer + offset);
        offset += sizeof(int32_t);
    }

    tensor->dims = MatShape(shape);
    return tensor;    
}

struct GGUFImporter 
{ 
        GGUFImporter(const char *filename);

        // Layerwise construction 
        void add_output_layer(); //, size_t blkN);

        // Parsing infos and metadata
        void parseMetadata(size_t& offset);
        void parseHeader(size_t& offset);
        void parseTensorInfo(size_t& offset);

        Mat parseTensor(std::string name);

        // Metadata Storage which will be used during layer construction
        std::map<std::string, std::unique_ptr<MetadataKeyValueNode>> metadata;    
        std::map<std::string, std::unique_ptr<TensorMetadata>> tensorsMetadata;
        std::string filename;

        // GGUF file header definitions
        uint32_t version; 
        uint64_t tensor_count;
        uint32_t magic;
        uint64_t metadata_kv_count;
        
        // File buffer
        std::vector<uint8_t> buffer; 

        // Offsets for parsing (helper variables - should be wrapped into something more structured)
        size_t offset = 0;
        size_t tensor_data_ofset =  -1;
        size_t metadata_offset = 0;

        // Net impl stuff
        Net net;
        Net::Impl* netimpl;
        std::vector<Ptr<Layer> > prog;
};

GGUFImporter::GGUFImporter(const char *filename) {
    netimpl = net.getImpl();

    this->filename = filename;

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: ");
    }

    // Get the size of the file and prepare a buffer
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    buffer.resize(size);

    // Read the file content into the buffer
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Error reading file: " );
    }
}

Mat GGUFImporter::parseTensor(std::string name) {
    Mat tensor;

    std::unique_ptr<TensorMetadata>& tensorMetadata = tensorsMetadata[name];
    tensor.create(tensorMetadata->dims, CV_32F);

    size_t start = tensor_data_ofset + tensorMetadata->data_offset;
    size_t end = start + tensorMetadata->size() * sizeof(float);

    memcpy(tensor.data, buffer.data() + start, end - start);
    return tensor;
}

void GGUFImporter::parse_output_layer() // , size_t blkN) 
{   
    LayerParams layerParams;

    std::string weightKey = "blk.0.attn_qkv.weight";
    std::string biasKey = "blk.0.attn_qkv.bias";
    std::string inputKey = "blk.0.attn_qkv.input";
    std::string outputKey = "blk.0.attn_qkv.output";

    Mat weight = parseTensor(weightKey);
    Mat bias = parseTensor(biasKey);

    netimpl->newConstArg(weightKey, weight);
    netimpl->newConstArg(biasKey, bias);
    Arg input_arg = netimpl->newArg(inputKey, DNN_ARG_INPUT);
    Arg output_arg = netimpl->newArg(outputKey, DNN_ARG_OUTPUT);

    std::vector<Arg> inputs = {input_arg};

    Ptr<Graph> graph =  netimpl->newGraph("GGUFGraph", inputs, 1);
    
    layerParams.type = "Gemm";
    layerParams.blobs.push_back(weight);
    layerParams.blobs.push_back(bias);

    Ptr<Layer> layer = LayerFactory::createLayerInstance(layerParams.type, layerParams);
    layer->inputs = inputs;
    layer->outputs = {output_arg};
    layer->netimpl = netimpl;

    graph->setProg({layer});

    netimpl->mainGraph = graph;
    netimpl->modelFormat = DNN_MODEL_GENERIC;
    netimpl->originalLayout = DATA_LAYOUT_UNKNOWN;

    netimpl->prepareForInference();

    net.dumpToStream(std::cout);

}

// Ptr<Graph> GGUFImporter::createGraph(){
//     return 
// }

void GGUFImporter::parseHeader(size_t& offset) {
    //uint32_t magic = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);

    version = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);

    tensor_count = *reinterpret_cast<const uint64_t*>(buffer.data() + offset);
    offset += sizeof(uint64_t);

    metadata_kv_count = *reinterpret_cast<const uint64_t*>(buffer.data() + offset);
    offset += sizeof(uint64_t);
}

void GGUFImporter::parseMetadata(size_t& offset) {
    // Loop through and parse each key–value pair.
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        // Create a new metadata key–value node.
        auto kv = std::make_unique<MetadataKeyValueNode>();

        // Parse the key (stored as a GGUF string).
        kv->key = parseGGUFString(buffer.data(), offset);
        CV_LOG_DEBUG(NULL, "Parsing metadata key: " << kv->key);
        
        // Parse the value node (which will read its type and then the data).
        kv->value = parseMetadataValueNode(buffer.data(), offset);

        // Store the parsed key–value node in the metadata map.
        metadata[kv->key] = std::move(kv);
    }
}




Net parseFromGGUF(const char *filename) {

    GGUFImporter importer(filename);
    
    size_t offset = 0;
    importer.parseHeader(offset);
    importer.parseMetadata(offset);

    for (size_t i = 0; i < importer.tensor_count; ++i) {
        importer.parseTensorInfo(offset);
    }

    // set the offset 
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn