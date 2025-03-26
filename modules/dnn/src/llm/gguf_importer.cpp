#include "../precomp.hpp"
#include "../net_impl.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "gguf_def.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


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

void parseMetadataKeyValuePair(const uint8_t* buffer, size_t offset, MetadataKeyValueNode& result)
{
    // Parse key
    result.key = parseGGUFString(buffer, offset);
    // offset += sizeof(uint64_t) + result.key.len;
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


class GGUFImporter 
{
    public:
        GGUFImporter();
        void parseFile(const char *filename);

    protected:
        void parse_attn_qkv(LayerParams& layerParams, size_t blkN);

        void addLayer(LayerParams& layerParams, int num_inputs);

        //Ptr<Graph> createGraph();

        void parseMetadata(size_t& offset);
        void parseHeader(size_t& offset);
        void parseTensorInfo(size_t& offset);

        Mat parseTensor(std::string name);

        std::map<std::string, std::unique_ptr<MetadataKeyValueNode>> metadata;    
        std::map<std::string, std::unique_ptr<TensorMetadata>> tensorsMetadata;
        std::string filename;

        uint32_t version; 
        uint64_t tensor_count;
        uint32_t magic;
        uint64_t metadata_kv_count;
        
        std::vector<uint8_t> buffer; 

        size_t n_tensors;
        size_t offset = 0;
        size_t tensor_data_ofset =  -1;
        size_t metadata_offset = 0;

        Net net;
        Net::Impl* netimpl;
        std::vector<Ptr<Layer> > prog;
};

GGUFImporter::GGUFImporter() {
    netimpl = net.getImpl();
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

// for now this is the test method
void GGUFImporter::parse_attn_qkv(LayerParams& layerParams, size_t blkN) 
{   
    std::string weightKey = "blk." + std::to_string(blkN) + ".attn_qkv.weight";
    std::string biasKey = "blk." + std::to_string(blkN) + ".attn_qkv.bias";
    std::string inputKey = "blk." + std::to_string(blkN) + ".attn_qkv.input";
    std::string outputKey = "blk." + std::to_string(blkN) + ".attn_qkv.output";

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

        // Parse the value node (which will read its type and then the data).
        kv->value = parseMetadataValueNode(buffer.data(), offset);

        // Store the parsed key–value node in the metadata map.
        metadata[kv->key] = std::move(kv);
    }
}


void GGUFImporter::parseFile(const char *filename) {
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

    size_t offset = 0;
    parseHeader(offset);
    parseMetadata(offset);

    for (size_t i = 0; i < tensor_count; ++i) {
        parseTensorInfo(offset);
    }

    // set the offset 
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn