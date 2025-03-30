#include "gguf_parser.hpp"
#include "gguf_def.hpp"
#include <fstream>
#include "opencv2/core.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
using namespace dnn;

std::string parseGGUFString(const uint8_t* buffer, size_t& offset)
{
    std::string result;
    
    uint64_t len = *reinterpret_cast<const uint64_t*>(buffer + offset);
    offset += sizeof(uint64_t);

    result.assign(reinterpret_cast<const char*>(buffer + offset), len);
    offset = offset + sizeof(char) * len;
    
    return result;
}

Ptr<MetadataValueNode> parseMetadataSingleValueNode(const uint8_t* buffer, size_t& offset, uint32_t type)
{
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8: {
            uint8_t val = *(buffer + offset);
            offset += sizeof(uint8_t);
            return makePtr<MetadataSingleValueNode<uint8_t>>(val, GGUF_METADATA_VALUE_TYPE_UINT8);
        }
        case GGUF_METADATA_VALUE_TYPE_INT8: {
            int8_t val = *reinterpret_cast<const int8_t*>(buffer + offset);
            offset += sizeof(int8_t);
            return makePtr<MetadataSingleValueNode<int8_t>>(val, GGUF_METADATA_VALUE_TYPE_INT8);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT16: {
            uint16_t val = *reinterpret_cast<const uint16_t*>(buffer + offset);
            offset += sizeof(uint16_t);
            return makePtr<MetadataSingleValueNode<uint16_t>>(val, GGUF_METADATA_VALUE_TYPE_UINT16);
        }
        case GGUF_METADATA_VALUE_TYPE_INT16: {
            int16_t val = *reinterpret_cast<const int16_t*>(buffer + offset);
            offset += sizeof(int16_t);
            return makePtr<MetadataSingleValueNode<int16_t>>(val, GGUF_METADATA_VALUE_TYPE_INT16);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT32: {
            uint32_t val = *reinterpret_cast<const uint32_t*>(buffer + offset);
            offset += sizeof(uint32_t);
            return makePtr<MetadataSingleValueNode<uint32_t>>(val, GGUF_METADATA_VALUE_TYPE_UINT32);
        }
        case GGUF_METADATA_VALUE_TYPE_INT32: {
            int32_t val = *reinterpret_cast<const int32_t*>(buffer + offset);
            offset += sizeof(int32_t);
            return makePtr<MetadataSingleValueNode<int32_t>>(val, GGUF_METADATA_VALUE_TYPE_INT32);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64: {
            uint64_t val = *reinterpret_cast<const uint64_t*>(buffer + offset);
            offset += sizeof(uint64_t);
            return makePtr<MetadataSingleValueNode<uint64_t>>(val, GGUF_METADATA_VALUE_TYPE_UINT64);
        }
        case GGUF_METADATA_VALUE_TYPE_INT64: {
            int64_t val = *reinterpret_cast<const int64_t*>(buffer + offset);
            offset += sizeof(int64_t);
            return makePtr<MetadataSingleValueNode<int64_t>>(val, GGUF_METADATA_VALUE_TYPE_INT64);
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
            float val = *reinterpret_cast<const float*>(buffer + offset);
            offset += sizeof(float);
            return makePtr<MetadataSingleValueNode<float>>(val, GGUF_METADATA_VALUE_TYPE_FLOAT32);
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
            double val = *reinterpret_cast<const double*>(buffer + offset);
            offset += sizeof(double);
            return makePtr<MetadataSingleValueNode<double>>(val, GGUF_METADATA_VALUE_TYPE_FLOAT64);
        }
        case GGUF_METADATA_VALUE_TYPE_BOOL: {
            uint8_t val = *(buffer + offset);
            offset += sizeof(uint8_t);
            bool boolVal = (val != 0);
            return makePtr<MetadataSingleValueNode<bool>>(boolVal, GGUF_METADATA_VALUE_TYPE_BOOL);
        }
        case GGUF_METADATA_VALUE_TYPE_STRING: {
            std::string str = parseGGUFString(buffer, offset);
            return makePtr<MetadataSingleValueNode<std::string>>(str, GGUF_METADATA_VALUE_TYPE_STRING);
        }
        default:
            throw std::runtime_error("Unknown metadata value type ID: " + std::to_string(type));
    }
}

Ptr<MetadataValueNode> parseMetadataValueNode(const uint8_t* buffer, size_t& offset) {
    uint32_t type = *reinterpret_cast<const uint32_t*>(buffer + offset);
    offset += sizeof(uint32_t);

    if (type != GGUF_METADATA_VALUE_TYPE_ARRAY) {
        return parseMetadataSingleValueNode(buffer, offset, type);
    }

    uint32_t elementType = *reinterpret_cast<const uint32_t*>(buffer + offset);
    offset += sizeof(uint32_t);

    uint64_t count = *reinterpret_cast<const uint64_t*>(buffer + offset);
    offset += sizeof(uint64_t);

    auto arrayNode = makePtr<MetadataArrayNode>(static_cast<gguf_metadata_value_type>(elementType));
    for (uint64_t i = 0; i < count; ++i) {
        arrayNode->array.push_back(parseMetadataSingleValueNode(buffer, offset, elementType));
    }

    return arrayNode;
}


void GGUFParser::prepareFile(const char *filename) {
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
    parseTensorInfo(offset);
};

void GGUFParser::parseHeader(size_t& offset) {
    //uint32_t magic = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);

    version = *reinterpret_cast<const uint32_t*>(buffer.data() + offset);
    offset += sizeof(uint32_t);

    tensor_count = *reinterpret_cast<const uint64_t*>(buffer.data() + offset);
    offset += sizeof(uint64_t);

    metadata_kv_count = *reinterpret_cast<const uint64_t*>(buffer.data() + offset);
    offset += sizeof(uint64_t);
};

void GGUFParser::parseMetadata(size_t& offset) {
    // Loop through and parse each key–value pair.
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        // Create a new metadata key–value node.
        auto kv = Ptr<MetadataKeyValueNode>();

        // Parse the key (stored as a GGUF string).
        kv->key = parseGGUFString(buffer.data(), offset);
        
        // Parse the value node (which will read its type and then the data).
        kv->value = parseMetadataValueNode(buffer.data(), offset);

        // Store the parsed key–value node in the metadata map.
        metadata[kv->key] = std::move(kv);
    }
}

void GGUFParser::parseTensorInfo(size_t& offset) {
    for(size_t t = 0; t < tensor_count; ++t) {
        auto tensor = parseTensorMetaData(buffer.data(), offset);
        tensorsMetadata[tensor->name] = std::move(tensor);
    }

    auto tensor = parseTensorMetaData(buffer.data(), offset);
    tensorsMetadata[tensor->name] = std::move(tensor);

    // @TODO: this is dangerous. Offset should be set in a more clear way.
    tensor_data_ofset = offset;
};

Mat GGUFParser::getTensor(std::string name) {
    Mat tensor;

    Ptr<TensorMetadata> tensorMetadata = tensorsMetadata[name];
    tensor.create(tensorMetadata->dims, CV_32F);

    size_t start = tensor_data_ofset + tensorMetadata->data_offset;
    size_t end = start + tensorMetadata->size() * sizeof(float);

    memcpy(tensor.data, buffer.data() + start, end - start);
    return tensor;
};

std::string GGUFParser::getStringMetadata(const std::string key) {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("Key not found in metadata: " + key);
    }

    MetadataValueNode* valueNode = it->second->value.get();
    if(valueNode->valueType != GGUF_METADATA_VALUE_TYPE_STRING) {
        throw std::runtime_error("Value is not a string type for key: " + key);
    }

    auto* singleValueNode =
        static_cast<MetadataSingleValueNode<std::string>*>(valueNode);
    
    return singleValueNode->value;
}

template<typename T>
T GGUFParser::getTypedMetadata(const std::string& key,
                               gguf_metadata_value_type expectedType)
{
    // 1. Check if key exists in the map
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        throw std::runtime_error("Key not found in metadata: " + key);
    }

    // 2. Check that node type matches what we expect
    MetadataValueNode* valueNode = it->second->value.get();
    if (valueNode->valueType != expectedType) {
        throw std::runtime_error(
            "Value type mismatch for key '" + key + "'. Expected: " +
            std::to_string(expectedType) + ", Found: " +
            std::to_string(valueNode->valueType)
        );
    }

    // 3. Safely cast to the correct single-value node type
    auto* singleValueNode = static_cast<MetadataSingleValueNode<T>*>(valueNode);

    // 4. Return the stored value
    return singleValueNode->value;
}


std::string GGUFParser::get_architecture(){
    return getStringMetadata("architecture");
};

CV__DNN_INLINE_NS_END

}}