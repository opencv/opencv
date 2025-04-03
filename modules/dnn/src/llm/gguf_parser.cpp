#include "../precomp.hpp"
#include "gguf_parser.hpp"
//#include "gguf_buffer.hpp"
#include "gguf_def.hpp"
#include <fstream>
#include "opencv2/core.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
using namespace dnn;

std::string parseGGUFString(GGUFBufferReader& reader)
{
    std::string result;
    
    uint64_t len = reader.readSingleValue<uint64_t>();
    return reader.readString(len);
    // result.assign(reinterpret_cast<const char*>(buffer + offset), len);
    // offset = offset + sizeof(char) * len;
    
    // return result;
}

Ptr<MetadataValueNode> parseMetadataSingleValueNode(GGUFBufferReader& reader, uint32_t type)
{
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:
            return makePtr<MetadataSingleValueNode<uint8_t>>(
                reader.readSingleValue<uint8_t>(), 
                GGUF_METADATA_VALUE_TYPE_UINT8
            );

        case GGUF_METADATA_VALUE_TYPE_INT8:
            return makePtr<MetadataSingleValueNode<int8_t>>(
                reader.readSingleValue<int8_t>(), 
                GGUF_METADATA_VALUE_TYPE_INT8
            );

        case GGUF_METADATA_VALUE_TYPE_UINT16:
            return makePtr<MetadataSingleValueNode<uint16_t>>(
                reader.readSingleValue<uint16_t>(), 
                GGUF_METADATA_VALUE_TYPE_UINT16
            );

        case GGUF_METADATA_VALUE_TYPE_INT16:
            return makePtr<MetadataSingleValueNode<int16_t>>(
                reader.readSingleValue<int16_t>(), 
                GGUF_METADATA_VALUE_TYPE_INT16
            );

        case GGUF_METADATA_VALUE_TYPE_UINT32:
            return makePtr<MetadataSingleValueNode<uint32_t>>(
                reader.readSingleValue<uint32_t>(), 
                GGUF_METADATA_VALUE_TYPE_UINT32
            );

        case GGUF_METADATA_VALUE_TYPE_INT32:
            return makePtr<MetadataSingleValueNode<int32_t>>(
                reader.readSingleValue<int32_t>(), 
                GGUF_METADATA_VALUE_TYPE_INT32
            );

        case GGUF_METADATA_VALUE_TYPE_UINT64:
            return makePtr<MetadataSingleValueNode<uint64_t>>(
                reader.readSingleValue<uint64_t>(), 
                GGUF_METADATA_VALUE_TYPE_UINT64
            );

        case GGUF_METADATA_VALUE_TYPE_INT64:
            return makePtr<MetadataSingleValueNode<int64_t>>(
                reader.readSingleValue<int64_t>(), 
                GGUF_METADATA_VALUE_TYPE_INT64
            );

        case GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return makePtr<MetadataSingleValueNode<float>>(
                reader.readSingleValue<float>(), 
                GGUF_METADATA_VALUE_TYPE_FLOAT32
            );

        case GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return makePtr<MetadataSingleValueNode<double>>(
                reader.readSingleValue<double>(), 
                GGUF_METADATA_VALUE_TYPE_FLOAT64
            );

        case GGUF_METADATA_VALUE_TYPE_BOOL: {
            uint8_t raw = reader.readSingleValue<uint8_t>();
            return makePtr<MetadataSingleValueNode<bool>>(
                raw != 0, 
                GGUF_METADATA_VALUE_TYPE_BOOL
            );
        }

        case GGUF_METADATA_VALUE_TYPE_STRING: {
            std::string str = parseGGUFString(reader);
            return makePtr<MetadataSingleValueNode<std::string>>(str, GGUF_METADATA_VALUE_TYPE_STRING);
        }

        default:
            throw std::runtime_error("Unknown metadata value type ID: " + std::to_string(type));
    }
}

Ptr<MetadataValueNode> parseMetadataValueNode(GGUFBufferReader& reader) {
    uint32_t type = reader.readSingleValue<uint32_t>();

    if (type != GGUF_METADATA_VALUE_TYPE_ARRAY) {
        return parseMetadataSingleValueNode(reader, type);
    }

    uint32_t elementType = reader.readSingleValue<uint32_t>();
    uint64_t count = reader.readSingleValue<uint64_t>();

    auto arrayNode = makePtr<MetadataArrayNode>(static_cast<gguf_metadata_value_type>(elementType));
    for (uint64_t i = 0; i < count; ++i) {
        arrayNode->array.push_back(
            parseMetadataSingleValueNode(reader, elementType)
        );
    }

    return arrayNode;
}

TensorMetadata parseTensorMetaData(GGUFBufferReader& reader) {
    auto tensor = TensorMetadata();

    tensor.name = parseGGUFString(reader);
    tensor.dims = MatShape();

    uint32_t dim_count = reader.readSingleValue<uint32_t>();

    for (uint32_t i = 0; i < dim_count; ++i) {
        tensor.dims.push_back( reader.readSingleValue<int64_t>());
    }

    tensor.type = reader.readSingleValue<uint32_t>();;
    // now we support only float32
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(tensor.type));
    }   

    tensor.type_size = sizeof(float);

    size_t tensor_data_offset = reader.readSingleValue<uint64_t>();
    tensor.data_offset = tensor_data_offset;

    return tensor;
}

size_t TensorMetadata::size() const {
    size_t size = 1;
    for (const auto& dim : dims) {
        size *= dim;
    }
    return size * type_size;
}

GGUFParser::GGUFParser(const String& ggufFileName) {
    buffer = makePtr<GGUFBuffer>(ggufFileName);
    GGUFBufferReader reader(buffer);
    magic = reader.readSingleValue<uint32_t>();
    version = reader.readSingleValue<uint32_t>();
    tensor_count = reader.readSingleValue<uint64_t>();
    metadata_kv_count = reader.readSingleValue<uint64_t>();

    // Loop through and parse each key–value pair.
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        // Create a new metadata key–value node.
        Ptr<MetadataKeyValueNode> kv = makePtr<MetadataKeyValueNode>();
        // Parse the key (stored as a GGUF string).
        kv->key = parseGGUFString(reader);
        // Parse the value node (which will read its type and then the data).
        kv->value = parseMetadataValueNode(reader);
        // Store the parsed key–value node in the metadata map.
        metadata[kv->key] = kv;
    }

    num_blocks = getTypedMetadata<int32_t>("num_blocks", GGUF_METADATA_VALUE_TYPE_INT32);

    for(size_t t = 0; t < tensor_count; ++t) {
        auto tensor = parseTensorMetaData(reader);
        tensorsMetadata[tensor.name] = tensor;
    }

    tensor_reader = makePtr<GGUFBufferReader>(buffer);
    tensor_reader->current_offset = reader.current_offset;
};

Mat GGUFParser::getTensor(std::string name) {
    Mat tensor;
    TensorMetadata tensorMetadata = tensorsMetadata[name];
    if (tensorMetadata.type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(tensorMetadata.type));
    }
    if (tensorMetadata.dims.size() == 2) 
        return tensor_reader->read2DMat(
            GGML_TYPE_F32,
            tensorMetadata.dims[0],
            tensorMetadata.dims[1],
            tensorMetadata.data_offset
        );
    if (tensorMetadata.dims.size() == 1) 
        return tensor_reader->read1DMat(
            GGML_TYPE_F32,
            tensorMetadata.dims[0],
            tensorMetadata.data_offset
        );

    throw std::runtime_error(
        "Unsupported tensor dimension: " + std::to_string(tensorMetadata.dims.size()));xw
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




std::string GGUFParser::get_architecture(){
    return getStringMetadata("architecture");
};

CV__DNN_INLINE_NS_END

}}