#ifndef __OPENCV_GGUFPARSER_HPP__
#define __OPENCV_GGUFPARSER_HPP__

#include <string>
#include <map>
#include "opencv2/core.hpp"
// #include <opencv2/core.hpp>
// #include <opencv2/dnn.hpp>
#include "gguf_def.hpp"
#include "gguf_buffer.hpp"


namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using namespace cv::dnn;

std::string parseGGUFString(GGUFBufferReader& reader);

struct MetadataValueNode{
    virtual ~MetadataValueNode() = default;
    gguf_metadata_value_type valueType;

    protected:
        MetadataValueNode(gguf_metadata_value_type type) : valueType(type) {}
};

struct MetadataKeyValueNode
{
    std::string key;
    Ptr<MetadataValueNode> value;
};

template<typename T>
struct MetadataSingleValueNode : public MetadataValueNode
{
    T value;    
    MetadataSingleValueNode(const T& val, gguf_metadata_value_type type): MetadataValueNode(type), value(val) {}
    // Add a constructor that initializes 'value'
    explicit MetadataSingleValueNode(const T &v) : value(v) {}

};

struct MetadataArrayNode : public MetadataValueNode
{
    std::vector<Ptr<MetadataValueNode>> array;
    gguf_metadata_value_type elementType;

    MetadataArrayNode(gguf_metadata_value_type elemType)
        : MetadataValueNode(GGUF_METADATA_VALUE_TYPE_ARRAY), elementType(elemType) {}
};

Ptr<MetadataValueNode> parseMetadataSingleValueNode(GGUFBufferReader& reader, uint32_t type);
Ptr<MetadataValueNode> parseMetadataValueNode(GGUFBufferReader& reader);

struct TensorMetadata {
    std::string name;
    MatShape dims;
    size_t data_offset;
    uint32_t type;
    uint32_t type_size;

    size_t size() const;
};

TensorMetadata parseTensorMetaData(GGUFBufferReader& reader);

struct GGUFParser 
{ 
    GGUFParser(const String& ggufFileName);

    // helpers which will be used in GGUFImporter
    Mat getTensor(std::string name);
    std::string get_architecture();
    std::string getStringMetadata(std::string key);
    
    template<typename T>
    T getTypedMetadata(const std::string& key, gguf_metadata_value_type expectedType);

    // Metadata Storage which will be used during layer construction
    std::map<std::string, Ptr<MetadataKeyValueNode>> metadata;    
    std::map<std::string, TensorMetadata> tensorsMetadata;

    // GGUF file header definitions
    uint32_t version; 
    uint64_t tensor_count;
    uint32_t magic;
    uint64_t metadata_kv_count;
    
    // File buffer
    Ptr<GGUFBuffer> buffer;
    Ptr<GGUFBufferReader> tensor_reader;

};


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


std::string parseGGUFString(GGUFBufferReader& reader);


CV__DNN_INLINE_NS_END
}}

#endif