#ifndef __OPENCV_GGUFPARSER_HPP__
#define __OPENCV_GGUFPARSER_HPP__

#include <string>
#include <map>
#include "opencv2/core.hpp"
// #include <opencv2/core.hpp>
// #include <opencv2/dnn.hpp>
#include "gguf_buffer.hpp"


namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using namespace cv::dnn;

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
    
    std::map<std::string, TensorMetadata> tensorsMetadata;
    Dict metadataDict;

    // GGUF file header definitions
    uint32_t version; 
    uint64_t tensor_count;
    uint32_t magic;
    uint64_t metadata_kv_count;
    
    // File buffer
    Ptr<GGUFBuffer> buffer;
    Ptr<GGUFBufferReader> tensor_reader;

};



CV__DNN_INLINE_NS_END
}}

#endif