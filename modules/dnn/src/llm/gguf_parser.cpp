#include "../precomp.hpp"
#include "gguf_parser.hpp"
//#include "gguf_buffer.hpp"
#include <fstream>
#include "opencv2/core.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
using namespace dnn;

TensorMetadata parseTensorMetaData(GGUFBufferReader& reader) {
    auto tensor = TensorMetadata();
    tensor.name = reader.readString();
    tensor.dims = MatShape();
    int dim_count = reader.readSingleValueInt<uint32_t>();
    for (uint32_t i = 0; i < dim_count; ++i) {
        tensor.dims.push_back(reader.readSingleValueInt<int64_t>());
    }
    tensor.type = reader.readSingleValueInt<uint32_t>();;
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(tensor.type));
    }   
    tensor.type_size = sizeof(float);
    size_t tensor_data_offset = reader.readSingleValueInt<uint64_t>();
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

    // Header
    magic = reader.readSingleValue<uint32_t,uint32_t>();
    version = reader.readSingleValueInt<uint32_t>();
    tensor_count = reader.readSingleValueInt<uint64_t>();
    metadata_kv_count = reader.readSingleValueInt<uint64_t>();

    // Metadata
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        auto key = reader.readString();
        int type = reader.readSingleValueInt<uint32_t>();
        if (type == GGUF_METADATA_VALUE_TYPE_ARRAY) {
            // parse array
            int batch_length = reader.readSingleValueInt<uint32_t>();
            gguf_metadata_value_type array_type = (gguf_metadata_value_type)reader.readSingleValueInt<uint32_t>();
            switch (array_type) {
                case GGUF_METADATA_VALUE_TYPE_UINT8:
                case GGUF_METADATA_VALUE_TYPE_INT8:
                case GGUF_METADATA_VALUE_TYPE_UINT16:
                case GGUF_METADATA_VALUE_TYPE_INT16:
                case GGUF_METADATA_VALUE_TYPE_UINT32:
                case GGUF_METADATA_VALUE_TYPE_INT32:
                case GGUF_METADATA_VALUE_TYPE_INT64:
                case GGUF_METADATA_VALUE_TYPE_UINT64:
                case GGUF_METADATA_VALUE_TYPE_BOOL:
                    metadataDict.set(key, reader.readIntArray(batch_length, array_type));
                    break;
                case GGUF_METADATA_VALUE_TYPE_FLOAT64:
                case GGUF_METADATA_VALUE_TYPE_FLOAT32:
                    metadataDict.set(key, reader.readRealArray(batch_length, array_type));
                    break;
                case GGUF_METADATA_VALUE_TYPE_STRING:
                    metadataDict.set(key, reader.readStringArray(batch_length));
                    break;
                default:
                    throw std::runtime_error("Unsupported metadata array type: " + std::to_string(array_type));
            }
        } else {
            // parse single value
            metadataDict.set(key, reader.readSingleValue(type));
        }
    }

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
        "Unsupported tensor dimension: " + std::to_string(tensorMetadata.dims.size()));
};

std::string GGUFParser::get_architecture(){
    return getStringMetadata("architecture");
};

CV__DNN_INLINE_NS_END

}}