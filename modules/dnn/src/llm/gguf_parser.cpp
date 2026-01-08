#include "../precomp.hpp"
#include "gguf_parser.hpp"
//#include "gguf_buffer.hpp"
#include <fstream>
#include "opencv2/core.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN
using namespace dnn;

tensor_role get_tensor_role(std::string name) {
    if (name == "attn_norm") return tensor_role::attn_norm;
    if (name == "attn_norm_2") return tensor_role::attn_norm_2;
    if (name == "attn_qkv") return tensor_role::attn_qkv;
    if (name == "attn_q") return tensor_role::attn_q;
    if (name == "attn_k") return tensor_role::attn_k;
    if (name == "attn_v") return tensor_role::attn_v;
    if (name == "attn_output") return tensor_role::attn_output;
    if (name == "ffn_norm") return tensor_role::ffn_norm;
    if (name == "ffn_up") return tensor_role::ffn_up;
    if (name == "ffn_gate") return tensor_role::ffn_gate;
    if (name == "ffn_down") return tensor_role::ffn_down;
    if (name == "ffn_gate_inp") return tensor_role::ffn_gate_inp;
    if (name == "ffn_gate_exp") return tensor_role::ffn_gate_exp;
    if (name == "ffn_down_exp") return tensor_role::ffn_down_exp;
    if (name == "ffn_up_exp") return tensor_role::ffn_up_exp;
    throw std::runtime_error("Unknown tensor role: " + name);
}

Mat TensorMetadata::getMat(Ptr<GGUFBufferReader> tensor_reader) {
    if (type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(type));
    }
    if (dims.size() == 2) 
        return tensor_reader->read2DMat(
            GGML_TYPE_F32, dims[0],dims[1], data_offset
        );
    if (dims.size() == 1) 
        return tensor_reader->read1DMat(
            GGML_TYPE_F32, dims[0], data_offset
        );

    throw std::runtime_error(
        "Unsupported tensor dimension: " + std::to_string(dims.size()));
};

std::string GGUFParser::get_architecture(){
    return getStringMetadata("architecture");
};

MatShape BlockMetadata::getInputShape() 
{
    std::vector<TensorMetadata> t = getTensorMetadata( tensor_role::attn_qkv, false, false);
    if (t.size() == 0) {
        throw std::runtime_error("No input tensors found");
    }

    MatShape inputShape(3);
    inputShape[0] = 1; inputShape[1] = -1;
    inputShape[2] = t[0].dims[1];
    return inputShape;
}

int BlockMetadata::getDhidden() 
{
    std::vector<TensorMetadata> t = getTensorMetadata( tensor_role::attn_qkv, false, false);
    if (t.size() == 0) {
        throw std::runtime_error("No input tensors found");
    }
    return t[0].dims[0] / 3;
}

MatShape BlockMetadata::getOutputShape() 
{
    std::vector<TensorMetadata> t = getTensorMetadata( tensor_role::attn_qkv, false, false);
    if (t.size() == 0) {
        throw std::runtime_error("No input tensors found");
    }

    MatShape outputShape(3);
    outputShape[0] = 1; outputShape[1] = -1;
    outputShape[2] = t[0].dims[1];
    return outputShape;
}

void BlockMetadata::getAttentionLayerParams(Ptr<GGUFBufferReader> tensor_reader, LayerParams& layerParams) 
{
    layerParams.type = "Attention";
    layerParams.name = "attention_block";
    layerParams.set("num_heads", n_heads);
    layerParams.set("blockn", blockn);
    int d_hidden = getDhidden();
    std::vector<int> qkv_hidden_sizes = {d_hidden, d_hidden, d_hidden};
    layerParams.set("qkv_hidden_sizes", DictValue::arrayInt(&qkv_hidden_sizes[0], 3));
    Mat qkv_weight = getTensorMetadata(tensor_role::attn_qkv, false, false)[0].getMat(tensor_reader);
    Mat qkv_bias = getTensorMetadata(tensor_role::attn_qkv, true, false)[0].getMat(tensor_reader);
    layerParams.blobs.push_back(qkv_weight.t());
    layerParams.blobs.push_back(qkv_bias);
}

std::vector<TensorMetadata> BlockMetadata::getTensorMetadata(tensor_role role, bool is_bias, bool allow_multiple) {
    std::vector<TensorMetadata> result;
    for (const auto & tensor : tensors) {
        if (tensor.role == role && tensor.is_bias == is_bias) {
            result.push_back(tensor);
        }
    }
    return result;
    if (!allow_multiple && result.size() > 1) {
        throw std::runtime_error("Multiple tensors found for role: " + std::to_string(role));
    }
    return result;
}

void GGUFParser::addTensorMetadata() {
    auto tensor = TensorMetadata();

    std::string tensor_name = reader.readString();
    std::regex layerRegex(R"(^blk\.(\d+)\.([a-zA-Z0-9_]+)\.(weight|bias)$)");
    std::smatch match;
    std::string layerName, paramType;

    int blockn;

    if (std::regex_match(tensor_name, match, layerRegex)) {
        blockn = std::stoi(match[1].str());
        layerName = match[2].str();
        paramType = match[3].str();
        if (paramType == "weight") {
            tensor.is_bias = false;
        } else if (paramType == "bias") {
            tensor.is_bias = true;
        } else {
            throw std::runtime_error("Unknown parameter type: " + paramType);
        }
        tensor.role = get_tensor_role(layerName);
        tensor.n_block = blockn;
    } else {
        throw std::runtime_error("Invalid tensor name format: " + tensor_name);
    }

    int dim_count = reader.readSingleValueInt<uint32_t>();
    tensor.dims = MatShape(dim_count);
    std::vector<int> dims;
    for (uint32_t i = 0; i < dim_count; ++i) {
        dims.push_back(reader.readSingleValueInt<int64_t>());    
    }
    // GGUF stores dims in reversed order 
    for (int i = 0; i < dim_count; ++i) {
        tensor.dims[i] = dims[dim_count - 1- i];
    }

    tensor.type = reader.readSingleValueInt<uint32_t>();;
    if (tensor.type != GGML_TYPE_F32) {
        throw std::runtime_error("Unsupported tensor type: " + std::to_string(tensor.type));
    }   
    tensor.type_size = sizeof(float);
    size_t tensor_data_offset = reader.readSingleValueInt<uint64_t>();
    tensor.data_offset = tensor_data_offset;

    if (blockn + 1 > blocks.size()) {
        int current_blockn = blocks.size();
        // @TODO this should be done more flexibly, eg GGUF allows [llm].attention.head_count and [llm].attention.head_count_kv 
        int n_heads = metadataDict.get("num_heads", 0);
        for (int i = current_blockn; i <= blockn; ++i) {
            blocks.push_back(BlockMetadata(i, n_heads));
        }
    }

    blocks[blockn].tensors.push_back(tensor);
}

size_t TensorMetadata::size() const {
    size_t size = 1;
    for (const auto& dim : dims) {
        size *= dim;
    }
    return size * type_size;
}

GGUFParser::GGUFParser(const String& ggufFileName) : 
buffer(makePtr<GGUFBuffer>(ggufFileName)) , reader(buffer)  
{
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

    for(size_t t = 0; t < tensor_count; ++t)
        addTensorMetadata();
    
    tensor_reader = makePtr<GGUFBufferReader>(buffer);
    tensor_reader->current_offset = reader.current_offset;
};

CV__DNN_INLINE_NS_END

}}