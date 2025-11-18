#ifndef __OPENCV_GGUFPARSER_HPP__
#define __OPENCV_GGUFPARSER_HPP__

#include <string>
#include <map>
#include "opencv2/core.hpp"
#include <regex>
// #include <opencv2/core.hpp>
// #include <opencv2/dnn.hpp>
#include "gguf_buffer.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using namespace cv::dnn;

enum tensor_role : uint32_t{
    attn_norm = 0,
    attn_norm_2 = 1,
    attn_qkv = 2,
    attn_q = 3,
    attn_k = 4,
    attn_v = 5,
    attn_output = 6,
    ffn_norm = 7,
    ffn_up = 8,
    ffn_gate = 9,
    ffn_down = 10,
    ffn_gate_inp = 11,
    ffn_gate_exp = 12,
    ffn_down_exp = 13,
    ffn_up_exp = 14
};

tensor_role get_tensor_role(std::string name);

struct TensorMetadata {
    std::string name;
    MatShape dims;
    size_t data_offset;
    uint32_t type;
    uint32_t type_size;
    int n_block;
    tensor_role role;
    size_t size() const;
    bool is_bias=false;

    Mat getMat(Ptr<GGUFBufferReader> tensor_reader);
};

/* 
* Block is a collection of tensors that are used in in one "block"
* The tensors are grouped by their role in the layer.
* The blockn is the index of the block in the model.
* Block can have different layout depending on architecture
* currently, we support blocks whcih consist only of attention layer
*/
struct BlockMetadata {
    BlockMetadata(int n_blocks, int n_heads) : blockn(n_blocks), n_heads(n_heads) {}
    MatShape getInputShape();
    MatShape getOutputShape();
    std::vector<TensorMetadata> getTensorMetadata(tensor_role role, bool is_bias, bool allow_multiple = false);
    void getAttentionLayerParams(Ptr<GGUFBufferReader> tensor_reader, LayerParams& layerParams);
    int getDhidden();
    int blockn;
    int n_heads;
    std::vector<TensorMetadata> tensors;
};

TensorMetadata parseTensorMetaData(GGUFBufferReader& reader);

struct GGUFParser 
{ 
    GGUFParser(const String& ggufFileName);
    // helpers which will be used in GGUFImporter
    Mat getTensor(std::string name);
    std::string get_architecture();
    std::string getStringMetadata(std::string key);
    
    void addTensorMetadata();
    
    std::vector<BlockMetadata> blocks;
    Dict metadataDict;

    // GGUF file header definitions
    uint32_t version; 
    uint64_t tensor_count;
    uint32_t magic;
    uint64_t metadata_kv_count;
    
    // File buffer
    Ptr<GGUFBuffer> buffer;
    GGUFBufferReader reader;
    Ptr<GGUFBufferReader> tensor_reader;
};

CV__DNN_INLINE_NS_END
}}

#endif