
#include "../precomp.hpp"
#include "gguf_importer.hpp"
#include "gguf_parser.hpp"
#include "gguf_def.hpp"
#include "../net_impl.hpp"
// #include <opencv2/core.hpp>


#include <fstream>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void ArchBlockConstructor::finalizeGraph() {
    graph->setProg(prog);
}


void VanillaArchBlockConstructor::initGraph(Net::Impl* netimpl) {
    // For now, we ignore automatic shape computation and require,
    // that input and output shape are stored in metadata

      // vanilla architecture blocks just consist of vanilla attention. 
    // hence input size is determined by the columns of the attention block
    MatShape inputshape(3);

    size_t k_hidden_size = ggufFile->getTypedMetadata<int32_t>(
        "k_hidden_size", GGUF_METADATA_VALUE_TYPE_INT32);
    size_t q_hidden_size = ggufFile->getTypedMetadata<int32_t>(
        "q_hidden_size", GGUF_METADATA_VALUE_TYPE_INT32);
    size_t v_hidden_size = ggufFile->getTypedMetadata<int32_t>(
        "v_hidden_size", GGUF_METADATA_VALUE_TYPE_INT32);

    inputshape[0] = 1;                                               // batch size
    inputshape[1] = -1;                                              // variable length
    inputshape[2] = k_hidden_size + q_hidden_size + v_hidden_size;   // qkv_hidden_size

    Arg input = netimpl->newArg("globInput",  DNN_ARG_INPUT );
    ArgData inputdata = netimpl->args.at(input.idx);
    inputdata.shape = inputshape;
    // @TODO set from a const from now to avoid multiple definitions
    inputdata.type = CV_32F;
    

    graph = netimpl->newGraph("VanillaAttention", {input}, true);

    // vanilla architecture blocks just consist of vanilla attention. 
    // hence input size is determined by the columns of the attention block
    MatShape outputshape(3);

    size_t output_size = ggufFile->getTypedMetadata<int32_t>(
        "output_size", GGUF_METADATA_VALUE_TYPE_INT32);
    
    outputshape[0] = 1;                                               // batch size
    outputshape[1] = -1;                                              // variable length
    outputshape[2] = output_size;   // qkv_hidden_size

    Arg output = netimpl->newArg("globOut",  DNN_ARG_OUTPUT );
    ArgData outputdata = netimpl->args.at(output.idx);
    outputdata.shape = outputshape;

    graph->setOutputs({output});
}


void VanillaArchBlockConstructor::AddAttentionBlock(Net::Impl* netimpl, int blockn) {
    // Add attention block to the net
    LayerParams layerParams;
    layerParams.type = "Attention";
    layerParams.name = "attention_block";
    Mat weight = ggufFile->getTensor("blk." + std::to_string(blockn) + ".attn_qkv.weight").t(); // opencv requires weight to be of shpae D_emb 
    Mat bias = ggufFile->getTensor("blk." + std::to_string(blockn) + ".attn_qkv.bias");
    int k_hidden_size = ggufFile->getTypedMetadata<int32_t>(
        "k_hidden_size", GGUF_METADATA_VALUE_TYPE_INT32);
    int v_hidden_size = ggufFile->getTypedMetadata<int32_t>(
        "v_hidden_size", GGUF_METADATA_VALUE_TYPE_INT32);
    int q_hidden_size = ggufFile->getTypedMetadata<int32_t>(
        "q_hidden_size", GGUF_METADATA_VALUE_TYPE_INT32);
    int num_heads = ggufFile->getTypedMetadata<int32_t>(
        "num_heads", GGUF_METADATA_VALUE_TYPE_INT32);
    
    MatShape weightshape = weight.shape();
    int hidden_size = weightshape[1];
    
    std::vector<int> qkv_hidden_sizes = {k_hidden_size, q_hidden_size, v_hidden_size};
    layerParams.set(
        "qkv_hidden_sizes",
        DictValue::arrayInt(&qkv_hidden_sizes[0], 3)
    );
    layerParams.set("num_heads", num_heads);
    // zero bias for now
    layerParams.blobs.push_back(weight);
    layerParams.blobs.push_back(bias);

    std::string inputName, outputName;
    if (blockn == 0){
        inputName = "globInput";
    } else {
        inputName = "blk." + std::to_string(blockn - 1) + ".attn_out";
    }

    size_t num_blocks = ggufFile->getTypedMetadata<int32_t>("num_blocks", GGUF_METADATA_VALUE_TYPE_INT32);

    if (blockn >= num_blocks - 1){
        outputName = "globOut";
    } else {
        outputName = "blk." + std::to_string(blockn) + ".attn_out";
    }

    if(inputName.empty() || outputName.empty()){
        CV_Error(Error::StsBadArg, "VanillaArchBlockConstructor: input or output name for a block is empty");
    }

    /////////////////////////////////
    // Set proper ArgData for input
    /////////////////////////////////    
    Arg input = netimpl->getArg(inputName);
    ArgData& inputAdata = netimpl->args.at(input.idx);
    // Infer shape
    inputAdata.shape.resize(3);
    inputAdata.shape[0] = -1; // batch size
    inputAdata.shape[1] = -1; // variable length
    inputAdata.shape[2] = hidden_size;
    inputAdata.type = CV_32F;


    /////////////////////////////////
    // Set proper ArgData for output
    /////////////////////////////////   
    Arg output =  netimpl->getArg(outputName); 
    ArgData& outputAdata = netimpl->args.at(input.idx);
    // Infer shape
    outputAdata.shape.resize(3);
    outputAdata.shape[0] = -1; // batch size
    outputAdata.shape[1] = -1; // variable length
    outputAdata.shape[2] = hidden_size;
    outputAdata.type = CV_32F;

    Ptr<Layer> layer = LayerFactory::createLayerInstance(layerParams.type, layerParams);
    layer->netimpl = netimpl;
    layer->inputs = {input};
    layer->outputs = {output};
    prog.push_back(layer);
}

Net GGUFImporter::constructNet() {
    VanillaArchBlockConstructor archBlockConstructor(ggufFile);
    archBlockConstructor.initGraph(netimpl);
    size_t num_blocks = ggufFile->getTypedMetadata<int32_t>("num_blocks", GGUF_METADATA_VALUE_TYPE_INT32);
    for (int i = 0; i < num_blocks; i++){
        archBlockConstructor.AddAttentionBlock(netimpl, i);
    }
    archBlockConstructor.finalizeGraph();

    netimpl->prepareForInference();
    return net;
}


Net readNetFromGGUF(const String& ggufFileName){
    GGUFImporter importer(ggufFileName);
    return importer.constructNet();
}

CV__DNN_INLINE_NS_END

}}