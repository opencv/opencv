
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

    size_t k_hidden_size = ggufFile->getTypedMetadata<size_t>(
        "k_hidden_size", GGUF_METADATA_VALUE_TYPE_UINT32);
    size_t q_hidden_size = ggufFile->getTypedMetadata<size_t>(
        "q_hidden_size", GGUF_METADATA_VALUE_TYPE_UINT32);
    size_t v_hidden_size = ggufFile->getTypedMetadata<size_t>(
        "v_hidden_size", GGUF_METADATA_VALUE_TYPE_UINT32);

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

    size_t output_size = ggufFile->getTypedMetadata<size_t>(
        "output_size", GGUF_METADATA_VALUE_TYPE_UINT32);
    
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
    layerParams.set("num_heads", 1);

    Mat weight = ggufFile->getTensor("blk." + std::to_string(blockn) + ".attn_qkv");
    // zero bias for now
    Mat bias = Mat::zeros(weight.size[0], weight.size[1], CV_32F);
    layerParams.blobs.push_back(weight);
    layerParams.blobs.push_back(bias);

    std::string inputName, outputName;
    if (blockn == 0){
        std::string inputName = "globInput";
    } else {
        std::string inputName = "blk." + std::to_string(blockn - 1) + ".attn_out";
    }

    if (blockn >= ggufFile->getTypedMetadata<size_t>("num_blocks", GGUF_METADATA_VALUE_TYPE_UINT32) - 1){
        std::string outputName = "globOut";
    } else {
        std::string outputName = "blk." + std::to_string(blockn) + ".attn_out";
    }

    if(inputName.empty() || outputName.empty()){
        CV_Error(Error::StsBadArg, "VanillaArchBlockConstructor: input or output name for a block is empty");
    }

    Arg input = netimpl->getArg(inputName);
    Arg output = netimpl->getArg(outputName);


    Ptr<Layer> layer = LayerFactory::createLayerInstance(layerParams.type, layerParams);
    layer->netimpl = netimpl;
    layer->inputs = {input};
    layer->outputs = {output};
    prog.push_back(layer);
}

Net GGUFImporter::constructNet() {
    VanillaArchBlockConstructor archBlockConstructor(ggufFile);
    archBlockConstructor.initGraph(netimpl);
    int num_blocks = ggufFile->getTypedMetadata<size_t>("num_blocks", GGUF_METADATA_VALUE_TYPE_UINT32);
    for (int i = 0; i < num_blocks; i++){
        archBlockConstructor.AddAttentionBlock(netimpl, i);
    }
    return net;
}


Net readNetFromGGUF(const String& ggufFileName){
    GGUFImporter importer(ggufFileName);
    return importer.constructNet();
}

CV__DNN_INLINE_NS_END

}}