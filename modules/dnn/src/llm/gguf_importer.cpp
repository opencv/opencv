
#include "../precomp.hpp"
#include "../net_impl.hpp"
#include <opencv2/dnn/layer_reg.private.hpp>
#include "gguf_importer.hpp"
#include "gguf_parser.hpp"
// #include <opencv2/core.hpp>


#include <fstream>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

GGUFImporter::GGUFImporter(const String& ggufFileName) {
    netimpl = net.getImpl();
    ggufFile = makePtr<GGUFParser>(ggufFileName);

    netInput = netimpl->newArg("input",  DNN_ARG_INPUT, true);
    netOutput = netimpl->newArg("output",  DNN_ARG_OUTPUT, true );
    ArgData inputdata = netimpl->args.at(netInput.idx);
    inputdata.shape = ggufFile->blocks[0].getInputShape();
    // @TODO make more flexible
    inputdata.type = CV_32F;

    ArgData outputdata = netimpl->args.at(netOutput.idx);
    outputdata.shape =  ggufFile->blocks[ggufFile->blocks.size() - 1].getOutputShape(); // output size;
    outputdata.type = CV_32F;

    netimpl->args[netInput.idx] = inputdata;
    netimpl->args[netOutput.idx] = outputdata;

    graph = netimpl->newGraph("VanillaAttention", {netInput}, true);
    graph->setOutputs({netOutput});
}

Net GGUFImporter::constructNet() {
    std::vector<Arg> blockInputs = {netInput};
    std::vector<Arg> blockOutputs = {netOutput};

    for (auto& block : ggufFile->blocks){
        addBlock(block, blockInputs, blockOutputs);
    }

    graph->setProg(prog);
    netimpl->prepareForInference();

    return net;
}

void GGUFImporter::addBlock(BlockMetadata block, std::vector<Arg>& blockInputs, std::vector<Arg>& blockOutputs) {
    bool is_final_block = block.blockn == ggufFile->blocks.size() - 1;
    // ArgKind outputArgKind = is_final_block ? DNN_ARG_OUTPUT : DNN_ARG_TEMP;
    // Add attention
    LayerParams layerParams;
    // @TODO need to rework the naming system
    std::string outArgName = is_final_block ? "output" : "";
    Arg out = netimpl->getArg(outArgName);
    if(!is_final_block) {
        ArgData outData = netimpl->args.at(out.idx);
        outData.shape = block.getOutputShape();
        // @TODO make more flexible
        outData.type = CV_32F;
        netimpl->args[out.idx] = outData;
    }

    // Attention layer
    block.getAttentionLayerParams(ggufFile->tensor_reader, layerParams);
    Ptr<Layer> layer = LayerFactory::createLayerInstance(layerParams.type, layerParams);
    layer->netimpl = netimpl;
    layer->inputs = blockInputs;
    layer->outputs = {out};
    prog.push_back(layer);

    blockOutputs = {out};
}

Net readNetFromGGUF(const String& ggufFileName){
    GGUFImporter importer(ggufFileName);
    return importer.constructNet();
}

CV__DNN_INLINE_NS_END

}}