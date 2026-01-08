

#ifndef __OPENCV_GGUFIMPORTER_HPP__
#define __OPENCV_GGUFIMPORTER_HPP__
#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"
#include "gguf_parser.hpp"
#include "../net_impl.hpp"


namespace cv { namespace dnn {

CV__DNN_INLINE_NS_BEGIN
using namespace cv::dnn;

/* Fabric for creating Net from GGUF file */
struct GGUFImporter
{   
    GGUFImporter(const String& ggufFileName);
    Net constructNet();
    void addBlock(BlockMetadata block, std::vector<Arg>& blockInputs, std::vector<Arg>& blockOutputs);
    // parser
    Ptr<GGUFParser> ggufFile;
    Arg netInput,netOutput;
    // Net impl stuff
    Net net;
    Net::Impl* netimpl;
    std::vector<Ptr<Layer>> prog;
    Ptr<Graph> graph;
};


Net readNetFromGGUF(const String& ggufFileName);

CV__DNN_INLINE_NS_END
}}

#endif