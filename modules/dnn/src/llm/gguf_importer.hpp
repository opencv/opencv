

#ifndef __OPENCV_GGUFIMPORTER_HPP__
#define __OPENCV_GGUFIMPORTER_HPP__
#include <opencv2/dnn.hpp>
#include "gguf_def.hpp"
#include "opencv2/core.hpp"
#include "gguf_parser.hpp"


namespace cv { namespace dnn {

CV__DNN_INLINE_NS_BEGIN
using namespace cv::dnn;

struct ArchBlockConstructor {
    ArchBlockConstructor(Ptr<GGUFParser> ggufFile) : ggufFile(ggufFile) {};

    virtual ~ArchBlockConstructor() = default;

    virtual void initGraph(Net::Impl* netimpl){};
    virtual void AddAttentionBlock(Net::Impl* netimpl,int blockn){};
    void finalizeGraph();

    Ptr<GGUFParser> ggufFile;
    Ptr<Graph> graph;
    std::vector<Ptr<Layer> > prog;
};

// Vanilla means just existing Attention from opencv dnn
struct VanillaArchBlockConstructor : public ArchBlockConstructor{
    using ArchBlockConstructor::ArchBlockConstructor;  // inherit constructors if needed

    void initGraph(Net::Impl* netimpl) override;
    void AddAttentionBlock(Net::Impl* netimpl, int blockn) override;
};

/* Fabric for creating Net from GGUF file */
struct GGUFImporter
{   
    GGUFImporter(const String& ggufFileName) {
        netimpl = net.getImpl();
        ggufFile = makePtr<GGUFParser>(ggufFileName);
    }
    // net construction
    Net constructNet();
    // parser
    Ptr<GGUFParser> ggufFile;
    // Net impl stuff
    Net net;
    Net::Impl* netimpl;
    std::vector<Ptr<Layer>> prog;
};

Net readNetFromGGUF(const String& ggufFileName);

CV__DNN_INLINE_NS_END
}}

#endif