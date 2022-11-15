// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

void Net2::dump()
{
    impl()->dump();
}

void Net2::Impl::dump() const
{
    dumpGraph(graph, "", false);
}

static const char* typ2str(int typ)
{
    return
        typ == CV_8U ? "U8" :
        typ == CV_8S ? "I8" :
        typ == CV_16U ? "U16" :
        typ == CV_16S ? "I16" :
        typ == CV_32S ? "I32" :
        typ == CV_32F ? "F32" :
        typ == CV_64F ? "F64" :
        typ == CV_16F ? "F16" : "<unknown_type>";
}

void Net2::Impl::dumpArgInfo(int argidx, const string& indent, bool comma) const
{
    const LayerArg& arg = args[argidx];
    const char* kind_str =
        arg.kind == DNN_ARG_CONST ? "const" :
        arg.kind == DNN_ARG_INPUT ? "input" :
        arg.kind == DNN_ARG_OUTPUT ? "output" :
        arg.kind == DNN_ARG_TEMP ? "temp" : "";

    printf("%s\"%s\"%s // %s <", indent.c_str(), arg.name.c_str(), (comma ? "," : ""), kind_str);
    int i, ndims = arg.shape.ndims;
    if (ndims < 0)
        printf("...>");
    else {
        for (i = 0; i < ndims; i++) {
            int size_i = (int)arg.shape.shape[i];
            if (i > 0)
                printf(" x ");
            if (size_i >= 0)
                printf("%d", size_i);
            else
                printf("%s", dimnames_[-size_i-1].c_str());
        }
        printf(">");
    }
    printf(" %s", typ2str(arg.typ));
    if (arg.kind == DNN_ARG_TEMP && argidx < bufidxs.size()) {
        printf(" (buf #%d)", bufidxs[argidx]);
    }
    printf("\n");
}

void Net2::Impl::dumpGraph(const Graph& graph, const string& indent, bool comma) const
{
    size_t ninputs = graph.inputs.size(), noutputs = graph.outputs.size();
    string subindent = indent + delta_indent;
    string argindent = subindent + delta_indent;
    printf("graph {\n%sinputs: {\n", subindent.c_str());

    for (size_t i = 0; i < ninputs; i++) {
        dumpArgInfo(graph.inputs[i], argindent, i+1 < ninputs);
    }
    printf("%s},\n%soutputs: [\n", subindent.c_str(), subindent.c_str());
    for (size_t i = 0; i < noutputs; i++) {
        dumpArgInfo(graph.outputs[i], argindent, i+1 < noutputs);
    }
    printf("%s],\n%sprog: [\n", subindent.c_str(), subindent.c_str());
    size_t nops = graph.prog.size();
    for (size_t i = 0; i < nops; i++) {
        printf("%s// op #%d\n", argindent.c_str(), (int)i);
        const Node& node = graph.prog[i];
        dumpNode(*node.op.get(), node.inputs, node.outputs,
                 node.subgraphs, argindent, i+1 < nops);
    }
    printf("%s]\n%s}%s\n", subindent.c_str(), indent.c_str(), (comma ? "," : ""));
}

void Net2::Impl::dumpAttrValue(const DictValue& dictv) const
{
    int i, n = (int)dictv.size();
    bool scalar = n == 1;
    if (!scalar) printf("[");
    if (dictv.isInt())
    {
        for (i = 0; i < n; i++)
            printf("%s%lld", (i == 0 ? "" : ", "), (long long)dictv.get<int64>(i));
    }
    else if (dictv.isReal())
    {
        for (i = 0; i < n; i++)
            printf("%s%.3g", (i == 0 ? "" : ", "), dictv.get<double>(i));
    }
    else if (dictv.isString())
    {
        for (i = 0; i < n; i++)
            printf("%s\"%s\"", (i == 0 ? "" : ", "), dictv.get<String>(i).c_str());
    }
    else if (dictv.isMat())
    {
        Mat m = dictv.getMat();
        TensorShape shape = TensorShape::fromArray(m, dictv.getDims());
        printf("<");
        int i, ndims = shape.ndims;
        for (i = 0; i < ndims; i++)
            printf("%s%d", (i == 0 ? "" : " x "), (int)shape.shape[i]);
        printf("> %s", typ2str(m.type()));
    }
    if (!scalar) printf("]");
}

void Net2::Impl::dumpNode(const Layer& layer,
                          const vector<int>& inpargs,
                          const vector<int>& outargs,
                          const vector<PGraph>& subgraphs,
                          const string& indent,
                          bool comma) const
{
    size_t ninputs = inpargs.size(), noutputs = outargs.size();
    size_t ngraphs = subgraphs.size();
    string subindent = indent + delta_indent;
    string argindent = subindent + delta_indent;
    printf("%s%s {\n%sname: \"%s\",\n",
           indent.c_str(), layer.type.c_str(), subindent.c_str(),
           layer.name.c_str());
    LayerParams params;
    layer.serialize(params);
    for (auto pair: params) {
        printf("%s%s: ", subindent.c_str(), pair.first.c_str());
        dumpAttrValue(pair.second);
        printf(",\n");
    }
    printf("%sinputs: [\n", subindent.c_str());
    for (size_t i = 0; i < ninputs; i++) {
        dumpArgInfo(inpargs[i], argindent, i+1 < ninputs);
    }
    printf("%s],\n%soutputs: [\n", subindent.c_str(), subindent.c_str());
    for (size_t i = 0; i < noutputs; i++) {
        dumpArgInfo(outargs[i], argindent, i+1 < noutputs);
    }
    printf("%s]%s\n", subindent.c_str(), (subgraphs.empty() ? "" : ","));

    if (!subgraphs.empty()) {
        vector<string> names;
        if (layer.type == "If")
            names = {"then", "else"};
        else if (layer.type == "Loop")
            names = {"body"};
        if (ngraphs != names.size())
            CV_Error(Error::StsError,
                     format("unsupported operation '%s' with subgraphs",
                            layer.type.c_str()));
        for (size_t i = 0; i < ngraphs; i++) {
            printf("%s%s: ", subindent.c_str(), names[i].c_str());
            dumpGraph(*subgraphs[i].get(), subindent + delta_indent, i+1 < ngraphs);
        }
    }
    printf("%s}%s\n", indent.c_str(), (comma ? "," : ""));
}

CV__DNN_INLINE_NS_END
}}
