// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

void Net2::dump() const
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

static const char* layout2str(char* buf, DataLayout layout, int c=0)
{
    if (layout == DNN_LAYOUT_NCHWc) {
        sprintf(buf, "NCHWc(%d)", c);
        return buf;
    }
    return  layout == DNN_LAYOUT_UNKNOWN ? "unknown" :
            layout == DNN_LAYOUT_ND ? "ND" :
            layout == DNN_LAYOUT_NCHW ? "NCHW" :
            layout == DNN_LAYOUT_NHWC ? "NHWC" : "???";
}

void Net2::Impl::dumpArgInfo(int argidx, const string& indent, bool comma) const
{
    const LayerArg& arg = args.at(argidx);
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
    //if (arg.kind == DNN_ARG_TEMP && argidx < bufidxs.size())
        printf(" (buf #%d)", bufidxs[argidx]);
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
        auto m = dictv.getMat();
        TensorShape shape = TensorShape::fromArray(m.first, m.second);
        printf("<");
        int i, ndims = shape.ndims;
        for (i = 0; i < ndims; i++)
            printf("%s%d", (i == 0 ? "" : " x "), (int)shape.shape[i]);
        printf("> %s", typ2str(m.first.type()));
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

void Net2::Impl::dumpArg(const String& prefix, int i,
                         int argidx, bool dumpdata) const
{
    char buf[128];
    const Tensor& t = tensors.at(argidx);
    const LayerArg& arg = args.at(argidx);
    printf("%s %d. Name: %s\n  Buf: %d\n  Type: %s\n  Shape: {",
           prefix.c_str(), i, arg.name.c_str(), bufidxs.at(argidx), typ2str(t.typ));
    for (int i = 0; i < t.shape.ndims; i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)t.shape.shape[i]);
    }
    printf("}\n  Layout: %s\n", layout2str(buf, t.shape.layout, t.shape.shape[t.shape.ndims-1]));
    if (dumpdata) {
        // [TODO] in the case of block layout probably we need to
        // make 'NCHW' copy of the tensor data before dumping it.
        cv::dnn::dump(t);
    }
}

template<typename _Tp> struct Fmt
{
    typedef _Tp temp_type;
    static const char* fmt() { return "%d"; }
};

template<> struct Fmt<int64_t>
{
    typedef long long temp_type;
    static const char* fmt() { return "%lld"; }
};

template<> struct Fmt<uint64_t>
{
    typedef unsigned long long temp_type;
    static const char* fmt() { return "%llu"; }
};

template<> struct Fmt<float>
{
    typedef float temp_type;
    static const char* fmt() { return "%.5g"; }
};

template<> struct Fmt<double>
{
    typedef double temp_type;
    static const char* fmt() { return "%.5g"; }
};

template<> struct Fmt<float16_t>
{
    typedef float temp_type;
    static const char* fmt() { return "%.5g"; }
};

template <typename _Tp>
static void dumpRow(const _Tp* ptr, int n, size_t ofs, int border)
{
    const char* fmt = Fmt<_Tp>::fmt();
    int i, ndump = border > 0 ? std::min(n, border*2+1) : n;
    if (border == 0)
        border = ndump;
    for (i = 0; i < ndump; i++) {
        int j = n == ndump || i < border ? i : i == border ? -1 : n-border*2-1+i;
        if (i > 0)
            printf(", ");
        if (j >= 0)
            printf(fmt, (typename Fmt<_Tp>::temp_type)ptr[ofs + j]);
        else
            printf("... ");
    }
}

static void dumpSlice(const Tensor& t, const size_t* step, int d, size_t ofs, int border)
{
    int ndims = t.shape.ndims;
    int n = d >= ndims ? 1 : (int)t.shape.shape[d];
    if (d >= ndims - 1) {
        int typ = t.typ;
        void* data = t.data();
        if (typ == CV_8U)
            dumpRow((const uint8_t*)data, n, ofs, border);
        else if (typ == CV_8S)
            dumpRow((const int8_t*)data, n, ofs, border);
        else if (typ == CV_16U)
            dumpRow((const uint16_t*)data, n, ofs, border);
        else if (typ == CV_16S)
            dumpRow((const int16_t*)data, n, ofs, border);
        else if (typ == CV_32S)
            dumpRow((const int*)data, n, ofs, border);
        else if (typ == CV_32F)
            dumpRow((const float*)data, n, ofs, border);
        else if (typ == CV_64F)
            dumpRow((const double*)data, n, ofs, border);
        else if (typ == CV_16F)
            dumpRow((const float16_t*)data, n, ofs, border);
        else {
            CV_Error(Error::StsNotImplemented, "unsupported type");
        }
    } else {
        int i, ndump = border > 0 ? std::min(n, border*2+1) : n;
        bool dots = false;
        for (i = 0; i < ndump; i++) {
            if (i > 0 && !dots) {
                int nempty_lines = ndims - 2 - d;
                for (int k = 0; k < nempty_lines; k++)
                    printf("\n");
            }
            if (i > 0)
                printf("\n");
            int j = n == ndump || i < border ? i :
                    i == border ? -1 :
                    n - border*2 - 1 + i;
            bool dots = j < 0;
            if (!dots)
                dumpSlice(t, step, d+1, ofs + j*step[d], border);
            else
                printf("...");
        }
    }
}

void dump(const Tensor& t, int border0, int maxsz_all, bool braces)
{
    const TensorShape& shape = t.shape;
    size_t szall = shape.total();
    if (szall == 0) {
        printf("no data");
    } else {
        int ndims = t.shape.ndims;
        int border = szall < (size_t)maxsz_all ? 0 : border0;
        size_t step[TensorShape::MAX_TENSOR_DIMS];
        step[ndims-1] = 1;
        for (int i = ndims-2; i >= 0; i--)
            step[i] = step[i+1]*t.shape.shape[i+1];
        if (braces)
            printf("[");
        dumpSlice(t, step, 0, 0, border);
        if (braces)
            printf("]\n");
        else
            printf("\n");
    }
}

void dump(InputArray m, int ndims, int border, int maxsz_all, bool braces)
{
    Tensor t(m, ndims, false);
    dump(t, border, maxsz_all, braces);
}

CV__DNN_INLINE_NS_END
}}
