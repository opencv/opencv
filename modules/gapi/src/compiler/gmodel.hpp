// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GMODEL_HPP
#define OPENCV_GAPI_GMODEL_HPP

#include <memory>           // shared_ptr
#include <unordered_map>
#include <functional>       // std::function

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passes/topological_sort.hpp>

// /!\ ATTENTION:
//
// No API includes like GMat, GNode, GCall here!
// This part of the system is API-unaware by its design.
//

#include <opencv2/gapi/util/any.hpp>

#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gkernel.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gislandmodel.hpp"

namespace cv { namespace gimpl {

// TODO: Document all metadata types

struct NodeType
{
    static const char *name() { return "NodeType"; }
    enum { OP, DATA } t;
};

struct Input
{
    static const char *name() { return "Input"; }
    std::size_t port;
};

struct Output
{
    static const char *name() { return "Output"; }
    std::size_t port;
};

struct Op
{
    static const char *name() { return "Op"; }
    cv::GKernel         k;
    std::vector<GArg>   args; // TODO: Introduce a new type for internal args?
    std::vector<RcDesc> outs; // TODO: Introduce a new type for resource references

    cv::gapi::GBackend  backend;
};

struct Data
{
    static const char *name() { return "Data"; }

    // FIXME: This is a _pure_ duplication of RcDesc now! (except storage)
    GShape   shape; // FIXME: Probably to be replaced by GMetaArg?
    int      rc;
    GMetaArg meta;
    HostCtor ctor;  // T-specific helper to deal with unknown types in our code
    // FIXME: Why rc+shape+meta is not represented as RcDesc here?

    enum class Storage
    {
        INTERNAL,   // data object is not listed in GComputation protocol
        INPUT,      // data object is listed in GComputation protocol as Input
        OUTPUT,     // data object is listed in GComputation protocol as Output
        CONST_VAL,  // data object is constant.
                    // Note: CONST is sometimes defined in Win sys headers
    };
    Storage storage;
};

struct ConstValue
{
    static const char *name() { return "ConstValue"; }
    GRunArg arg;
};

// This metadata is valid for both DATA and OP kinds of nodes
// FIXME: Rename to IslandTag
struct Island
{
    static const char *name() { return "Island"; }
    std::string island; // can be set by user, otherwise is set by fusion
};

struct Protocol
{
    static const char *name() { return "Protocol"; }
    // TODO: Replace the whole thing with a "Protocol" object
    std::vector<RcDesc> inputs;
    std::vector<RcDesc> outputs;

    std::vector<ade::NodeHandle> in_nhs;
    std::vector<ade::NodeHandle> out_nhs;
};

// The original metadata the graph has been compiled for.
// - For regular GCompiled, this information always present and
//   is NOT updated on reshape()
// - For GStreamingCompiled, this information may be missing.
//   It means that compileStreaming() was called without meta.
struct OriginalInputMeta
{
    static const char *name() { return "OriginalInputMeta"; }
    GMetaArgs inputMeta;
};

struct OutputMeta
{
    static const char *name() { return "OutputMeta"; }
    GMetaArgs outMeta;
};

struct Journal
{
    static const char *name() { return "Journal"; }
    std::vector<std::string> messages;
};

// Unique data object counter (per-type)
class DataObjectCounter
{
public:
    static const char* name() { return "DataObjectCounter"; }
    int GetNewId(GShape shape) { return m_next_data_id[shape]++; }
private:
    std::unordered_map<cv::GShape, int> m_next_data_id;
};

// A projected graph of Islands (generated from graph of Operations)
struct IslandModel
{
    static const char* name() { return "IslandModel"; }
    std::shared_ptr<ade::Graph> model;
};

// List of backends selected for current graph execution
struct ActiveBackends
{
    static const char *name() { return "ActiveBackends"; }
    std::unordered_set<cv::gapi::GBackend> backends;
};

// This is a graph-global flag indicating this graph is compiled for
// the streaming case.  Streaming-neutral passes (i.e. nearly all of
// them) can ignore this flag safely.
//
// FIXME: Probably a better design can be suggested.
struct Streaming
{
    static const char *name() { return "StreamingFlag"; }
};

// Backend-specific inference parameters for a neural network.
// Since these parameters are set on compilation stage (not
// on a construction stage), these parameters are bound lately
// to the operation node.
// NB: These parameters are not included into GModel by default
// since it is not used regularly by all parties.
struct NetworkParams
{
    static const char *name() { return "NetworkParams"; }
    cv::util::any opaque;
};

// This is a custom metadata handling operator.
// Sometimes outMeta() can't be bound to input parameters only
// so several backends (today -- mainly inference) may find this useful.
// If provided, the meta inference pass uses this function instead of
// OP.k.outMeta.
struct CustomMetaFunction
{
    static const char *name() { return "CustomMetaFunction"; }
    using CM = std::function< cv::GMetaArgs( const ade::Graph      &,
                                             const ade::NodeHandle &,
                                             const cv::GMetaArgs   &,
                                             const cv::GArgs       &)>;
    CM customOutMeta;
};

namespace GModel
{
    using Graph = ade::TypedGraph
        < NodeType
        , Input
        , Output
        , Op
        , Data
        , ConstValue
        , Island
        , Protocol
        , OriginalInputMeta
        , OutputMeta
        , Journal
        , ade::passes::TopologicalSortData
        , DataObjectCounter
        , IslandModel
        , ActiveBackends
        , CustomMetaFunction
        , Streaming
        >;

    // FIXME: How to define it based on GModel???
    using ConstGraph = ade::ConstTypedGraph
        < NodeType
        , Input
        , Output
        , Op
        , Data
        , ConstValue
        , Island
        , Protocol
        , OriginalInputMeta
        , OutputMeta
        , Journal
        , ade::passes::TopologicalSortData
        , DataObjectCounter
        , IslandModel
        , ActiveBackends
        , CustomMetaFunction
        , Streaming
        >;

    // FIXME:
    // Export a single class, not a bunch of functions inside a namespace

    // User should initialize graph before using it
    // GAPI_EXPORTS for tests
    GAPI_EXPORTS void init (Graph& g);

    GAPI_EXPORTS ade::NodeHandle mkOpNode(Graph &g, const GKernel &k, const std::vector<GArg>& args, const std::string &island);
    // Isn't used by the framework or default backends, required for external backend development
    GAPI_EXPORTS ade::NodeHandle mkDataNode(Graph &g, const GShape shape);

    // Adds a string message to a node. Any node can be subject of log, messages then
    // appear in the dumped .dot file.x
    GAPI_EXPORTS void log(Graph &g, ade::NodeHandle op, std::string &&message, ade::NodeHandle updater = ade::NodeHandle());
    GAPI_EXPORTS void log(Graph &g, ade::EdgeHandle op, std::string &&message, ade::NodeHandle updater = ade::NodeHandle());
    // Clears logged messages of a node.
    GAPI_EXPORTS void log_clear(Graph &g, ade::NodeHandle node);

    GAPI_EXPORTS void linkIn   (Graph &g, ade::NodeHandle op,     ade::NodeHandle obj, std::size_t in_port);
    GAPI_EXPORTS void linkOut  (Graph &g, ade::NodeHandle op,     ade::NodeHandle obj, std::size_t out_port);

    GAPI_EXPORTS void redirectReaders(Graph &g, ade::NodeHandle from, ade::NodeHandle to);
    GAPI_EXPORTS void redirectWriter (Graph &g, ade::NodeHandle from, ade::NodeHandle to);

    GAPI_EXPORTS std::vector<ade::NodeHandle> orderedInputs (const ConstGraph &g, ade::NodeHandle nh);
    GAPI_EXPORTS std::vector<ade::NodeHandle> orderedOutputs(const ConstGraph &g, ade::NodeHandle nh);

    // Returns input meta array for given op node
    // Array is sparse, as metadata for non-gapi input objects is empty
    // TODO:
    // Cover with tests!!
    GAPI_EXPORTS GMetaArgs collectInputMeta(const GModel::ConstGraph &cg, ade::NodeHandle node);
    GAPI_EXPORTS GMetaArgs collectOutputMeta(const GModel::ConstGraph &cg, ade::NodeHandle node);

    GAPI_EXPORTS ade::EdgeHandle getInEdgeByPort(const GModel::ConstGraph& cg, const ade::NodeHandle& nh, std::size_t in_port);

    // Returns true if the given backend participates in the execution
    GAPI_EXPORTS bool isActive(const GModel::Graph &cg, const cv::gapi::GBackend &backend);
} // namespace GModel


}} // namespace cv::gimpl

#endif // OPENCV_GAPI_GMODEL_HPP
