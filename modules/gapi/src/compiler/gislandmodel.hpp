// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation


#ifndef OPENCV_GAPI_GISLANDMODEL_HPP
#define OPENCV_GAPI_GISLANDMODEL_HPP

#include <unordered_set>
#include <memory>        // shared_ptr

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>
#include <ade/passes/topological_sort.hpp>

#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/gkernel.hpp>

#include "compiler/gobjref.hpp"

namespace cv { namespace gimpl {


// FIXME: GAPI_EXPORTS only because of tests!
class GAPI_EXPORTS GIsland
{
public:
    using node_set = std::unordered_set
         < ade::NodeHandle
         , ade::HandleHasher<ade::Node>
         >;

    // Initial constructor (constructs a single-op Island)
    GIsland(const gapi::GBackend &bknd,
            ade::NodeHandle op,
            util::optional<std::string>&& user_tag);

    // Merged constructor
    GIsland(const gapi::GBackend &bknd,
            node_set &&all,
            node_set &&in_ops,
            node_set &&out_ops,
            util::optional<std::string>&& user_tag);

    const node_set& contents() const;
    const node_set& in_ops() const;
    const node_set& out_ops() const;

    std::string name() const;
    gapi::GBackend backend() const;

    /**
     * Returns all GModel operation node handles which are _reading_
     * from a GModel data object associated (wrapped in) the given
     * Slot object.
     *
     * @param g an ade::Graph with GIslandModel information inside
     * @param slot_nh Slot object node handle of interest
     * @return a set of GModel operation node handles
     */
    node_set consumers(const ade::Graph &g,
                       const ade::NodeHandle &slot_nh) const;

    /**
     * Returns a GModel operation node handle which is _writing_
     * to a GModel data object associated (wrapped in) the given
     * Slot object.
     *
     * @param g an ade::Graph with GIslandModel information inside
     * @param slot_nh Slot object node handle of interest
     * @return a node handle of original GModel
     */
    ade::NodeHandle producer(const ade::Graph &g,
                             const ade::NodeHandle &slot_nh) const;

    void debug() const;
    bool is_user_specified() const;

protected:
    gapi::GBackend m_backend; // backend which handles this Island execution

    node_set m_all;     // everything (data + operations) within an island
    node_set m_in_ops;  // operations island begins with
    node_set m_out_ops; // operations island ends with

    // has island name IF specified by user. Empty for internal (inferred) islands
    util::optional<std::string> m_user_tag;
};



// GIslandExecutable - a backend-specific thing which executes
// contents of an Island
// * Is instantiated by the last step of the Islands fusion procedure;
// * Is orchestrated by a GExecutor instance.
//

class GIslandExecutable
{
public:
    using InObj  = std::pair<RcDesc, cv::GRunArg>;
    using OutObj = std::pair<RcDesc, cv::GRunArgP>;

    // FIXME: now run() requires full input vector to be available.
    // actually, parts of subgraph may execute even if there's no all data
    // slots in place.
    // TODO: Add partial execution capabilities
    virtual void run(std::vector<InObj>  &&input_objs,
                     std::vector<OutObj> &&output_objs) = 0;

    virtual bool canReshape() const = 0;
    virtual void reshape(ade::Graph& g, const GCompileArgs& args) = 0;

    virtual ~GIslandExecutable() = default;
};

// GIslandEmitter - a backend-specific thing which feeds data into
// the pipeline. This one is just an interface, implementations are executor-defined.
class GIslandEmitter
{
public:
    // Obtain next value from the emitter
    virtual bool pull(GRunArg &) = 0;
    virtual ~GIslandEmitter() = default;
};

// Couldn't reuse NodeType here - FIXME unify (move meta to a shared place)
struct NodeKind
{
    static const char *name() { return "NodeKind"; }
    enum { ISLAND, SLOT, EMIT, SINK} k;
};

// FIXME: Rename to Island (as soon as current GModel::Island is renamed
// to IslandTag).
struct FusedIsland
{
    static const char *name() { return "FusedIsland"; }
    std::shared_ptr<GIsland> object;
};

struct DataSlot
{
    static const char *name() { return "DataSlot"; }
    ade::NodeHandle original_data_node; // direct link to GModel
};

struct IslandExec
{
    static const char *name() { return "IslandExecutable"; }
    std::shared_ptr<GIslandExecutable> object;
};

struct Emitter
{
    static const char *name() { return "Emitter"; }
    std::size_t proto_index;
    std::shared_ptr<GIslandEmitter> object;
};

struct Sink
{
    static const char *name() { return "Sink"; }
    std::size_t proto_index;
};

// This flag is set in graph's own metadata if compileIsland was successful
struct IslandsCompiled
{
    static const char *name() { return "IslandsCompiled"; }
};

namespace GIslandModel
{
    using Graph = ade::TypedGraph
        < NodeKind
        , FusedIsland
        , DataSlot
        , IslandExec
        , Emitter
        , Sink
        , IslandsCompiled
        , ade::passes::TopologicalSortData
        >;

    // FIXME: derive from TypedGraph
    using ConstGraph = ade::ConstTypedGraph
        < NodeKind
        , FusedIsland
        , DataSlot
        , IslandExec
        , Emitter
        , Sink
        , IslandsCompiled
        , ade::passes::TopologicalSortData
        >;

    // Top-level function
    void generateInitial(Graph &g, const ade::Graph &src_g);
    // "Building blocks"
    ade::NodeHandle mkSlotNode(Graph &g, const ade::NodeHandle &data_nh);
    ade::NodeHandle mkIslandNode(Graph &g, const gapi::GBackend &bknd, const ade::NodeHandle &op_nh, const ade::Graph &orig_g);
    ade::NodeHandle mkIslandNode(Graph &g, std::shared_ptr<GIsland>&& isl);
    ade::NodeHandle mkEmitNode(Graph &g, std::size_t in_idx); // streaming-related
    ade::NodeHandle mkSinkNode(Graph &g, std::size_t out_idx); // streaming-related

    // GIslandModel API
    void syncIslandTags(Graph &g, ade::Graph &orig_g);
    void compileIslands(Graph &g, const ade::Graph &orig_g, const GCompileArgs &args);

    // Debug routines
    // producerOf - returns an Island handle which produces given data object
    //     from the original model (! don't mix with DataSlot)
    // FIXME: GAPI_EXPORTS because of tests only!
    ade::NodeHandle GAPI_EXPORTS producerOf(const ConstGraph &g, ade::NodeHandle &data_nh);

} // namespace GIslandModel

}} // namespace cv::gimpl

#endif // OPENCV_GAPI_GISLANDMODEL_HPP
