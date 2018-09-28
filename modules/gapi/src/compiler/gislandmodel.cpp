// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <sstream>
#include <unordered_set>
#include <unordered_map>

#include <ade/util/checked_cast.hpp>

#include "api/gbackend_priv.hpp" // GBackend::Priv().compile()
#include "compiler/gmodel.hpp"
#include "compiler/gislandmodel.hpp"
#include "logger.hpp"    // GAPI_LOG

namespace cv { namespace gimpl {

GIsland::GIsland(const gapi::GBackend &bknd,
                 ade::NodeHandle op,
                 util::optional<std::string> &&user_tag)
    : m_backend(bknd)
    , m_user_tag(std::move(user_tag))
{
    m_all.insert(op);
    m_in_ops.insert(op);
    m_out_ops.insert(op);
}

// _ because of gcc4.8 wanings on ARM
GIsland::GIsland(const gapi::GBackend &_bknd,
                 node_set &&_all,
                 node_set &&_in_ops,
                 node_set &&_out_ops,
                 util::optional<std::string> &&_user_tag)
    : m_backend(_bknd)
    , m_all(std::move(_all))
    , m_in_ops(std::move(_in_ops))
    , m_out_ops(std::move(_out_ops))
    , m_user_tag(std::move(_user_tag))
{
}

const GIsland::node_set& GIsland::contents() const
{
    return m_all;
}

const GIsland::node_set& GIsland::in_ops() const
{
    return m_in_ops;
}

const GIsland::node_set& GIsland::out_ops() const
{
    return m_out_ops;
}

gapi::GBackend GIsland::backend() const
{
    return m_backend;
}

bool GIsland::is_user_specified() const
{
    return m_user_tag.has_value();
}

void GIsland::debug() const
{
    std::stringstream stream;
    stream << name() << " {{\n  input ops: ";
    for (const auto& nh : m_in_ops) stream << nh << "; ";
    stream << "\n  output ops: ";
    for (const auto& nh : m_out_ops) stream << nh << "; ";
    stream << "\n  contents: ";
    for (const auto& nh : m_all) stream << nh << "; ";
    stream << "\n}}" << std::endl;
    GAPI_LOG_INFO(NULL, stream.str());
}

GIsland::node_set GIsland::consumers(const ade::Graph &g,
                                     const ade::NodeHandle &slot_nh) const
{
    GIslandModel::ConstGraph gim(g);
    auto data_nh = gim.metadata(slot_nh).get<DataSlot>().original_data_node;
    GIsland::node_set result;
    for (const auto& in_op : m_in_ops)
    {
        auto it = std::find(in_op->inNodes().begin(),
                            in_op->inNodes().end(),
                            data_nh);
        if (it != in_op->inNodes().end())
            result.insert(in_op);
    }
    return result;
}

ade::NodeHandle GIsland::producer(const ade::Graph &g,
                                  const ade::NodeHandle &slot_nh) const
{
    GIslandModel::ConstGraph gim(g);
    auto data_nh = gim.metadata(slot_nh).get<DataSlot>().original_data_node;
    for (const auto& out_op : m_out_ops)
    {
        auto it = std::find(out_op->outNodes().begin(),
                            out_op->outNodes().end(),
                            data_nh);
        if (it != out_op->outNodes().end())
            return out_op;
    }
    // Consistency: A GIsland requested for producer() of slot_nh should
    // always had the appropriate GModel node handle in its m_out_ops vector.
    GAPI_Assert(false);
    return ade::NodeHandle();
}

std::string GIsland::name() const
{
    if (is_user_specified())
        return m_user_tag.value();

    std::stringstream ss;
    ss << "island_#" << std::hex << static_cast<const void*>(this);
    return ss.str();
}

void GIslandModel::generateInitial(GIslandModel::Graph &g,
                                   const ade::Graph &src_graph)
{
    const GModel::ConstGraph src_g(src_graph);

    // Initially GIslandModel is a 1:1 projection from GModel:
    // 1) Every GModel::OP becomes a separate GIslandModel::FusedIsland;
    // 2) Every GModel::DATA becomes GIslandModel::DataSlot;
    // 3) Single-operation FusedIslands are connected with DataSlots in the
    //    same way as OPs and DATA (edges with the same metadata)

    using node_set = std::unordered_set
        < ade::NodeHandle
        , ade::HandleHasher<ade::Node>
        >;
    using node_map = std::unordered_map
        < ade::NodeHandle
        , ade::NodeHandle
        , ade::HandleHasher<ade::Node>
        >;

    node_set all_operations;
    node_map data_to_slot;

    // First, list all operations and build create DataSlots in <g>
    for (auto src_nh : src_g.nodes())
    {
        switch (src_g.metadata(src_nh).get<NodeType>().t)
        {
        case NodeType::OP:   all_operations.insert(src_nh);                break;
        case NodeType::DATA: data_to_slot[src_nh] = mkSlotNode(g, src_nh); break;
        default: GAPI_Assert(false); break;
        }
    } // for (src_g.nodes)

    // Now put single-op islands and connect it with DataSlots
    for (auto src_op_nh : all_operations)
    {
        auto nh = mkIslandNode(g, src_g.metadata(src_op_nh).get<Op>().backend, src_op_nh, src_graph);
        for (auto in_edge : src_op_nh->inEdges())
        {
            auto src_data_nh = in_edge->srcNode();
            auto isl_slot_nh = data_to_slot.at(src_data_nh);
            g.link(isl_slot_nh, nh); // no other data stored yet
        }
        for (auto out_edge : src_op_nh->outEdges())
        {
            auto dst_data_nh = out_edge->dstNode();
            auto isl_slot_nh = data_to_slot.at(dst_data_nh);
            g.link(nh, isl_slot_nh);
        }
    } // for(all_operations)
}

ade::NodeHandle GIslandModel::mkSlotNode(Graph &g, const ade::NodeHandle &data_nh)
{
    auto nh = g.createNode();
    g.metadata(nh).set(DataSlot{data_nh});
    g.metadata(nh).set(NodeKind{NodeKind::SLOT});
    return nh;
}

ade::NodeHandle GIslandModel::mkIslandNode(Graph &g, const gapi::GBackend& bknd, const ade::NodeHandle &op_nh, const ade::Graph &orig_g)
{
    const GModel::ConstGraph src_g(orig_g);
    util::optional<std::string> user_tag;
    if (src_g.metadata(op_nh).contains<Island>())
    {
        user_tag = util::make_optional(src_g.metadata(op_nh).get<Island>().island);
    }

    auto nh = g.createNode();
    std::shared_ptr<GIsland> island(new GIsland(bknd, op_nh, std::move(user_tag)));
    g.metadata(nh).set(FusedIsland{std::move(island)});
    g.metadata(nh).set(NodeKind{NodeKind::ISLAND});
    return nh;
}

ade::NodeHandle GIslandModel::mkIslandNode(Graph &g, std::shared_ptr<GIsland>&& isl)
{
    ade::NodeHandle nh = g.createNode();
    g.metadata(nh).set(cv::gimpl::NodeKind{cv::gimpl::NodeKind::ISLAND});
    g.metadata(nh).set<cv::gimpl::FusedIsland>({std::move(isl)});
    return nh;
}

void GIslandModel::syncIslandTags(Graph &g, ade::Graph &orig_g)
{
    GModel::Graph gm(orig_g);
    for (auto nh : g.nodes())
    {
        if (NodeKind::ISLAND == g.metadata(nh).get<NodeKind>().k)
        {
            auto island = g.metadata(nh).get<FusedIsland>().object;
            auto isl_tag = island->name();
            for (const auto& orig_nh_inside : island->contents())
            {
                gm.metadata(orig_nh_inside).set(Island{isl_tag});
            }
        }
    }
}

void GIslandModel::compileIslands(Graph &g, const ade::Graph &orig_g, const GCompileArgs &args)
{
    GModel::ConstGraph gm(orig_g);

    auto original_sorted = gm.metadata().get<ade::passes::TopologicalSortData>();
    for (auto nh : g.nodes())
    {
        if (NodeKind::ISLAND == g.metadata(nh).get<NodeKind>().k)
        {
            auto island_obj = g.metadata(nh).get<FusedIsland>().object;
            auto island_ops = island_obj->contents();

            std::vector<ade::NodeHandle> topo_sorted_list;
            ade::util::copy_if(original_sorted.nodes(),
                               std::back_inserter(topo_sorted_list),
                               [&](ade::NodeHandle sorted_nh) {
                                   return ade::util::contains(island_ops, sorted_nh);
                               });

            auto island_exe = island_obj->backend().priv()
                .compile(orig_g, args, topo_sorted_list);
            GAPI_Assert(nullptr != island_exe);
            g.metadata(nh).set(IslandExec{std::move(island_exe)});
        }
    }
}

ade::NodeHandle GIslandModel::producerOf(const ConstGraph &g, ade::NodeHandle &data_nh)
{
    for (auto nh : g.nodes())
    {
        // find a data slot...
        if (NodeKind::SLOT == g.metadata(nh).get<NodeKind>().k)
        {
            // which is associated with the given data object...
            if (data_nh == g.metadata(nh).get<DataSlot>().original_data_node)
            {
                // which probably has a produrer...
                if (0u != nh->inNodes().size())
                {
                    // ...then the answer is that producer
                    return nh->inNodes().front();
                }
                else return ade::NodeHandle(); // input data object?
                                               // return empty to break the cycle
            }
        }
    }
    // No appropriate data slot found - probably, the object has been
    // optimized out during fusion
    return ade::NodeHandle();
}

} // namespace cv
} // namespace gimpl
