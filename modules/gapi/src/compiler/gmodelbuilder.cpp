// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


////////////////////////////////////////////////////////////////////////////////
//
//    FIXME: "I personally hate this file"
//                                        - Dmitry
//
////////////////////////////////////////////////////////////////////////////////
#include "precomp.hpp"

#include <utility>              // tuple
#include <stack>                // stack
#include <vector>               // vector
#include <unordered_set>        // unordered_set
#include <type_traits>          // is_same

#include <ade/util/zip_range.hpp>   // util::indexed

#include "api/gorigin.hpp"
#include "api/gproto_priv.hpp"  // descriptor_of and other GProtoArg-related
#include "api/gcall_priv.hpp"
#include "api/gnode_priv.hpp"

#include "compiler/gmodelbuilder.hpp"
#include "compiler/gmodel_priv.hpp"

namespace {


// TODO: move to helpers and cover with internal tests?
template<typename T> struct GVisited
{
    typedef std::unordered_set<T> VTs;

    bool visited(const T& t) const { return m_visited.find(t) != m_visited.end(); }
    void visit  (const T& t)       { m_visited.insert(t); }
    const VTs& visited()     const { return m_visited; }

private:
    VTs m_visited;
};

template<typename T, typename U = T> struct GVisitedTracker: protected GVisited<T>
{
    typedef std::vector<U> TUs;

    void  visit(const T& t, const U& u) { GVisited<T>::visit(t); m_tracked.push_back(u); }
    const TUs& tracked() const          { return m_tracked; }
    using GVisited<T>::visited;

private:
    TUs m_tracked;
};

} // namespace


cv::gimpl::Unrolled cv::gimpl::unrollExpr(const GProtoArgs &ins,
                                          const GProtoArgs &outs)
{
    // FIXME: Who's gonna check if ins/outs are not EMPTY?
    // FIXME: operator== for GObjects? (test if the same object or not)
    using GObjId = const cv::GOrigin*;

    GVisitedTracker<const GNode::Priv*, cv::GNode> ops;
    GVisited<GObjId> reached_sources;
    cv::GOriginSet   origins;

    // Cache input argument objects for a faster look-up
    // While the only reliable way to identify a Data object is Origin
    // (multiple data objects may refer to the same Origin as result of
    // multuple yield() calls), input objects can be uniquely identified
    // by its `priv` address. Here we rely on this to verify if the expression
    // we unroll actually matches the protocol specified to us by user.
    std::unordered_set<GObjId> in_objs_p;
    for (const auto& in_obj : ins)
    {
        // Objects are guaranteed to remain alive while this method
        // is working, so it is safe to keep pointers here and below
        in_objs_p.insert(&proto::origin_of(in_obj));
    }

    // Recursive expression traversal
    std::stack<cv::GProtoArg> data_objs(std::deque<cv::GProtoArg>(outs.begin(), outs.end()));
    while (!data_objs.empty())
    {
        const auto  obj   = data_objs.top();
        const auto &obj_p = proto::origin_of(obj);
        data_objs.pop();

        const auto &origin = obj_p;
        origins.insert(origin); // TODO: Put Object description here later on

        // If this Object is listed in the protocol, don't dive deeper (even
        // if it is in fact a result of operation). Our computation is
        // bounded by this data slot, so terminate this recursion path early.
        if (in_objs_p.find(&obj_p) != in_objs_p.end())
        {
            reached_sources.visit(&obj_p);
            continue;
        }

        const cv::GNode &node = origin.node;
        switch (node.shape())
        {
        case cv::GNode::NodeShape::EMPTY:
            // TODO: Own exception type?
            util::throw_error(std::logic_error("Empty node reached!"));
            break;

        case cv::GNode::NodeShape::PARAM:
        case cv::GNode::NodeShape::CONST_BOUNDED:
            // No preceding operation to this data object - so the data object is either a GComputation
            // parameter or a constant (compile-time) value
            // Record it to check if protocol matches expression tree later
            if (!reached_sources.visited(&obj_p))
                reached_sources.visit(&obj_p);
            break;

        case cv::GNode::NodeShape::CALL:
            if (!ops.visited(&node.priv()))
            {
                // This operation hasn't been visited yet - mark it so,
                // then add its operands to stack to continue recursion.
                ops.visit(&node.priv(), node);

                const cv::GCall&        call   = origin.node.call();
                const cv::GCall::Priv&  call_p = call.priv();

                // Put the outputs object description of the node
                // so that they are not lost if they are not consumed by other operations
                GAPI_Assert(call_p.m_k.outCtors.size() == call_p.m_k.outShapes.size());
                for (const auto it : ade::util::indexed(call_p.m_k.outShapes))
                {
                    std::size_t port  = ade::util::index(it);
                    GShape shape      = ade::util::value(it);

                    // FIXME: then use ZIP
                    HostCtor ctor     = call_p.m_k.outCtors[port];

                    // NB: Probably this fixes all other "missing host ctor"
                    // problems.
                    // TODO: Clean-up the old workarounds if it really is.
                    GOrigin org {shape, node, port, std::move(ctor), origin.kind};
                    origins.insert(org);
                }

                for (const auto &arg : call_p.m_args)
                {
                    if (proto::is_dynamic(arg))
                    {
                        data_objs.push(proto::rewrap(arg)); // Dive deeper
                    }
                }
            }
            break;

        default:
            // Unsupported node shape
            GAPI_Error("InternalError");
            break;
        }
    }

    // Check if protocol mentions data_objs which weren't reached during traversal
    const auto missing_reached_sources = [&reached_sources](GObjId p) {
        return reached_sources.visited().find(p) == reached_sources.visited().end();
    };
    if (ade::util::any_of(in_objs_p, missing_reached_sources))
    {
        // TODO: Own exception type or a return code?
      util::throw_error(std::logic_error("Data object listed in Protocol "
                                     "wasn\'t reached during unroll"));
    }

    // Check if there endpoint (parameter) data_objs which are not listed in protocol
    const auto missing_in_proto = [&in_objs_p](GObjId p) {
        return p->node.shape() != cv::GNode::NodeShape::CONST_BOUNDED &&
               in_objs_p.find(p) == in_objs_p.end();
    };
    if (ade::util::any_of(reached_sources.visited(), missing_in_proto))
    {
        // TODO: Own exception type or a return code?
      util::throw_error(std::logic_error("Data object reached during unroll "
                                     "wasn\'t found in Protocol"));
    }

    return cv::gimpl::Unrolled{ops.tracked(), origins};
}


cv::gimpl::GModelBuilder::GModelBuilder(ade::Graph &g)
    : m_g(g), m_gm(g)
{
}

cv::gimpl::GModelBuilder::ProtoSlots
cv::gimpl::GModelBuilder::put(const GProtoArgs &ins, const GProtoArgs &outs)
{
    const auto unrolled = cv::gimpl::unrollExpr(ins, outs);

    // First, put all operations and its arguments into graph.
    for (const auto &op_expr_node : unrolled.all_ops)
    {
        GAPI_Assert(op_expr_node.shape() == GNode::NodeShape::CALL);
        const GCall&        call    = op_expr_node.call();
        const GCall::Priv&  call_p  = call.priv();
        ade::NodeHandle     call_h  = put_OpNode(op_expr_node);

        for (const auto it : ade::util::indexed(call_p.m_args))
        {
            const auto  in_port = ade::util::index(it);
            const auto& in_arg  = ade::util::value(it);

            if (proto::is_dynamic(in_arg))
            {
                ade::NodeHandle data_h = put_DataNode(proto::origin_of(in_arg));
                cv::gimpl::GModel::linkIn(m_gm, call_h, data_h, in_port);
            }
        }
    }

    // Then iterate via all "origins", instantiate (if not yet) Data graph nodes
    // and connect these nodes with their producers in graph
    for (const auto &origin : unrolled.all_data)
    {
        const cv::GNode& prod = origin.node;
        GAPI_Assert(prod.shape() != cv::GNode::NodeShape::EMPTY);

        ade::NodeHandle data_h = put_DataNode(origin);
        if (prod.shape() == cv::GNode::NodeShape::CALL)
        {
            ade::NodeHandle call_h = put_OpNode(prod);
            cv::gimpl::GModel::linkOut(m_gm, call_h, data_h, origin.port);
        }
    }

    // Mark graph data nodes as INPUTs and OUTPUTs respectively (according to the protocol)
    for (const auto &arg : ins)
    {
        ade::NodeHandle nh = put_DataNode(proto::origin_of(arg));
        m_gm.metadata(nh).get<Data>().storage = Data::Storage::INPUT;
    }
    for (const auto &arg : outs)
    {
        ade::NodeHandle nh = put_DataNode(proto::origin_of(arg));
        m_gm.metadata(nh).get<Data>().storage = Data::Storage::OUTPUT;
    }

    // And, finally, store data object layout in meta
    GModel::LayoutGraph lg(m_g);
    lg.metadata().set(Layout{m_graph_data});

    // After graph is generated, specify which data objects are actually
    // computation entry/exit points.
    using NodeDescr = std::pair<std::vector<RcDesc>,
                                std::vector<ade::NodeHandle> >;

    const auto get_proto_slots = [&](const GProtoArgs &proto) -> NodeDescr
    {
        NodeDescr slots;

        slots.first.reserve(proto.size());
        slots.second.reserve(proto.size());

        for (const auto &arg : proto)
        {
            ade::NodeHandle nh = put_DataNode(proto::origin_of(arg));
            const auto &desc = m_gm.metadata(nh).get<Data>();
            //These extra empty {} are to please GCC (-Wmissing-field-initializers)
            slots.first.push_back(RcDesc{desc.rc, desc.shape, {}});
            slots.second.push_back(nh);
        }
        return slots;
    };

    auto in_slots  = get_proto_slots(ins);
    auto out_slots = get_proto_slots(outs);
    return ProtoSlots{in_slots.first,  out_slots.first,
                      in_slots.second, out_slots.second};
}

ade::NodeHandle cv::gimpl::GModelBuilder::put_OpNode(const cv::GNode &node)
{
    const auto& node_p = node.priv();
    const auto  it     = m_graph_ops.find(&node_p);
    if (it == m_graph_ops.end())
    {
        GAPI_Assert(node.shape() == GNode::NodeShape::CALL);
        const auto &call_p = node.call().priv();
        auto nh = cv::gimpl::GModel::mkOpNode(m_gm, call_p.m_k, call_p.m_args, call_p.m_params, node_p.m_island);
        m_graph_ops[&node_p] = nh;
        return nh;
    }
    else return it->second;
}

// FIXME: rename to get_DataNode (and same for Op)
ade::NodeHandle cv::gimpl::GModelBuilder::put_DataNode(const GOrigin &origin)
{
    const auto it = m_graph_data.find(origin);
    if (it == m_graph_data.end())
    {
        auto nh = cv::gimpl::GModel::mkDataNode(m_gm, origin);
        m_graph_data[origin] = nh;
        return nh;
    }
    else
    {
        // FIXME: One of the ugliest workarounds ever
        if (it->first.ctor.index() == it->first.ctor.index_of<cv::util::monostate>()
            && origin.ctor.index() !=    origin.ctor.index_of<cv::util::monostate>()) {
            // meanwhile update existing object
            m_gm.metadata(it->second).get<Data>().ctor = origin.ctor;
        }
        return it->second;
    }
}
