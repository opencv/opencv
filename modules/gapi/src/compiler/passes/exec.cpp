// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <string>
#include <list> // list
#include <iomanip> // setw, etc
#include <fstream> // ofstream
#include <memory>
#include <functional>

#include <ade/util/algorithm.hpp>   // contains
#include <ade/util/chain_range.hpp> // chain

#include <opencv2/gapi/util/optional.hpp>  // util::optional
#include "logger.hpp"    // GAPI_LOG

#include "api/gbackend_priv.hpp" // for canMerge()
#include "compiler/gmodel.hpp"
#include "compiler/gislandmodel.hpp"
#include "compiler/passes/passes.hpp"
#include "compiler/passes/helpers.hpp"
#include "compiler/transactions.hpp"

////////////////////////////////////////////////////////////////////////////////
//
// N.B.
// Merge is a binary operation (LHS `Merge` RHS) where LHS may be arbitrary
//
// After every merge, the procedure starts from the beginning (in the topological
// order), thus trying to merge next "unmerged" island to the latest merged one.
//
////////////////////////////////////////////////////////////////////////////////

// Uncomment to dump more info on merge process
// FIXME: make it user-configurable run-time option
// #define DEBUG_MERGE

namespace cv
{
namespace gimpl
{
namespace
{
    bool fusionIsTrivial(const ade::Graph &src_graph)
    {
        // Fusion is considered trivial if there only one
        // active backend and no user-defined islands
        // FIXME:
        // Also check the cases backend can't handle
        // (e.x. GScalar connecting two fluid ops should split the graph)
        const GModel::ConstGraph g(src_graph);
        if (g.metadata().contains<Desynchronized>()) {
            // Fusion of a graph having a desynchronized path is
            // definitely non-trivial
            return false;
        }
        const auto& active_backends = g.metadata().get<ActiveBackends>().backends;
        if (active_backends.size() != 1u) {
            // More than 1 backend involved - non-trivial
            return false;
        }
        const auto& has_island_tags = [&](ade::NodeHandle nh) {
            return g.metadata(nh).contains<Island>();
        };
        if (ade::util::any_of(g.nodes(), has_island_tags)) {
            // There are user-defined islands - non-trivial
            return false;
        }
        if (active_backends.begin()->priv().controlsMerge()) {
            // If the only backend controls Island Fusion on its own - non-trivial
            return false;
        }
        return true;
    }

    void fuseTrivial(GIslandModel::Graph &g, const ade::Graph &src_graph)
    {
        const GModel::ConstGraph src_g(src_graph);

        const auto& backend = *src_g.metadata().get<ActiveBackends>().backends.cbegin();
        const auto& proto = src_g.metadata().get<Protocol>();
        GIsland::node_set all, in_ops, out_ops, in_cvals;

        all.insert(src_g.nodes().begin(), src_g.nodes().end());

        for (const auto& nh : proto.in_nhs)
        {
            all.erase(nh);
            in_ops.insert(nh->outNodes().begin(), nh->outNodes().end());
        }
        for (const auto& nh : proto.out_nhs)
        {
            all.erase(nh);
            out_ops.insert(nh->inNodes().begin(), nh->inNodes().end());
        }
        for (const auto& nh : src_g.nodes())
        {
            if (src_g.metadata(nh).get<NodeType>().t == NodeType::DATA)
            {
                const auto &d = src_g.metadata(nh).get<Data>();
                if (d.storage == Data::Storage::CONST_VAL
                    && !backend.priv().supportsConst(d.shape)) {
                    // don't put this node into the island's graph - so the island
                    // executable don't need to handle value-initialized G-type manually.
                    // Still mark its readers as inputs
                    all.erase(nh);
                    in_cvals.insert(nh);
                    in_ops.insert(nh->outNodes().begin(), nh->outNodes().end());
                }
            }
        }
        auto isl = std::make_shared<GIsland>(backend,
                                             std::move(all),
                                             std::move(in_ops),
                                             std::move(out_ops),
                                             util::optional<std::string>{});

        auto ih = GIslandModel::mkIslandNode(g, std::move(isl));

        for (const auto& nh : ade::util::chain(ade::util::toRange(proto.in_nhs),
                                               ade::util::toRange(in_cvals)))
        {
            auto slot = GIslandModel::mkSlotNode(g, nh);
            g.link(slot, ih);
        }
        for (const auto& nh : proto.out_nhs)
        {
            auto slot = GIslandModel::mkSlotNode(g, nh);
            g.link(ih, slot);
        }
    }

    struct MergeContext
    {
        using CycleCausers = std::pair< std::shared_ptr<GIsland>,
                                        std::shared_ptr<GIsland> >;

        struct CycleHasher final
        {
            std::size_t operator()(const CycleCausers& p) const
            {
                std::size_t a = std::hash<GIsland*>()(p.first.get());
                std::size_t b = std::hash<GIsland*>()(p.second.get());
                return a ^ (b << 1);
            }
        };

        // Set of Islands pairs which cause a cycle if merged.
        // Every new merge produces a new Island, and if Islands were
        // merged (and thus dropped from GIslandModel), the objects may
        // still be alive as included into this set.
        std::unordered_set<CycleCausers, CycleHasher> cycle_causers;
    };

#if defined(__GNUC__) && (__GNUC__ >= 13)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

    bool canMerge(const GIslandModel::Graph &g,
                  const ade::NodeHandle &a_nh,
                  const ade::NodeHandle &slot_nh,
                  const ade::NodeHandle &b_nh,
                  const MergeContext &ctx = MergeContext())
    {
        auto a_ptr = g.metadata(a_nh).get<FusedIsland>().object;
        auto b_ptr = g.metadata(b_nh).get<FusedIsland>().object;
        GAPI_Assert(a_ptr.get());
        GAPI_Assert(b_ptr.get());

        // Islands with different affinity can't be merged
        if (a_ptr->backend() != b_ptr->backend())
            return false;

        // Islands which cause a cycle can't be merged as well
        // (since the flag is set, the procedure already tried to
        // merge these islands in the past)
        if (   ade::util::contains(ctx.cycle_causers, std::make_pair(a_ptr, b_ptr))
            || ade::util::contains(ctx.cycle_causers, std::make_pair(b_ptr, a_ptr)))
            return false;

        // There may be user-defined islands. Initially user-defined
        // islands also are built from single operations and then merged
        // by this procedure, but there is some exceptions.
        // User-specified island can't be merged to an internal island
        if (   ( a_ptr->is_user_specified() && !b_ptr->is_user_specified())
            || (!a_ptr->is_user_specified() &&  b_ptr->is_user_specified()))
        {
            return false;
        }
        else if (a_ptr->is_user_specified() && b_ptr->is_user_specified())
        {
            // These islads are _different_ user-specified Islands
            // FIXME: today it may only differ by name
            if (a_ptr->name() != b_ptr->name())
                return false;
        }

        // If available, run the backend-specified merge checker
        const auto &this_backend_p = a_ptr->backend().priv();
        if (    this_backend_p.controlsMerge()
            && !this_backend_p.allowsMerge(g, a_nh, slot_nh, b_nh))
        {
            return false;
        }
        return true;
    }

#if defined(__GNUC__) && (__GNUC__ == 13)
#pragma GCC diagnostic pop
#endif

    inline bool isProducedBy(const ade::NodeHandle &slot,
                             const ade::NodeHandle &island)
    {
        // A data slot may have only 0 or 1 producer
        if (slot->inNodes().size() == 0)
            return false;

        return slot->inNodes().front() == island;
    }

    inline bool isConsumedBy(const ade::NodeHandle &slot,
                             const ade::NodeHandle &island)
    {
        auto it = std::find_if(slot->outNodes().begin(),
                               slot->outNodes().end(),
                               [&](const ade::NodeHandle &nh) {
                                   return nh == island;
                               });
        return it != slot->outNodes().end();
    }

    /**
     * Find a candidate Island for merge for the given Island nh.
     *
     * @param g Island Model where merge occurs
     * @param nh GIsland node, either LHS or RHS of probable merge
     * @param ctx Merge context, may contain some cached stuff to avoid
     *      double/triple/etc checking
     * @return Tuple of Island handle, Data slot handle (which connects them),
     *      and a position of found handle with respect to nh (IN/OUT)
     */
    std::tuple<ade::NodeHandle, ade::NodeHandle, Direction>
    findCandidate(const GIslandModel::Graph &g,
                  ade::NodeHandle nh,
                  const MergeContext &ctx = MergeContext())
    {
        using namespace std::placeholders;

        // Before checking for candidates, find and ban neighbor nodes
        // (input or outputs) which are connected via desynchronized
        // edges.
        GIsland::node_set nodes_with_desync_edges;
        for (const auto& in_eh : nh->inEdges()) {
            if (g.metadata(in_eh).contains<DesyncIslEdge>()) {
                nodes_with_desync_edges.insert(in_eh->srcNode());
            }
        }
        for (const auto& output_data_nh : nh->outNodes()) {
            for (const auto &out_reader_eh : output_data_nh->outEdges()) {
                if (g.metadata(out_reader_eh).contains<DesyncIslEdge>()) {
                    nodes_with_desync_edges.insert(out_reader_eh->dstNode());
                }
            }
        }

        // Find a first matching candidate GIsland for merge
        // among inputs
        for (const auto& in_eh : nh->inEdges())
        {
            if (ade::util::contains(nodes_with_desync_edges, in_eh->srcNode())) {
                continue; // desync edges can never be fused
            }
            const auto& input_data_nh = in_eh->srcNode();
            if (input_data_nh->inNodes().size() != 0)
            {
                // Data node must have a single producer only
                GAPI_DbgAssert(input_data_nh->inNodes().size() == 1);
                auto input_data_prod_nh = input_data_nh->inNodes().front();
                if (canMerge(g, input_data_prod_nh, input_data_nh, nh, ctx))
                    return std::make_tuple(input_data_prod_nh,
                                           input_data_nh,
                                           Direction::In);
            }
        } // for(inNodes)

        // Ok, now try to find it among the outputs
        for (const auto& output_data_nh : nh->outNodes())
        {
            auto mergeTest = [&](ade::EdgeHandle cons_eh) -> bool {
                if (ade::util::contains(nodes_with_desync_edges, cons_eh->dstNode())) {
                    return false;  // desync edges can never be fused
                }
                return canMerge(g, nh, output_data_nh, cons_eh->dstNode(), ctx);
            };
            auto cand_it = std::find_if(output_data_nh->outEdges().begin(),
                                        output_data_nh->outEdges().end(),
                                        mergeTest);
            if (cand_it != output_data_nh->outEdges().end())
                return std::make_tuple((*cand_it)->dstNode(),
                                       output_data_nh,
                                       Direction::Out);
        } // for(outNodes)
        // Empty handle, no good candidates
        return std::make_tuple(ade::NodeHandle(),
                               ade::NodeHandle(),
                               Direction::Invalid);
    }

    // A cancellable merge of two GIslands, "a" and "b", connected via "slot"
    class MergeAction
    {
        ade::Graph &m_g;
        const ade::Graph &m_orig_g;
        GIslandModel::Graph m_gim;
        ade::NodeHandle m_prod;
        ade::NodeHandle m_slot;
        ade::NodeHandle m_cons;

        using Change = ChangeT<DesyncIslEdge>;
        Change::List m_changes;

        struct MergeObjects
        {
            using NS = GIsland::node_set;
            NS all;      // same as in GIsland
            NS in_ops;   // same as in GIsland
            NS out_ops;  // same as in GIsland
            NS opt_interim_slots;    // can be dropped (optimized out)
            NS non_opt_interim_slots;// can't be dropped (extern. link)
        };
        MergeObjects identify() const;

    public:
        MergeAction(ade::Graph &g,
                    const ade::Graph &orig_g,
                    ade::NodeHandle prod,
                    ade::NodeHandle slot,
                    ade::NodeHandle cons)
            : m_g(g)
            , m_orig_g(orig_g)
            , m_gim(GIslandModel::Graph(m_g))
            , m_prod(prod)
            , m_slot(slot)
            , m_cons(cons)
        {
        }

        void tryMerge(); // Try to merge islands Prod and Cons
        void rollback(); // Roll-back changes if merge has been done but broke the model
        void commit();   // Fix changes in the model after successful merge
    };

    // Merge proceduce is a replacement of two Islands, Prod and Cons,
    // connected via !Slot!, with a new Island, which contain all Prod
    // nodes + all Cons nodes, and reconnected in the graph properly:
    //
    // Merge(Prod, !Slot!, Cons)
    //
    //                                  [Slot 2]
    //                                    :
    //                                    v
    //     ... [Slot 0] -> Prod -> !Slot! -> Cons -> [Slot 3] -> ...
    //     ... [Slot 1] -'           :           '-> [Slot 4] -> ...
    //                               V
    //                              ...
    // results into:
    //
    //     ... [Slot 0] -> Merged  -> [Slot 3] ...
    //     ... [Slot 1] :         :-> [Slot 4] ...
    //     ... [Slot 2] '         '-> !Slot! ...
    //
    // The rules are the following:
    // 1) All Prod input slots become Merged input slots;
    //    - Example: Slot 0 Slot 1
    // 2) Any Cons input slots which come from Islands different to Prod
    //    also become Merged input slots;
    //    - Example: Slot 2
    // 3) All Cons output slots become Merged output slots;
    //    - Example: Slot 3, Slot 4
    // 4) All Prod output slots which are not consumed by Cons
    //    also become Merged output slots;
    //    - (not shown on the example)
    // 5) If the !Slot! which connects Prod and Cons is consumed
    //    exclusively by Cons, it is optimized out (dropped) from the model;
    // 6) If the !Slot! is used by consumers other by Cons, it
    //    becomes an extra output of Merged
    // 7) !Slot! may be not the only connection between Prod and Cons,
    //    but as a result of merge operation, all such connections
    //    should be handles as described for !Slot!

    MergeAction::MergeObjects MergeAction::identify() const
    {
        auto lhs = m_gim.metadata(m_prod).get<FusedIsland>().object;
        auto rhs = m_gim.metadata(m_cons).get<FusedIsland>().object;

        GIsland::node_set interim_slots;

        GIsland::node_set merged_all(lhs->contents());
        merged_all.insert(rhs->contents().begin(), rhs->contents().end());

        GIsland::node_set merged_in_ops(lhs->in_ops());     // (1)
        for (auto cons_in_slot_nh : m_cons->inNodes())      // (2)
        {
            if (isProducedBy(cons_in_slot_nh, m_prod))
            {
                interim_slots.insert(cons_in_slot_nh);
                // at this point, interim_slots are not sync with merged_all
                // (merged_all will be updated with interim_slots which
                // will be optimized out).
            }
            else
            {
                const auto extra_in_ops = rhs->consumers(m_g, cons_in_slot_nh);
                merged_in_ops.insert(extra_in_ops.begin(), extra_in_ops.end());
            }
        }

        GIsland::node_set merged_out_ops(rhs->out_ops());   // (3)
        for (auto prod_out_slot_nh : m_prod->outNodes())    // (4)
        {
            if (!isConsumedBy(prod_out_slot_nh, m_cons))
            {
                merged_out_ops.insert(lhs->producer(m_g, prod_out_slot_nh));
            }
        }

        // (5,6,7)
        GIsland::node_set opt_interim_slots;
        GIsland::node_set non_opt_interim_slots;

        auto is_non_opt = [&](const ade::NodeHandle &slot_nh) {
            // If a data object associated with this slot is a part
            // of GComputation _output_ protocol, it can't be optimzied out
            const auto data_nh = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
            const auto &data = GModel::ConstGraph(m_orig_g).metadata(data_nh).get<Data>();
            if (data.storage == Data::Storage::OUTPUT)
                return true;

            // Otherwise, a non-optimizeable data slot is the one consumed
            // by some other island than "cons"
            const auto it = std::find_if(slot_nh->outNodes().begin(),
                                         slot_nh->outNodes().end(),
                                         [&](ade::NodeHandle &&nh)
                                         {return nh != m_cons;});
            return it != slot_nh->outNodes().end();
        };
        for (auto slot_nh : interim_slots)
        {
            // Put all intermediate data nodes (which are BOTH optimized
            // and not-optimized out) to Island contents.
            merged_all.insert(m_gim.metadata(slot_nh)
                              .get<DataSlot>().original_data_node);

            GIsland::node_set &dst = is_non_opt(slot_nh)
                ? non_opt_interim_slots // there are consumers other than m_cons
                : opt_interim_slots;    // there's no consumers other than m_cons
            dst.insert(slot_nh);
        }

        // (4+6).
        // BTW, (4) could be "All Prod output slots read NOT ONLY by Cons"
        for (auto non_opt_slot_nh : non_opt_interim_slots)
        {
            merged_out_ops.insert(lhs->producer(m_g, non_opt_slot_nh));
        }
        return MergeObjects{
            merged_all, merged_in_ops, merged_out_ops,
            opt_interim_slots, non_opt_interim_slots
        };
    }

    // FIXME(DM): Probably this procedure will be refactored dramatically one day...
    void MergeAction::tryMerge()
    {
        // _: Understand the contents and I/O connections of a new merged Island
        MergeObjects mo = identify();
        auto lhs_obj = m_gim.metadata(m_prod).get<FusedIsland>().object;
        auto rhs_obj = m_gim.metadata(m_cons).get<FusedIsland>().object;
        GAPI_Assert(   ( lhs_obj->is_user_specified() &&  rhs_obj->is_user_specified())
                    || (!lhs_obj->is_user_specified() && !rhs_obj->is_user_specified()));
        cv::util::optional<std::string> maybe_user_tag;
        if (lhs_obj->is_user_specified() && rhs_obj->is_user_specified())
        {
            GAPI_Assert(lhs_obj->name() == rhs_obj->name());
            maybe_user_tag = cv::util::make_optional(lhs_obj->name());
        }

        // A: Create a new Island and add it to the graph
        auto backend = m_gim.metadata(m_prod).get<FusedIsland>()
            .object->backend();
        auto merged = std::make_shared<GIsland>(backend,
                                                std::move(mo.all),
                                                std::move(mo.in_ops),
                                                std::move(mo.out_ops),
                                                std::move(maybe_user_tag));
        // FIXME: move this debugging to some user-controllable log-level
#ifdef DEBUG_MERGE
        merged->debug();
#endif
        auto new_nh = GIslandModel::mkIslandNode(m_gim, std::move(merged));
        m_changes.enqueue<Change::NodeCreated>(new_nh);

        // B: Disconnect all Prod's input Slots from Prod,
        //    connect it to Merged
        std::vector<ade::EdgeHandle> input_edges(m_prod->inEdges().begin(),
                                                 m_prod->inEdges().end());
        for (auto in_edge : input_edges)
        {
            // FIXME: Introduce a Relink primitive instead?
            // (combining the both actions into one?)
            m_changes.enqueue<Change::NewLink>(m_g, in_edge->srcNode(), new_nh, in_edge);
            m_changes.enqueue<Change::DropLink>(m_g, m_prod, in_edge);
        }

        // C: Disconnect all Cons' output Slots from Cons,
        //    connect it to Merged
        std::vector<ade::EdgeHandle> output_edges(m_cons->outEdges().begin(),
                                                  m_cons->outEdges().end());
        for (auto out_edge : output_edges)
        {
            m_changes.enqueue<Change::NewLink>(m_g, new_nh, out_edge->dstNode(), out_edge);
            m_changes.enqueue<Change::DropLink>(m_g, m_cons, out_edge);
        }

        // D: Process the intermediate slots (between Prod n Cons).
        // D/1 - Those which are optimized out are just removed from the model
        for (auto opt_slot_nh : mo.opt_interim_slots)
        {
            GAPI_Assert(1      == opt_slot_nh->inNodes().size() );
            GAPI_Assert(m_prod == opt_slot_nh->inNodes().front());

            std::vector<ade::EdgeHandle> edges_to_drop;
            ade::util::copy(ade::util::chain(opt_slot_nh->inEdges(),
                                             opt_slot_nh->outEdges()),
                            std::back_inserter(edges_to_drop));
            for (auto eh : edges_to_drop)
            {
                m_changes.enqueue<Change::DropLink>(m_g, opt_slot_nh, eh);
            }
            m_changes.enqueue<Change::DropNode>(opt_slot_nh);
        }
        // D/2 - Those which are used externally are connected to new nh
        //       as outputs.
        for (auto non_opt_slot_nh : mo.non_opt_interim_slots)
        {
            GAPI_Assert(1      == non_opt_slot_nh->inNodes().size() );
            GAPI_Assert(m_prod == non_opt_slot_nh->inNodes().front());
            m_changes.enqueue<Change::DropLink>(m_g, non_opt_slot_nh,
                                                non_opt_slot_nh->inEdges().front());

            std::vector<ade::EdgeHandle> edges_to_probably_drop
                (non_opt_slot_nh->outEdges().begin(),
                 non_opt_slot_nh->outEdges().end());;
            for (auto eh : edges_to_probably_drop)
            {
                if (eh->dstNode() == m_cons)
                {
                    // drop only edges to m_cons, as there's other consumers
                    m_changes.enqueue<Change::DropLink>(m_g, non_opt_slot_nh, eh);
                }
            }
            // FIXME: No metadata copied here (from where??)
            // For DesyncIslEdges it still works, as these tags are
            // placed to Data->Op edges and this one is an Op->Data
            // edge.
            m_changes.enqueue<Change::NewLink>(m_g, new_nh, non_opt_slot_nh);
        }

        // E. All Prod's output edges which are directly related to Merge (e.g.
        //    connected to Cons) were processed on step (D).
        //    Relink the remaining output links
        std::vector<ade::EdgeHandle>  prod_extra_out_edges
            (m_prod->outEdges().begin(),
             m_prod->outEdges().end());
        for (auto extra_out : prod_extra_out_edges)
        {
            m_changes.enqueue<Change::NewLink>(m_g, new_nh, extra_out->dstNode(), extra_out);
            m_changes.enqueue<Change::DropLink>(m_g, m_prod, extra_out);
        }

        // F. All Cons' input edges which are directly related to merge (e.g.
        //    connected to Prod) were processed on step (D) as well,
        //    remaining should become Merged island's input edges
        std::vector<ade::EdgeHandle> cons_extra_in_edges
            (m_cons->inEdges().begin(),
             m_cons->inEdges().end());
        for (auto extra_in : cons_extra_in_edges)
        {
            m_changes.enqueue<Change::NewLink>(m_g, extra_in->srcNode(), new_nh, extra_in);
            m_changes.enqueue<Change::DropLink>(m_g, m_cons, extra_in);
        }

        // G. Finally, drop the original Island nodes. DropNode() does
        //    the sanity check for us (both nodes should have 0 edges).
        m_changes.enqueue<Change::DropNode>(m_prod);
        m_changes.enqueue<Change::DropNode>(m_cons);
    }

    void MergeAction::rollback()
    {
        m_changes.rollback(m_g);
    }
    void MergeAction::commit()
    {
        m_changes.commit(m_g);
    }

#ifdef DEBUG_MERGE
    void merge_debug(const ade::Graph &g, int iteration)
    {
        std::stringstream filename;
        filename << "fusion_" << static_cast<const void*>(&g)
                 << "_" << std::setw(4) << std::setfill('0') << iteration
                 << ".dot";
        std::ofstream ofs(filename.str());
        passes::dumpDot(g, ofs);
    }
#endif

    void fuseGeneral(ade::Graph& im, const ade::Graph& g)
    {
        GIslandModel::Graph gim(im);
        MergeContext mc;

        bool there_was_a_merge = false;
#ifdef DEBUG_MERGE
        std::size_t iteration = 0u;
#endif
        do
        {
            there_was_a_merge = false;

            // FIXME: move this debugging to some user-controllable log level
#ifdef DEBUG_MERGE
            GAPI_LOG_INFO(NULL, "Before next merge attempt " << iteration << "...");
            merge_debug(g, iteration);
            iteration++;
#endif
            auto sorted = pass_helpers::topoSort(im);
            for (auto nh : sorted)
            {
                if (NodeKind::ISLAND == gim.metadata(nh).get<NodeKind>().k)
                {
                    ade::NodeHandle cand_nh;
                    ade::NodeHandle cand_slot;
                    Direction dir = Direction::Invalid;
                    std::tie(cand_nh, cand_slot, dir) = findCandidate(gim, nh, mc);
                    if (cand_nh != nullptr && dir != Direction::Invalid)
                    {
                        auto lhs_nh = (dir == Direction::In  ? cand_nh : nh);
                        auto rhs_nh = (dir == Direction::Out ? cand_nh : nh);

                        auto l_obj = gim.metadata(lhs_nh).get<FusedIsland>().object;
                        auto r_obj = gim.metadata(rhs_nh).get<FusedIsland>().object;
                        GAPI_LOG_INFO(NULL, r_obj->name() << " can be merged into " << l_obj->name());
                        // Try to do a merge. If merge was successful, check if the
                        // graph have cycles (cycles are prohibited at this point).
                        // If there are cycles, roll-back the merge and mark a pair of
                        // these Islands with a special tag - "cycle-causing".
                        MergeAction action(im, g, lhs_nh, cand_slot, rhs_nh);
                        action.tryMerge();
                        if (pass_helpers::hasCycles(im))
                        {
                            GAPI_LOG_INFO(NULL,
                                          "merge(" << l_obj->name() << "," << r_obj->name() <<
                                          ") caused cycle, rolling back...");
                            action.rollback();
                            // don't try to merge these two islands next time (findCandidate will use that)
                            mc.cycle_causers.insert({l_obj, r_obj});
                        }
                        else
                        {
                            GAPI_LOG_INFO(NULL,
                                          "merge(" << l_obj->name() << "," << r_obj->name() <<
                                          ") was successful!");
                            action.commit();
#ifdef DEBUG_MERGE
                            GIslandModel::syncIslandTags(gim, g);
#endif
                            there_was_a_merge = true;
                            break; // start do{}while from the beginning
                        }
                    } // if(can merge)
                } // if(ISLAND)
            } // for(all nodes)
        }
        while (there_was_a_merge);
    }
}  // anonymous namespace

void passes::fuseIslands(ade::passes::PassContext &ctx)
{
    std::shared_ptr<ade::Graph> gptr(new ade::Graph);
    GIslandModel::Graph gim(*gptr);

    if (fusionIsTrivial(ctx.graph))
    {
        fuseTrivial(gim, ctx.graph);
    }
    else
    {
        GIslandModel::generateInitial(gim, ctx.graph);
        fuseGeneral(*gptr.get(), ctx.graph);
    }
    GModel::Graph(ctx.graph).metadata().set(IslandModel{std::move(gptr)});
}

void passes::syncIslandTags(ade::passes::PassContext &ctx)
{
    GModel::Graph gm(ctx.graph);
    std::shared_ptr<ade::Graph> gptr(gm.metadata().get<IslandModel>().model);
    GIslandModel::Graph gim(*gptr);
    GIslandModel::syncIslandTags(gim, ctx.graph);
}

void passes::topoSortIslands(ade::passes::PassContext &ctx)
{
    GModel::Graph gm(ctx.graph);
    std::shared_ptr<ade::Graph> gptr(gm.metadata().get<IslandModel>().model);
    auto pass_ctx = ade::passes::PassContext{*gptr};
    ade::passes::TopologicalSort{}(pass_ctx);
}
}} // namespace cv::gimpl
