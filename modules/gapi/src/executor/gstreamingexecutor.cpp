// This file is part of OpenCV project.

// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"


#include <iostream>

#include <ade/util/zip_range.hpp>

#if !defined(GAPI_STANDALONE)
#include <opencv2/videoio.hpp>   // FIXME: MOVE THIS DEPENDENCY OUT!!1
#endif // GAPI_STANDALONE
#include <opencv2/gapi/opencv_includes.hpp>
#include "executor/gstreamingexecutor.hpp"
#include "compiler/passes/passes.hpp"

namespace
{
#if !defined(GAPI_STANDALONE)
    class VideoEmitter final: public cv::gimpl::GIslandEmitter {
        // FIXME: This is a huge dependency for core G-API library!
        // It needs to be moved out to some separate module.
        cv::VideoCapture vcap;

        virtual bool pull(cv::GRunArg &arg) override {
            // FIXME: probably we can maintain a pool of (then) pre-allocated
            // buffers to avoid runtime allocations.
            // Pool size can be determined given the internal queue size.
            cv::Mat nextFrame;
            vcap >> nextFrame;
            if (nextFrame.empty()) {
                return false;
            }
            arg = std::move(nextFrame);
            return true;
        }
    public:

        explicit VideoEmitter(const cv::GRunArg &arg) {
            const auto &param = cv::util::get<cv::gapi::GVideoCapture>(arg);
            GAPI_Assert(vcap.open(param.path));
        }
    };
#endif // GAPI_STANDALONE

    class ConstEmitter final: public cv::gimpl::GIslandEmitter {
        cv::GRunArg m_arg;

        virtual bool pull(cv::GRunArg &arg) override {
            arg = m_arg;
            return true;
        }
    public:

        explicit ConstEmitter(const cv::GRunArg &arg) : m_arg(arg) {
        }
    };
} // anonymous namespace

cv::gimpl::GStreamingExecutor::GStreamingExecutor(std::unique_ptr<ade::Graph> &&g_model)
    : m_orig_graph(std::move(g_model))
    , m_island_graph(GModel::Graph(*m_orig_graph).metadata()
                     .get<IslandModel>().model)
    , m_gm(*m_orig_graph)
    , m_gim(*m_island_graph)
{
    // NB: Right now GIslandModel is acyclic, and all the below code assumes that.
    // NB: This naive execution code is taken from GExecutor nearly "as-is"

    const auto proto = m_gm.metadata().get<Protocol>();
    m_emitters.resize(proto.in_nhs.size());

    auto sorted = m_gim.metadata().get<ade::passes::TopologicalSortData>();
    for (auto nh : sorted.nodes())
    {
        switch (m_gim.metadata(nh).get<NodeKind>().k)
        {
        case NodeKind::ISLAND:
            {
                std::vector<RcDesc> input_rcs;
                std::vector<RcDesc> output_rcs;
                input_rcs.reserve(nh->inNodes().size());
                output_rcs.reserve(nh->outNodes().size());

                auto xtract = [&](ade::NodeHandle slot_nh, std::vector<RcDesc> &vec) {
                    const auto orig_data_nh
                        = m_gim.metadata(slot_nh).get<DataSlot>().original_data_node;
                    const auto &orig_data_info
                        = m_gm.metadata(orig_data_nh).get<Data>();
                    vec.emplace_back(RcDesc{ orig_data_info.rc
                                           , orig_data_info.shape
                                           , orig_data_info.ctor});
                };
                // (3)
                for (auto in_slot_nh  : nh->inNodes())  xtract(in_slot_nh,  input_rcs);
                for (auto out_slot_nh : nh->outNodes()) xtract(out_slot_nh, output_rcs);

                m_ops.emplace_back(OpDesc{ std::move(input_rcs)
                                         , std::move(output_rcs)
                                         , m_gim.metadata(nh).get<IslandExec>().object});
            }
            break;

        case NodeKind::SLOT:
            {
                const auto orig_data_nh
                    = m_gim.metadata(nh).get<DataSlot>().original_data_node;
                initResource(orig_data_nh);
                m_slots.emplace_back(DataDesc{nh, orig_data_nh});
            }
            break;

        case NodeKind::EMIT:
            {
                const auto emitter_idx
                    = m_gim.metadata(nh).get<Emitter>().proto_index;
                GAPI_Assert(emitter_idx < m_emitters.size());
                m_emitters[emitter_idx] = nh;
            }
            break;

        default:
            GAPI_Assert(false);
            break;
        } // switch(kind)
    } // for(gim nodes)
}

void cv::gimpl::GStreamingExecutor::initResource(const ade::NodeHandle &orig_nh)
{
    const Data &d = m_gm.metadata(orig_nh).get<Data>();

    if (   d.storage != Data::Storage::INTERNAL
        && d.storage != Data::Storage::CONST_VAL)
        return;

    // INTERNALS+CONST only! no need to allocate/reset output objects
    // to as it is bound externally (e.g. already in the m_res)

    switch (d.shape)
    {
    case GShape::GMAT:
        {
            const auto desc = util::get<cv::GMatDesc>(d.meta);
            auto& mat = m_res.slot<cv::gapi::own::Mat>()[d.rc];
            createMat(desc, mat);
        }
        break;

    case GShape::GSCALAR:
        if (d.storage == Data::Storage::CONST_VAL)
        {
            auto rc = RcDesc{d.rc, d.shape, d.ctor};
            magazine::bindInArg(m_res, rc, m_gm.metadata(orig_nh).get<ConstValue>().arg);
        }
        break;

    case GShape::GARRAY:
        // Constructed on Reset, do nothing here
        break;

    default:
        GAPI_Assert(false);
    }
}

void cv::gimpl::GStreamingExecutor::setSource(GRunArgs &&ins)
{
    bool has_video = false;

    // Walk through the protocol, set-up emitters appropriately
    // There's a 1:1 mapping between emitters and corresponding data inputs.
    for (auto it : ade::util::zip(ade::util::toRange(m_emitters),
                                  ade::util::toRange(ins)))
    {
        auto  emit_nh  = std::get<0>(it);
        auto& emit_arg = std::get<1>(it);
        auto& emitter  = m_gim.metadata(emit_nh).get<Emitter>().object;

        switch (emit_arg.index())
        {
        // Create a streaming emitter.
        // Produces the next video frame when pulled.
        case emit_arg.index_of<cv::gapi::GVideoCapture>():
            if (has_video)
                util::throw_error(std::logic_error("Only one video source is"
                                                   " currently supported!"));
            has_video = true;
#if !defined(GAPI_STANDALONE)
            emitter.reset(new VideoEmitter{emit_arg});
#else
            util::throw_error(std::logic_error("Video is not supported in the "
                                               "standalone mode"));
#endif
            break;

        default:
            // Create a constant emitter.
            // Produces always the same ("constant") value when pulled.
            emitter.reset(new ConstEmitter{emit_arg});
            break;
        }
    }
}

void cv::gimpl::GStreamingExecutor::start()
{
    // FIXME: there's nothig to do here now,
    // but in fact this method should activate emitters to work
}

bool cv::gimpl::GStreamingExecutor::pull(cv::GRunArgsP &&outs)
{
    // FIXME: This method should only obtain data which is already processed,
    // or block until it arrives.
    // Currently it is more like GExecutor::run() with some changes

    const auto& proto = m_gm.metadata().get<Protocol>();

    // First, poll data emitters to obtain "next data frame"
    std::vector<GRunArg> this_shot;
    this_shot.resize(proto.inputs.size());
    for (auto it : ade::util::indexed(m_emitters))
    {
        auto idx = ade::util::index(it);
        auto eh  = ade::util::value(it);

        if (!m_gim.metadata(eh).get<Emitter>().object->pull(this_shot[idx])) {
            // Emitter pull failure means end-of-stream
            return false;
        }
    }
    // FIXME: probably these two cycles can be merged into one
    for (auto it : ade::util::zip(ade::util::toRange(proto.inputs),
                                  ade::util::toRange(this_shot)))
    {
        magazine::bindInArg(m_res, std::get<0>(it), std::get<1>(it));
    }

    //ensure that output Mat parameters are correctly allocated
    for (auto index : ade::util::iota(proto.out_nhs.size()) )     //FIXME: avoid copy of NodeHandle and GRunRsltComp ?
    {
        auto& nh = proto.out_nhs.at(index);
        const Data &d = m_gm.metadata(nh).get<Data>();
        if (d.shape == GShape::GMAT)
        {
            using cv::util::get;
            const auto desc = get<cv::GMatDesc>(d.meta);

            auto check_own_mat = [&desc, &outs, &index]()
            {
                auto& out_mat = *get<cv::gapi::own::Mat*>(outs.at(index));
                GAPI_Assert(out_mat.data != nullptr &&
                        desc.canDescribe(out_mat));
            };

#if !defined(GAPI_STANDALONE)
            if (cv::util::holds_alternative<cv::Mat*>(outs.at(index)))
            {
                auto& out_mat = *get<cv::Mat*>(outs.at(index));
                createMat(desc, out_mat);
            }
            // In the case of own::Mat never reallocated, checked to perfectly fit required meta
            else
#endif
            {
                check_own_mat();
            }
        }
    }
    // Bind results to storage
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(outs)))
    {
        magazine::bindOutArg(m_res, std::get<0>(it), std::get<1>(it));
    }

    // Reset internal data
    for (auto &sd : m_slots)
    {
        const auto& data = m_gm.metadata(sd.data_nh).get<Data>();
        magazine::resetInternalData(m_res, data);
    }

    // Run the script
    for (auto &op : m_ops)
    {
        // (5)
        using InObj  = GIslandExecutable::InObj;
        using OutObj = GIslandExecutable::OutObj;
        std::vector<InObj>  in_objs;
        std::vector<OutObj> out_objs;
        in_objs.reserve (op.in_objects.size());
        out_objs.reserve(op.out_objects.size());

        for (const auto &rc : op.in_objects)
        {
            in_objs.emplace_back(InObj{rc, magazine::getArg(m_res, rc)});
        }
        for (const auto &rc : op.out_objects)
        {
            out_objs.emplace_back(OutObj{rc, magazine::getObjPtr(m_res, rc)});
        }

        // (6)
        op.isl_exec->run(std::move(in_objs), std::move(out_objs));
    }

    // (7)
    for (auto it : ade::util::zip(ade::util::toRange(proto.outputs),
                                  ade::util::toRange(outs)))
    {
        magazine::writeBack(m_res, std::get<0>(it), std::get<1>(it));
    }

    return true; // Obtained data succesfully
}

void cv::gimpl::GStreamingExecutor::stop()
{
    // FIXME: there's nothing to do here now,
    // but in fact once pipeline is stopped explicitly,
    // pull() should start doing nothing and return false.
    // FIXME: ...and how to deal with still-unread data then?
}
