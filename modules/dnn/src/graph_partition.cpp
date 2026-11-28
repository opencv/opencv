// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "graph_partition.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

static inline bool isPlacedOp(const std::vector<Ptr<LayerInfo> >& prog,
                              const std::vector<uchar>& hostOp, size_t i)
{
    return prog[i] && !hostOp[i];
}

void demoteShortRuns(const std::vector<Ptr<LayerInfo> >& prog, const std::vector<uchar>& hostOp,
                     std::vector<int>& placement, size_t minSize)
{
    const size_t nops = prog.size();
    CV_Assert(hostOp.size() == nops && placement.size() == nops);
    bool haveCPU = false, haveAccel = false;
    for (size_t i = 0; i < nops; i++) {
        if (!isPlacedOp(prog, hostOp, i))
            continue;
        if (placement[i] == DNN_BACKEND_OPENCV)
            haveCPU = true;
        else
            haveAccel = true;
    }
    if (!haveCPU || !haveAccel || minSize <= 1)
        return;

    size_t i = 0;
    while (i < nops) {
        if (!isPlacedOp(prog, hostOp, i) || placement[i] == DNN_BACKEND_OPENCV) {
            i++;
            continue;
        }
        const int backend = placement[i];
        const size_t first = i;
        size_t last = i, count = 0;
        for (; i < nops; i++) {
            if (!isPlacedOp(prog, hostOp, i))
                continue;
            if (placement[i] != backend)
                break;
            count++;
            last = i;
        }
        if (count < minSize) {
            for (size_t k = first; k <= last; k++) {
                if (isPlacedOp(prog, hostOp, k) && placement[k] == backend)
                    placement[k] = DNN_BACKEND_OPENCV;
            }
        }
    }
}

void buildSegments(const std::vector<Ptr<LayerInfo> >& prog, const std::vector<uchar>& hostOp,
                   const std::vector<int>& placement, const std::vector<Arg>& graphOutputs,
                   const std::vector<uchar>& constArg,
                   std::vector<GraphSegment>& segments, std::vector<ArgTransition>& transitions)
{
    const size_t nops = prog.size();
    CV_Assert(hostOp.size() == nops && placement.size() == nops);
    segments.clear();
    transitions.clear();

    int pendingFirst = -1, lastHostOp = -1;
    for (size_t i = 0; i < nops; i++) {
        if (!prog[i])
            continue;
        if (hostOp[i]) {
            if (!segments.empty())
                segments.back().last = (int)i;
            else if (pendingFirst < 0)
                pendingFirst = (int)i;
            lastHostOp = (int)i;
            continue;
        }
        if (!segments.empty() && segments.back().backend == placement[i]) {
            segments.back().last = (int)i;
            segments.back().nops++;
        } else {
            segments.push_back(GraphSegment{placement[i], pendingFirst >= 0 ? pendingFirst : (int)i, (int)i, 1});
            pendingFirst = -1;
        }
    }
    if (segments.empty() && pendingFirst >= 0)
        segments.push_back(GraphSegment{DNN_BACKEND_OPENCV, pendingFirst, lastHostOp, 0});

    // where each value currently lives; anything not produced here comes from the host
    std::map<int, int> producedOn;
    std::set<std::pair<int, int> > recorded;
    for (size_t i = 0; i < nops; i++) {
        const Ptr<LayerInfo>& op = prog[i];
        if (!op)
            continue;
        const int consumerBackend = hostOp[i] ? (int)DNN_BACKEND_OPENCV : placement[i];
        const bool headerOnly = hostOp[i] && op->type == "Shape";
        for (const Arg& inp : op->inputs) {
            if (inp.empty() || headerOnly || ((size_t)inp.idx < constArg.size() && constArg[inp.idx]))
                continue;
            auto it = producedOn.find(inp.idx);
            const int from = it != producedOn.end() ? it->second : (int)DNN_BACKEND_OPENCV;
            if (from != consumerBackend && recorded.insert(std::make_pair(inp.idx, consumerBackend)).second)
                transitions.push_back(ArgTransition{inp, from, consumerBackend, (int)i});
        }
        for (const Arg& out : op->outputs) {
            if (!out.empty())
                producedOn[out.idx] = consumerBackend;
        }
    }
    for (const Arg& out : graphOutputs) {
        auto it = producedOn.find(out.idx);
        if (it != producedOn.end() && it->second != DNN_BACKEND_OPENCV &&
            recorded.insert(std::make_pair(out.idx, (int)DNN_BACKEND_OPENCV)).second)
            transitions.push_back(ArgTransition{out, it->second, DNN_BACKEND_OPENCV, -1});
    }
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
