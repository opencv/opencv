// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <functional>
#include <iostream>
#include <iomanip> // std::fixed, std::setprecision
#include <set>
#include <unordered_set>
#include <stack>

#include <ade/util/algorithm.hpp>
#include <ade/util/chain_range.hpp>
#include <ade/util/iota_range.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>

#include <ade/typed_graph.hpp>
#include <ade/execution_engine/execution_engine.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include "logger.hpp"

#include <opencv2/gapi/gmat.hpp>    //for version of descr_of
// PRIVATE STUFF!
#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/fluid/gfluidbuffer_priv.hpp"
#include "backends/fluid/gfluidbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GFluidModel = ade::TypedGraph
    < cv::gimpl::FluidUnit
    , cv::gimpl::FluidData
    , cv::gimpl::Protocol
    , cv::gimpl::FluidUseOwnBorderBuffer
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstFluidModel = ade::ConstTypedGraph
    < cv::gimpl::FluidUnit
    , cv::gimpl::FluidData
    , cv::gimpl::Protocol
    , cv::gimpl::FluidUseOwnBorderBuffer
    >;

// FluidBackend middle-layer implementation ////////////////////////////////////
namespace
{
    class GFluidBackendImpl final: public cv::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const cv::GKernelImpl &impl) override
        {
            GFluidModel fm(graph);
            auto fluid_impl = cv::util::any_cast<cv::GFluidKernel>(impl.opaque);
            fm.metadata(op_node).set(cv::gimpl::FluidUnit{fluid_impl, {}, 0, -1, {}, 0.0});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &args,
                             const std::vector<ade::NodeHandle> &nodes) const override
        {
            using namespace cv::gimpl;
            GModel::ConstGraph g(graph);
            auto isl_graph = g.metadata().get<IslandModel>().model;
            GIslandModel::Graph gim(*isl_graph);

            const auto num_islands = std::count_if
                (gim.nodes().begin(), gim.nodes().end(),
                 [&](const ade::NodeHandle &nh) {
                    return gim.metadata(nh).get<NodeKind>().k == NodeKind::ISLAND;
                });

            const auto out_rois = cv::gapi::getCompileArg<cv::GFluidOutputRois>(args);
            if (num_islands > 1 && out_rois.has_value())
                cv::util::throw_error(std::logic_error("GFluidOutputRois feature supports only one-island graphs"));

            auto rois = out_rois.value_or(cv::GFluidOutputRois());

            auto graph_data = fluidExtractInputDataFromGraph(graph, nodes);
            const auto parallel_out_rois = cv::gapi::getCompileArg<cv::GFluidParallelOutputRois>(args);
            const auto gpfor             = cv::gapi::getCompileArg<cv::GFluidParallelFor>(args);

#if !defined(GAPI_STANDALONE)
            auto default_pfor = [](std::size_t count, std::function<void(std::size_t)> f){
                struct Body : cv::ParallelLoopBody {
                    decltype(f) func;
                    Body( decltype(f) && _f) : func(_f){}
                    virtual void operator() (const cv::Range& r) const CV_OVERRIDE
                    {
                        for (std::size_t i : ade::util::iota(r.start, r.end))
                        {
                            func(i);
                        }
                    }
                };
                cv::parallel_for_(cv::Range{0,static_cast<int>(count)}, Body{std::move(f)});
            };
#else
            auto default_pfor = [](std::size_t count, std::function<void(std::size_t)> f){
                for (auto i : ade::util::iota(count)){
                    f(i);
                }
            };
#endif

            auto pfor  = gpfor.has_value() ? gpfor.value().parallel_for : default_pfor;

            return parallel_out_rois.has_value() ?
                       EPtr{new cv::gimpl::GParallelFluidExecutable (graph, graph_data, std::move(parallel_out_rois.value().parallel_rois), pfor)}
                     : EPtr{new cv::gimpl::GFluidExecutable         (graph, graph_data, std::move(rois.rois))}
            ;
        }

        virtual void addMetaSensitiveBackendPasses(ade::ExecutionEngineSetupContext &ectx) override;

    };
}

cv::gapi::GBackend cv::gapi::fluid::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GFluidBackendImpl>());
    return this_backend;
}

// FluidAgent implementation ///////////////////////////////////////////////////

namespace cv { namespace gimpl {
struct FluidMapper
{
    FluidMapper(double ratio, int lpi) : m_ratio(ratio), m_lpi(lpi) {}
    virtual ~FluidMapper() = default;
    virtual int firstWindow(int outCoord, int lpi) const = 0;
    virtual std::pair<int,int> linesReadAndNextWindow(int outCoord, int lpi) const = 0;

protected:
    double m_ratio = 0.0;
    int    m_lpi   = 0;
};

struct FluidDownscaleMapper : public FluidMapper
{
    virtual int firstWindow(int outCoord, int lpi) const override;
    virtual std::pair<int,int> linesReadAndNextWindow(int outCoord, int lpi) const override;
    using FluidMapper::FluidMapper;
};

struct FluidUpscaleMapper : public FluidMapper
{
    virtual int firstWindow(int outCoord, int lpi) const override;
    virtual std::pair<int,int> linesReadAndNextWindow(int outCoord, int lpi) const override;
    FluidUpscaleMapper(double ratio, int lpi, int inHeight) : FluidMapper(ratio, lpi), m_inHeight(inHeight) {}
private:
    int m_inHeight = 0;
};

struct FluidFilterAgent : public FluidAgent
{
private:
    virtual int firstWindow(std::size_t inPort) const override;
    virtual std::pair<int,int> linesReadAndnextWindow(std::size_t inPort) const override;
    virtual void setRatio(double) override { /* nothing */ }
public:
    using FluidAgent::FluidAgent;
    int m_window;

    FluidFilterAgent(const ade::Graph &g, ade::NodeHandle nh)
        : FluidAgent(g, nh)
        , m_window(GConstFluidModel(g).metadata(nh).get<FluidUnit>().window)
    {}
};

struct FluidResizeAgent : public FluidAgent
{
private:
    virtual int firstWindow(std::size_t inPort) const override;
    virtual std::pair<int,int> linesReadAndnextWindow(std::size_t inPort) const override;
    virtual void setRatio(double ratio) override;

    std::unique_ptr<FluidMapper> m_mapper;
public:
    using FluidAgent::FluidAgent;
};

struct Fluid420toRGBAgent : public FluidAgent
{
private:
    virtual int firstWindow(std::size_t inPort) const override;
    virtual std::pair<int,int> linesReadAndnextWindow(std::size_t inPort) const override;
    virtual void setRatio(double) override { /* nothing */ }
public:
    using FluidAgent::FluidAgent;
};
}} // namespace cv::gimpl

cv::gimpl::FluidAgent::FluidAgent(const ade::Graph &g, ade::NodeHandle nh)
    : k(GConstFluidModel(g).metadata(nh).get<FluidUnit>().k)        // init(0)
    , op_handle(nh)                                                 // init(1)
    , op_name(GModel::ConstGraph(g).metadata(nh).get<Op>().k.name)  // init(2)
{
    std::set<int> out_w;
    std::set<int> out_h;
    GModel::ConstGraph cm(g);
    for (auto out_data : nh->outNodes())
    {
        const auto  &d      = cm.metadata(out_data).get<Data>();
        cv::GMatDesc d_meta = cv::util::get<cv::GMatDesc>(d.meta);
        out_w.insert(d_meta.size.width);
        out_h.insert(d_meta.size.height);
    }

    // Different output sizes are not supported
    GAPI_Assert(out_w.size() == 1 && out_h.size() == 1);
}

void cv::gimpl::FluidAgent::reset()
{
    m_producedLines = 0;

    for (const auto it : ade::util::indexed(in_views))
    {
        auto& v = ade::util::value(it);
        if (v)
        {
            auto idx = ade::util::index(it);
            auto lines = firstWindow(idx);
            v.priv().reset(lines);
        }
    }
}

namespace {
static int calcGcd (int n1, int n2)
{
    return (n2 == 0) ? n1 : calcGcd (n2, n1 % n2);
}

// This is an empiric formula and this is not 100% guaranteed
// that it produces correct results in all possible cases
// FIXME:
// prove correctness or switch to some trusted method
//
// When performing resize input/output pixels form a cyclic
// pattern where inH/gcd input pixels are mapped to outH/gcd
// output pixels (pattern repeats gcd times).
//
// Output pixel can partually cover some of the input pixels.
// There are 3 possible cases:
//
// :___ ___:    :___ _:_ ___:    :___ __: ___ :__ ___:
// |___|___|    |___|_:_|___|    |___|__:|___|:__|___|
// :       :    :     :     :    :      :     :      :
//
// 1) No partial coverage, max window = scaleFactor;
// 2) Partial coverage occurs on the one side of the output pixel,
//    max window = scaleFactor + 1;
// 3) Partial coverage occurs at both sides of the output pixel,
//    max window = scaleFactor + 2;
//
// Type of the coverage is determined by remainder of
// inPeriodH/outPeriodH division, but it's an heuristic
// (howbeit didn't found the proof of the opposite so far).

static int calcResizeWindow(int inH, int outH)
{
    GAPI_Assert(inH >= outH);
    auto gcd = calcGcd(inH, outH);
    int  inPeriodH =  inH/gcd;
    int outPeriodH = outH/gcd;
    int scaleFactor = inPeriodH / outPeriodH;

    switch ((inPeriodH) % (outPeriodH))
    {
    case 0:  return scaleFactor;     break;
    case 1:  return scaleFactor + 1; break;
    default: return scaleFactor + 2;
    }
}

static int maxLineConsumption(const cv::GFluidKernel::Kind kind, int window, int inH, int outH, int lpi, std::size_t inPort)
{
    switch (kind)
    {
    case cv::GFluidKernel::Kind::Filter: return window + lpi - 1; break;
    case cv::GFluidKernel::Kind::Resize:
    {
        if  (inH >= outH)
        {
            // FIXME:
            // This is a suboptimal value, can be reduced
            return calcResizeWindow(inH, outH) * lpi;
        }
        else
        {
            // FIXME:
            // This is a suboptimal value, can be reduced
            return (inH == 1) ? 1 : 2 + lpi - 1;
        }
    } break;
    case cv::GFluidKernel::Kind::YUV420toRGB: return inPort == 0 ? 2 : 1; break;
    default: GAPI_Assert(false); return 0;
    }
}

static int borderSize(const cv::GFluidKernel::Kind kind, int window)
{
    switch (kind)
    {
    case cv::GFluidKernel::Kind::Filter: return (window - 1) / 2; break;
    // Resize never reads from border pixels
    case cv::GFluidKernel::Kind::Resize: return 0; break;
    case cv::GFluidKernel::Kind::YUV420toRGB: return 0; break;
    default: GAPI_Assert(false); return 0;
    }
}

inline double inCoord(int outIdx, double ratio)
{
    return outIdx * ratio;
}

inline int windowStart(int outIdx, double ratio)
{
    return static_cast<int>(inCoord(outIdx, ratio) + 1e-3);
}

inline int windowEnd(int outIdx, double ratio)
{
    return static_cast<int>(std::ceil(inCoord(outIdx + 1, ratio) - 1e-3));
}

inline double inCoordUpscale(int outCoord, double ratio)
{
    // Calculate the projection of output pixel's center
    return (outCoord + 0.5) * ratio - 0.5;
}

inline int upscaleWindowStart(int outCoord, double ratio)
{
    int start = static_cast<int>(inCoordUpscale(outCoord, ratio));
    GAPI_DbgAssert(start >= 0);
    return start;
}

inline int upscaleWindowEnd(int outCoord, double ratio, int inSz)
{
    int end = static_cast<int>(std::ceil(inCoordUpscale(outCoord, ratio)) + 1);
    if (end > inSz)
    {
        end = inSz;
    }
    return end;
}
} // anonymous namespace

int cv::gimpl::FluidDownscaleMapper::firstWindow(int outCoord, int lpi) const
{
    return windowEnd(outCoord + lpi - 1, m_ratio) - windowStart(outCoord, m_ratio);
}

std::pair<int,int> cv::gimpl::FluidDownscaleMapper::linesReadAndNextWindow(int outCoord, int lpi) const
{
    auto nextStartIdx = outCoord + 1 + m_lpi - 1;
    auto nextEndIdx   = nextStartIdx + lpi - 1;

    auto currStart = windowStart(outCoord, m_ratio);
    auto nextStart = windowStart(nextStartIdx, m_ratio);
    auto nextEnd   = windowEnd(nextEndIdx, m_ratio);

    auto lines_read = nextStart - currStart;
    auto next_window = nextEnd - nextStart;

    return std::make_pair(lines_read, next_window);
}

int cv::gimpl::FluidUpscaleMapper::firstWindow(int outCoord, int lpi) const
{
    return upscaleWindowEnd(outCoord + lpi - 1, m_ratio, m_inHeight) - upscaleWindowStart(outCoord, m_ratio);
}

std::pair<int,int> cv::gimpl::FluidUpscaleMapper::linesReadAndNextWindow(int outCoord, int lpi) const
{
    auto nextStartIdx = outCoord + 1 + m_lpi - 1;
    auto nextEndIdx   = nextStartIdx + lpi - 1;

    auto currStart = upscaleWindowStart(outCoord, m_ratio);
    auto nextStart = upscaleWindowStart(nextStartIdx, m_ratio);
    auto nextEnd   = upscaleWindowEnd(nextEndIdx, m_ratio, m_inHeight);

    auto lines_read = nextStart - currStart;
    auto next_window = nextEnd - nextStart;

    return std::make_pair(lines_read, next_window);
}

int cv::gimpl::FluidFilterAgent::firstWindow(std::size_t) const
{
    int lpi = std::min(k.m_lpi, m_outputLines - m_producedLines);
    return m_window + lpi - 1;
}

std::pair<int,int> cv::gimpl::FluidFilterAgent::linesReadAndnextWindow(std::size_t) const
{
    int lpi = std::min(k.m_lpi, m_outputLines - m_producedLines - k.m_lpi);
    return std::make_pair(k.m_lpi, m_window - 1 + lpi);
}

int cv::gimpl::FluidResizeAgent::firstWindow(std::size_t) const
{
    auto outIdx = out_buffers[0]->priv().y();
    auto lpi = std::min(m_outputLines - m_producedLines, k.m_lpi);
    return m_mapper->firstWindow(outIdx, lpi);
}

std::pair<int,int> cv::gimpl::FluidResizeAgent::linesReadAndnextWindow(std::size_t) const
{
    auto outIdx = out_buffers[0]->priv().y();
    auto lpi = std::min(m_outputLines - m_producedLines - k.m_lpi, k.m_lpi);
    return m_mapper->linesReadAndNextWindow(outIdx, lpi);
}

int cv::gimpl::Fluid420toRGBAgent::firstWindow(std::size_t inPort) const
{
    // 2 lines for Y, 1 for UV
    return inPort == 0 ? 2 : 1;
}

std::pair<int,int> cv::gimpl::Fluid420toRGBAgent::linesReadAndnextWindow(std::size_t inPort) const
{
    // 2 lines for Y, 1 for UV
    return inPort == 0 ? std::make_pair(2, 2) : std::make_pair(1, 1);
}

void cv::gimpl::FluidResizeAgent::setRatio(double ratio)
{
    if (ratio >= 1.0)
    {
        m_mapper.reset(new FluidDownscaleMapper(ratio, k.m_lpi));
    }
    else
    {
        m_mapper.reset(new FluidUpscaleMapper(ratio, k.m_lpi, in_views[0].meta().size.height));
    }
}

bool cv::gimpl::FluidAgent::canRead() const
{
    // An agent can work if every input buffer have enough data to start
    for (const auto& in_view : in_views)
    {
        if (in_view)
        {
            if (!in_view.ready())
                return false;
        }
    }
    return true;
}

bool cv::gimpl::FluidAgent::canWrite() const
{
    // An agent can work if there is space to write in its output
    // allocated buffers
    GAPI_DbgAssert(!out_buffers.empty());
    auto out_begin = out_buffers.begin();
    auto out_end   = out_buffers.end();
    if (k.m_scratch) out_end--;
    for (auto it = out_begin; it != out_end; ++it)
    {
        if ((*it)->priv().full())
        {
            return false;
        }
    }
    return true;
}

bool cv::gimpl::FluidAgent::canWork() const
{
    return canRead() && canWrite();
}

void cv::gimpl::FluidAgent::doWork()
{
    GAPI_DbgAssert(m_outputLines > m_producedLines);
    for (auto& in_view : in_views)
    {
        if (in_view) in_view.priv().prepareToRead();
    }

    k.m_f(in_args, out_buffers);

    for (const auto it : ade::util::indexed(in_views))
    {
        auto& in_view = ade::util::value(it);

        if (in_view)
        {
            auto idx = ade::util::index(it);
            auto pair = linesReadAndnextWindow(idx);
            in_view.priv().readDone(pair.first, pair.second);
        };
    }

    for (auto* out_buf : out_buffers)
    {
        out_buf->priv().writeDone();
        // FIXME WARNING: Scratch buffers rotated here too!
    }

    m_producedLines += k.m_lpi;
}

bool cv::gimpl::FluidAgent::done() const
{
    // m_producedLines is a multiple of LPI, while original
    // height may be not.
    return m_producedLines >= m_outputLines;
}

void cv::gimpl::FluidAgent::debug(std::ostream &os)
{
    os << "Fluid Agent " << std::hex << this
       << " (" << op_name << ") --"
       << " canWork=" << std::boolalpha << canWork()
       << " canRead=" << std::boolalpha << canRead()
       << " canWrite=" << std::boolalpha << canWrite()
       << " done="    << done()
       << " lines="   << std::dec << m_producedLines << "/" << m_outputLines
       << " {{\n";
    for (auto out_buf : out_buffers)
    {
        out_buf->debug(os);
    }
    std::cout << "}}" << std::endl;
}

// GCPUExcecutable implementation //////////////////////////////////////////////

void cv::gimpl::GFluidExecutable::initBufferRois(std::vector<int>& readStarts,
                                                 std::vector<cv::Rect>& rois,
                                                 const std::vector<cv::Rect>& out_rois)
{
    GConstFluidModel fg(m_g);
    auto proto = m_gm.metadata().get<Protocol>();
    std::stack<ade::NodeHandle> nodesToVisit;

    // FIXME?
    // There is possible case when user pass the vector full of default Rect{}-s,
    // Can be diagnosed and handled appropriately
    if (proto.outputs.size() != out_rois.size())
    {
        GAPI_Assert(out_rois.size() == 0);
        // No inference required, buffers will obtain roi from meta
        return;
    }

    // First, initialize rois for output nodes, add them to traversal stack
    for (const auto it : ade::util::indexed(proto.out_nhs))
    {
        const auto idx = ade::util::index(it);
        const auto& nh  = ade::util::value(it);

        const auto &d  = m_gm.metadata(nh).get<Data>();

        // This is not our output
        if (m_id_map.count(d.rc) == 0)
        {
            continue;
        }

        if (d.shape == GShape::GMAT)
        {
            auto desc = util::get<GMatDesc>(d.meta);
            auto id = m_id_map.at(d.rc);
            readStarts[id] = 0;

            const auto& out_roi = out_rois[idx];
            if (out_roi == cv::Rect{})
            {
                rois[id] = cv::Rect{ 0, 0, desc.size.width, desc.size.height };
            }
            else
            {
                GAPI_Assert(out_roi.height > 0);
                GAPI_Assert(out_roi.y + out_roi.height <= desc.size.height);

                // Only slices are supported at the moment
                GAPI_Assert(out_roi.x == 0);
                GAPI_Assert(out_roi.width == desc.size.width);
                rois[id] = out_roi;
            }

            nodesToVisit.push(nh);
        }
    }

    // Perform a wide search from each of the output nodes
    // And extend roi of buffers by border_size
    // Each node can be visited multiple times
    // (if node has been already visited, the check that inferred rois are the same is performed)
    while (!nodesToVisit.empty())
    {
        const auto startNode = nodesToVisit.top();
        nodesToVisit.pop();

        if (!startNode->inNodes().empty())
        {
            GAPI_Assert(startNode->inNodes().size() == 1);
            const auto& oh = startNode->inNodes().front();

            const auto& data = m_gm.metadata(startNode).get<Data>();
            // only GMats participate in the process so it's valid to obtain GMatDesc
            const auto& meta = util::get<GMatDesc>(data.meta);

            for (const auto& in_edge : oh->inEdges())
            {
                const auto& in_node = in_edge->srcNode();
                const auto& in_data = m_gm.metadata(in_node).get<Data>();

                if (in_data.shape == GShape::GMAT && fg.metadata(in_node).contains<FluidData>())
                {
                    const auto& in_meta = util::get<GMatDesc>(in_data.meta);
                    const auto& fd = fg.metadata(in_node).get<FluidData>();

                    auto adjFilterRoi = [](cv::Rect produced, int b, int max_height) {
                        // Extend with border roi which should be produced, crop to logical image size
                        cv::Rect roi = {produced.x, produced.y - b, produced.width, produced.height + 2*b};
                        cv::Rect fullImg{ 0, 0, produced.width, max_height };
                        return roi & fullImg;
                    };

                    auto adjResizeRoi = [](cv::Rect produced, cv::Size inSz, cv::Size outSz) {
                        auto map = [](int outCoord, int producedSz, int inSize, int outSize) {
                            double ratio = (double)inSize / outSize;
                            int w0 = 0, w1 = 0;
                            if (ratio >= 1.0)
                            {
                                w0 = windowStart(outCoord, ratio);
                                w1 = windowEnd  (outCoord + producedSz - 1, ratio);
                            }
                            else
                            {
                                w0 = upscaleWindowStart(outCoord, ratio);
                                w1 = upscaleWindowEnd(outCoord + producedSz - 1, ratio, inSize);
                            }
                            return std::make_pair(w0, w1);
                        };

                        auto mapY = map(produced.y, produced.height, inSz.height, outSz.height);
                        auto y0 = mapY.first;
                        auto y1 = mapY.second;

                        auto mapX = map(produced.x, produced.width, inSz.width, outSz.width);
                        auto x0 = mapX.first;
                        auto x1 = mapX.second;

                        cv::Rect roi = {x0, y0, x1 - x0, y1 - y0};
                        return roi;
                    };

                    auto adj420Roi = [&](cv::Rect produced, std::size_t port) {
                        GAPI_Assert(produced.x % 2 == 0);
                        GAPI_Assert(produced.y % 2 == 0);
                        GAPI_Assert(produced.width % 2 == 0);
                        GAPI_Assert(produced.height % 2 == 0);

                        cv::Rect roi;
                        switch (port) {
                        case 0: roi = produced; break;
                        case 1:
                        case 2: roi = cv::Rect{ produced.x/2, produced.y/2, produced.width/2, produced.height/2 }; break;
                        default: GAPI_Assert(false);
                        }
                        return roi;
                    };

                    cv::Rect produced = rois[m_id_map.at(data.rc)];

                    // Apply resize-specific roi transformations
                    cv::Rect resized;
                    switch (fg.metadata(oh).get<FluidUnit>().k.m_kind)
                    {
                    case GFluidKernel::Kind::Filter:      resized = produced; break;
                    case GFluidKernel::Kind::Resize:      resized = adjResizeRoi(produced, in_meta.size, meta.size); break;
                    case GFluidKernel::Kind::YUV420toRGB: resized = adj420Roi(produced, m_gm.metadata(in_edge).get<Input>().port); break;
                    default: GAPI_Assert(false);
                    }

                    // All below transformations affect roi of the writer, preserve read start position here
                    int readStart = resized.y;

                    // Extend required input roi (both y and height) to be even if it's produced by CS420toRGB
                    if (!in_node->inNodes().empty()) {
                        auto in_data_producer = in_node->inNodes().front();
                        if (fg.metadata(in_data_producer).get<FluidUnit>().k.m_kind == GFluidKernel::Kind::YUV420toRGB) {
                            if (resized.y % 2 != 0) {
                                resized.y--;
                                resized.height++;
                            }

                            if (resized.height % 2 != 0) {
                                resized.height++;
                            }
                        }
                    }

                    // Apply filter-specific roi transformations, clip to image size
                    // Note: done even for non-filter kernels as applies border-related transformations
                    // (required in the case when there are multiple readers with different border requirements)
                    auto roi = adjFilterRoi(resized, fd.border_size, in_meta.size.height);

                    auto in_id = m_id_map.at(in_data.rc);
                    if (rois[in_id] == cv::Rect{})
                    {
                        readStarts[in_id] = readStart;
                        rois[in_id] = roi;
                        // Continue traverse on internal (w.r.t Island) data nodes only.
                        if (fd.internal) nodesToVisit.push(in_node);
                    }
                    else
                    {
                        GAPI_Assert(readStarts[in_id] == readStart);
                        GAPI_Assert(rois[in_id] == roi);
                    }
                } // if (in_data.shape == GShape::GMAT)
            } // for (const auto& in_edge : oh->inEdges())
        } // if (!startNode->inNodes().empty())
    } // while (!nodesToVisit.empty())
}

cv::gimpl::FluidGraphInputData cv::gimpl::fluidExtractInputDataFromGraph(const ade::Graph &g, const std::vector<ade::NodeHandle> &nodes)
{
    decltype(FluidGraphInputData::m_agents_data)       agents_data;
    decltype(FluidGraphInputData::m_scratch_users)     scratch_users;
    decltype(FluidGraphInputData::m_id_map)            id_map;
    decltype(FluidGraphInputData::m_all_gmat_ids)      all_gmat_ids;
    std::size_t                                        mat_count = 0;

    GConstFluidModel fg(g);
    GModel::ConstGraph m_gm(g);

    // Initialize vector of data buffers, build list of operations
    // FIXME: There _must_ be a better way to [query] count number of DATA nodes

    auto grab_mat_nh = [&](ade::NodeHandle nh) {
        auto rc = m_gm.metadata(nh).get<Data>().rc;
        if (id_map.count(rc) == 0)
        {
            all_gmat_ids[mat_count] = nh;
            id_map[rc] = mat_count++;
        }
    };

    std::size_t last_agent = 0;

    for (const auto &nh : nodes)
    {
        switch (m_gm.metadata(nh).get<NodeType>().t)
        {
        case NodeType::DATA:
            if (m_gm.metadata(nh).get<Data>().shape == GShape::GMAT)
                grab_mat_nh(nh);
            break;

        case NodeType::OP:
        {
            const auto& fu = fg.metadata(nh).get<FluidUnit>();

            agents_data.push_back({fu.k.m_kind, nh, {}, {}});
            // NB.: in_buffer_ids size is equal to Arguments size, not Edges size!!!
            agents_data.back().in_buffer_ids.resize(m_gm.metadata(nh).get<Op>().args.size(), -1);
            for (auto eh : nh->inEdges())
            {
                // FIXME Only GMats are currently supported (which can be represented
                // as fluid buffers
                if (m_gm.metadata(eh->srcNode()).get<Data>().shape == GShape::GMAT)
                {
                    const auto in_port = m_gm.metadata(eh).get<Input>().port;
                    const int  in_buf  = m_gm.metadata(eh->srcNode()).get<Data>().rc;

                    agents_data.back().in_buffer_ids[in_port] = in_buf;
                    grab_mat_nh(eh->srcNode());
                }
            }
            // FIXME: Assumption that all operation outputs MUST be connected
            agents_data.back().out_buffer_ids.resize(nh->outEdges().size(), -1);
            for (auto eh : nh->outEdges())
            {
                const auto& data = m_gm.metadata(eh->dstNode()).get<Data>();
                const auto out_port = m_gm.metadata(eh).get<Output>().port;
                const int  out_buf  = data.rc;

                agents_data.back().out_buffer_ids[out_port] = out_buf;
                if (data.shape == GShape::GMAT) grab_mat_nh(eh->dstNode());
            }
            if (fu.k.m_scratch)
                scratch_users.push_back(last_agent);
            last_agent++;
            break;
        }
        default: GAPI_Assert(false);
        }
    }

    // Check that IDs form a continiuos set (important for further indexing)
    GAPI_Assert(id_map.size() >  0);
    GAPI_Assert(id_map.size() == static_cast<size_t>(mat_count));

    return FluidGraphInputData {std::move(agents_data), std::move(scratch_users), std::move(id_map), std::move(all_gmat_ids), mat_count};
}

cv::gimpl::GFluidExecutable::GFluidExecutable(const ade::Graph                       &g,
                                              const cv::gimpl::FluidGraphInputData   &traverse_res,
                                              const std::vector<cv::Rect> &outputRois)
    : m_g(g), m_gm(m_g),
      m_num_int_buffers (traverse_res.m_mat_count),
      m_scratch_users   (traverse_res.m_scratch_users),
      m_id_map          (traverse_res.m_id_map),
      m_all_gmat_ids    (traverse_res.m_all_gmat_ids),
      m_buffers(m_num_int_buffers + m_scratch_users.size())
{
    GConstFluidModel fg(m_g);

    auto create_fluid_agent = [&g](agent_data_t const& agent_data) -> std::unique_ptr<FluidAgent> {
        std::unique_ptr<FluidAgent> agent_ptr;
        switch (agent_data.kind)
        {
            case GFluidKernel::Kind::Filter:      agent_ptr.reset(new FluidFilterAgent(g, agent_data.nh));      break;
            case GFluidKernel::Kind::Resize:      agent_ptr.reset(new FluidResizeAgent(g, agent_data.nh));      break;
            case GFluidKernel::Kind::YUV420toRGB: agent_ptr.reset(new Fluid420toRGBAgent(g, agent_data.nh));    break;
            default: GAPI_Assert(false);
        }
        std::tie(agent_ptr->in_buffer_ids, agent_ptr->out_buffer_ids) = std::tie(agent_data.in_buffer_ids, agent_data.out_buffer_ids);
        return agent_ptr;
    };

    for (auto const& agent_data : traverse_res.m_agents_data){
        m_agents.push_back(create_fluid_agent(agent_data));
    }

    // Actually initialize Fluid buffers
    GAPI_LOG_INFO(NULL, "Initializing " << m_num_int_buffers << " fluid buffer(s)" << std::endl);

    // After buffers are allocated, repack: ...
    for (auto &agent : m_agents)
    {
        // a. Agent input parameters with View pointers (creating Views btw)
        const auto &op = m_gm.metadata(agent->op_handle).get<Op>();
        const auto &fu =   fg.metadata(agent->op_handle).get<FluidUnit>();
        agent->in_args.resize(op.args.size());
        agent->in_views.resize(op.args.size());
        for (auto it : ade::util::indexed(ade::util::toRange(agent->in_buffer_ids)))
        {
            auto in_idx  = ade::util::index(it);
            auto buf_idx = ade::util::value(it);

            if (buf_idx >= 0)
            {
                // IF there is input buffer, register a view (every unique
                // reader has its own), and store it in agent Args
                gapi::fluid::Buffer &buffer = m_buffers.at(m_id_map.at(buf_idx));

                auto inEdge = GModel::getInEdgeByPort(m_g, agent->op_handle, in_idx);
                auto ownStorage = fg.metadata(inEdge).get<FluidUseOwnBorderBuffer>().use;

                // NB: It is safe to keep ptr as view lifetime is buffer lifetime
                agent->in_views[in_idx] = buffer.mkView(fu.border_size, ownStorage);
                agent->in_args[in_idx]  = GArg(&agent->in_views[in_idx]);
                buffer.addView(&agent->in_views[in_idx]);
            }
            else
            {
                // Copy(FIXME!) original args as is
                agent->in_args[in_idx] = op.args[in_idx];
            }
        }

        // b. Agent output parameters with Buffer pointers.
        agent->out_buffers.resize(agent->op_handle->outEdges().size(), nullptr);
        for (auto it : ade::util::indexed(ade::util::toRange(agent->out_buffer_ids)))
        {
            auto out_idx = ade::util::index(it);
            auto buf_idx = m_id_map.at(ade::util::value(it));
            agent->out_buffers.at(out_idx) = &m_buffers.at(buf_idx);
        }
    }

    // After parameters are there, initialize scratch buffers
    const std::size_t num_scratch = m_scratch_users.size();
    if (num_scratch)
    {
        GAPI_LOG_INFO(NULL, "Initializing " << num_scratch << " scratch buffer(s)" << std::endl);
        std::size_t last_scratch_id = 0;

        for (auto i : m_scratch_users)
        {
            auto &agent = m_agents.at(i);
            GAPI_Assert(agent->k.m_scratch);
            const std::size_t new_scratch_idx = m_num_int_buffers + last_scratch_id;
            agent->out_buffers.emplace_back(&m_buffers[new_scratch_idx]);
            last_scratch_id++;
        }
    }

    makeReshape(outputRois);

    GAPI_LOG_INFO(NULL, "Internal buffers: " << std::fixed << std::setprecision(2) << static_cast<float>(total_buffers_size())/1024 << " KB\n");
}

std::size_t cv::gimpl::GFluidExecutable::total_buffers_size() const
{
    GConstFluidModel fg(m_g);
    std::size_t total_size = 0;
    for (const auto i : ade::util::indexed(m_buffers))
    {
        // Check that all internal and scratch buffers are allocated
        const auto  idx = ade::util::index(i);
        const auto& b   = ade::util::value(i);
        if (idx >= m_num_int_buffers ||
            fg.metadata(m_all_gmat_ids.at(idx)).get<FluidData>().internal == true)
        {
            GAPI_Assert(b.priv().size() > 0);
        }

        // Buffers which will be bound to real images may have size of 0 at this moment
        // (There can be non-zero sized const border buffer allocated in such buffers)
        total_size += b.priv().size();
    }
    return total_size;
}

namespace
{
    void resetFluidData(ade::Graph& graph)
    {
        using namespace cv::gimpl;
        GModel::Graph g(graph);
        GFluidModel fg(graph);
        for (const auto& node : g.nodes())
        {
            if (g.metadata(node).get<NodeType>().t == NodeType::DATA)
            {
                auto& fd = fg.metadata(node).get<FluidData>();
                fd.latency         = 0;
                fd.skew            = 0;
                fd.max_consumption = 0;
            }

            GModel::log_clear(g, node);
        }
    }

    void initFluidUnits(ade::Graph& graph)
    {
        using namespace cv::gimpl;
        GModel::Graph g(graph);
        GFluidModel fg(graph);

        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
        for (auto node : sorted)
        {
            if (fg.metadata(node).contains<FluidUnit>())
            {
                std::set<int> in_hs, out_ws, out_hs;

                for (const auto& in : node->inNodes())
                {
                    const auto& d = g.metadata(in).get<Data>();
                    if (d.shape == cv::GShape::GMAT)
                    {
                        const auto& meta = cv::util::get<cv::GMatDesc>(d.meta);
                        in_hs.insert(meta.size.height);
                    }
                }

                for (const auto& out : node->outNodes())
                {
                    const auto& d = g.metadata(out).get<Data>();
                    if (d.shape == cv::GShape::GMAT)
                    {
                        const auto& meta = cv::util::get<cv::GMatDesc>(d.meta);
                        out_ws.insert(meta.size.width);
                        out_hs.insert(meta.size.height);
                    }
                }

                auto &fu = fg.metadata(node).get<FluidUnit>();

                GAPI_Assert((out_ws.size() == 1 && out_hs.size() == 1) &&
                            ((in_hs.size() == 1) ||
                            ((in_hs.size() == 2) && fu.k.m_kind == cv::GFluidKernel::Kind::YUV420toRGB)));

                const auto &op = g.metadata(node).get<Op>();
                fu.line_consumption.resize(op.args.size(), 0);

                auto in_h  = *in_hs .cbegin();
                auto out_h = *out_hs.cbegin();

                fu.ratio = (double)in_h / out_h;

                // Set line consumption for each image (GMat) input
                for (const auto& in_edge : node->inEdges())
                {
                    const auto& d = g.metadata(in_edge->srcNode()).get<Data>();
                    if (d.shape == cv::GShape::GMAT)
                    {
                        auto port = g.metadata(in_edge).get<Input>().port;
                        fu.line_consumption[port] = maxLineConsumption(fu.k.m_kind, fu.window, in_h, out_h, fu.k.m_lpi, port);

                        GModel::log(g, node, "Line consumption (port " + std::to_string(port) + "): "
                                    + std::to_string(fu.line_consumption[port]));
                    }
                }

                fu.border_size = borderSize(fu.k.m_kind, fu.window);
                GModel::log(g, node, "Border size: " + std::to_string(fu.border_size));
            }
        }
    }

    // FIXME!
    // Split into initLineConsumption and initBorderSizes,
    // call only consumption related stuff during reshape
    void initLineConsumption(ade::Graph& graph)
    {
        using namespace cv::gimpl;
        GModel::Graph g(graph);
        GFluidModel fg(graph);

        for (const auto &node : g.nodes())
        {
            if (fg.metadata(node).contains<FluidUnit>())
            {
                const auto &fu = fg.metadata(node).get<FluidUnit>();

                for (const auto &in_edge : node->inEdges())
                {
                    const auto &in_data_node = in_edge->srcNode();
                    auto port = g.metadata(in_edge).get<Input>().port;

                    auto &fd = fg.metadata(in_data_node).get<FluidData>();

                    // Update (not Set) fields here since a single data node may be
                    // accessed by multiple consumers
                    fd.max_consumption = std::max(fu.line_consumption[port], fd.max_consumption);
                    fd.border_size     = std::max(fu.border_size, fd.border_size);

                    GModel::log(g, in_data_node, "Line consumption: " + std::to_string(fd.max_consumption)
                                + " (upd by " + std::to_string(fu.line_consumption[port]) + ")", node);
                    GModel::log(g, in_data_node, "Border size: " + std::to_string(fd.border_size), node);
                }
            }
        }
    }

    void calcLatency(ade::Graph& graph)
    {
        using namespace cv::gimpl;
        GModel::Graph g(graph);
        GFluidModel fg(graph);

        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
        for (const auto &node : sorted)
        {
            if (fg.metadata(node).contains<FluidUnit>())
            {
                const auto &fu = fg.metadata(node).get<FluidUnit>();

                GModel::log(g, node, "LPI: " + std::to_string(fu.k.m_lpi));

                // Output latency is max(input_latency) + own_latency
                int out_latency = 0;
                for (const auto &in_edge: node->inEdges())
                {
                    // FIXME: ASSERT(DATA), ASSERT(FLUIDDATA)
                    const auto port = g.metadata(in_edge).get<Input>().port;
                    const auto own_latency = fu.line_consumption[port] - fu.border_size;
                    const auto in_latency = fg.metadata(in_edge->srcNode()).get<FluidData>().latency;
                    out_latency = std::max(out_latency, in_latency + own_latency);
                }

                for (const auto &out_data_node : node->outNodes())
                {
                    // FIXME: ASSERT(DATA), ASSERT(FLUIDDATA)
                    auto &fd     = fg.metadata(out_data_node).get<FluidData>();
                    // If fluid node is external, it will be bound to a real image without
                    // fluid buffer allocation, so set its latency to 0 not to confuse later latency propagation.
                    // Latency is used in fluid buffer allocation process and is not used by the scheduler
                    // so latency doesn't affect the execution and setting it to 0 is legal
                    fd.latency   = fd.internal ? out_latency : 0;
                    fd.lpi_write = fu.k.m_lpi;
                    GModel::log(g, out_data_node, "Latency: " + std::to_string(fd.latency));
                }
            }
        }
    }

    void calcSkew(ade::Graph& graph)
    {
        using namespace cv::gimpl;
        GModel::Graph g(graph);
        GFluidModel fg(graph);

        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
        for (const auto &node : sorted)
        {
            if (fg.metadata(node).contains<FluidUnit>())
            {
                int max_latency = 0;
                for (const auto &in_data_node : node->inNodes())
                {
                    // FIXME: ASSERT(DATA), ASSERT(FLUIDDATA)
                    max_latency = std::max(max_latency, fg.metadata(in_data_node).get<FluidData>().latency);
                }
                for (const auto &in_data_node : node->inNodes())
                {
                    // FIXME: ASSERT(DATA), ASSERT(FLUIDDATA)
                    auto &fd = fg.metadata(in_data_node).get<FluidData>();

                    // Update (not Set) fields here since a single data node may be
                    // accessed by multiple consumers
                    fd.skew = std::max(fd.skew, max_latency - fd.latency);

                    GModel::log(g, in_data_node, "Skew: " + std::to_string(fd.skew), node);
                }
            }
        }
    }
}

void cv::gimpl::GFluidExecutable::makeReshape(const std::vector<cv::Rect> &out_rois)
{
    GConstFluidModel fg(m_g);

    // Calculate rois for each fluid buffer
    std::vector<int> readStarts(m_num_int_buffers);
    std::vector<cv::Rect> rois(m_num_int_buffers);
    initBufferRois(readStarts, rois, out_rois);

    // NB: Allocate ALL buffer object at once, and avoid any further reallocations
    // (since raw pointers-to-elements are taken)
    for (const auto &it : m_all_gmat_ids)
    {
        auto id = it.first;
        auto nh = it.second;
        const auto & d  = m_gm.metadata(nh).get<Data>();
        const auto &fd  = fg.metadata(nh).get<FluidData>();
        const auto meta = cv::util::get<GMatDesc>(d.meta);

        m_buffers[id].priv().init(meta, fd.lpi_write, readStarts[id], rois[id]);

        // TODO:
        // Introduce Storage::INTERNAL_GRAPH and Storage::INTERNAL_ISLAND?
        if (fd.internal == true)
        {
            // FIXME: do max_consumption calculation properly (e.g. in initLineConsumption)
            int max_consumption = 0;
            if (nh->outNodes().empty()) {
                // nh is always a DATA node, so it is safe to get inNodes().front() since there's
                // always a single writer (OP node)
                max_consumption = fg.metadata(nh->inNodes().front()).get<FluidUnit>().k.m_lpi;
            } else {
                max_consumption = fd.max_consumption;
            }
            m_buffers[id].priv().allocate(fd.border, fd.border_size, max_consumption, fd.skew);
            std::stringstream stream;
            m_buffers[id].debug(stream);
            GAPI_LOG_INFO(NULL, stream.str());
        }
    }

    // Allocate views, initialize agents
    for (auto &agent : m_agents)
    {
        const auto &fu = fg.metadata(agent->op_handle).get<FluidUnit>();
        for (auto it : ade::util::indexed(ade::util::toRange(agent->in_buffer_ids)))
        {
            auto in_idx  = ade::util::index(it);
            auto buf_idx = ade::util::value(it);

            if (buf_idx >= 0)
            {
                agent->in_views[in_idx].priv().allocate(fu.line_consumption[in_idx], fu.border);
            }
        }

        agent->setRatio(fu.ratio);
        agent->m_outputLines = agent->out_buffers.front()->priv().outputLines();
    }

    // Initialize scratch buffers
    if (m_scratch_users.size())
    {
        for (auto i : m_scratch_users)
        {
            auto &agent = m_agents.at(i);
            GAPI_Assert(agent->k.m_scratch);

            // Trigger Scratch buffer initialization method
            agent->k.m_is(GModel::collectInputMeta(m_gm, agent->op_handle), agent->in_args, *agent->out_buffers.back());
            std::stringstream stream;
            agent->out_buffers.back()->debug(stream);
            GAPI_LOG_INFO(NULL, stream.str());
        }
    }

    // FIXME: calculate the size (lpi * ..)
    m_script.clear();
    m_script.reserve(10000);
}

void cv::gimpl::GFluidExecutable::reshape(ade::Graph &g, const GCompileArgs &args)
{
    // FIXME: Probably this needs to be integrated into common pass re-run routine
    // Backends may want to mark with passes to re-run on reshape and framework could
    // do it system-wide (without need in every backend handling reshape() directly).
    // This design needs to be analyzed for implementation.
    resetFluidData(g);
    initFluidUnits(g);
    initLineConsumption(g);
    calcLatency(g);
    calcSkew(g);
    const auto out_rois = cv::gapi::getCompileArg<cv::GFluidOutputRois>(args).value_or(cv::GFluidOutputRois());
    makeReshape(out_rois.rois);
}

// FIXME: Document what it does
void cv::gimpl::GFluidExecutable::bindInArg(const cv::gimpl::RcDesc &rc, const GRunArg &arg)
{
    magazine::bindInArg(m_res, rc, arg);
    if (rc.shape == GShape::GMAT) {
        auto& mat = m_res.slot<cv::Mat>()[rc.id];
        // fluid::Buffer::bindTo() is not connected to magazine::bindIn/OutArg and unbind() calls,
        // it's simply called each run() without any requirement to call some fluid-specific
        // unbind() at the end of run()
        m_buffers[m_id_map.at(rc.id)].priv().bindTo(mat, true);
    }
}

void cv::gimpl::GFluidExecutable::bindOutArg(const cv::gimpl::RcDesc &rc, const GRunArgP &arg)
{
    // Only GMat is supported as return type
    if (rc.shape != GShape::GMAT) {
        util::throw_error(std::logic_error("Unsupported return GShape type"));
    }
    magazine::bindOutArg(m_res, rc, arg);
    auto& mat = m_res.slot<cv::Mat>()[rc.id];
    m_buffers[m_id_map.at(rc.id)].priv().bindTo(mat, false);
}

void cv::gimpl::GFluidExecutable::packArg(cv::GArg &in_arg, const cv::GArg &op_arg)
{
    GAPI_Assert(op_arg.kind != cv::detail::ArgKind::GMAT
           && op_arg.kind != cv::detail::ArgKind::GSCALAR
           && op_arg.kind != cv::detail::ArgKind::GARRAY
           && op_arg.kind != cv::detail::ArgKind::GOPAQUE);

    if (op_arg.kind == cv::detail::ArgKind::GOBJREF)
    {
        const cv::gimpl::RcDesc &ref = op_arg.get<cv::gimpl::RcDesc>();
        if (ref.shape == GShape::GSCALAR)
        {
            in_arg = GArg(m_res.slot<cv::Scalar>()[ref.id]);
        }
        else if (ref.shape == GShape::GARRAY)
        {
            in_arg = GArg(m_res.slot<cv::detail::VectorRef>()[ref.id]);
        }
        else if (ref.shape == GShape::GOPAQUE)
        {
            in_arg = GArg(m_res.slot<cv::detail::OpaqueRef>()[ref.id]);
        }
    }
}

void cv::gimpl::GFluidExecutable::run(std::vector<InObj>  &&input_objs,
                                      std::vector<OutObj> &&output_objs)
{
    run(input_objs, output_objs);
}
void cv::gimpl::GFluidExecutable::run(std::vector<InObj>  &input_objs,
                                      std::vector<OutObj> &output_objs)
{
    // Bind input buffers from parameters
    for (auto& it : input_objs)  bindInArg(it.first, it.second);
    for (auto& it : output_objs) bindOutArg(it.first, it.second);

    // Reset Buffers and Agents state before we go
    for (auto &buffer : m_buffers)
        buffer.priv().reset();

    for (auto &agent : m_agents)
    {
        agent->reset();
        // Pass input cv::Scalar's to agent argument
        const auto& op = m_gm.metadata(agent->op_handle).get<Op>();
        for (const auto it : ade::util::indexed(op.args))
        {
            const auto& arg = ade::util::value(it);
            packArg(agent->in_args[ade::util::index(it)], arg);
        }
    }

    // Explicitly reset Scratch buffers, if any
    for (auto scratch_i : m_scratch_users)
    {
        auto &agent = m_agents[scratch_i];
        GAPI_DbgAssert(agent->k.m_scratch);
        agent->k.m_rs(*agent->out_buffers.back());
    }

    // Now start executing our stuff!
    // Fluid execution is:
    // - run through list of Agents from Left to Right
    // - for every Agent:
    //   - if all input Buffers have enough data to fulfill
    //     Agent's window - trigger Agent
    //     - on trigger, Agent takes all input lines from input buffers
    //       and produces a single output line
    //     - once Agent finishes, input buffers get "readDone()",
    //       and output buffers get "writeDone()"
    //   - if there's not enough data, Agent is skipped
    // Yes, THAT easy!

    if (m_script.empty())
    {
        bool complete = true;
        do {
            complete = true;
            bool work_done=false;
            for (auto &agent : m_agents)
            {
                // agent->debug(std::cout);
                if (!agent->done())
                {
                    if (agent->canWork())
                    {
                        agent->doWork(); work_done=true;
                        m_script.push_back(agent.get());
                    }
                    if (!agent->done())   complete = false;
                }
            }
            GAPI_Assert(work_done || complete);
        } while (!complete); // FIXME: number of iterations can be calculated statically
    }
    else
    {
        for (auto &agent : m_script)
        {
            agent->doWork();
        }
    }

    // In/Out args clean-up is mandatory now with RMat
    for (auto &it : input_objs) magazine::unbind(m_res, it.first);
    for (auto &it : output_objs) magazine::unbind(m_res, it.first);
}

cv::gimpl::GParallelFluidExecutable::GParallelFluidExecutable(const ade::Graph                      &g,
                                                              const FluidGraphInputData             &graph_data,
                                                              const std::vector<GFluidOutputRois>   &parallelOutputRois,
                                                              const decltype(parallel_for)          &pfor)
: parallel_for(pfor)
{
    for (auto&& rois : parallelOutputRois){
        tiles.emplace_back(new GFluidExecutable(g, graph_data, rois.rois));
    }
}


void cv::gimpl::GParallelFluidExecutable::reshape(ade::Graph&, const GCompileArgs& )
{
    //TODO: implement ?
    GAPI_Assert(false && "Not Implemented;");
}

void cv::gimpl::GParallelFluidExecutable::run(std::vector<InObj>  &&input_objs,
                                              std::vector<OutObj> &&output_objs)
{
    parallel_for(tiles.size(), [&, this](std::size_t index){
        GAPI_Assert((bool)tiles[index]);
        tiles[index]->run(input_objs, output_objs);
    });
}


// FIXME: these passes operate on graph global level!!!
// Need to fix this for heterogeneous (island-based) processing
void GFluidBackendImpl::addMetaSensitiveBackendPasses(ade::ExecutionEngineSetupContext &ectx)
{
    using namespace cv::gimpl;

    // FIXME: all passes were moved to "exec" stage since Fluid
    // should check Islands configuration first (which is now quite
    // limited), and only then continue with all other passes.
    //
    // The passes/stages API must be streamlined!
    ectx.addPass("exec", "init_fluid_data", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        auto isl_graph = g.metadata().get<IslandModel>().model;
        GIslandModel::Graph gim(*isl_graph);

        GFluidModel fg(ctx.graph);

        const auto setFluidData = [&](ade::NodeHandle nh, bool internal) {
            FluidData fd;
            fd.internal = internal;
            fg.metadata(nh).set(fd);
        };

        for (const auto& nh : gim.nodes())
        {
            switch (gim.metadata(nh).get<NodeKind>().k)
            {
            case NodeKind::ISLAND:
            {
                const auto isl = gim.metadata(nh).get<FusedIsland>().object;
                if (isl->backend() == cv::gapi::fluid::backend())
                {
                    // Add FluidData to all data nodes inside island,
                    // set internal = true if node is not a slot in terms of higher-level GIslandModel
                    for (const auto& node : isl->contents())
                    {
                        if (g.metadata(node).get<NodeType>().t == NodeType::DATA &&
                            !fg.metadata(node).contains<FluidData>())
                            setFluidData(node, true);
                    }
                } // if (fluid backend)
            } break; // case::ISLAND
            case NodeKind::SLOT:
            {
                // add FluidData to slot if it's read/written by fluid
                // regardless if it is one fluid island (both writing to and reading from this object)
                // or two distinct islands (both fluid)
                auto isFluidIsland = [&](const ade::NodeHandle& node) {
                    // With Streaming, Emitter islands may have no FusedIsland thing in meta.
                    // FIXME: Probably this is a concept misalignment
                    if (!gim.metadata(node).contains<FusedIsland>()) {
                        const auto kind = gim.metadata(node).get<NodeKind>().k;
                        GAPI_Assert(kind == NodeKind::EMIT || kind == NodeKind::SINK);
                        return false;
                    }
                    const auto isl = gim.metadata(node).get<FusedIsland>().object;
                    return isl->backend() == cv::gapi::fluid::backend();
                };

                if (ade::util::any_of(ade::util::chain(nh->inNodes(), nh->outNodes()), isFluidIsland))
                {
                    auto data_node = gim.metadata(nh).get<DataSlot>().original_data_node;
                    setFluidData(data_node, false);
                }
            } break; // case::SLOT
            case NodeKind::EMIT:
            case NodeKind::SINK:
                break; // do nothing for Streaming nodes
            default: GAPI_Assert(false);
            } // switch
        } // for (gim.nodes())
    });
    // FIXME:
    // move to unpackKernel method
    // when https://gitlab-icv.inn.intel.com/G-API/g-api/merge_requests/66 is merged
    ectx.addPass("exec", "init_fluid_unit_windows_and_borders", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        GFluidModel fg(ctx.graph);

        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
        for (auto node : sorted)
        {
            if (fg.metadata(node).contains<FluidUnit>())
            {
                // FIXME: check that op has only one data node on input
                auto &fu = fg.metadata(node).get<FluidUnit>();
                const auto &op = g.metadata(node).get<Op>();
                auto inputMeta = GModel::collectInputMeta(fg, node);

                // Trigger user-defined "getWindow" callback
                fu.window = fu.k.m_gw(inputMeta, op.args);

                // Trigger user-defined "getBorder" callback
                fu.border = fu.k.m_b(inputMeta, op.args);
            }
        }
    });
    ectx.addPass("exec", "init_fluid_units", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        initFluidUnits(ctx.graph);
    });
    ectx.addPass("exec", "init_line_consumption", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        initLineConsumption(ctx.graph);
    });
    ectx.addPass("exec", "calc_latency", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        calcLatency(ctx.graph);
    });
    ectx.addPass("exec", "calc_skew", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        calcSkew(ctx.graph);
    });

    ectx.addPass("exec", "init_buffer_borders", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        GFluidModel fg(ctx.graph);
        auto sorted = g.metadata().get<ade::passes::TopologicalSortData>().nodes();
        for (auto node : sorted)
        {
            if (fg.metadata(node).contains<FluidData>())
            {
                auto &fd = fg.metadata(node).get<FluidData>();

                // Assign border stuff to FluidData

                // In/out data nodes are bound to user data directly,
                // so cannot be extended with a border
                if (fd.internal == true)
                {
                    // For now border of the buffer's storage is the border
                    // of the first reader whose border size is the same.
                    // FIXME: find more clever strategy of border picking
                    // (it can be a border which is common for majority of the
                    // readers, also we can calculate the number of lines which
                    // will be copied by views on each iteration and base our choice
                    // on this criteria)
                    auto readers = node->outNodes();

                    // There can be a situation when __internal__ nodes produced as part of some
                    // operation are unused later in the graph:
                    //
                    // in -> OP1
                    //        |------> internal_1  // unused node
                    //        |------> internal_2 -> OP2
                    //                                |------> out
                    //
                    // To allow graphs like the one above, skip nodes with empty outNodes()
                    if (readers.empty()) {
                        continue;
                    }

                    const auto &candidate = ade::util::find_if(readers, [&](ade::NodeHandle nh) {
                        return fg.metadata(nh).contains<FluidUnit>() &&
                               fg.metadata(nh).get<FluidUnit>().border_size == fd.border_size;
                    });

                    GAPI_Assert(candidate != readers.end());

                    const auto &fu = fg.metadata(*candidate).get<FluidUnit>();
                    fd.border = fu.border;
                }

                if (fd.border)
                {
                    GModel::log(g, node, "Border type: " + std::to_string(fd.border->type), node);
                }
            }
        }
    });
    ectx.addPass("exec", "init_view_borders", [](ade::passes::PassContext &ctx)
    {
        GModel::Graph g(ctx.graph);
        if (!GModel::isActive(g, cv::gapi::fluid::backend()))  // FIXME: Rearchitect this!
            return;

        GFluidModel fg(ctx.graph);
        for (auto node : g.nodes())
        {
            if (fg.metadata(node).contains<FluidData>())
            {
                auto &fd = fg.metadata(node).get<FluidData>();
                for (auto out_edge : node->outEdges())
                {
                    const auto dstNode = out_edge->dstNode();
                    if (fg.metadata(dstNode).contains<FluidUnit>())
                    {
                        const auto &fu = fg.metadata(dstNode).get<FluidUnit>();

                        // There is no need in own storage for view if it's border is
                        // the same as the buffer's (view can have equal or smaller border
                        // size in this case)
                        if (fu.border_size == 0 ||
                                (fu.border && fd.border && (*fu.border == *fd.border)))
                        {
                            GAPI_Assert(fu.border_size <= fd.border_size);
                            fg.metadata(out_edge).set(FluidUseOwnBorderBuffer{false});
                        }
                        else
                        {
                            fg.metadata(out_edge).set(FluidUseOwnBorderBuffer{true});
                            GModel::log(g, out_edge, "OwnBufferStorage: true");
                        }
                    }
                }
            }
        }
    });
}
