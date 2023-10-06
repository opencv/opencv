// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <mutex>

#if !defined(GAPI_STANDALONE)
#include <opencv2/imgproc.hpp>
#endif // !defined(GAPI_STANDALONE)

#include <opencv2/gapi/util/throw.hpp> // throw_error
#include <opencv2/gapi/streaming/format.hpp> // kernels

#include "logger.hpp"
#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"

#include "gstreamingbackend.hpp"
#include "gstreamingkernel.hpp"

namespace {

struct StreamingCreateFunction
{
    static const char *name() { return "StreamingCreateFunction";  }
    cv::gapi::streaming::CreateActorFunction createActorFunction;
};

using StreamingGraph = ade::TypedGraph
    < cv::gimpl::Op
    , StreamingCreateFunction
    >;

using ConstStreamingGraph = ade::ConstTypedGraph
    < cv::gimpl::Op
    , StreamingCreateFunction
    >;

class GStreamingIntrinExecutable final: public cv::gimpl::GIslandExecutable
{
    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override {
        GAPI_Error("Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    virtual bool allocatesOutputs() const override { return true; }
    // Return an empty RMat since we will reuse the input.
    // There is no need to allocate and copy 4k image here.
    virtual cv::RMat allocate(const cv::GMatDesc&) const override { return {}; }

    virtual bool canReshape() const override { return true; }
    virtual void reshape(ade::Graph&, const cv::GCompileArgs&) override {
        // Do nothing here
    }

public:
    GStreamingIntrinExecutable(const ade::Graph                   &,
                               const cv::GCompileArgs             &,
                               const std::vector<ade::NodeHandle> &);

    const ade::Graph& m_g;
    cv::gimpl::GModel::ConstGraph m_gm;
    cv::gapi::streaming::IActor::Ptr m_actor;
};

void GStreamingIntrinExecutable::run(GIslandExecutable::IInput  &in,
                                     GIslandExecutable::IOutput &out)
{
    m_actor->run(in, out);
}

class GStreamingBackendImpl final: public cv::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
        StreamingGraph gm(graph);
        const auto &kimpl  = cv::util::any_cast<cv::gapi::streaming::GStreamingKernel>(impl.opaque);
        gm.metadata(op_node).set(StreamingCreateFunction{kimpl.createActorFunction});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{new GStreamingIntrinExecutable(graph, args, nodes)};
    }

    virtual bool controlsMerge() const override
    {
        return true;
    }

    virtual bool allowsMerge(const cv::gimpl::GIslandModel::Graph &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &) const override
    {
        return false;
    }
};

GStreamingIntrinExecutable::GStreamingIntrinExecutable(const ade::Graph& g,
                                                       const cv::GCompileArgs& args,
                                                       const std::vector<ade::NodeHandle>& nodes)
    : m_g(g), m_gm(m_g)
{
    using namespace cv::gimpl;
    const auto is_op = [this](const ade::NodeHandle &nh)
    {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    auto it = std::find_if(nodes.begin(), nodes.end(), is_op);
    GAPI_Assert(it != nodes.end() && "No operators found for this island?!");

    ConstStreamingGraph cag(m_g);
    m_actor = cag.metadata(*it).get<StreamingCreateFunction>().createActorFunction(args);

    // Ensure this the only op in the graph
    if (std::any_of(it+1, nodes.end(), is_op))
    {
        cv::util::throw_error
            (std::logic_error
             ("Internal error: Streaming subgraph has multiple operations"));
    }
}

} // anonymous namespace

cv::gapi::GBackend cv::gapi::streaming::backend()
{
    static cv::gapi::GBackend this_backend(std::make_shared<GStreamingBackendImpl>());
    return this_backend;
}

struct Copy: public cv::detail::KernelTag
{
    using API = cv::gimpl::streaming::GCopy;

    static cv::gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public cv::gapi::streaming::IActor
    {
        public:
            explicit Actor(const cv::GCompileArgs&) {}
            virtual void run(cv::gimpl::GIslandExecutable::IInput  &in,
                             cv::gimpl::GIslandExecutable::IOutput &out) override;
    };

    static cv::gapi::streaming::IActor::Ptr create(const cv::GCompileArgs& args)
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor(args));
    }

    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void Copy::Actor::run(cv::gimpl::GIslandExecutable::IInput  &in,
                      cv::gimpl::GIslandExecutable::IOutput &out)
{
    const auto in_msg = in.get();
    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg))
    {
        out.post(cv::gimpl::EndOfStream{});
        return;
    }

    GAPI_DbgAssert(cv::util::holds_alternative<cv::GRunArgs>(in_msg));
    const cv::GRunArgs &in_args = cv::util::get<cv::GRunArgs>(in_msg);
    GAPI_Assert(in_args.size() == 1u);

    const auto& in_arg = in_args[0];
    auto out_arg = out.get(0);
    using cv::util::get;
    switch (in_arg.index()) {
    case cv::GRunArg::index_of<cv::RMat>():
        *get<cv::RMat*>(out_arg) = get<cv::RMat>(in_arg);
        break;
    case cv::GRunArg::index_of<cv::MediaFrame>():
        *get<cv::MediaFrame*>(out_arg) = get<cv::MediaFrame>(in_arg);
        break;
    // FIXME: Add support for remaining types
    default:
        GAPI_Error("Copy: unsupported data type");
    }
    out.meta(out_arg, in_arg.meta);
    out.post(std::move(out_arg));
}

cv::GKernelPackage cv::gimpl::streaming::kernels()
{
    return cv::gapi::kernels<Copy>();
}

#if !defined(GAPI_STANDALONE)

class GAccessorActorBase : public cv::gapi::streaming::IActor {
public:
    explicit GAccessorActorBase(const cv::GCompileArgs&) {}
    virtual void run(cv::gimpl::GIslandExecutable::IInput  &in,
                     cv::gimpl::GIslandExecutable::IOutput &out) override {
        const auto in_msg = in.get();
        if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg))
        {
            out.post(cv::gimpl::EndOfStream{});
            return;
        }

        GAPI_Assert(cv::util::holds_alternative<cv::GRunArgs>(in_msg));
        const cv::GRunArgs &in_args = cv::util::get<cv::GRunArgs>(in_msg);
        GAPI_Assert(in_args.size() == 1u);
        auto frame = cv::util::get<cv::MediaFrame>(in_args[0]);

        cv::GRunArgP out_arg = out.get(0);
        auto& rmat = *cv::util::get<cv::RMat*>(out_arg);

        extractRMat(frame, rmat);

        out.meta(out_arg, in_args[0].meta);
        out.post(std::move(out_arg));
    }

    virtual void extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat) = 0;

protected:
    std::once_flag m_warnFlag;
};

struct GOCVBGR: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GBGR;
    static cv::gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public GAccessorActorBase
    {
    public:
        using GAccessorActorBase::GAccessorActorBase;
        virtual void extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat) override;
    };

    static cv::gapi::streaming::IActor::Ptr create(const cv::GCompileArgs& args)
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void GOCVBGR::Actor::extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat)
{
    const auto& desc = frame.desc();
    switch (desc.fmt)
    {
        case cv::MediaFormat::BGR:
        {
            rmat = cv::make_rmat<cv::gimpl::RMatMediaFrameAdapter>(frame,
            [](const cv::GFrameDesc& d){ return cv::GMatDesc(CV_8U, 3, d.size); },
            [](const cv::GFrameDesc& d, const cv::MediaFrame::View& v){
                return cv::Mat(d.size, CV_8UC3, v.ptr[0], v.stride[0]);
            });
            break;
        }
        case cv::MediaFormat::NV12:
        {
            std::call_once(m_warnFlag,
                [](){
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from NV12 to BGR will happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve cv::Mat-s from NV12 cv::MediaFrame for free, you may use "
                        "cv::gapi::streaming::Y and cv::gapi::streaming::UV accessors.\n");
                });

            cv::Mat bgr;
            auto view = frame.access(cv::MediaFrame::Access::R);
            cv::Mat y_plane (desc.size,     CV_8UC1, view.ptr[0], view.stride[0]);
            cv::Mat uv_plane(desc.size / 2, CV_8UC2, view.ptr[1], view.stride[1]);
            cv::cvtColorTwoPlane(y_plane, uv_plane, bgr, cv::COLOR_YUV2BGR_NV12);
            rmat = cv::make_rmat<cv::gimpl::RMatOnMat>(bgr);
            break;
        }
        case cv::MediaFormat::GRAY:
        {
            std::call_once(m_warnFlag,
                []() {
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from GRAY to BGR will happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve cv::Mat from GRAY cv::MediaFrame for free, you may use "
                        "cv::gapi::streaming::Y.\n");
                });
            cv::Mat bgr;
            auto view = frame.access(cv::MediaFrame::Access::R);
            cv::Mat gray(desc.size, CV_8UC1, view.ptr[0], view.stride[0]);
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            rmat = cv::make_rmat<cv::gimpl::RMatOnMat>(bgr);
            break;
        }

        default:
            cv::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for cv::gapi::streaming::BGR"));
    }
}

struct GOCVY: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GY;
    static cv::gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public GAccessorActorBase
    {
    public:
        using GAccessorActorBase::GAccessorActorBase;
        virtual void extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat) override;
    };

    static cv::gapi::streaming::IActor::Ptr create(const cv::GCompileArgs& args)
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void GOCVY::Actor::extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat)
{
    const auto& desc = frame.desc();
    switch (desc.fmt)
    {
        case cv::MediaFormat::BGR:
        {
            std::call_once(m_warnFlag,
                [](){
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from BGR to NV12 Y plane will "
                        "happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve cv::Mat from BGR cv::MediaFrame for free, you may use "
                        "cv::gapi::streaming::BGR accessor.\n");
                });

            auto view = frame.access(cv::MediaFrame::Access::R);
            cv::Mat tmp_bgr(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            cv::Mat yuv;
            cvtColor(tmp_bgr, yuv, cv::COLOR_BGR2YUV_I420);
            rmat = cv::make_rmat<cv::gimpl::RMatOnMat>(yuv.rowRange(0, desc.size.height));
            break;
        }
        case cv::MediaFormat::NV12:
        {
            rmat = cv::make_rmat<cv::gimpl::RMatMediaFrameAdapter>(frame,
            [](const cv::GFrameDesc& d){ return cv::GMatDesc(CV_8U, 1, d.size); },
            [](const cv::GFrameDesc& d, const cv::MediaFrame::View& v){
                return cv::Mat(d.size, CV_8UC1, v.ptr[0], v.stride[0]);
            });
            break;
        }
        case cv::MediaFormat::GRAY:
        {
            rmat = cv::make_rmat<cv::gimpl::RMatMediaFrameAdapter>(frame,
            [](const cv::GFrameDesc& d) { return cv::GMatDesc(CV_8U, 1, d.size); },
            [](const cv::GFrameDesc& d, const cv::MediaFrame::View& v) {
                return cv::Mat(d.size, CV_8UC1, v.ptr[0], v.stride[0]);
            });
            break;
        }
        default:
            cv::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for cv::gapi::streaming::Y"));
    }
}

struct GOCVUV: public cv::detail::KernelTag
{
    using API = cv::gapi::streaming::GUV;
    static cv::gapi::GBackend backend() { return cv::gapi::streaming::backend(); }

    class Actor final: public GAccessorActorBase
    {
    public:
        using GAccessorActorBase::GAccessorActorBase;
        virtual void extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat) override;
    };

    static cv::gapi::streaming::IActor::Ptr create(const cv::GCompileArgs& args)
    {
        return cv::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static cv::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void GOCVUV::Actor::extractRMat(const cv::MediaFrame& frame, cv::RMat& rmat)
{
    const auto& desc = frame.desc();
    switch (desc.fmt)
    {
        case cv::MediaFormat::BGR:
        {
            std::call_once(m_warnFlag,
                [](){
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from BGR to NV12 UV plane will "
                        "happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve cv::Mat from BGR cv::MediaFrame for free, you may use "
                        "cv::gapi::streaming::BGR accessor.\n");
                });

            auto view = frame.access(cv::MediaFrame::Access::R);

            cv::Mat tmp_bgr(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            cv::Mat yuv;
            cvtColor(tmp_bgr, yuv, cv::COLOR_BGR2YUV_I420);

            cv::Mat uv;
            std::vector<int> dims = { desc.size.height / 2,
                                        desc.size.width / 2  };
            auto start = desc.size.height;
            auto range_h = desc.size.height / 4;
            std::vector<cv::Mat> uv_planes = {
                yuv.rowRange(start, start + range_h).reshape(0, dims),
                yuv.rowRange(start + range_h, start + range_h * 2).reshape(0, dims)
            };
            cv::merge(uv_planes, uv);
            rmat = cv::make_rmat<cv::gimpl::RMatOnMat>(uv);
            break;
        }
        case cv::MediaFormat::NV12:
        {
            rmat = cv::make_rmat<cv::gimpl::RMatMediaFrameAdapter>(frame,
            [](const cv::GFrameDesc& d){ return cv::GMatDesc(CV_8U, 2, d.size / 2); },
            [](const cv::GFrameDesc& d, const cv::MediaFrame::View& v){
                return cv::Mat(d.size / 2, CV_8UC2, v.ptr[1], v.stride[1]);
            });
            break;
        }
        case cv::MediaFormat::GRAY:
        {
            cv::Mat uv(desc.size / 2, CV_8UC2, cv::Scalar::all(127));
            rmat = cv::make_rmat<cv::gimpl::RMatOnMat>(uv);
            break;
        }
        default:
            cv::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for cv::gapi::streaming::UV"));
    }
}

cv::GKernelPackage cv::gapi::streaming::kernels()
{
    return cv::gapi::kernels<GOCVBGR, GOCVY, GOCVUV>();
}

#else

cv::GKernelPackage cv::gapi::streaming::kernels()
{
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("cv::gapi::streaming::kernels() isn't supported in standalone"));
}

#endif // !defined(GAPI_STANDALONE)

cv::GMat cv::gapi::copy(const cv::GMat& in) {
    return cv::gimpl::streaming::GCopy::on<cv::GMat>(in);
}

cv::GFrame cv::gapi::copy(const cv::GFrame& in) {
    return cv::gimpl::streaming::GCopy::on<cv::GFrame>(in);
}
