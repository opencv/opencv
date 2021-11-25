// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation

#include "precomp.hpp"

// needs to be included regardless if IE is present or not
// (cv::gapi::ie::backend() is still there and is defined always)
#include "backends/ie/giebackend.hpp"

#ifdef HAVE_INF_ENGINE

#if INF_ENGINE_RELEASE <= 2019010000
#   error G-API IE module supports only OpenVINO IE >= 2019 R1
#endif

#include <functional>
#include <unordered_set>
#include <atomic>

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/own/convert.hpp>
#include <opencv2/gapi/gframe.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/ie/util.hpp"
#include "backends/ie/giebackend/giewrapper.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!
#include "logger.hpp"

#if INF_ENGINE_RELEASE < 2021010000
#include "ie_compound_blob.h"
#endif

#if defined(HAVE_TBB)
#  include <tbb/concurrent_queue.h> // FIXME: drop it from here!
template<typename T> using QueueClass = tbb::concurrent_bounded_queue<T>;
#else
#  include "executor/conc_queue.hpp"
template<typename T> using QueueClass = cv::gapi::own::concurrent_bounded_queue<T>;
#endif // TBB

#include "utils/itt.hpp"

namespace IE = InferenceEngine;

namespace {

inline IE::ROI toIE(const cv::Rect &rc) {
    return IE::ROI
        { 0u
        , static_cast<std::size_t>(rc.x)
        , static_cast<std::size_t>(rc.y)
        , static_cast<std::size_t>(rc.width)
        , static_cast<std::size_t>(rc.height)
        };
}

inline IE::SizeVector toIE(const cv::MatSize &sz) {
    return cv::to_own<IE::SizeVector::value_type>(sz);
}
inline std::vector<int> toCV(const IE::SizeVector &vsz) {
    std::vector<int> result;
    result.reserve(vsz.size());
    for (auto sz : vsz) {
        result.push_back(ade::util::checked_cast<int>(sz));
    }
    return result;
}

inline IE::Layout toIELayout(const std::size_t ndims) {
    static const IE::Layout lts[] = {
        IE::Layout::SCALAR,
        IE::Layout::C,
        IE::Layout::NC,
        IE::Layout::CHW,
        IE::Layout::NCHW,
        IE::Layout::NCDHW,
    };
    // FIXME: This is not really a good conversion,
    // since it may also stand for NHWC/HW/CN/NDHWC data
    CV_Assert(ndims < sizeof(lts) / sizeof(lts[0]));
    return lts[ndims];
}

inline IE::Precision toIE(int depth) {
    switch (depth) {
    case CV_8U:  return IE::Precision::U8;
    case CV_32S: return IE::Precision::I32;
    case CV_32F: return IE::Precision::FP32;
    case CV_16F: return IE::Precision::FP16;
    default:     GAPI_Assert(false && "IE. Unsupported data type");
    }
    return IE::Precision::UNSPECIFIED;
}
inline int toCV(IE::Precision prec) {
    switch (prec) {
    case IE::Precision::U8:   return CV_8U;
    case IE::Precision::FP32: return CV_32F;
    case IE::Precision::I32:  return CV_32S;
    case IE::Precision::I64:  return CV_32S;
    case IE::Precision::FP16: return CV_16F;
    default:     GAPI_Assert(false && "IE. Unsupported data type");
    }
    return -1;
}

inline IE::TensorDesc toIE(const cv::Mat &mat, cv::gapi::ie::TraitAs hint) {
    const auto &sz = mat.size;
    // NB: For some reason RGB image is 2D image
    // (since channel component is not counted here).
    // Note: regular 2D vectors also fall into this category
    if (sz.dims() == 2 && hint == cv::gapi::ie::TraitAs::IMAGE)
    {
        // NB: This logic is mainly taken from IE samples
        const size_t channels = mat.channels();
        const size_t height   = mat.size().height;
        const size_t width    = mat.size().width;

        const size_t strideH  = mat.step1();
        IE::BlockingDesc bdesc({1, height, width, channels} /* blocking dims */,
                               {0, 2, 3, 1} /* order for NHWC   */,
                               0            /* offset           */,
                               {0, 0, 0, 0} /* offsets for dims */,
                               {strideH * height, strideH, channels, 1} /* strides for dims */);

        return IE::TensorDesc(toIE(mat.depth()),
                              IE::SizeVector{1, channels, height, width}, bdesc);
    }
    return IE::TensorDesc(toIE(mat.depth()), toIE(sz), toIELayout(sz.dims()));
}

inline IE::Blob::Ptr wrapIE(const cv::Mat &mat, cv::gapi::ie::TraitAs hint) {
    const auto tDesc = toIE(mat, hint);
    switch (mat.depth()) {
        // NB: Seems there's no way to create an untyped (T-less) Blob::Ptr
        // in IE given only precision via TensorDesc. So we have to do this:
#define HANDLE(E,T) \
        case CV_##E: return IE::make_shared_blob<T>(tDesc, const_cast<T*>(mat.ptr<T>()))
        HANDLE(8U, uint8_t);
        HANDLE(32F, float);
        HANDLE(32S, int);
        HANDLE(16F, int16_t);
#undef HANDLE
    default: GAPI_Assert(false && "IE. Unsupported data type");
    }
    return IE::Blob::Ptr{};
}

inline IE::Blob::Ptr wrapIE(const cv::MediaFrame::View& view,
                            const cv::GFrameDesc& desc) {

    switch (desc.fmt) {
        case cv::MediaFormat::BGR: {
            auto bgr = cv::Mat(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            return wrapIE(bgr, cv::gapi::ie::TraitAs::IMAGE);
        }
        case cv::MediaFormat::NV12: {
            auto y_plane  = cv::Mat(desc.size, CV_8UC1, view.ptr[0], view.stride[0]);
            auto uv_plane = cv::Mat(desc.size / 2, CV_8UC2, view.ptr[1], view.stride[1]);
            return cv::gapi::ie::util::to_ie(y_plane, uv_plane);
        }
        default:
            GAPI_Assert(false && "Unsupported media format for IE backend");
    }
    GAPI_Assert(false);
}

template<class MatType>
inline void copyFromIE(const IE::Blob::Ptr &blob, MatType &mat) {
    switch (blob->getTensorDesc().getPrecision()) {
#define HANDLE(E,T)                                                 \
        case IE::Precision::E: std::copy_n(blob->buffer().as<T*>(), \
                                           mat.total(),             \
                                           reinterpret_cast<T*>(mat.data)); \
            break;
        HANDLE(U8, uint8_t);
        HANDLE(FP32, float);
        HANDLE(I32, int);
        HANDLE(FP16, cv::float16_t);
#undef HANDLE
        case IE::Precision::I64: {
            GAPI_LOG_WARNING(NULL, "INT64 isn't supported for cv::Mat. Conversion to INT32 is used.");
            cv::gimpl::convertInt64ToInt32(blob->buffer().as<int64_t*>(),
                                           reinterpret_cast<int*>(mat.data),
                                           mat.total());
            break;
        }
    default: GAPI_Assert(false && "IE. Unsupported data type");
    }
}

// IE-specific metadata, represents a network with its parameters
struct IEUnit {
    static const char *name() { return "IEModelConfig"; }

    cv::gapi::ie::detail::ParamDesc params;
    IE::CNNNetwork net;

    IE::ExecutableNetwork this_network;
    cv::gimpl::ie::wrap::Plugin this_plugin;

    InferenceEngine::RemoteContext::Ptr rctx = nullptr;

    // FIXME: Unlike loadNetwork case, importNetwork requires that preprocessing
    // should be passed as ExecutableNetwork::SetBlob method, so need to collect
    // and store this information at the graph compilation stage (outMeta) and use in runtime.
    using PreProcMap = std::unordered_map<std::string, IE::PreProcessInfo>;
    PreProcMap preproc_map;

    explicit IEUnit(const cv::gapi::ie::detail::ParamDesc &pp)
        : params(pp) {
        InferenceEngine::ParamMap* ctx_params =
                            cv::util::any_cast<InferenceEngine::ParamMap>(&params.context_config);
        if (ctx_params != nullptr) {
            auto ie_core = cv::gimpl::ie::wrap::getCore();
            rctx = ie_core.CreateContext(params.device_id, *ctx_params);
        }

        if (params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
            net = cv::gimpl::ie::wrap::readNetwork(params);
            // NB: Set batch size only if user asked. (don't set by default)
            if (params.batch_size.has_value())  {
                net.setBatchSize(params.batch_size.value());
            }
        } else if (params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Import) {
            this_plugin = cv::gimpl::ie::wrap::getPlugin(params);
            this_network = cv::gimpl::ie::wrap::importNetwork(this_plugin, params, rctx);
            if (!params.reshape_table.empty() || !params.layer_names_to_reshape.empty()) {
                GAPI_LOG_WARNING(NULL, "Reshape isn't supported for imported network");
            }
        } else {
            cv::util::throw_error(std::logic_error("Unsupported ParamDesc::Kind"));
        }

        // The practice shows that not all inputs and not all outputs
        // are mandatory to specify in IE model.
        // So what we're concerned here about is:
        // if operation's (not topology's) input/output number is
        // greater than 1, then we do care about input/output layer
        // names. Otherwise, names are picked up automatically.
        // TODO: Probably this check could be done at the API entry point? (gnet)
        if (params.num_in > 1u && params.num_in != params.input_names.size()) {
            cv::util::throw_error(std::logic_error("Please specify input layer names for "
                                                   + params.model_path));
        }
        if (params.num_out > 1u && params.num_out != params.output_names.size()) {
            cv::util::throw_error(std::logic_error("Please specify output layer names for "
                                                   + params.model_path));
        }
        if (params.num_in == 1u && params.input_names.empty()) {
            if (params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
                params.input_names = { net.getInputsInfo().begin()->first };
            } else {
                params.input_names = { this_network.GetInputsInfo().begin()->first };
            }
        }
        if (params.num_out == 1u && params.output_names.empty()) {
            if (params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
                params.output_names = { net.getOutputsInfo().begin()->first };
            } else {
                params.output_names = { this_network.GetOutputsInfo().begin()->first };
            }
        }
        if (!params.reshape_table.empty()) {
            GAPI_Assert((params.reshape_table.size() + params.layer_names_to_reshape.size()) <=
                         params.num_in &&
                        "Number of layers to reshape must be less than or equal to number of inputs");
        }
    }

    // This method is [supposed to be] called at Island compilation stage
    cv::gimpl::ie::IECompiled compile() const {
        IEUnit* non_const_this = const_cast<IEUnit*>(this);
        // FIXME: LoadNetwork must be called only after all necessary model
        // inputs information is set, since it's done in outMeta and compile called after that,
        // this place seems to be suitable, but consider another place not to break const agreements.
        if (params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
            non_const_this->this_plugin  = cv::gimpl::ie::wrap::getPlugin(params);
            non_const_this->this_network = cv::gimpl::ie::wrap::loadNetwork(non_const_this->this_plugin,
                                                                            net, params, rctx);
        }

        return {params, this_plugin, this_network};
    }
};

class IECallContext
{
public:
    IECallContext(const IEUnit                                      &  unit,
                  cv::gimpl::GIslandExecutable::IOutput             &  output,
                  const cv::GArgs                                   &  args,
                  const std::vector<cv::gimpl::RcDesc>              &  outs,
                  std::vector<cv::gimpl::GIslandExecutable::InObj>  && input_objs,
                  std::vector<cv::gimpl::GIslandExecutable::OutObj> && output_objs);

    const cv::GArgs& inArgs() const;

    // Generic accessor API
    template<typename T>
    const T& inArg(std::size_t input) const {
        return m_args.at(input).get<T>();
    }

    template<typename T>
    std::vector<T>& outVecR(std::size_t output) {
        return outVecRef(output).wref<T>();
    }

    // Syntax sugar
          cv::GShape      inShape(std::size_t input) const;
    const cv::Mat&        inMat  (std::size_t input) const;
    const cv::MediaFrame& inFrame(std::size_t input) const;

    const cv::GRunArg& input  (std::size_t idx) const;
          cv::GRunArgP output (std::size_t idx);
          cv::Mat&     outMatR(std::size_t idx);

    const IEUnit                          &uu;
    cv::gimpl::GIslandExecutable::IOutput &out;

    // NB: Need to gurantee that MediaFrame::View don't die until request is over.
    using Views = std::vector<std::unique_ptr<cv::MediaFrame::View>>;
    Views views;

private:
    cv::detail::VectorRef& outVecRef(std::size_t idx);

    cv::GArg packArg(const cv::GArg &arg);

    // To store input/output data from frames
    std::vector<cv::gimpl::GIslandExecutable::InObj>  m_input_objs;
    std::vector<cv::gimpl::GIslandExecutable::OutObj> m_output_objs;

    // To simplify access to cv::Mat inside cv::RMat
    cv::gimpl::Mag m_res;

    // FIXME: avoid conversion of arguments from internal representation to OpenCV one on each call
    //to OCV kernel. (This can be achieved by a two single time conversions in GCPUExecutable::run,
    //once on enter for input and output arguments, and once before return for output arguments only
    // FIXME: check if the above applies to this backend (taken from CPU)
    std::unordered_map<std::size_t, cv::GRunArgP> m_results;

    // Input parameters passed to an inference operation.
    cv::GArgs m_args;
    cv::GShapes m_in_shapes;
};

IECallContext::IECallContext(const IEUnit                                      &  unit,
                             cv::gimpl::GIslandExecutable::IOutput             &  output,
                             const cv::GArgs                                   &  args,
                             const std::vector<cv::gimpl::RcDesc>              &  outs,
                             std::vector<cv::gimpl::GIslandExecutable::InObj>  && input_objs,
                             std::vector<cv::gimpl::GIslandExecutable::OutObj> && output_objs)
: uu(unit), out(output), m_input_objs(std::move(input_objs)), m_output_objs(std::move(output_objs))
{
    for (auto& it : m_input_objs)  cv::gimpl::magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : m_output_objs) cv::gimpl::magazine::bindOutArg(m_res, it.first, it.second);

    m_args.reserve(args.size());
    using namespace std::placeholders;
    ade::util::transform(args,
                         std::back_inserter(m_args),
                         std::bind(&IECallContext::packArg, this, _1));

    ade::util::transform(args, std::back_inserter(m_in_shapes),
            [](const cv::GArg& arg) {
                return arg.get<cv::gimpl::RcDesc>().shape;
            });

    for (const auto out_it : ade::util::indexed(outs)) {
        // FIXME: Can the same GArg type resolution mechanism be reused here?
        const auto port  = ade::util::index(out_it);
        const auto desc  = ade::util::value(out_it);
        m_results[port] = cv::gimpl::magazine::getObjPtr(m_res, desc);
    }
}

const cv::GArgs& IECallContext::inArgs() const {
    return m_args;
}

cv::GShape IECallContext::inShape(std::size_t i) const {
    return m_in_shapes[i];
}

const cv::Mat& IECallContext::inMat(std::size_t input) const {
    return inArg<cv::Mat>(input);
}

const cv::MediaFrame& IECallContext::inFrame(std::size_t input) const {
    return inArg<cv::MediaFrame>(input);
}

cv::Mat& IECallContext::outMatR(std::size_t idx) {
    return *cv::util::get<cv::Mat*>(m_results.at(idx));
}

cv::GRunArgP IECallContext::output(std::size_t idx) {
    return m_output_objs[idx].second;
};

const cv::GRunArg& IECallContext::input(std::size_t idx) const {
    return m_input_objs[idx].second;
}

cv::detail::VectorRef& IECallContext::outVecRef(std::size_t idx) {
    return cv::util::get<cv::detail::VectorRef>(m_results.at(idx));
}

cv::GArg IECallContext::packArg(const cv::GArg &arg) {
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY);

    if (arg.kind != cv::detail::ArgKind::GOBJREF) {
        cv::util::throw_error(std::logic_error("Inference supports G-types ONLY!"));
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case cv::GShape::GMAT: return cv::GArg(m_res.slot<cv::Mat>()[ref.id]);

    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case cv::GShape::GARRAY:  return cv::GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));

    // Note: .at() is intentional for GOpaque as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case cv::GShape::GOPAQUE:  return cv::GArg(m_res.slot<cv::detail::OpaqueRef>().at(ref.id));

    case cv::GShape::GFRAME:  return cv::GArg(m_res.slot<cv::MediaFrame>().at(ref.id));

    default:
        cv::util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

struct IECallable {
    static const char *name() { return "IERequestCallable"; }
    using Run = std::function<void(std::shared_ptr<IECallContext>, cv::gimpl::ie::RequestPool&)>;
    Run run;
};

struct KImpl {
    cv::gimpl::CustomMetaFunction::CM customMetaFunc;
    IECallable::Run run;
};

// FIXME: Is there a way to take a typed graph (our GModel),
// and create a new typed graph _ATOP_ of that (by extending with a couple of
// new types?).
// Alternatively, is there a way to compose types graphs?
//
// If not, we need to introduce that!
using GIEModel = ade::TypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    , IEUnit
    , IECallable
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGIEModel = ade::ConstTypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    , IEUnit
    , IECallable
    >;

inline IE::Blob::Ptr extractRemoteBlob(IECallContext& ctx, std::size_t i) {
    GAPI_Assert(ctx.inShape(i) == cv::GShape::GFRAME &&
                "Remote blob is supported for MediaFrame only");

    cv::util::any any_blob_params = ctx.inFrame(i).blobParams();
    auto ie_core = cv::gimpl::ie::wrap::getCore();

    using ParamType = std::pair<InferenceEngine::TensorDesc,
                                InferenceEngine::ParamMap>;

    ParamType* blob_params = cv::util::any_cast<ParamType>(&any_blob_params);
    if (blob_params == nullptr) {
        GAPI_Assert(false && "Incorrect type of blobParams: "
                              "expected std::pair<InferenceEngine::TensorDesc,"
                                                 "InferenceEngine::ParamMap>");
    }

    return ctx.uu.rctx->CreateBlob(blob_params->first,
                                   blob_params->second);
}

inline IE::Blob::Ptr extractBlob(IECallContext& ctx,
                                 std::size_t i,
                                 cv::gapi::ie::TraitAs hint) {
    if (ctx.uu.rctx != nullptr) {
        return extractRemoteBlob(ctx, i);
    }

    switch (ctx.inShape(i)) {
        case cv::GShape::GFRAME: {
            const auto& frame = ctx.inFrame(i);
            ctx.views.emplace_back(new cv::MediaFrame::View(frame.access(cv::MediaFrame::Access::R)));
            return wrapIE(*(ctx.views.back()), frame.desc());
        }
        case cv::GShape::GMAT: {
            return wrapIE(ctx.inMat(i), hint);
        }
        default:
            GAPI_Assert("Unsupported input shape for IE backend");
    }
    GAPI_Assert(false);
}


static void setBlob(InferenceEngine::InferRequest& req,
                    const std::string&             layer_name,
                    const IE::Blob::Ptr&           blob,
                    const IECallContext&           ctx) {
    // TODO: Ideally we shouldn't do SetBlob() but GetBlob() instead,
    // and redirect our data producers to this memory
    // (A memory dialog comes to the picture again)
    using namespace cv::gapi::ie::detail;
    if (ctx.uu.params.kind == ParamDesc::Kind::Load) {
        req.SetBlob(layer_name, blob);
    } else {
        GAPI_Assert(ctx.uu.params.kind == ParamDesc::Kind::Import);
        req.SetBlob(layer_name, blob, ctx.uu.preproc_map.at(layer_name));
    }
}

} // anonymous namespace

std::vector<InferenceEngine::InferRequest> cv::gimpl::ie::IECompiled::createInferRequests() {
    std::vector<InferenceEngine::InferRequest> requests;
    requests.reserve(params.nireq);

    for (size_t i = 0; i < params.nireq; ++i) {
        requests.push_back(this_network.CreateInferRequest());
        auto& request = requests.back();
        // Bind const data to infer request
        for (auto &&p : params.const_inputs) {
            // FIXME: SetBlob is known to be inefficient,
            // it is worth to make a customizable "initializer" and pass the
            // cv::Mat-wrapped blob there to support IE's optimal "GetBlob idiom"
            // Still, constant data is to set only once.
            request.SetBlob(p.first, wrapIE(p.second.first, p.second.second));
        }
    }

    return requests;
}

class cv::gimpl::ie::RequestPool {
public:
    using RunF      = std::function<void(InferenceEngine::InferRequest&)>;
    using CallbackF = std::function<void(InferenceEngine::InferRequest&)>;

    // NB: The task is represented by:
    // RunF      - function which is set blobs and run async inference.
    // CallbackF - function which is obtain output blobs and post it to output.
    struct Task {
        RunF run;
        CallbackF callback;
    };

    explicit RequestPool(std::vector<InferenceEngine::InferRequest>&& requests);

    void execute(Task&& t);
    void waitAll();

private:
    void callback(Task task, InferenceEngine::InferRequest& request, size_t id);
    void setup();

    QueueClass<size_t>                         m_idle_ids;
    std::vector<InferenceEngine::InferRequest> m_requests;
};

// RequestPool implementation //////////////////////////////////////////////
cv::gimpl::ie::RequestPool::RequestPool(std::vector<InferenceEngine::InferRequest>&& requests)
    : m_requests(std::move(requests)) {
        setup();
    }

void cv::gimpl::ie::RequestPool::setup() {
    for (size_t i = 0; i < m_requests.size(); ++i) {
        m_idle_ids.push(i);
    }
}

void cv::gimpl::ie::RequestPool::execute(cv::gimpl::ie::RequestPool::Task&& t) {
    size_t id = 0u;
    m_idle_ids.pop(id);

    auto& request = m_requests[id];

    request.SetCompletionCallback(
            std::bind(&cv::gimpl::ie::RequestPool::callback, this, t, std::ref(request), id));
    t.run(request);
}

void cv::gimpl::ie::RequestPool::callback(cv::gimpl::ie::RequestPool::Task task,
                                          InferenceEngine::InferRequest& request,
                                          size_t id) {
    task.callback(request);
    // NB: IE::InferRequest keeps the callback until the new one is set.
    // Since user's callback might keep resources that should be released,
    // need to destroy its after execution.
    // Let's set the empty one to cause the destruction of a callback.
    request.SetCompletionCallback([](){});
    m_idle_ids.push(id);
}

// NB: Not thread-safe.
void cv::gimpl::ie::RequestPool::waitAll() {
    // NB: It will be blocked if at least one request is busy.
    for (size_t i = 0; i < m_requests.size(); ++i) {
        size_t id = 0u;
        m_idle_ids.pop(id);
    }
    setup();
}

// GCPUExcecutable implementation //////////////////////////////////////////////
cv::gimpl::ie::GIEExecutable::GIEExecutable(const ade::Graph &g,
                                            const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g) {
    // FIXME: Currently this backend is capable to run a single inference node only.
    // Need to extend our island fusion with merge/not-to-merge decision making parametrization
    GConstGIEModel iem(g);

    for (auto &nh : nodes) {
        switch (m_gm.metadata(nh).get<NodeType>().t) {
        case NodeType::OP:
            if (this_nh == nullptr) {
                this_nh = nh;
                this_iec = iem.metadata(this_nh).get<IEUnit>().compile();
                m_reqPool.reset(new RequestPool(this_iec.createInferRequests()));
            }
            else
                util::throw_error(std::logic_error("Multi-node inference is not supported!"));
            break;

        case NodeType::DATA: {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST_VAL) {
                util::throw_error(std::logic_error("No const data please!"));
            }
            if (desc.storage == Data::Storage::INTERNAL) {
                util::throw_error(std::logic_error("No internal data please!"));
            }
            break;
        }
        default: util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
    }
}

void cv::gimpl::ie::GIEExecutable::run(cv::gimpl::GIslandExecutable::IInput  &in,
                                       cv::gimpl::GIslandExecutable::IOutput &out) {
    // General alghoritm:
    //     1. Collect island inputs/outputs.
    //     2. Create kernel context. (Every kernel has his own context).
    //     3. If the EndOfStream message is recieved, wait until all passed task are done.
    //     4.
    //        5.1 Run the kernel.
    //        5.2 Kernel wait for all nececcary infer requests and start asynchronous execution.
    //        5.3 After the kernel is finished continue processing next frame.
    //
    //     5. If graph is compiled in non-streaming mode, wait until all tasks are done.

    std::vector<InObj>  input_objs;
    std::vector<OutObj> output_objs;

    const auto &in_desc  = in.desc();
    const auto  in_msg   = in.get();

    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg))
    {
        // (3) Wait until all passed task are done.
        m_reqPool->waitAll();
        out.post(cv::gimpl::EndOfStream{});
        return;
    }

    GAPI_Assert(cv::util::holds_alternative<cv::GRunArgs>(in_msg));
    const auto in_vector = cv::util::get<cv::GRunArgs>(in_msg);

    // (1) Collect island inputs/outputs
    input_objs.reserve(in_desc.size());
    for (auto &&it: ade::util::zip(ade::util::toRange(in_desc),
                    ade::util::toRange(in_vector)))
    {
        input_objs.emplace_back(std::get<0>(it), std::get<1>(it));
    }

    const auto &out_desc = out.desc();
    output_objs.reserve(out_desc.size());
    for (auto &&it: ade::util::indexed(ade::util::toRange(out_desc)))
    {
        output_objs.emplace_back(ade::util::value(it),
                out.get(ade::util::checked_cast<int>(ade::util::index(it))));
    }

    GConstGIEModel giem(m_g);
    const auto &uu = giem.metadata(this_nh).get<IEUnit>();
    const auto &op = m_gm.metadata(this_nh).get<Op>();
    // (2) Create kernel context
    auto ctx = std::make_shared<IECallContext>(uu, out, op.args, op.outs,
            std::move(input_objs), std::move(output_objs));

    const auto &kk = giem.metadata(this_nh).get<IECallable>();

    // (4) Run the kernel.
    kk.run(ctx, *m_reqPool);

    // (5) In non-streaming mode need to wait until the all tasks are done
    // FIXME: Is there more graceful way to handle this case ?
    if (!m_gm.metadata().contains<Streaming>()) {
        m_reqPool->waitAll();
    }
}

namespace cv {
namespace gimpl {
namespace ie {
static void configureInputReshapeByImage(const IE::InputInfo::Ptr& ii,
                                         const cv::GMetaArg mm,
                                         IE::ICNNNetwork::InputShapes& input_reshape_table) {
    const auto& layer_name = ii->name();
    // Finding name in reshape table
    const auto name_pos_in_table = input_reshape_table.find(layer_name);
    // If contains then reshape for this layer already configured by shapes
    // otherwise create a new element of reshape table with name and dimension
    // which based on input image size.
    if (name_pos_in_table != input_reshape_table.end()) {
        GAPI_Assert(false &&
                    "Names of layers for reshape with specified dimensions shouldn't intersect with names for reshape by image");
    }
    cv::Size image_sz;
    switch (mm.index()) {
        case cv::GMetaArg::index_of<cv::GMatDesc>():
            {
                const auto &meta = util::get<cv::GMatDesc>(mm);
                image_sz = meta.size;
                break;
            }
        case cv::GMetaArg::index_of<cv::GFrameDesc>():
            {
                const auto &meta = util::get<cv::GFrameDesc>(mm);
                image_sz = meta.size;
                break;
            }
        default:
            util::throw_error(std::runtime_error("Unsupported input meta for IE backend"));
    }
    auto input_dims = ii->getTensorDesc().getDims();
    const auto size = input_dims.size();
    if (size <= 1) {
        GAPI_Assert(false && "Unsupported number of dimensions for reshape by image");
    }
    input_dims.at(size - 2) = static_cast<size_t>(image_sz.height);
    input_dims.at(size - 1) = static_cast<size_t>(image_sz.width);
    // Adding new element to reshape table
    input_reshape_table.emplace(layer_name, input_dims);
}

static void configureInputInfo(const IE::InputInfo::Ptr& ii, const cv::GMetaArg mm) {
    switch (mm.index()) {
        case cv::GMetaArg::index_of<cv::GMatDesc>():
        {
            ii->setPrecision(toIE(util::get<cv::GMatDesc>(mm).depth));
            break;
        }
        case cv::GMetaArg::index_of<cv::GFrameDesc>():
        {
            const auto &meta = util::get<cv::GFrameDesc>(mm);
            switch (meta.fmt) {
                case cv::MediaFormat::NV12:
                    ii->getPreProcess().setColorFormat(IE::ColorFormat::NV12);
                    break;
                case cv::MediaFormat::BGR:
                    // NB: Do nothing
                    break;
                default:
                    GAPI_Assert(false && "Unsupported media format for IE backend");
            }
            ii->setPrecision(toIE(CV_8U));
            break;
        }
        default:
            util::throw_error(std::runtime_error("Unsupported input meta for IE backend"));
    }
}

static IE::PreProcessInfo configurePreProcInfo(const IE::InputInfo::CPtr& ii,
                                               const cv::GMetaArg&        mm) {
    IE::PreProcessInfo info;
    if (cv::util::holds_alternative<cv::GFrameDesc>(mm)) {
        auto desc = cv::util::get<cv::GFrameDesc>(mm);
        if (desc.fmt == cv::MediaFormat::NV12) {
            info.setColorFormat(IE::ColorFormat::NV12);
        }
    }
    const auto layout = ii->getTensorDesc().getLayout();
    if (layout == IE::Layout::NCHW ||
        layout == IE::Layout::NHWC) {
        info.setResizeAlgorithm(IE::RESIZE_BILINEAR);
    }
    return info;
}

// NB: This is a callback used by async infer
// to post outputs blobs (cv::GMat's).
static void PostOutputs(InferenceEngine::InferRequest   &request,
                        std::shared_ptr<IECallContext>   ctx) {
    GAPI_ITT_STATIC_LOCAL_HANDLE(ie_cb_post_outputs_hndl, "IE_async_callback_PostOutputs");
    GAPI_ITT_AUTO_TRACE_GUARD(ie_cb_post_outputs_hndl);

    for (auto i : ade::util::iota(ctx->uu.params.num_out))
    {
        auto& out_mat = ctx->outMatR(i);
        IE::Blob::Ptr this_blob = request.GetBlob(ctx->uu.params.output_names[i]);
        copyFromIE(this_blob, out_mat);
        auto output = ctx->output(i);
        ctx->out.meta(output, ctx->input(0).meta);
        ctx->out.post(std::move(output));
    }
}

class PostOutputsList {
public:
    PostOutputsList(size_t size,
                    std::shared_ptr<IECallContext> ctx,
                    std::vector<std::vector<int>>&& cached_dims);

    void operator()(InferenceEngine::InferRequest &request, size_t pos) const;

private:
    struct Priv {
        size_t              size;
        std::atomic<size_t> finished{0u};
        std::shared_ptr<IECallContext> ctx;
        std::vector<std::vector<int>> cached_dims;
    };
    std::shared_ptr<Priv> m_priv;
};

PostOutputsList::PostOutputsList(size_t size,
                                 std::shared_ptr<IECallContext> ctx,
                                 std::vector<std::vector<int>>&& cached_dims)
    : m_priv(new Priv()) {
    m_priv->size = size;
    m_priv->ctx = ctx;
    m_priv->cached_dims = std::move(cached_dims);
}

void PostOutputsList::operator()(InferenceEngine::InferRequest &req, size_t pos) const {
    auto&& ctx         = m_priv->ctx;
    auto&& cached_dims = m_priv->cached_dims;
    auto&& finished    = m_priv->finished;
    auto&& size        = m_priv->size;
    for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
        std::vector<cv::Mat> &out_vec = ctx->outVecR<cv::Mat>(i);

        IE::Blob::Ptr out_blob = req.GetBlob(ctx->uu.params.output_names[i]);
        GAPI_Assert(out_blob);

        // FIXME: Avoid data copy. Not sure if it is possible though
        out_vec[pos].create(cached_dims[i], toCV(out_blob->getTensorDesc().getPrecision()));
        copyFromIE(out_blob, out_vec[pos]);
    }
    ++finished;

    if (finished == size) {
        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            auto output = ctx->output(i);
            ctx->out.meta(output, ctx->input(0).meta);
            ctx->out.post(std::move(output));
        }
    }
}

struct Infer: public cv::detail::KernelTag {
    using API = cv::GInferBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        // Specify network's output layer metadata to the framework
        // Also specify the input information to the IE from the framework
        // NB: Have no clue if network's input [dimensions] may ever define
        // its output dimensions. It seems possible with OpenCV DNN APIs

        cv::GMetaArgs result;

        GConstGIEModel gm(gr);
        const auto &uu = gm.metadata(nh).get<IEUnit>();
        IE::ICNNNetwork::InputShapes input_reshape_table = uu.params.reshape_table;

        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == in_metas.size()
                    && "Known input layers count doesn't match input meta count");

        // NB: Configuring input precision and network reshape must be done
        // only in the loadNetwork case.
        using namespace cv::gapi::ie::detail;
        if (uu.params.kind == ParamDesc::Kind::Load) {
            auto inputs = uu.net.getInputsInfo();
            for (auto &&it : ade::util::zip(ade::util::toRange(uu.params.input_names),
                                            ade::util::toRange(in_metas))) {
                    const auto &input_name = std::get<0>(it);
                    auto ii = inputs.at(input_name);
                    const auto & mm = std::get<1>(it);

                    configureInputInfo(ii, mm);
                    if (uu.params.layer_names_to_reshape.find(input_name) !=
                        uu.params.layer_names_to_reshape.end()) {
                        configureInputReshapeByImage(ii, mm, input_reshape_table);
                    }

                    // NB: Configure resize only for NCHW/NHWC layout,
                    // since it isn't supposed to work with others.
                    auto layout = ii->getTensorDesc().getLayout();
                    if (layout == IE::Layout::NCHW ||
                        layout == IE::Layout::NHWC) {
                        ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
                    }
            }

            // FIXME: This isn't the best place to call reshape function.
            // Сorrect solution would be to do this in compile() method of network,
            // but now input meta isn't passed to compile() method.
            if (!input_reshape_table.empty()) {
                const_cast<IE::CNNNetwork *>(&uu.net)->reshape(input_reshape_table);
            }
        } else {
            GAPI_Assert(uu.params.kind == ParamDesc::Kind::Import);
            auto inputs = uu.this_network.GetInputsInfo();
            // FIXME: This isn't the best place to collect PreProcMap.
            auto* non_const_prepm = const_cast<IEUnit::PreProcMap*>(&uu.preproc_map);
            for (auto &&it : ade::util::zip(ade::util::toRange(uu.params.input_names),
                                            ade::util::toRange(in_metas))) {
                const auto &input_name = std::get<0>(it);
                auto ii = inputs.at(input_name);
                const auto & mm = std::get<1>(it);
                non_const_prepm->emplace(input_name, configurePreProcInfo(ii, mm));
            }
        }

        // FIXME: It would be nice here to have an exact number of network's
        // input/output parameters. Probably GCall should store it here for us.
        // It doesn't, as far as I know..
        for (const auto &out_name : uu.params.output_names) {
            // NOTE: our output_names vector follows the API order
            // of this operation's outputs
            const auto& desc =
                uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load
                    ? uu.net.getOutputsInfo().at(out_name)->getTensorDesc()
                    : uu.this_network.GetOutputsInfo().at(out_name)->getTensorDesc();

            cv::GMatDesc outm(toCV(desc.getPrecision()),
                              toCV(desc.getDims()));
            result.emplace_back(outm);
        }
        return result;
    }

    static void run(std::shared_ptr<IECallContext>  ctx,
                    cv::gimpl::ie::RequestPool     &reqPool) {
        using namespace std::placeholders;
        reqPool.execute(
                cv::gimpl::ie::RequestPool::Task {
                    [ctx](InferenceEngine::InferRequest &req) {
                        // non-generic version for now:
                        // - assumes all inputs/outputs are always Mats
                        for (auto i : ade::util::iota(ctx->uu.params.num_in)) {
                            const auto& layer_name = ctx->uu.params.input_names[i];
                            auto layout =
                                ctx->uu.this_network.GetInputsInfo().
                                    at(layer_name)->getTensorDesc().getLayout();
                            auto hint =
                                (layout == IE::Layout::NCHW || layout == IE::Layout::NHWC)
                                ? cv::gapi::ie::TraitAs::IMAGE : cv::gapi::ie::TraitAs::TENSOR;

                            IE::Blob::Ptr this_blob = extractBlob(*ctx, i, hint);
                            setBlob(req, layer_name, this_blob, *ctx);
                        }
                        // FIXME: Should it be done by kernel ?
                        // What about to do that in RequestPool ?
                        req.StartAsync();
                    },
                    std::bind(PostOutputs, _1, ctx)
                }
        );
    }
};

struct InferROI: public cv::detail::KernelTag {
    using API = cv::GInferROIBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        cv::GMetaArgs result;

        GConstGIEModel gm(gr);
        const auto &uu = gm.metadata(nh).get<IEUnit>();
        IE::ICNNNetwork::InputShapes input_reshape_table = uu.params.reshape_table;

        // Initialize input information
        // FIXME: So far it is pretty limited
        GAPI_Assert(1u == uu.params.input_names.size());
        GAPI_Assert(2u == in_metas.size());

        const auto &input_name = uu.params.input_names.at(0);
        auto &&mm = in_metas.at(1u);
        // NB: Configuring input precision and network reshape must be done
        // only in the loadNetwork case.
        if (uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
            // 0th is ROI, 1st is input image
            auto ii = uu.net.getInputsInfo().at(input_name);
            configureInputInfo(ii, mm);
            if (uu.params.layer_names_to_reshape.find(input_name) !=
                uu.params.layer_names_to_reshape.end()) {
                configureInputReshapeByImage(ii, mm, input_reshape_table);
            }
            ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);

            // FIXME: This isn't the best place to call reshape function.
            // Сorrect solution would be to do this in compile() method of network,
            // but now input meta isn't passed to compile() method.
            if (!input_reshape_table.empty()) {
                const_cast<IE::CNNNetwork *>(&uu.net)->reshape(input_reshape_table);
            }
        } else {
            GAPI_Assert(uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Import);
            auto inputs = uu.this_network.GetInputsInfo();
            // FIXME: This isn't the best place to collect PreProcMap.
            auto* non_const_prepm = const_cast<IEUnit::PreProcMap*>(&uu.preproc_map);
            auto ii = inputs.at(input_name);
            non_const_prepm->emplace(input_name, configurePreProcInfo(ii, mm));
        }

        // FIXME: It would be nice here to have an exact number of network's
        // input/output parameters. Probably GCall should store it here for us.
        // It doesn't, as far as I know..
        for (const auto &out_name : uu.params.output_names) {
            // NOTE: our output_names vector follows the API order
            // of this operation's outputs
            const auto& desc =
                uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load
                    ? uu.net.getOutputsInfo().at(out_name)->getTensorDesc()
                    : uu.this_network.GetOutputsInfo().at(out_name)->getTensorDesc();

            cv::GMatDesc outm(toCV(desc.getPrecision()),
                              toCV(desc.getDims()));
            result.emplace_back(outm);
        }
        return result;
    }

    static void run(std::shared_ptr<IECallContext>  ctx,
                    cv::gimpl::ie::RequestPool     &reqPool) {
        using namespace std::placeholders;
        reqPool.execute(
                cv::gimpl::ie::RequestPool::Task {
                    [ctx](InferenceEngine::InferRequest &req) {
                        GAPI_Assert(ctx->uu.params.num_in == 1);
                        auto&& this_roi = ctx->inArg<cv::detail::OpaqueRef>(0).rref<cv::Rect>();

                        // NB: This blob will be used to make roi from its, so
                        // it should be treated as image
                        IE::Blob::Ptr this_blob =
                            extractBlob(*ctx, 1, cv::gapi::ie::TraitAs::IMAGE);
                        setBlob(req,
                                *(ctx->uu.params.input_names.begin()),
                                IE::make_shared_blob(this_blob, toIE(this_roi)),
                                *ctx);
                        // FIXME: Should it be done by kernel ?
                        // What about to do that in RequestPool ?
                        req.StartAsync();
                    },
                    std::bind(PostOutputs, _1, ctx)
                }
        );
    }
};


struct InferList: public cv::detail::KernelTag {
    using API = cv::GInferListBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        // Specify the input information to the IE from the framework
        // NB: Have no clue if network's input [dimensions] may ever define
        // its output dimensions. It seems possible with OpenCV DNN APIs

        GConstGIEModel gm(gr);
        const auto &uu = gm.metadata(nh).get<IEUnit>();
        IE::ICNNNetwork::InputShapes input_reshape_table = uu.params.reshape_table;

        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == (in_metas.size() - 1u)
                    && "Known input layers count doesn't match input meta count");

        // NB: Configuring input precision and network reshape must be done
        // only in the loadNetwork case.
        if (uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
            std::size_t idx = 1u;
            auto inputs = uu.net.getInputsInfo();
            for (auto &&input_name : uu.params.input_names) {
                auto ii = inputs.at(input_name);
                const auto & mm = in_metas[idx++];
                configureInputInfo(ii, mm);
                if (uu.params.layer_names_to_reshape.find(input_name) !=
                    uu.params.layer_names_to_reshape.end()) {
                    configureInputReshapeByImage(ii, mm, input_reshape_table);
                }
                ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
            }

            // FIXME: This isn't the best place to call reshape function.
            // Сorrect solution would be to do this in compile() method of network,
            // but now input meta isn't passed to compile() method.
            if (!input_reshape_table.empty()) {
                const_cast<IE::CNNNetwork *>(&uu.net)->reshape(input_reshape_table);
            }
        } else {
            GAPI_Assert(uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Import);
            std::size_t idx = 1u;
            auto inputs = uu.this_network.GetInputsInfo();
            auto* non_const_prepm = const_cast<IEUnit::PreProcMap*>(&uu.preproc_map);
            for (auto &&input_name : uu.params.input_names) {
                auto ii = inputs.at(input_name);
                const auto & mm = in_metas[idx++];
                non_const_prepm->emplace(input_name, configurePreProcInfo(ii, mm));
            }
        }

        // roi-list version is much easier at the moment.
        // All our outputs are vectors which don't have
        // metadata at the moment - so just create a vector of
        // "empty" array metadatas of the required size.
        return cv::GMetaArgs(uu.params.output_names.size(),
                             cv::GMetaArg{cv::empty_array_desc()});
    }

    static void run(std::shared_ptr<IECallContext>  ctx,
                    cv::gimpl::ie::RequestPool     &reqPool) {

        const auto& in_roi_vec = ctx->inArg<cv::detail::VectorRef>(0u).rref<cv::Rect>();
        // NB: In case there is no input data need to post output anyway
        if (in_roi_vec.empty()) {
            for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
                auto output = ctx->output(i);
                ctx->out.meta(output, ctx->input(0).meta);
                ctx->out.post(std::move(output));
            }
            return;
        }

        // NB: This blob will be used to make roi from its, so
        // it should be treated as image
        IE::Blob::Ptr this_blob = extractBlob(*ctx, 1, cv::gapi::ie::TraitAs::IMAGE);

        std::vector<std::vector<int>> cached_dims(ctx->uu.params.num_out);
        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            const auto& out_name = ctx->uu.params.output_names[i];
            const auto& desc =
                ctx->uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load
                    ? ctx->uu.net.getOutputsInfo().at(out_name)->getTensorDesc()
                    : ctx->uu.this_network.GetOutputsInfo().at(out_name)->getTensorDesc();
            cached_dims[i] = toCV(desc.getDims());
            // FIXME: Isn't this should be done automatically
            // by some resetInternalData(), etc? (Probably at the GExecutor level)
            auto& out_vec = ctx->outVecR<cv::Mat>(i);
            out_vec.clear();
            out_vec.resize(in_roi_vec.size());
        }

        PostOutputsList callback(in_roi_vec.size(), ctx, std::move(cached_dims));
        for (auto&& it : ade::util::indexed(in_roi_vec)) {
                  auto  pos = ade::util::index(it);
            const auto& rc  = ade::util::value(it);
            reqPool.execute(
                cv::gimpl::ie::RequestPool::Task {
                    [ctx, rc, this_blob](InferenceEngine::InferRequest &req) {
                        IE::Blob::Ptr roi_blob = IE::make_shared_blob(this_blob, toIE(rc));
                        setBlob(req, ctx->uu.params.input_names[0u], roi_blob, *ctx);
                        req.StartAsync();
                    },
                    std::bind(callback, std::placeholders::_1, pos)
                }
            );
        }
    }
};

struct InferList2: public cv::detail::KernelTag {
    using API = cv::GInferList2Base;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        // Specify the input information to the IE from the framework
        // NB: Have no clue if network's input [dimensions] may ever define
        // its output dimensions. It seems possible with OpenCV DNN APIs

        GConstGIEModel gm(gr);
        const auto &uu = gm.metadata(nh).get<IEUnit>();
        IE::ICNNNetwork::InputShapes input_reshape_table = uu.params.reshape_table;

        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == (in_metas.size() - 1u)
                    && "Known input layers count doesn't match input meta count");

        const auto &op = gm.metadata(nh).get<Op>();

        // In contrast to InferList, the InferList2 has only one
        // "full-frame" image argument, and all the rest are arrays of
        // ether ROI or blobs. So here we set the 0th arg image format
        // to all inputs which are ROI-based (skipping the
        // "blob"-based ones)
        // FIXME: this is filtering not done, actually! GArrayDesc has
        // no hint for its underlying type!
        const auto &mm_0 = in_metas[0u];
        switch (in_metas[0u].index()) {
            case cv::GMetaArg::index_of<cv::GMatDesc>(): {
                const auto &meta_0 = util::get<cv::GMatDesc>(mm_0);
                GAPI_Assert(   !meta_0.isND()
                        && !meta_0.planar
                        && "Only images are supported as the 0th argument");
                break;
            }
            case cv::GMetaArg::index_of<cv::GFrameDesc>(): {
                // FIXME: Is there any validation for GFrame ?
                break;
            }
            default:
                util::throw_error(std::runtime_error("Unsupported input meta for IE backend"));
        }

        if (util::holds_alternative<cv::GMatDesc>(mm_0)) {
            const auto &meta_0 = util::get<cv::GMatDesc>(mm_0);
            GAPI_Assert(   !meta_0.isND()
                    && !meta_0.planar
                    && "Only images are supported as the 0th argument");
        }

        std::size_t idx = 1u;
        for (auto &&input_name : uu.params.input_names) {
            const auto &mm = in_metas[idx];
            GAPI_Assert(util::holds_alternative<cv::GArrayDesc>(mm)
                        && "Non-array inputs are not supported");

            if (op.k.inKinds[idx] == cv::detail::OpaqueKind::CV_RECT) {
                // NB: Configuring input precision and network reshape must be done
                // only in the loadNetwork case.
                if (uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load) {
                    // This is a cv::Rect -- configure the IE preprocessing
                    auto ii = uu.net.getInputsInfo().at(input_name);
                    configureInputInfo(ii, mm_0);
                    if (uu.params.layer_names_to_reshape.find(input_name) !=
                        uu.params.layer_names_to_reshape.end()) {
                        configureInputReshapeByImage(ii, mm_0, input_reshape_table);
                    }
                    ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);

                    // FIXME: This isn't the best place to call reshape function.
                    // Сorrect solution would be to do this in compile() method of network,
                    // but now input meta isn't passed to compile() method.
                    if (!input_reshape_table.empty()) {
                        const_cast<IE::CNNNetwork *>(&uu.net)->reshape(input_reshape_table);
                    }
                } else {
                    GAPI_Assert(uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Import);
                    auto inputs = uu.this_network.GetInputsInfo();
                    auto* non_const_prepm = const_cast<IEUnit::PreProcMap*>(&uu.preproc_map);
                    auto ii = inputs.at(input_name);
                    non_const_prepm->emplace(input_name, configurePreProcInfo(ii, mm_0));
                }
            } else {
                // This is a cv::GMat (equals to: cv::Mat)
                // Just validate that it is really the type
                // (other types are prohibited here)
                GAPI_Assert(op.k.inKinds[idx] == cv::detail::OpaqueKind::CV_MAT);
            }
            idx++; // NB: Never forget to increment the counter
        }

        // roi-list version is much easier at the moment.
        // All our outputs are vectors which don't have
        // metadata at the moment - so just create a vector of
        // "empty" array metadatas of the required size.
        return cv::GMetaArgs(uu.params.output_names.size(),
                             cv::GMetaArg{cv::empty_array_desc()});
    }

    static void run(std::shared_ptr<IECallContext> ctx,
                    cv::gimpl::ie::RequestPool    &reqPool) {
        GAPI_Assert(ctx->inArgs().size() > 1u
                && "This operation must have at least two arguments");
        // NB: This blob will be used to make roi from its, so
        // it should be treated as image
        IE::Blob::Ptr blob_0 = extractBlob(*ctx, 0, cv::gapi::ie::TraitAs::IMAGE);
        const auto list_size = ctx->inArg<cv::detail::VectorRef>(1u).size();
        if (list_size == 0u) {
            for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
                auto output = ctx->output(i);
                ctx->out.meta(output, ctx->input(0).meta);
                ctx->out.post(std::move(output));
            }
            return;
        }
        // FIXME: This could be done ONCE at graph compile stage!
        std::vector< std::vector<int> > cached_dims(ctx->uu.params.num_out);
        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            const auto& out_name = ctx->uu.params.output_names[i];
            const auto& desc =
                ctx->uu.params.kind == cv::gapi::ie::detail::ParamDesc::Kind::Load
                    ? ctx->uu.net.getOutputsInfo().at(out_name)->getTensorDesc()
                    : ctx->uu.this_network.GetOutputsInfo().at(out_name)->getTensorDesc();
            cached_dims[i] = toCV(desc.getDims());
            // FIXME: Isn't this should be done automatically
            // by some resetInternalData(), etc? (Probably at the GExecutor level)
            auto& out_vec = ctx->outVecR<cv::Mat>(i);
            out_vec.clear();
            out_vec.resize(list_size);
        }

        PostOutputsList callback(list_size, ctx, std::move(cached_dims));
        for (const auto &list_idx : ade::util::iota(list_size)) {
            reqPool.execute(
                cv::gimpl::ie::RequestPool::Task {
                    [ctx, list_idx, list_size, blob_0](InferenceEngine::InferRequest &req) {
                        for (auto in_idx : ade::util::iota(ctx->uu.params.num_in)) {
                            const auto &this_vec = ctx->inArg<cv::detail::VectorRef>(in_idx+1u);
                            GAPI_Assert(this_vec.size() == list_size);
                            IE::Blob::Ptr this_blob;
                            if (this_vec.getKind() == cv::detail::OpaqueKind::CV_RECT) {
                                const auto &vec = this_vec.rref<cv::Rect>();
                                this_blob = IE::make_shared_blob(blob_0, toIE(vec[list_idx]));
                            } else if (this_vec.getKind() == cv::detail::OpaqueKind::CV_MAT) {
                                const auto &vec = this_vec.rref<cv::Mat>();
                                const auto &mat = vec[list_idx];
                                this_blob = wrapIE(mat, cv::gapi::ie::TraitAs::TENSOR);
                            } else {
                                GAPI_Assert(false &&
                                        "Only Rect and Mat types are supported for infer list 2!");
                            }
                            setBlob(req, ctx->uu.params.input_names[in_idx], this_blob, *ctx);
                        }
                        req.StartAsync();
                    },
                    std::bind(callback, std::placeholders::_1, list_idx)
                } // task
            );
        } // for
    }
};

} // namespace ie
} // namespace gapi
} // namespace cv


// IE backend implementation of GBackend::Priv ///////////////////////
namespace {
    class GIEBackendImpl final: public cv::gapi::GBackend::Priv {
        virtual void unpackKernel(ade::Graph            &gr,
                                  const ade::NodeHandle &nh,
                                  const cv::GKernelImpl &ii) override {
            using namespace cv::gimpl;
            // FIXME: Introduce a DNNBackend interface which'd specify
            // the framework for this???
            GIEModel gm(gr);
            auto &np = gm.metadata(nh).get<NetworkParams>();
            auto &pp = cv::util::any_cast<cv::gapi::ie::detail::ParamDesc>(np.opaque);
            const auto &ki = cv::util::any_cast<KImpl>(ii.opaque);

            GModel::Graph model(gr);
            auto& op = model.metadata(nh).get<Op>();

            // NB: In case generic infer, info about in/out names is stored in operation (op.params)
            if (pp.is_generic)
            {
                auto& info      = cv::util::any_cast<cv::detail::InOutInfo>(op.params);
                pp.input_names  = info.in_names;
                pp.output_names = info.out_names;
                pp.num_in       = info.in_names.size();
                pp.num_out      = info.out_names.size();
            }

            gm.metadata(nh).set(IEUnit{pp});
            gm.metadata(nh).set(IECallable{ki.run});
            gm.metadata(nh).set(CustomMetaFunction{ki.customMetaFunc});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override {
            return EPtr{new cv::gimpl::ie::GIEExecutable(graph, nodes)};
        }

        virtual cv::gapi::GKernelPackage auxiliaryKernels() const override {
            return cv::gapi::kernels< cv::gimpl::ie::Infer
                                    , cv::gimpl::ie::InferROI
                                    , cv::gimpl::ie::InferList
                                    , cv::gimpl::ie::InferList2
                                    >();
        }

        virtual bool controlsMerge() const override {
            return true;
        }

        virtual bool allowsMerge(const cv::gimpl::GIslandModel::Graph &,
                                 const ade::NodeHandle &,
                                 const ade::NodeHandle &,
                                 const ade::NodeHandle &) const override {
            return false;
        }
    };
}

cv::gapi::GBackend cv::gapi::ie::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GIEBackendImpl>());
    return this_backend;
}

cv::Mat cv::gapi::ie::util::to_ocv(IE::Blob::Ptr blob) {
    const auto& tdesc = blob->getTensorDesc();
    return cv::Mat(toCV(tdesc.getDims()),
                   toCV(tdesc.getPrecision()),
                   blob->buffer().as<uint8_t*>());
}

std::vector<int> cv::gapi::ie::util::to_ocv(const IE::SizeVector &dims) {
    return toCV(dims);
}

IE::Blob::Ptr cv::gapi::ie::util::to_ie(const cv::Mat &blob) {
    return wrapIE(blob, cv::gapi::ie::TraitAs::IMAGE);
}

IE::Blob::Ptr cv::gapi::ie::util::to_ie(const cv::Mat &y_plane, const cv::Mat &uv_plane) {
    auto y_blob   = wrapIE(y_plane,  cv::gapi::ie::TraitAs::IMAGE);
    auto uv_blob  = wrapIE(uv_plane, cv::gapi::ie::TraitAs::IMAGE);
#if INF_ENGINE_RELEASE >= 2021010000
    return IE::make_shared_blob<IE::NV12Blob>(y_blob, uv_blob);
#else
    return IE::make_shared_blob<InferenceEngine::NV12Blob>(y_blob, uv_blob);
#endif
}

#else // HAVE_INF_ENGINE

cv::gapi::GBackend cv::gapi::ie::backend() {
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("G-API has been compiled without OpenVINO IE support"));
}
#endif // HAVE_INF_ENGINE
