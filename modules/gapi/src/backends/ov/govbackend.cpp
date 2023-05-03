// needs to be included regardless if IE is present or not
// (cv::gapi::ov::backend() is still there and is defined always)
#include "backends/ov/govbackend.hpp"
#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

#include <opencv2/gapi/gcommon.hpp>
// Include anyway - cv::gapi::ov::backend() still needs to be defined
#include <opencv2/gapi/infer/ov.hpp>

#include <openvino/openvino.hpp>

#include <ade/util/zip_range.hpp>

static ov::Core getCore() {
    static ov::Core core;
    return core;
}

static ov::element::Type toOV(int depth) {
    switch (depth) {
        case CV_8U:  return ov::element::u8;
        case CV_32S: return ov::element::i32;
        case CV_32F: return ov::element::f32;
        case CV_16F: return ov::element::f16;
        default: GAPI_Error("OV. Unsupported data type");
    }
    return ov::element::undefined;
}

static std::vector<int> toCV(const ov::Shape &shape) {
    std::vector<int> result;
    result.reserve(shape.size());
    for (auto dim : shape) {
        result.push_back(ade::util::checked_cast<int>(dim));
    }
    return result;
}

static int toCV(const ov::element::Type &type) {
    switch (type) {
        case ov::element::u8:  return CV_8U;
        case ov::element::f32: return CV_32F;
        case ov::element::i32: return CV_32S;
        case ov::element::i64: return CV_32S;
        case ov::element::f16: return CV_16F;
        default: GAPI_Error("OV. Unsupported data type");
    }
    return -1;
}

struct OVUnit {
    static const char *name() { return "OVUnit"; }

    explicit OVUnit(const cv::gapi::ov::detail::ParamDesc &_params)
        : params(_params) {
        model = getCore().read_model(params.xml_path, params.bin_path);
        GAPI_Assert(model);

        if (params.num_in == 1u && params.input_names.empty()) {
            params.input_names = { model->inputs().begin()->get_any_name() };
        }

        if (params.num_out == 1u && params.output_names.empty()) {
            params.output_names = { model->outputs().begin()->get_any_name() };
        }
    };

    cv::gimpl::ov::OVCompiled compile() {
        ov::CompiledModel compiled =
            getCore().compile_model(model, params.device);
        return {compiled.create_infer_request()};
    }

    cv::gapi::ov::detail::ParamDesc params;
    std::shared_ptr<ov::Model> model;
};

class OVCallContext
{
public:
    OVCallContext(const OVUnit                                      &  unit,
                  cv::gimpl::GIslandExecutable::IOutput             &  output,
                  const cv::GArgs                                   &  args,
                  const std::vector<cv::gimpl::RcDesc>              &  outs,
                  cv::GRunArg::Meta                                 && meta,
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

    cv::GRunArgP output (std::size_t idx);
    cv::Mat&     outMatR(std::size_t idx);

    const OVUnit                          &uu;
    cv::gimpl::GIslandExecutable::IOutput &out;

    // NB: Need to guarantee that MediaFrame::View doesn't die until request is over.
    using Views = std::vector<std::unique_ptr<cv::MediaFrame::View>>;
    Views views;

    // To store exception appeared in callback.
    std::exception_ptr eptr;

    const cv::GRunArg::Meta& getMeta() { return m_meta; };

    using req_key_t = void*;
    cv::MediaFrame* prepareKeepAliveFrameSlot(req_key_t key);
    size_t releaseKeepAliveFrame(req_key_t key);
private:
    cv::detail::VectorRef& outVecRef(std::size_t idx);

    cv::GArg packArg(const cv::GArg &arg);

    // To propagate accumulated meta from all inputs to output.
    cv::GRunArg::Meta m_meta;

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

    // keep alive preprocessed frames
    std::mutex keep_alive_frames_mutex;
    std::unordered_map<req_key_t, cv::MediaFrame> keep_alive_pp_frames;
};

OVCallContext::OVCallContext(const OVUnit                                      &  unit,
                             cv::gimpl::GIslandExecutable::IOutput             &  output,
                             const cv::GArgs                                   &  args,
                             const std::vector<cv::gimpl::RcDesc>              &  outs,
                             cv::GRunArg::Meta                                 && meta,
                             std::vector<cv::gimpl::GIslandExecutable::InObj>  && input_objs,
                             std::vector<cv::gimpl::GIslandExecutable::OutObj> && output_objs)
: uu(unit), out(output), m_meta(std::move(meta)),
  m_input_objs(std::move(input_objs)), m_output_objs(std::move(output_objs))
{
    for (auto& it : m_input_objs)  cv::gimpl::magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : m_output_objs) cv::gimpl::magazine::bindOutArg(m_res, it.first, it.second);

    m_args.reserve(args.size());
    using namespace std::placeholders;
    ade::util::transform(args,
                         std::back_inserter(m_args),
                         std::bind(&OVCallContext::packArg, this, _1));

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

const cv::GArgs& OVCallContext::inArgs() const {
    return m_args;
}

cv::GShape OVCallContext::inShape(std::size_t i) const {
    return m_in_shapes[i];
}

const cv::Mat& OVCallContext::inMat(std::size_t input) const {
    return inArg<cv::Mat>(input);
}

const cv::MediaFrame& OVCallContext::inFrame(std::size_t input) const {
    return inArg<cv::MediaFrame>(input);
}

cv::Mat& OVCallContext::outMatR(std::size_t idx) {
    return *cv::util::get<cv::Mat*>(m_results.at(idx));
}

cv::GRunArgP OVCallContext::output(std::size_t idx) {
    return m_output_objs[idx].second;
};

cv::detail::VectorRef& OVCallContext::outVecRef(std::size_t idx) {
    return cv::util::get<cv::detail::VectorRef>(m_results.at(idx));
}

cv::GArg OVCallContext::packArg(const cv::GArg &arg) {
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

struct OVCallable {
    static const char *name() { return "OVRequestCallable"; }
    using Run = std::function<void(std::shared_ptr<OVCallContext>,
                                   ov::InferRequest&)>;
    Run run;
};

struct KImpl {
    cv::gimpl::CustomMetaFunction::CM customMetaFunc;
    OVCallable::Run run;
};

using GOVModel = ade::TypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    , OVUnit
    , OVCallable
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGOVModel = ade::ConstTypedGraph
    < cv::gimpl::Protocol
    , cv::gimpl::Op
    , cv::gimpl::NetworkParams
    , cv::gimpl::CustomMetaFunction
    , OVUnit
    , OVCallable
    >;

namespace cv {
namespace gimpl {
namespace ov {

// NB: To avoid namespaces conflict
using ::ov::preprocess::PrePostProcessor;

struct Infer: public cv::detail::KernelTag {
    using API = cv::GInferBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        std::cout << "cv::gimpl::ov::Infer::outMeta" << std::endl;
        cv::GMetaArgs result;

        GConstGOVModel gm(gr);
        const auto &uu = gm.metadata(nh).get<OVUnit>();
        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == in_metas.size()
                    && "Known input layers count doesn't match input meta count");
        // NB: Configuring input/output precision and network reshape must be done
        // only in the loadNetwork case.
        using namespace cv::gapi::ie::detail;
        {
            PrePostProcessor ppp(uu.model);
            for (auto &&it : ade::util::zip(ade::util::toRange(uu.params.input_names),
                                            ade::util::toRange(in_metas))) {
                const auto &mm = std::get<1>(it);
                GAPI_Assert(cv::util::holds_alternative<cv::GMatDesc>(mm));
                const auto &matdesc = cv::util::get<cv::GMatDesc>(mm);

                const auto &input_name = std::get<0>(it);
                auto &input_info = ppp.input(input_name);
                input_info.tensor().set_element_type(toOV(matdesc.depth));
                // NB: For some reason RGB image is 2D image
                // (since channel component is not counted here).
                // Note: regular 2D vectors also fall into this category
                //
                // Need to somehow distinguish 2D tensor from image
                // in order to decide whether configure resize or not.
                //
                // Image (not tensor) isND -> false
                // a) cv::Mat(H, W, CV_8UC3) -> GMatDesc{CV_8U, 3, H, W}
                //
                // Must be tensor as well but dims == 2. isND -> false
                // b) cv::Mat({32, 32}, CV_8U) -> GMatDesc{CV_8U, 1, 32, 32}
                //
                // Tensors isND -> true:
                // c) cv::Mat({32, 32, 32}, CV_8U) -> GMatDesc{CV_8U, {32, 32, 32}}
                // d) cv::Mat({32}, CV_8U)         -> GMatDesc{CV_8U, {32}}
                //
                // 1. If matdesc is ND - definitely not image. Cases: (c) and (d)
                // 2. If ov::Model expects 4D tensor for that input and user
                // provided U8 data (matdesc.dept) - most likely this is the image case...
                //
                // Corner case - model with 4D & U8 input (don't recall any)
                if (!matdesc.isND()        &&
                    matdesc.depth == CV_8U &&
                    uu.model->input(input_name).get_shape().size() == 4u ) {
                    input_info.tensor().set_layout(::ov::Layout("NHWC"))
                                       .set_shape({1,
                                                   matdesc.size.height,
                                                   matdesc.size.width,
                                                   matdesc.chan});
                    input_info.model().set_layout(::ov::Layout("NCHW"));
                    // NB: Failing with DNN models (obsolete models ???)
                    //input_info.preprocess()
                        //.resize(::ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
                }
            }
            // FIXME: There must be separate flow to change
            // ov::Model outside outMeta method.
            const_cast<std::shared_ptr<::ov::Model>&>(uu.model) = ppp.build();
        }

        for (const auto &out_name : uu.params.output_names) {
            // NOTE: our output_names vector follows the API order
            // of this operation's outputs
            const auto &out = uu.model->output(out_name);
            cv::GMatDesc outm(toCV(out.get_element_type()),
                              toCV(out.get_shape()));
            result.emplace_back(std::move(outm));
        }
        return result;
    }

    static void run(std::shared_ptr<OVCallContext> ctx,
                    ::ov::InferRequest             &infer_request) {
        for (auto i : ade::util::iota(ctx->uu.params.num_in)) {
            const auto& input_name = ctx->uu.params.input_names[i];
            auto input_tensor = infer_request.get_tensor(input_name);

            cv::Mat mat = ctx->inMat(i);
            std::copy_n(reinterpret_cast<uint8_t*>(mat.ptr<uint8_t>()),
                        input_tensor.get_byte_size(),
                        reinterpret_cast<uint8_t*>(input_tensor.data()));
        }

        infer_request.infer();

        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            const auto& out_name = ctx->uu.params.output_names[i];
            auto out_tensor = infer_request.get_tensor(out_name);

            auto& out_mat = ctx->outMatR(i);
            std::copy_n(reinterpret_cast<uint8_t*>(out_tensor.data()),
                        out_tensor.get_byte_size(),
                        out_mat.data);

            auto output = ctx->output(i);
            ctx->out.meta(output, ctx->getMeta());
            ctx->out.post(std::move(output), ctx->eptr);
        }
    }
};

} // namespace ov
} // namespace gimpl
} // namespace cv

// IE backend implementation of GBackend::Priv ///////////////////////
namespace {
    class GOVBackendImpl final: public cv::gapi::GBackend::Priv {
        virtual void unpackKernel(ade::Graph            &gr,
                                  const ade::NodeHandle &nh,
                                  const cv::GKernelImpl &ii) override {
            using namespace cv::gimpl;
            // FIXME: Introduce a DNNBackend interface which'd specify
            // the framework for this???
            GOVModel gm(gr);
            auto &np = gm.metadata(nh).get<NetworkParams>();
            auto &pp = cv::util::any_cast<cv::gapi::ov::detail::ParamDesc>(np.opaque);
            const auto &ki = cv::util::any_cast<KImpl>(ii.opaque);

            GModel::Graph model(gr);

            gm.metadata(nh).set(OVUnit{pp});
            gm.metadata(nh).set(OVCallable{ki.run});
            gm.metadata(nh).set(CustomMetaFunction{ki.customMetaFunc});
        }

        virtual EPtr compile(const ade::Graph &graph,
                             const cv::GCompileArgs &,
                             const std::vector<ade::NodeHandle> &nodes) const override {
            return EPtr{new cv::gimpl::ov::GOVExecutable(graph, nodes)};
        }

        virtual cv::GKernelPackage auxiliaryKernels() const override {
            return cv::gapi::kernels< cv::gimpl::ov::Infer >();
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

cv::gapi::GBackend cv::gapi::ov::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GOVBackendImpl>());
    return this_backend;
}

// GOVExecutable implementation //////////////////////////////////////////////
cv::gimpl::ov::GOVExecutable::GOVExecutable(const ade::Graph &g,
                                            const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g) {
    std::cout << "cv::gimpl::ov::GOVExecutable::GOVExecutable" << std::endl;

    // FIXME: Currently this backend is capable to run a single inference node only.
    // Need to extend our island fusion with merge/not-to-merge decision making parametrization
    GConstGOVModel ovm(g);

    for (auto &nh : nodes) {
        switch (m_gm.metadata(nh).get<NodeType>().t) {
        case NodeType::OP:
            if (this_nh == nullptr) {
                this_nh = nh;
                compiled = const_cast<OVUnit&>(ovm.metadata(this_nh).get<OVUnit>()).compile();
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

void cv::gimpl::ov::GOVExecutable::run(cv::gimpl::GIslandExecutable::IInput  &in,
                                       cv::gimpl::GIslandExecutable::IOutput &out) {
    std::vector<InObj>  input_objs;
    std::vector<OutObj> output_objs;

    const auto &in_desc = in.desc();
          auto  in_msg  = in.get();

    if (cv::util::holds_alternative<cv::gimpl::EndOfStream>(in_msg))
    {
        out.post(cv::gimpl::EndOfStream{});
        return;
    }

    GAPI_Assert(cv::util::holds_alternative<cv::GRunArgs>(in_msg));
    const auto in_vector = cv::util::get<cv::GRunArgs>(in_msg);
    // NB: Collect meta from all inputs.
    cv::GRunArg::Meta stub_meta;
    for (auto &&in_arg : in_vector)
    {
        stub_meta.insert(in_arg.meta.begin(), in_arg.meta.end());
    }

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

    GConstGOVModel giem(m_g);
    const auto &uu = giem.metadata(this_nh).get<OVUnit>();
    const auto &op = m_gm.metadata(this_nh).get<Op>();

    auto ctx = std::make_shared<OVCallContext>(uu, out, op.args, op.outs,
            std::move(stub_meta), std::move(input_objs), std::move(output_objs));

    const auto &kk = giem.metadata(this_nh).get<OVCallable>();

    // (5) Run the kernel.
    try {
        kk.run(ctx, compiled.infer_request);
    } catch (...) {
        auto eptr = std::current_exception();
        for (auto i : ade::util::iota(ctx->uu.params.num_out))
        {
            auto output = ctx->output(i);
            ctx->out.meta(output, ctx->getMeta());
            ctx->out.post(std::move(output), eptr);
        }
        return;
    }
}
