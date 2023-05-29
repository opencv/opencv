// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "precomp.hpp"

// needs to be included regardless if IE is present or not
// (cv::gapi::ov::backend() is still there and is defined always)
#include "backends/ov/govbackend.hpp"

#ifdef HAVE_INF_ENGINE

#include "backends/ov/util.hpp"
#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/infer/ov.hpp>

#include <ade/util/zip_range.hpp>

#include <openvino/openvino.hpp>

#include <fstream>

using ParamDesc = cv::gapi::ov::detail::ParamDesc;
using ParamsM   = ParamDesc::Model;
using ParamsCM  = ParamDesc::CompiledModel;

static ov::Core getCore() {
    static ov::Core core;
    return core;
}

static ov::AnyMap toOV(const ParamDesc::PluginConfigT &config) {
    return {config.begin(), config.end()};
}

static std::map<std::string, ::ov::PartialShape>
toOV(const std::map<std::string, std::vector<size_t>> &shapes) {
    std::map<std::string, ::ov::PartialShape> ov_shapes;
    for (const auto &it : shapes) {
        ov_shapes.emplace(it.first, ::ov::Shape(it.second));
    }
    return ov_shapes;
}

static ov::element::Type toOV(int depth) {
    switch (depth) {
        case CV_8U:  return ov::element::u8;
        case CV_32S: return ov::element::i32;
        case CV_32F: return ov::element::f32;
        case CV_16F: return ov::element::f16;
        default: GAPI_Error("OV Backend: Unsupported data type");
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
        default: GAPI_Error("OV Backend: Unsupported data type");
    }
    return -1;
}

static void copyFromOV(const ov::Tensor &tensor, cv::Mat &mat) {
    if (tensor.get_byte_size() != mat.total() * mat.elemSize()) {
        cv::util::throw_error(std::logic_error
            ("OV Backend: Failed to copy data from ov::Tensor to cv::Mat! "
             "Number of bytes mismatch. "
             "(cv::Mat " + std::to_string(mat.total() * mat.elemSize()) + ", "
             "ov::Tensor: " + std::to_string(tensor.get_byte_size()) + ")"));
    }
    std::copy_n(reinterpret_cast<uint8_t*>(tensor.data()),
                tensor.get_byte_size(),
                mat.ptr<uint8_t>());
}

static void copyToOV(const cv::Mat &mat, ov::Tensor &tensor) {
    if (tensor.get_byte_size() != mat.total() * mat.elemSize()) {
        cv::util::throw_error(std::logic_error
            ("OV Backend: Failed to copy data from cv::Mat to ov::Tensor! "
             "Number of bytes mismatch. "
             "(cv::Mat " + std::to_string(mat.total() * mat.elemSize()) + ", "
             "ov::Tensor: " + std::to_string(tensor.get_byte_size()) + ")"));
    }
    std::copy_n(mat.ptr<uint8_t>(),
                tensor.get_byte_size(),
                reinterpret_cast<uint8_t*>(tensor.data()));
}

std::vector<int> cv::gapi::ov::util::to_ocv(const ::ov::Shape &shape) {
    return toCV(shape);
}

int cv::gapi::ov::util::to_ocv(const ::ov::element::Type &type) {
    return toCV(type);
}

struct OVUnit {
    static const char *name() { return "OVUnit"; }

    explicit OVUnit(const ParamDesc &pd)
        : params(pd) {

        if (cv::util::holds_alternative<ParamDesc::Model>(params.kind)) {
            const auto desc = cv::util::get<ParamDesc::Model>(params.kind);
            model = getCore().read_model(desc.model_path, desc.bin_path);
            GAPI_Assert(model);

            if (params.num_in == 1u && params.input_names.empty()) {
                params.input_names = { model->inputs().begin()->get_any_name() };
            }
            if (params.num_out == 1u && params.output_names.empty()) {
                params.output_names = { model->outputs().begin()->get_any_name() };
            }

        } else {
            GAPI_Assert(cv::util::holds_alternative<ParamDesc::CompiledModel>(params.kind));
            std::ifstream file(cv::util::get<ParamDesc::CompiledModel>(params.kind).blob_path,
                               std::ios_base::in | std::ios_base::binary);
            GAPI_Assert(file.is_open());
            compiled_model = getCore().import_model(file,
                                                    params.device,
                                                    toOV(params.config));

            if (params.num_in == 1u && params.input_names.empty()) {
                params.input_names = { compiled_model.inputs().begin()->get_any_name() };
            }
            if (params.num_out == 1u && params.output_names.empty()) {
                params.output_names = { compiled_model.outputs().begin()->get_any_name() };
            }
        }
    };

    cv::gimpl::ov::OVCompiled compile() {
        if (cv::util::holds_alternative<ParamDesc::Model>(params.kind)) {
            compiled_model = getCore().compile_model(model,
                                                     params.device,
                                                     toOV(params.config));
        }
        return {compiled_model.create_infer_request()};
    }

    cv::gapi::ov::detail::ParamDesc params;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
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

    // To store exception appeared in callback.
    std::exception_ptr eptr;

    const cv::GRunArg::Meta& getMeta() { return m_meta; };
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

template <typename Attr> ParamsM::AttrMap<Attr>
broadcastLayerAttr(const ParamsM::LayerVariantAttr<Attr> &layer_attr,
                   const std::vector<std::string>        &layer_names) {
    ParamsM::AttrMap<Attr> map;
    if (cv::util::holds_alternative<ParamsM::AttrMap<Attr>>(layer_attr)) {
        map = cv::util::get<ParamsM::AttrMap<Attr>>(layer_attr);
        // NB: Validate map:
        std::unordered_set<std::string> existing_layers =
            {layer_names.begin(), layer_names.end()};
        for (const auto &p : map) {
            const auto it = existing_layers.find(p.first);
            if (it == existing_layers.end()) {
                cv::util::throw_error(
                        std::logic_error("OV Backend: Failed to"
                                         " find layer with name: " + p.first));
            }
        }
    } else if (cv::util::holds_alternative<Attr>(layer_attr)) {
        // NB: Broadcast value to all layers.
        auto elem = cv::util::get<Attr>(layer_attr);
        for (auto &&layer_name : layer_names) {
            map.emplace(layer_name, elem);
        }
    }
    return map;
}

template <typename K, typename V>
cv::optional<V> lookUp(const std::map<K, V> &map, const K& key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        return {};
    }
    return cv::util::make_optional(std::move(it->second));
}

static bool isImage(const cv::GMatDesc &desc,
                    const ::ov::Shape  &input_shape) {
    return (input_shape.size() == 4u)                      &&
           (!desc.isND())  /* dims == 2 */                 &&
           (desc.chan == 1 || desc.chan == 3)              &&
           (desc.size.height != 1 && desc.size.width != 1) &&
           (desc.depth == CV_8U);
}

struct Infer: public cv::detail::KernelTag {
    using API = cv::GInferBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ov::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &in_metas,
                                 const cv::GArgs       &/*in_args*/) {
        cv::GMetaArgs result;

        GConstGOVModel gm(gr);
        const auto &uu = gm.metadata(nh).get<OVUnit>();
        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == in_metas.size()
                    && "Known input layers count doesn't match input meta count");

        // NB: Pre/Post processing configuration avaiable only for read models.
        if (cv::util::holds_alternative<ParamsM>(uu.params.kind)) {
            const auto &model_info = cv::util::get<ParamsM>(uu.params.kind);

            const auto new_shapes =
                broadcastLayerAttr(model_info.new_shapes,
                                   uu.params.input_names);
            const_cast<std::shared_ptr<::ov::Model>&>(uu.model)->reshape(toOV(new_shapes));

            const auto input_tensor_layout =
                broadcastLayerAttr(model_info.input_tensor_layout,
                                   uu.params.input_names);
            const auto input_model_layout =
                broadcastLayerAttr(model_info.input_model_layout,
                                   uu.params.input_names);

            ::ov::preprocess::PrePostProcessor ppp(uu.model);
            for (auto &&it : ade::util::zip(ade::util::toRange(uu.params.input_names),
                                            ade::util::toRange(in_metas))) {
                const auto &mm = std::get<1>(it);
                GAPI_Assert(cv::util::holds_alternative<cv::GMatDesc>(mm));
                const auto &matdesc = cv::util::get<cv::GMatDesc>(mm);

                const auto &input_name = std::get<0>(it);
                auto &input_info = ppp.input(input_name);
                input_info.tensor().set_element_type(toOV(matdesc.depth));

                const auto explicit_in_model_layout = lookUp(input_model_layout , input_name);
                if (explicit_in_model_layout) {
                    input_info.model().set_layout(::ov::Layout(*explicit_in_model_layout));
                }

                const auto explicit_in_tensor_layout = lookUp(input_tensor_layout, input_name);
                if (explicit_in_tensor_layout) {
                    input_info.model().set_layout(::ov::Layout(*explicit_in_tensor_layout));
                }

                // NB: Image case - all necessary preprocessng:
                // layout conversion, resize is configured automatically.
                // FIXME: What if user provided tensor layout that isn't equal to NHWC?
                // Current solution just disable all automatic preprocessing...
                const auto &input_shape = uu.model->input(input_name).get_shape();
                if (isImage(matdesc, input_shape) &&
                    ((!explicit_in_tensor_layout) ||
                     (explicit_in_tensor_layout && (*explicit_in_tensor_layout == "NHWC")))) {
                    // NB: If user didn't provide input model layout explicitly set it to
                    // "NCHW" in order to perform conversion to image layout (NHWC).
                    if (!explicit_in_model_layout) {
                        input_info.model().set_layout(::ov::Layout("NCHW"));
                    }
                    // Image has NHWC layout.
                    input_info.tensor().set_layout(::ov::Layout("NHWC"))
                                       .set_shape({1,
                                                   matdesc.size.height,
                                                   matdesc.size.width,
                                                   matdesc.chan});
                    // NB: Configure resize since image may has
                    // different size than model expects.
                    input_info.preprocess()
                        .resize(::ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
                }
            }

            const auto output_tensor_layout =
                broadcastLayerAttr(model_info.output_tensor_layout,
                                   uu.params.output_names);
            const auto output_model_layout =
                broadcastLayerAttr(model_info.output_model_layout,
                                   uu.params.output_names);
            const auto output_tensor_precision =
                broadcastLayerAttr(model_info.output_tensor_precision,
                                   uu.params.output_names);

            for (const auto &output_name : uu.params.output_names) {
                const auto explicit_out_tensor_layout =
                    lookUp(output_tensor_layout, output_name);
                if (explicit_out_tensor_layout) {
                    ppp.output(output_name).tensor()
                        .set_layout(::ov::Layout(*explicit_out_tensor_layout));
                }

                const auto explicit_out_model_layout =
                    lookUp(output_model_layout, output_name);
                if (explicit_out_model_layout) {
                    ppp.output(output_name).model()
                        .set_layout(::ov::Layout(*explicit_out_model_layout));
                }

                const auto explicit_out_tensor_prec =
                    lookUp(output_tensor_precision, output_name);
                if (explicit_out_tensor_prec) {
                    ppp.output(output_name).tensor()
                        .set_element_type(toOV(*explicit_out_tensor_prec));
                }
            }
            // FIXME: outMeta is the only method now
            // where pre/post processing can be configured.
            const_cast<std::shared_ptr<::ov::Model>&>(uu.model) = ppp.build();
        }

        for (const auto &out_name : uu.params.output_names) {
            cv::GMatDesc outm;
            if (cv::util::holds_alternative<ParamsM>(uu.params.kind)) {
                const auto &out = uu.model->output(out_name);
                outm = cv::GMatDesc(toCV(out.get_element_type()),
                                    toCV(out.get_shape()));
            } else {
                GAPI_Assert(cv::util::holds_alternative<ParamsCM>(uu.params.kind));
                const auto &out = uu.compiled_model.output(out_name);
                outm = cv::GMatDesc(toCV(out.get_element_type()),
                                    toCV(out.get_shape()));
            }
            result.emplace_back(std::move(outm));
        }

        return result;
    }

    static void run(std::shared_ptr<OVCallContext> ctx,
                    ::ov::InferRequest             &infer_request) {
        for (auto i : ade::util::iota(ctx->uu.params.num_in)) {
            const auto& input_name = ctx->uu.params.input_names[i];
            auto input_tensor = infer_request.get_tensor(input_name);
            copyToOV(ctx->inMat(i), input_tensor);
        }

        infer_request.infer();

        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            const auto& out_name = ctx->uu.params.output_names[i];
            copyFromOV(infer_request.get_tensor(out_name),
                       ctx->outMatR(i));
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
            auto &pp = cv::util::any_cast<ParamDesc>(np.opaque);
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
    cv::GRunArg::Meta stub_meta;
    for (auto &&in_arg : in_vector)
    {
        stub_meta.insert(in_arg.meta.begin(), in_arg.meta.end());
    }

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

#else // HAVE_INF_ENGINE

cv::gapi::GBackend cv::gapi::ov::backend() {
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("G-API has been compiled without OpenVINO support"));
}

#endif // HAVE_INF_ENGINE
