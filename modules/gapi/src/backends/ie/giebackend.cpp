// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "precomp.hpp"

#include <functional>
#include <unordered_set>

#include <ade/util/algorithm.hpp>

#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/util/chain_range.hpp>
#include <ade/typed_graph.hpp>

#include <inference_engine.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/garray.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/ie/giebackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

namespace IE = InferenceEngine;

namespace {

// Taken from IE samples
IE::Blob::Ptr wrapMatToBlob(const cv::Mat &mat) {
    // FIXME: This function is very ugly
    if (mat.size.dims() == 2) {
        const size_t channels = mat.channels();
        const size_t height   = mat.size().height;
        const size_t width    = mat.size().width;

        const size_t strideH  = mat.step.buf[0];
        const size_t strideW  = mat.step.buf[1];

        const bool is_dense =
                strideW == channels &&
                strideH == channels * width;

        if (!is_dense)
            cv::util::throw_error(std::logic_error("Doesn't support conversion"
                                                   " from not dense cv::Mat"));

        // FIXME: Proper format conversion!
        IE::TensorDesc tDesc(IE::Precision::U8,
                             {1, channels, height, width},
                             IE::Layout::NHWC);
        return IE::make_shared_blob<uint8_t>(tDesc, mat.data);
    }

    const auto &sz = mat.size;
    CV_Assert(sz.dims() > 2);
    IE::TensorDesc tDesc(IE::Precision::FP32, // FIXME:!!!
                         {sz[0], sz[1], sz[2], sz[3]},
                         IE::Layout::NCHW);
    // FIXME: UGLY!!!!
    return IE::make_shared_blob<float>(tDesc, reinterpret_cast<float*>(mat.data));
}

inline IE::ROI toROI(const cv::Rect &rc) {
    return IE::ROI
        { 0u
        , static_cast<std::size_t>(rc.x)
        , static_cast<std::size_t>(rc.y)
        , static_cast<std::size_t>(rc.width)
        , static_cast<std::size_t>(rc.height)
        };
}

// IE-specific metadata, represents a network with its parameters
struct IEUnit {
    static const char *name() { return "IEModelConfig"; }

    cv::gapi::ie::detail::ParamDesc params;
    IE::CNNNetwork net;
    IE::InputsDataMap inputs;
    IE::OutputsDataMap outputs;

    explicit IEUnit(const cv::gapi::ie::detail::ParamDesc &pp)
        : params(pp) {

        IE::CNNNetReader reader;
        reader.ReadNetwork(params.model_path);
        reader.ReadWeights(params.weights_path);
        net = reader.getNetwork();
        inputs = net.getInputsInfo();
        outputs = net.getOutputsInfo();

        // The practice shows that not all inputs and not all outputs
        // are mandatory to specify in IE model.
        // So what we're concerned here about is:
        // if opeation's (not topology's) input/output number is
        // greater than 1, then we do care about input/output layer
        // names. Otherwise, names are picked up automatically.
        // FIXE: Probably this check could be done at the API entry point? (gnet)
        if (params.num_in > 1u && params.num_in != params.input_names.size()) {
            cv::util::throw_error(std::logic_error("Please specify input layer names for "
                                                   + params.model_path));
        }
        if (params.num_out > 1u && params.num_out != params.output_names.size()) {
            cv::util::throw_error(std::logic_error("Please specify output layer names for "
                                                   + params.model_path));
        }
        if (params.num_in == 1u && params.input_names.empty()) {
            params.input_names = { inputs.begin()->first };
        }
        if (params.num_out == 1u && params.output_names.empty()) {
            params.output_names = { outputs.begin()->first };
        }

        // Initialize input information
        // FIXME: This should happen at the metadata configuration stage!
        // given the real data
        for (auto &&in: params.input_names) {
            auto &&ii = inputs.at(in);
            // FIXME: Currently hardcoded for the given sample
            ii->getPreProcess().setResizeAlgorithm(IE::RESIZE_BILINEAR);
            ii->setLayout(IE::Layout::NCHW);
            ii->setPrecision(IE::Precision::U8);
        }
    }

    cv::gimpl::ie::IECompiled compile() const {
        auto this_plugin = IE::PluginDispatcher().getPluginByDevice(params.device_id);
        auto this_network = this_plugin.LoadNetwork(net, {});
        auto this_request = this_network.CreateInferRequest();

        // Bind const data to infer request
        for (auto &&p : params.const_inputs) {
            this_request.SetBlob(p.first, wrapMatToBlob(p.second));
        }

        return {this_plugin, this_network, this_request};
    }
};

struct IECallContext
{
    // Input parameters passed to an inference operation.
    std::vector<cv::GArg> args;

    //FIXME: avoid conversion of arguments from internal representaion to OpenCV one on each call
    //to OCV kernel. (This can be achieved by a two single time conversions in GCPUExecutable::run,
    //once on enter for input and output arguments, and once before return for output arguments only
    //FIXME: check if the above applies to this backend (taken from CPU)
    std::unordered_map<std::size_t, cv::GRunArgP> results;

    // Generic accessor API
    template<typename T>
    const T& inArg(int input) { return args.at(input).get<T>(); }

    // Syntax sugar
    const cv::gapi::own::Mat&   inMat(int input) {
        return inArg<cv::gapi::own::Mat>(input);
    }
    cv::gapi::own::Mat&         outMatR(int output) {
        return *cv::util::get<cv::gapi::own::Mat*>(results.at(output));
    }

    template<typename T> std::vector<T>& outVecR(int output) { // FIXME: the same issue
        return outVecRef(output).wref<T>();
    }
    cv::detail::VectorRef& outVecRef(int output) {
        return cv::util::get<cv::detail::VectorRef>(results.at(output));
    }
};

struct IECallable {
    static const char *name() { return "IERequestCallable"; }
    // FIXME: Make IECallContext manage them all? (3->1)
    using Run = std::function<void(cv::gimpl::ie::IECompiled &, const IEUnit &, IECallContext &)>;
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
} // anonymous namespace

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
            }
            else
                util::throw_error(std::logic_error("Multi-node inference is not supported!"));
            break;

        case NodeType::DATA: {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST) {
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

// FIXME: Document what it does
cv::GArg cv::gimpl::ie::GIEExecutable::packArg(const cv::GArg &arg) {
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != cv::detail::ArgKind::GMAT
                && arg.kind != cv::detail::ArgKind::GSCALAR
                && arg.kind != cv::detail::ArgKind::GARRAY);

    if (arg.kind != cv::detail::ArgKind::GOBJREF) {
        util::throw_error(std::logic_error("Inference supports G-types ONLY!"));
    }
    GAPI_Assert(arg.kind == cv::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const cv::gimpl::RcDesc &ref = arg.get<cv::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:    return GArg(m_res.slot<cv::gapi::own::Mat>()[ref.id]);

    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:  return GArg(m_res.slot<cv::detail::VectorRef>().at(ref.id));

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

void cv::gimpl::ie::GIEExecutable::run(std::vector<InObj>  &&input_objs,
                                       std::vector<OutObj> &&output_objs) {
    // Update resources with run-time information - what this Island
    // has received from user (or from another Island, or mix...)
    // FIXME: Check input/output objects against GIsland protocol

    for (auto& it : input_objs)   magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : output_objs)  magazine::bindOutArg(m_res, it.first, it.second);

    // FIXME: Running just a single node now.
    // Not sure if need to support many of them, though
    // FIXME: Make this island-unmergeable?
    const auto &op = m_gm.metadata(this_nh).get<Op>();

    // Initialize kernel's execution context:
    // - Input parameters
    IECallContext context;
    context.args.reserve(op.args.size());
    using namespace std::placeholders;
    ade::util::transform(op.args,
                          std::back_inserter(context.args),
                          std::bind(&GIEExecutable::packArg, this, _1));

    // - Output parameters.
    for (const auto &out_it : ade::util::indexed(op.outs)) {
        // FIXME: Can the same GArg type resolution mechanism be reused here?
        const auto out_port  = ade::util::index(out_it);
        const auto out_desc  = ade::util::value(out_it);
        context.results[out_port] = magazine::getObjPtr(m_res, out_desc);
    }

    // And now trigger the execution
    GConstGIEModel giem(m_g);
    const auto &uu = giem.metadata(this_nh).get<IEUnit>();
    const auto &kk = giem.metadata(this_nh).get<IECallable>();
    kk.run(this_iec, uu, context);

    for (auto &it : output_objs) magazine::writeBack(m_res, it.first, it.second);
}

namespace cv {
namespace gimpl {
namespace ie {

struct Infer: public cv::detail::KernelTag {
    using API = cv::GInferBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &/*in_metas*/,
                                 const cv::GArgs       &/*in_args*/) {
        // Specify network's output layer metadata to the framework
        // NB: Have no clue if network's input [dimensions] may ever define
        // its output dimensions

        cv::GMetaArgs result;

        GConstGIEModel gm(gr);
        const auto &uu = gm.metadata(nh).get<IEUnit>();

        // FIXME: It would be nice here to have an exact number of network's
        // input/output parameters. Probably GCall should store it here for us.
        // It doesn't, as far as I know..
        for (const auto &out_name : uu.params.output_names) {
            // NOTE: our output_names vector follows the API order
            // of this operation's outputs
            const IE::DataPtr& ie_out = uu.outputs.at(out_name);
            const IE::SizeVector dims = ie_out->getTensorDesc().getDims();

            cv::GMatDesc this_meta;
            this_meta.depth = CV_32F; // FIXME: how we figure this out?
            this_meta.chan = -1;
            this_meta.size = cv::gapi::own::Size{-1,-1};
            this_meta.dims.clear();
            this_meta.dims.reserve(dims.size());
            for (auto d : dims) { this_meta.dims.push_back(static_cast<int>(d)); }
            result.emplace_back(this_meta);
        }
        return result;
    }

    static void run(IECompiled &iec, const IEUnit &uu, IECallContext &ctx) {
        // non-generic version for now:
        // - assumes all inputs/outputs are always Mats
        for (auto i : ade::util::iota(uu.params.num_in)) {
            const cv::Mat this_mat = to_ocv(ctx.inMat(i));
            IE::Blob::Ptr this_blob = wrapMatToBlob(this_mat);

            // FIXME: Ideally we shouldn't do SetBlob() but GetBlob() instead,
            // and redirect our data producers to this memory
            // (A memory dialog comes to the picture again)
            iec.this_request.SetBlob(uu.params.input_names[i], this_blob);
        }
        iec.this_request.Infer();
        for (auto i : ade::util::iota(uu.params.num_out)) {
            cv::gapi::own::Mat& out_mat = ctx.outMatR(i);
            CV_Assert(out_mat.type() == CV_32F);

            IE::Blob::Ptr this_blob = iec.this_request.GetBlob(uu.params.output_names[i]);
            std::copy_n(this_blob->buffer().as<float*>(),   // FIXME: type
                        out_mat.total(),
                        reinterpret_cast<float*>(out_mat.data)); // FIXME: type(!!!)
        }
    }
};

struct InferList: public cv::detail::KernelTag {
    using API = cv::GInferListBase;
    static cv::gapi::GBackend backend()  { return cv::gapi::ie::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static cv::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const cv::GMetaArgs   &/*in_metas*/,
                                 const cv::GArgs       &/*in_args*/) {
        // roi-list version is much easier at the moment.
        cv::GMetaArgs result;
        GConstGIEModel gm(gr);
        const auto &uu = gm.metadata(nh).get<IEUnit>();
        for (const auto &out_name : uu.params.output_names) {
            result.emplace_back(cv::empty_array_desc());
        }
        return result;
    }

    static void run(IECompiled &iec, const IEUnit &uu, IECallContext &ctx) {
        // non-generic version for now:
        // - assumes zero input is always ROI list
        // - assumes all inputs/outputs are always Mats
        CV_Assert(uu.params.num_in == 1); // roi list is not counted in net's inputs

        const auto& in_roi_vec = ctx.inArg<cv::detail::VectorRef>(0u).rref<cv::Rect>();
        const cv::Mat this_mat = to_ocv(ctx.inMat(1u));
        IE::Blob::Ptr this_blob = wrapMatToBlob(this_mat);

        // FIXME: This could be done ONCE at graph compile stage!
        std::vector< std::vector<int> > cached_dims(uu.params.num_out);
        for (auto i : ade::util::iota(uu.params.num_out)) {
            const IE::DataPtr& ie_out = uu.outputs.at(uu.params.output_names[i]);
            const IE::SizeVector ie_dims = ie_out->getTensorDesc().getDims();
            for (auto d : ie_dims) { cached_dims[i].push_back(static_cast<int>(d)); }
            ctx.outVecR<cv::Mat>(i).clear();
            // FIXME: Isn't this should be done automatically
            // by some resetInternalData(), etc? (Probably at the GExecutor level)
        }

        for (const auto &rc : in_roi_vec) {
            // FIXME: Assumed only 1 input
            IE::Blob::Ptr roi_blob = IE::make_shared_blob(this_blob, toROI(rc));
            iec.this_request.SetBlob(uu.params.input_names[0u], roi_blob);
            iec.this_request.Infer();

            // While input is fixed to be 1,
            // there may be still multiple outputs
            for (auto i : ade::util::iota(uu.params.num_out)) {
                std::vector<cv::Mat> &out_vec = ctx.outVecR<cv::Mat>(i);

                IE::Blob::Ptr out_blob = iec.this_request.GetBlob(uu.params.output_names[i]);

                cv::Mat out_mat(cached_dims[i],
                                CV_32F,                            // FIXME: type
                                out_blob->buffer().as<float*>()); // FIXME: type(!!!)
                // FIXME: Avoid data copy.
                // Not sure if this could be avoided, though
                out_vec.push_back(out_mat.clone());
            }
        }
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
            const auto &np = gm.metadata(nh).get<NetworkParams>();
            const auto &pp = cv::util::any_cast<cv::gapi::ie::detail::ParamDesc>(np.opaque);
            const auto &ki = cv::util::any_cast<KImpl>(ii.opaque);
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
                                    , cv::gimpl::ie::InferList
                                    >();
        }
    };
}

cv::gapi::GBackend cv::gapi::ie::backend() {
    static cv::gapi::GBackend this_backend(std::make_shared<GIEBackendImpl>());
    return this_backend;
}
