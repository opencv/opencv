// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2023 Intel Corporation

#ifndef OPENCV_GAPI_INFER_IE_HPP
#define OPENCV_GAPI_INFER_IE_HPP

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>
#include <tuple> // tuple, tuple_size
#include <map>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/any.hpp>

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage
#include <opencv2/gapi/infer.hpp>   // Generic
#include <opencv2/gapi/streaming/onevpl/accel_types.hpp> // Preproc Dev & Ctx

namespace cv {
namespace gapi {
// FIXME: introduce a new sub-namespace for NN?

/**
 * @brief This namespace contains G-API OpenVINO backend functions,
 * structures, and symbols.
 */
namespace ie {

GAPI_EXPORTS cv::gapi::GBackend backend();

/**
 * Specifies how G-API and IE should trait input data
 *
 * In OpenCV, the same cv::Mat is used to represent both
 * image and tensor data. Sometimes those are hardly distinguishable,
 * so this extra parameter is used to give G-API a hint.
 *
 * This hint controls how G-API reinterprets the data when converting
 * it to IE Blob format (and which layout/etc is assigned to this data).
 */
enum class TraitAs: int
{
    TENSOR, //!< G-API traits an associated cv::Mat as a raw tensor and passes dimensions as-is
    IMAGE   //!< G-API traits an associated cv::Mat as an image so creates an "image" blob (NCHW/NHWC, etc)
};

using IEConfig = std::map<std::string, std::string>;

enum InferMode {Sync, Async};

namespace detail {

template <typename T>
using AttrMap = std::map<std::string, T>;
// NB: This type is used to hold in/out layers
// attributes such as precision, layout, shape etc.
//
// User can provide attributes either:
// 1. cv::util::monostate - No value specified explicitly.
// 2. Attr - value specified explicitly that should be broadcasted to all layers.
// 3. AttrMap[str->T] - map specifies value for particular layer.
template <typename Attr>
using LayerVariantAttr = cv::util::variant< cv::util::monostate
                                          , AttrMap<Attr>
                                          , Attr>;

struct ParamDesc {
    std::string model_path;
    std::string weights_path;
    std::string device_id;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    using ConstInput = std::pair<cv::Mat, TraitAs>;
    std::unordered_map<std::string, ConstInput> const_inputs;

    std::size_t num_in;
    std::size_t num_out;

    enum class Kind {Load, Import};
    Kind kind;
    bool is_generic;
    IEConfig config;

    std::map<std::string, std::vector<std::size_t>> reshape_table;
    std::unordered_set<std::string> layer_names_to_reshape;

    // NB: Number of asyncrhonious infer requests
    size_t nireq;

    // NB: An optional config to setup RemoteContext for IE
    cv::util::any context_config;

    // NB: batch_size can't be equal to 1 by default, because some of models
    // have 2D (Layout::NC) input and if the first dimension not equal to 1
    // net.setBatchSize(1) will overwrite it.
    cv::optional<size_t> batch_size;

    cv::optional<cv::gapi::wip::onevpl::Device> vpl_preproc_device;
    cv::optional<cv::gapi::wip::onevpl::Context> vpl_preproc_ctx;

    InferMode mode;

    using PrecisionT = int;
    using PrecisionMapT = std::unordered_map<std::string, PrecisionT>;
    // NB: This parameter can contain:
    // 1. cv::util::monostate - Don't specify precision, but use default from IR/Blob.
    // 2. PrecisionT (CV_8U, CV_32F, ...) - Specifies precision for all output layers.
    // 3. PrecisionMapT ({{"layer0", CV_32F}, {"layer1", CV_16F}} - Specifies precision for certain output layer.
    // cv::util::monostate is default value that means precision wasn't specified.
    using PrecisionVariantT = cv::util::variant<cv::util::monostate,
                                                PrecisionT,
                                                PrecisionMapT>;

    PrecisionVariantT output_precision;
    LayerVariantAttr<std::string> input_layout;
    LayerVariantAttr<std::string> output_layout;
    LayerVariantAttr<int>         interpolation;
};
} // namespace detail

// FIXME: this is probably a shared (reusable) thing
template<typename Net>
struct PortCfg {
    using In = std::array
        < std::string
        , std::tuple_size<typename Net::InArgs>::value >;
    using Out = std::array
        < std::string
        , std::tuple_size<typename Net::OutArgs>::value >;
};

/**
 * @brief This structure provides functions
 * that fill inference parameters for "OpenVINO Toolkit" model.
 */
template<typename Net> class Params {
public:
    /** @brief Class constructor.

    Constructs Params based on model information and specifies default values for other
    inference description parameters. Model is loaded and compiled using "OpenVINO Toolkit".

    @param model Path to topology IR (.xml file).
    @param weights Path to weights (.bin file).
    @param device target device to use.
    */
    Params(const std::string &model,
           const std::string &weights,
           const std::string &device)
        : desc{ model, weights, device, {}, {}, {}
              , std::tuple_size<typename Net::InArgs>::value  // num_in
              , std::tuple_size<typename Net::OutArgs>::value // num_out
              , detail::ParamDesc::Kind::Load
              , false
              , {}
              , {}
              , {}
              , 1u
              , {}
              , {}
              , {}
              , {}
              , InferMode::Async
              , {}
              , {}
              , {}
              , {} } {
    }

    /** @overload
    Use this constructor to work with pre-compiled network.
    Model is imported from a pre-compiled blob.

    @param model Path to model.
    @param device target device to use.
    */
    Params(const std::string &model,
           const std::string &device)
        : desc{ model, {}, device, {}, {}, {}
              , std::tuple_size<typename Net::InArgs>::value  // num_in
              , std::tuple_size<typename Net::OutArgs>::value // num_out
              , detail::ParamDesc::Kind::Import
              , false
              , {}
              , {}
              , {}
              , 1u
              , {}
              , {}
              , {}
              , {}
              , InferMode::Async
              , {}
              , {}
              , {}
              , {} } {
    }

    /** @brief Specifies sequence of network input layers names for inference.

    The function is used to associate cv::gapi::infer<> inputs with the model inputs.
    Number of names has to match the number of network inputs as defined in G_API_NET().
    In case a network has only single input layer, there is no need to specify name manually.

    @param layer_names std::array<std::string, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Contains names of input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputLayers(const typename PortCfg<Net>::In &layer_names) {
        desc.input_names.clear();
        desc.input_names.reserve(layer_names.size());
        std::copy(layer_names.begin(), layer_names.end(),
                  std::back_inserter(desc.input_names));
        return *this;
    }

    /** @brief Specifies sequence of network output layers names for inference.

    The function is used to associate cv::gapi::infer<> outputs with the model outputs.
    Number of names has to match the number of network outputs as defined in G_API_NET().
    In case a network has only single output layer, there is no need to specify name manually.

    @param layer_names std::array<std::string, N> where N is the number of outputs
    as defined in the @ref G_API_NET. Contains names of output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutputLayers(const typename PortCfg<Net>::Out &layer_names) {
        desc.output_names.clear();
        desc.output_names.reserve(layer_names.size());
        std::copy(layer_names.begin(), layer_names.end(),
                  std::back_inserter(desc.output_names));
        return *this;
    }

    /** @brief Specifies a constant input.

    The function is used to set a constant input. This input has to be
    a preprocessed tensor if its type is TENSOR. Need to provide name of the
    network layer which will receive provided data.

    @param layer_name Name of network layer.
    @param data cv::Mat that contains data which will be associated with network layer.
    @param hint Input type @sa cv::gapi::ie::TraitAs.
    @return reference to this parameter structure.
    */
    Params<Net>& constInput(const std::string &layer_name,
                            const cv::Mat &data,
                            TraitAs hint = TraitAs::TENSOR) {
        desc.const_inputs[layer_name] = {data, hint};
        return *this;
    }

    /** @brief Specifies OpenVINO plugin configuration.

    The function is used to set configuration for OpenVINO plugin. Some parameters
    can be different for each plugin. Please follow https://docs.openvinotoolkit.org/latest/index.html
    to check information about specific plugin.

    @param cfg Map of pairs: (config parameter name, config parameter value).
    @return reference to this parameter structure.
    */
    Params& pluginConfig(const IEConfig& cfg) {
        desc.config = cfg;
        return *this;
    }

    /** @overload
    Function with a rvalue parameter.

    @param cfg rvalue map of pairs: (config parameter name, config parameter value).
    @return reference to this parameter structure.
    */
    Params& pluginConfig(IEConfig&& cfg) {
        desc.config = std::move(cfg);
        return *this;
    }

    /** @brief Specifies configuration for RemoteContext in InferenceEngine.

    When RemoteContext is configured the backend imports the networks using the context.
    It also expects cv::MediaFrames to be actually remote, to operate with blobs via the context.

    @param ctx_cfg cv::util::any value which holds InferenceEngine::ParamMap.
    @return reference to this parameter structure.
    */
    Params& cfgContextParams(const cv::util::any& ctx_cfg) {
        desc.context_config = ctx_cfg;
        return *this;
    }

    /** @overload
    Function with an rvalue parameter.

    @param ctx_cfg cv::util::any value which holds InferenceEngine::ParamMap.
    @return reference to this parameter structure.
    */
    Params& cfgContextParams(cv::util::any&& ctx_cfg) {
        desc.context_config = std::move(ctx_cfg);
        return *this;
    }

    /** @brief Specifies number of asynchronous inference requests.

    @param nireq Number of inference asynchronous requests.
    @return reference to this parameter structure.
    */
    Params& cfgNumRequests(size_t nireq) {
        GAPI_Assert(nireq > 0 && "Number of infer requests must be greater than zero!");
        desc.nireq = nireq;
        return *this;
    }

    /** @brief Specifies new input shapes for the network inputs.

    The function is used to specify new input shapes for the network inputs.
    Follow https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1networkNetwork.html
    for additional information.

    @param reshape_table Map of pairs: name of corresponding data and its dimension.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputReshape(const std::map<std::string, std::vector<std::size_t>>& reshape_table) {
        desc.reshape_table = reshape_table;
        return *this;
    }

    /** @overload */
    Params<Net>& cfgInputReshape(std::map<std::string, std::vector<std::size_t>>&& reshape_table) {
        desc.reshape_table = std::move(reshape_table);
        return *this;
    }

    /** @overload

    @param layer_name Name of layer.
    @param layer_dims New dimensions for this layer.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputReshape(const std::string& layer_name, const std::vector<size_t>& layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    /** @overload */
    Params<Net>& cfgInputReshape(std::string&& layer_name, std::vector<size_t>&& layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    /** @overload

    @param layer_names set of names of network layers that will be used for network reshape.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputReshape(const std::unordered_set<std::string>& layer_names) {
        desc.layer_names_to_reshape = layer_names;
        return *this;
    }

    /** @overload

    @param layer_names rvalue set of the selected layers will be reshaped automatically
    its input image size.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputReshape(std::unordered_set<std::string>&& layer_names) {
        desc.layer_names_to_reshape = std::move(layer_names);
        return *this;
    }

    /** @brief Specifies the inference batch size.

    The function is used to specify inference batch size.
    Follow https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1CNNNetwork.html#a8e9d19270a48aab50cb5b1c43eecb8e9 for additional information

    @param size batch size which will be used.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgBatchSize(const size_t size) {
        desc.batch_size = cv::util::make_optional(size);
        return *this;
    }

    Params<Net>& cfgPreprocessingParams(const cv::gapi::wip::onevpl::Device &device,
                                        const cv::gapi::wip::onevpl::Context &ctx) {
        desc.vpl_preproc_device = cv::util::make_optional(device);
        desc.vpl_preproc_ctx = cv::util::make_optional(ctx);
        return *this;
    }

    /** @brief Specifies which api will be used to run inference.

    The function is used to specify mode for OpenVINO inference.
    OpenVINO has two options to run inference:
    1. Asynchronous (using StartAsync: https://docs.openvino.ai/latest/classInferenceEngine_1_1InferRequest.html#doxid-class-inference-engine-1-1-infer-request-1a405293e8423d82a5b45f642a3bef0d24)
    2. Synchronous (using Infer: https://docs.openvino.ai/latest/classInferenceEngine_1_1InferRequest.html#doxid-class-inference-engine-1-1-infer-request-1a3391ce30894abde730523e9ca9371ce8)
    By default asynchronous mode is used.

    @param mode Inference mode which will be used.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInferMode(InferMode mode) {
        desc.mode = mode;
        return *this;
    }

    /** @brief Specifies the output precision for model.

    The function is used to set an output precision for model.

    @param precision Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutputPrecision(detail::ParamDesc::PrecisionT precision) {
        desc.output_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutputPrecision(detail::ParamDesc::PrecisionMapT precision_map) {
        desc.output_precision = precision_map;
        return *this;
    }

    /** @brief Specifies the input layout for model.

    The function is used to set an input layout for model.

    @param layout Layout in string representation ("NCHW", "NHWC", etc)
    will be applied to all input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputLayout(std::string layout) {
        desc.input_layout = std::move(layout);
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding input layer
    and its layout in string representation ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInputLayout(detail::AttrMap<std::string> layout_map) {
        desc.input_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Layout in string representation ("NCHW", "NHWC", etc)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutputLayout(std::string layout) {
        desc.output_layout = std::move(layout);
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in string representation ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutputLayout(detail::AttrMap<std::string> layout_map) {
        desc.output_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies resize interpolation algorithm.
     *
    The function is used to configure resize preprocessing for input layer.

    @param interpolation Resize interpolation algorithm.
    Supported algorithms: #INTER_LINEAR, #INTER_AREA.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgResize(int interpolation) {
        desc.interpolation = interpolation;
        return *this;
    }

    /** @overload

    @param interpolation Map of pairs: name of corresponding input layer
    and its resize algorithm.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgResize(detail::AttrMap<int> interpolation) {
        desc.interpolation = std::move(interpolation);
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend()    const { return cv::gapi::ie::backend();  }
    std::string   tag()        const { return Net::tag(); }
    cv::util::any params()     const { return { desc }; }
    // END(G-API's network parametrization API)

protected:
    detail::ParamDesc desc;
};

/*
* @brief This structure provides functions for generic network type that
* fill inference parameters.
* @see struct Generic
*/
template<>
class Params<cv::gapi::Generic> {
public:
    /** @brief Class constructor.

    Constructs Params based on model information and sets default values for other
    inference description parameters. Model is loaded and compiled using OpenVINO Toolkit.

    @param tag string tag of the network for which these parameters are intended.
    @param model path to topology IR (.xml file).
    @param weights path to weights (.bin file).
    @param device target device to use.
    */
    Params(const std::string &tag,
           const std::string &model,
           const std::string &weights,
           const std::string &device)
        : desc{ model, weights, device, {}, {}, {}, 0u, 0u,
                detail::ParamDesc::Kind::Load, true, {}, {}, {}, 1u,
                {}, {}, {}, {}, InferMode::Async, {}, {}, {}, {} },
          m_tag(tag) {
    }

    /** @overload

    This constructor for pre-compiled networks. Model is imported from pre-compiled
    blob.

    @param tag string tag of the network for which these parameters are intended.
    @param model path to model.
    @param device target device to use.
    */
    Params(const std::string &tag,
           const std::string &model,
           const std::string &device)
        : desc{ model, {}, device, {}, {}, {}, 0u, 0u,
                detail::ParamDesc::Kind::Import, true, {}, {}, {}, 1u,
                {}, {}, {}, {}, InferMode::Async, {}, {}, {}, {} },
          m_tag(tag) {
    }

    /** @see ie::Params::pluginConfig. */
    Params& pluginConfig(const IEConfig& cfg) {
        desc.config = cfg;
        return *this;
    }

    /** @overload */
    Params& pluginConfig(IEConfig&& cfg) {
        desc.config = std::move(cfg);
        return *this;
    }

    /** @see ie::Params::constInput. */
    Params& constInput(const std::string &layer_name,
                       const cv::Mat &data,
                       TraitAs hint = TraitAs::TENSOR) {
        desc.const_inputs[layer_name] = {data, hint};
        return *this;
    }

    /** @see ie::Params::cfgNumRequests. */
    Params& cfgNumRequests(size_t nireq) {
        GAPI_Assert(nireq > 0 && "Number of infer requests must be greater than zero!");
        desc.nireq = nireq;
        return *this;
    }

    /** @see ie::Params::cfgInputReshape */
    Params& cfgInputReshape(const std::map<std::string, std::vector<std::size_t>>&reshape_table) {
        desc.reshape_table = reshape_table;
        return *this;
    }

    /** @overload */
    Params& cfgInputReshape(std::map<std::string, std::vector<std::size_t>> && reshape_table) {
        desc.reshape_table = std::move(reshape_table);
        return *this;
    }

    /** @overload */
    Params& cfgInputReshape(std::string && layer_name, std::vector<size_t> && layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    /** @overload */
    Params& cfgInputReshape(const std::string & layer_name, const std::vector<size_t>&layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    /** @overload */
    Params& cfgInputReshape(std::unordered_set<std::string> && layer_names) {
        desc.layer_names_to_reshape = std::move(layer_names);
        return *this;
    }

    /** @overload */
    Params& cfgInputReshape(const std::unordered_set<std::string>&layer_names) {
        desc.layer_names_to_reshape = layer_names;
        return *this;
    }

    /** @see ie::Params::cfgBatchSize */
    Params& cfgBatchSize(const size_t size) {
        desc.batch_size = cv::util::make_optional(size);
        return *this;
    }

    /** @see ie::Params::cfgInferAPI */
    Params& cfgInferMode(InferMode mode) {
        desc.mode = mode;
        return *this;
    }

    /** @see ie::Params::cfgOutputPrecision */
    Params& cfgOutputPrecision(detail::ParamDesc::PrecisionT precision) {
        desc.output_precision = precision;
        return *this;
    }

    /** @overload */
    Params&
    cfgOutputPrecision(detail::ParamDesc::PrecisionMapT precision_map) {
        desc.output_precision = precision_map;
        return *this;
    }

    /** @see ie::Params::cfgInputLayout */
    Params& cfgInputLayout(std::string layout) {
        desc.input_layout = std::move(layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgInputLayout(detail::AttrMap<std::string> layout_map) {
        desc.input_layout = std::move(layout_map);
        return *this;
    }

    /** @see ie::Params::cfgOutputLayout */
    Params& cfgOutputLayout(std::string layout) {
        desc.output_layout = std::move(layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgOutputLayout(detail::AttrMap<std::string> layout_map) {
        desc.output_layout = std::move(layout_map);
        return *this;
    }

    /** @see ie::Params::cfgResize */
    Params& cfgResize(int interpolation) {
        desc.interpolation = interpolation;
        return *this;
    }

    /** @overload */
    Params& cfgResize(detail::AttrMap<int> interpolation) {
        desc.interpolation = std::move(interpolation);
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend()    const { return cv::gapi::ie::backend();  }
    std::string   tag()        const { return m_tag; }
    cv::util::any params()     const { return { desc }; }
    // END(G-API's network parametrization API)

protected:
    detail::ParamDesc desc;
    std::string m_tag;
};

} // namespace ie
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_IE_HPP
