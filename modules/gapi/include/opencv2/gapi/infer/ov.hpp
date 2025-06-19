// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_INFER_OV_HPP
#define OPENCV_GAPI_INFER_OV_HPP

#include <string>

#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp>     // GKernelType[M], GBackend
#include <opencv2/gapi/infer.hpp>       // Generic

#include <map>

namespace cv {
namespace gapi {

/**
 * @brief This namespace contains G-API OpenVINO 2.0 backend functions,
 * structures, and symbols.
 */
namespace ov {

GAPI_EXPORTS cv::gapi::GBackend backend();

namespace detail {

template <typename T>
using AttrMap = std::map<std::string, T>;
// NB: This type is supposed to be used to hold in/out layers
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
    struct Model {

        Model(const std::string &model_path_,
              const std::string &bin_path_)
            : model_path(model_path_), bin_path(bin_path_) {
        }

        std::string model_path;
        std::string bin_path;

        LayerVariantAttr<std::string> input_tensor_layout;
        LayerVariantAttr<std::string> input_model_layout;
        LayerVariantAttr<std::string> output_tensor_layout;
        LayerVariantAttr<std::string> output_model_layout;
        LayerVariantAttr<int>         output_tensor_precision;

        LayerVariantAttr<std::vector<size_t>> new_shapes;

        LayerVariantAttr<std::vector<float>> mean_values;
        LayerVariantAttr<std::vector<float>> scale_values;

        LayerVariantAttr<int> interpolation;
    };

    struct CompiledModel {
        std::string blob_path;
    };

    using Kind = cv::util::variant<Model, CompiledModel>;

    ParamDesc(Kind              &&kind_,
              const std::string &device_,
              const bool        is_generic_,
              const size_t      num_in_,
              const size_t      num_out_)
        : kind(std::move(kind_)), device(device_),
          is_generic(is_generic_),
          num_in(num_in_), num_out(num_out_) {
    }

    Kind kind;

    std::string device;
    bool is_generic;

    std::size_t num_in;
    std::size_t num_out;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    using PluginConfigT = std::map<std::string, std::string>;
    PluginConfigT config;

    size_t nireq = 1;
};

// NB: Just helper to avoid code duplication.
static detail::ParamDesc::Model&
getModelToSetAttrOrThrow(detail::ParamDesc::Kind  &kind,
                         const std::string        &attr_name) {
    if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(kind)) {
        cv::util::throw_error(
                std::logic_error("Specifying " + attr_name + " isn't"
                                 " possible for compiled model."));
    }
    GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(kind));
    return cv::util::get<detail::ParamDesc::Model>(kind);
}

} // namespace detail

/**
 * @brief This structure provides functions
 * that fill inference parameters for "OpenVINO Toolkit" model.
 */
template<typename Net> struct Params {
public:
    /** @brief Class constructor.

    Constructs Params based on model information and specifies default values for other
    inference description parameters. Model is loaded and compiled using "OpenVINO Toolkit".

    @param model_path Path to a model.
    @param bin_path Path to a data file.
    For IR format (*.bin):
    If path is empty, will try to read a bin file with the same name as xml.
    If the bin file with the same name is not found, will load IR without weights.
    For PDPD (*.pdmodel) and ONNX (*.onnx) formats bin_path isn't used.
    @param device target device to use.
    */
    Params(const std::string &model_path,
           const std::string &bin_path,
           const std::string &device)
        : m_desc( detail::ParamDesc::Kind{detail::ParamDesc::Model{model_path, bin_path}}
                 , device
                 , false /* is generic */
                 , std::tuple_size<typename Net::InArgs>::value
                 , std::tuple_size<typename Net::OutArgs>::value) {
    }

    /** @overload
    Use this constructor to work with pre-compiled network.
    Model is imported from a pre-compiled blob.

    @param blob_path path to the compiled model (*.blob).
    @param device target device to use.
    */
    Params(const std::string &blob_path,
           const std::string &device)
        : m_desc( detail::ParamDesc::Kind{detail::ParamDesc::CompiledModel{blob_path}}
                 , device
                 , false /* is generic */
                 , std::tuple_size<typename Net::InArgs>::value
                 , std::tuple_size<typename Net::OutArgs>::value) {
    }

    /** @brief Specifies sequence of network input layers names for inference.

    The function is used to associate cv::gapi::infer<> inputs with the model inputs.
    Number of names has to match the number of network inputs as defined in G_API_NET().
    In case a network has only single input layer, there is no need to specify name manually.

    @param layer_names std::array<std::string, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Contains names of input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputLayers(const std::vector<std::string> &layer_names) {
        m_desc.input_names = layer_names;
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
    Params<Net>& cfgOutputLayers(const std::vector<std::string> &layer_names) {
        m_desc.output_names = layer_names;
        return *this;
    }

    /** @brief Specifies OpenVINO plugin configuration.

    The function is used to set configuration for OpenVINO plugin. Some parameters
    can be different for each plugin. Please follow https://docs.openvinotoolkit.org/latest/index.html
    to check information about specific plugin.

    @param config Map of pairs: (config parameter name, config parameter value).
    @return reference to this parameter structure.
    */
    Params<Net>& cfgPluginConfig(const detail::ParamDesc::PluginConfigT &config) {
        m_desc.config = config;
        return *this;
    }

    /** @brief Specifies tensor layout for an input layer.

    The function is used to set tensor layout for an input layer.

    @param layout Tensor layout ("NCHW", "NWHC", etc)
    will be applied to all input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputTensorLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input tensor layout")
            .input_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload
    @param layout_map Map of pairs: name of corresponding input layer
    and its tensor layout represented in std::string ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInputTensorLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input tensor layout")
            .input_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies model layout for an input layer.

    The function is used to set model layout for an input layer.

    @param layout Model layout ("NCHW", "NHWC", etc)
    will be applied to all input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInputModelLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input model layout")
            .input_model_layout = std::move(layout);
        return *this;
    }

    /** @overload
    @param layout_map Map of pairs: name of corresponding input layer
    and its model layout ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInputModelLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input model layout")
            .input_model_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies tensor layout for an output layer.

    The function is used to set tensor layout for an output layer.

    @param layout Tensor layout ("NCHW", "NWHC", etc)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutputTensorLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor layout")
            .output_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload
    @param layout_map Map of pairs: name of corresponding output layer
    and its tensor layout represented in std::string ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutputTensorLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor layout")
            .output_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies model layout for an output layer.

    The function is used to set model layout for an output layer.

    @param layout Model layout ("NCHW", "NHWC", etc)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutputModelLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output model layout")
            .output_model_layout = std::move(layout);
        return *this;
    }

    /** @overload
    @param layout_map Map of pairs: name of corresponding output layer
    and its model layout ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutputModelLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output model layout")
            .output_model_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies tensor precision for an output layer.

    The function is used to set tensor precision for an output layer..

    @param precision Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutputTensorPrecision(int precision) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor precision")
            .output_tensor_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutputTensorPrecision(detail::AttrMap<int> precision_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor precision")
            .output_tensor_precision = std::move(precision_map);
        return *this;
    }

    /** @brief Specifies the new shape for input layers.

    The function is used to set new shape for input layers.

    @param new_shape New shape will be applied to all input layers.
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgReshape(std::vector<size_t> new_shape) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "reshape")
            .new_shapes = std::move(new_shape);
        return *this;
    }

    /** @overload

    @param new_shape_map Map of pairs: name of corresponding output layer
    and its new shape.
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgReshape(detail::AttrMap<std::vector<size_t>> new_shape_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "reshape")
            .new_shapes = std::move(new_shape_map);
        return *this;
    }

    /** @brief Specifies number of asynchronous inference requests.

    @param nireq Number of inference asynchronous requests.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgNumRequests(const size_t nireq) {
        if (nireq == 0) {
            cv::util::throw_error(
                    std::logic_error("Number of inference requests"
                                     " must be greater than zero."));
        }
        m_desc.nireq = nireq;
        return *this;
    }

    /** @brief Specifies mean values for preprocessing.
     *
    The function is used to set mean values for input layer preprocessing.

    @param mean_values Float vector contains mean values
    @return reference to this parameter structure.
    */
    Params<Net>& cfgMean(std::vector<float> mean_values) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "mean values")
            .mean_values = std::move(mean_values);
        return *this;
    }

    /** @overload

    @param mean_map Map of pairs: name of corresponding input layer
    and its mean values.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgMean(detail::AttrMap<std::vector<float>> mean_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "mean values")
            .mean_values = std::move(mean_map);
        return *this;
    }

    /** @brief Specifies scale values for preprocessing.
     *
    The function is used to set scale values for input layer preprocessing.

    @param scale_values Float vector contains scale values
    @return reference to this parameter structure.
    */
    Params<Net>& cfgScale(std::vector<float> scale_values) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "scale values")
            .scale_values = std::move(scale_values);
        return *this;
    }

    /** @overload

    @param scale_map Map of pairs: name of corresponding input layer
    and its mean values.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgScale(detail::AttrMap<std::vector<float>> scale_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "scale values")
            .scale_values = std::move(scale_map);
        return *this;
    }

    /** @brief Specifies resize interpolation algorithm.
     *
    The function is used to configure resize preprocessing for input layer.

    @param interpolation Resize interpolation algorithm.
    Supported algorithms: #INTER_NEAREST, #INTER_LINEAR, #INTER_CUBIC.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgResize(int interpolation) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "resize preprocessing")
            .interpolation = std::move(interpolation);
        return *this;
    }

    /** @overload

    @param interpolation Map of pairs: name of corresponding input layer
    and its resize algorithm.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgResize(detail::AttrMap<int> interpolation) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "resize preprocessing")
            .interpolation = std::move(interpolation);
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::ov::backend(); }
    std::string   tag()     const { return Net::tag(); }
    cv::util::any params()  const { return { m_desc }; }
    // END(G-API's network parametrization API)

protected:
    detail::ParamDesc m_desc;
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

    Constructs Params based on model information and specifies default values for other
    inference description parameters. Model is loaded and compiled using "OpenVINO Toolkit".

    @param tag string tag of the network for which these parameters are intended.
    @param model_path Path to a model.
    @param bin_path Path to a data file.
    For IR format (*.bin):
    If path is empty, will try to read a bin file with the same name as xml.
    If the bin file with the same name is not found, will load IR without weights.
    For PDPD (*.pdmodel) and ONNX (*.onnx) formats bin_path isn't used.
    @param device target device to use.
    */
    Params(const std::string &tag,
           const std::string &model_path,
           const std::string &bin_path,
           const std::string &device)
        : m_tag(tag),
          m_desc( detail::ParamDesc::Kind{detail::ParamDesc::Model{model_path, bin_path}}
                , device
                , true /* is generic */
                , 0u
                , 0u) {
    }

    /** @overload

    This constructor for pre-compiled networks. Model is imported from pre-compiled
    blob.

    @param tag string tag of the network for which these parameters are intended.
    @param blob_path path to the compiled model (*.blob).
    @param device target device to use.
    */
    Params(const std::string &tag,
           const std::string &blob_path,
           const std::string &device)
        : m_tag(tag),
          m_desc( detail::ParamDesc::Kind{detail::ParamDesc::CompiledModel{blob_path}}
                , device
                , true /* is generic */
                , 0u
                , 0u) {
    }

    /** @see ov::Params::cfgPluginConfig. */
    Params& cfgPluginConfig(const detail::ParamDesc::PluginConfigT &config) {
        m_desc.config = config;
        return *this;
    }

    /** @see ov::Params::cfgInputTensorLayout. */
    Params& cfgInputTensorLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input tensor layout")
            .input_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgInputTensorLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input tensor layout")
            .input_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @see ov::Params::cfgInputModelLayout. */
    Params& cfgInputModelLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input model layout")
            .input_model_layout = std::move(layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgInputModelLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "input model layout")
            .input_model_layout = std::move(layout_map);
        return *this;
    }

    /** @see ov::Params::cfgOutputTensorLayout. */
    Params& cfgOutputTensorLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor layout")
            .output_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgOutputTensorLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor layout")
            .output_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @see ov::Params::cfgOutputModelLayout. */
    Params& cfgOutputModelLayout(std::string layout) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output model layout")
            .output_model_layout = std::move(layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgOutputModelLayout(detail::AttrMap<std::string> layout_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output model layout")
            .output_model_layout = std::move(layout_map);
        return *this;
    }

    /** @see ov::Params::cfgOutputTensorPrecision. */
    Params& cfgOutputTensorPrecision(int precision) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor precision")
            .output_tensor_precision = precision;
        return *this;
    }

    /** @overload */
    Params&
    cfgOutputTensorPrecision(detail::AttrMap<int> precision_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "output tensor precision")
            .output_tensor_precision = std::move(precision_map);
        return *this;
    }

    /** @see ov::Params::cfgReshape. */
    Params& cfgReshape(std::vector<size_t> new_shape) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "reshape")
            .new_shapes = std::move(new_shape);
        return *this;
    }

    /** @overload */
    Params&
    cfgReshape(detail::AttrMap<std::vector<size_t>> new_shape_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "reshape")
            .new_shapes = std::move(new_shape_map);
        return *this;
    }

    /** @see ov::Params::cfgNumRequests. */
    Params& cfgNumRequests(const size_t nireq) {
        if (nireq == 0) {
            cv::util::throw_error(
                    std::logic_error("Number of inference requests"
                                     " must be greater than zero."));
        }
        m_desc.nireq = nireq;
        return *this;
    }

    /** @see ov::Params::cfgMean. */
    Params& cfgMean(std::vector<float> mean_values) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "mean values")
            .mean_values = std::move(mean_values);
        return *this;
    }

    /** @overload */
    Params& cfgMean(detail::AttrMap<std::vector<float>> mean_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "mean values")
            .mean_values = std::move(mean_map);
        return *this;
    }

    /** @see ov::Params::cfgScale. */
    Params& cfgScale(std::vector<float> scale_values) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "scale values")
            .scale_values = std::move(scale_values);
        return *this;
    }

    /** @overload */
    Params& cfgScale(detail::AttrMap<std::vector<float>> scale_map) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "scale values")
            .scale_values = std::move(scale_map);
        return *this;
    }

    /** @see ov::Params::cfgResize. */
    Params& cfgResize(int interpolation) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "resize preprocessing")
            .interpolation = std::move(interpolation);
        return *this;
    }

    /** @overload */
    Params& cfgResize(detail::AttrMap<int> interpolation) {
        detail::getModelToSetAttrOrThrow(m_desc.kind, "resize preprocessing")
            .interpolation = std::move(interpolation);
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::ov::backend(); }
    std::string   tag()     const { return m_tag; }
    cv::util::any params()  const { return { m_desc }; }
    // END(G-API's network parametrization API)

protected:
    std::string m_tag;
    detail::ParamDesc m_desc;
};

} // namespace ov

namespace wip { namespace ov {
/**
 * @brief Ask G-API OpenVINO backend to run only inference of model provided.
 *
 * G-API OpenVINO backend will perform only the inference of the model provided
 * without populating input and copying back output data.
 * This mode is used to evaluate the pure inference performance of the model without
 * taking into account the i/o data transfer.
 */
struct benchmark_mode { };

struct workload_type {
    using callback = std::function<void(const unsigned int)>;
    using listener = std::pair<int, callback>;
    std::shared_ptr<void> addListener(callback cb){
        int id = nextId++;
        listeners.emplace_back(id, std::move(cb));

        auto remover = [this, id](void*){ removeListener(id);};

        return std::shared_ptr<void>(nullptr, remover);
    }
    void setWorkloadType(const unsigned int type) {
        for(const listener& l : listeners) {
            l.second(type);
        }
    }
 private:
    std::vector<listener> listeners;
    int nextId = 0;
    void removeListener(int id) {
        listeners.erase(std::remove_if(listeners.begin(), listeners.end(), [=](listener& pair){return pair.first == id;}), listeners.end());
    }
};
} // namespace ov
} // namespace wip

} // namespace gapi

namespace detail
{
    template<> struct CompileArgTag<cv::gapi::wip::ov::benchmark_mode>
    {
        static const char* tag() { return "gapi.wip.ov.benchmark_mode"; }
    };
    template<> struct CompileArgTag<cv::gapi::wip::ov::workload_type>
    {
        static const char* tag() { return "gapi.wip.ov.workload_type"; }
    };
    template<> struct CompileArgTag<std::reference_wrapper<cv::gapi::wip::ov::workload_type>>
    {
        static const char* tag() { return "gapi.wip.ov.workload_type_ref"; }
    };
}

} // namespace cv

#endif // OPENCV_GAPI_INFER_OV_HPP
