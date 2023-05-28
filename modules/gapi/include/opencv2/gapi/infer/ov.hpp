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

struct ParamDesc {
    struct Model {
        Model(const std::string &model_path_,
              const std::string &bin_path_);

        std::string model_path;
        std::string bin_path;

        template <typename T>
        using Map = std::unordered_map<std::string, T>;
        // NB: This type is supposed to be used to hold in/out layers
        // attributes such as precision, layout, shape etc.
        //
        // User can provide attributes either:
        // 1. cv::util::monostate - No value specified explicitly.
        // 2. T - value specified explicitly that should be broadcasted to all layers.
        // 3. Map[str->T] - map specifies value for particular layer.
        template <typename T>
        using VariantMapT = cv::util::variant<cv::util::monostate, Map<T>, T>;

        VariantMapT<std::string> input_tensor_layout;
        VariantMapT<std::string> input_model_layout;
        VariantMapT<int>         output_precision;
    };

    struct CompiledModel {
        std::string blob_path;
    };

    using Kind = cv::util::variant<Model, CompiledModel>;

    ParamDesc(Kind              &&kind_,
              const std::string &device_,
              const bool        is_generic_,
              const size_t      num_in_,
              const size_t      num_out_);

    Kind kind;

    std::string device;
    bool is_generic;

    std::size_t num_in;
    std::size_t num_out;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    using PluginConfigT = std::map<std::string, std::string>;
    PluginConfigT config;
};

ParamDesc::Model::Model(const std::string &model_path_,
                        const std::string &bin_path_)
    : model_path(model_path_), bin_path(bin_path_) {
}

ParamDesc::ParamDesc(ParamDesc::Kind   &&kind_,
                     const std::string &device_,
                     const bool        is_generic_,
                     const size_t      num_in_,
                     const size_t      num_out_)
    : kind(std::move(kind_)), device(device_),
      is_generic(is_generic_),
      num_in(num_in_), num_out(num_out_) {
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
        : desc( detail::ParamDesc::Kind{detail::ParamDesc::Model{model_path, bin_path}}
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
        : desc( detail::ParamDesc::Kind{detail::ParamDesc::CompiledModel{blob_path}}
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
    Params<Net>& cfgInputLayers(const std::vector<std::string> &input_names) {
        desc.input_names = input_names;
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
    Params<Net>& cfgOutputLayers(const std::vector<std::string> &output_names) {
        desc.output_names = output_names;
        return *this;
    }

    /** @brief Specifies OpenVINO plugin configuration.

    The function is used to set configuration for OpenVINO plugin. Some parameters
    can be different for each plugin. Please follow https://docs.openvinotoolkit.org/latest/index.html
    to check information about specific plugin.

    @param cfg Map of pairs: (config parameter name, config parameter value).
    @return reference to this parameter structure.
    */
    Params<Net>& cfgPluginConfig(const detail::ParamDesc::PluginConfigT &config) {
        desc.config = config;
        return *this;
    }

    /** @brief Specifies the output precision for a model.

    The function is used to set an output precision for model.

    @param precision Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutTensorPrecision(int precision) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying output tensor precision isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(desc.kind);
        model.output_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutTensorPrecision(detail::ParamDesc::Model::Map<int> precision_map) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying output tensor precision isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(desc.kind);
        model.output_precision = precision_map;
        return *this;
    }

    /** @brief Specifies the layout for an input tensor.

    The function is used to set layot for an input tensor.

    @param layout Tensor layout ("NCHW", "NWHC", etc)
    will be applied to all input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInTensorLayout(std::string layout) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input tensor layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(desc.kind);
        model.input_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload
    @param layout_map Map of pairs: name of corresponding input layer
    and its tensor layout represented in std::string ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInTensorLayout(detail::ParamDesc::Model::Map<std::string> layout_map) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input tensor layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(desc.kind);
        model.input_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies an input layout for a model.

    The function is used to set an input layout for a model.

    @param layout Model layout ("NCHW", "NHWC", etc)
    will be applied to all input layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInModelLayout(std::string model_layout) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input model layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(desc.kind);
        model.input_model_layout = std::move(model_layout);
        return *this;
    }

    /** @overload
    @param layout_map Map of pairs: name of corresponding input layer
    and its model layout ("NCHW", "NHWC", etc)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInModelLayout(detail::ParamDesc::Model::Map<std::string> layout_map) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input model layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(desc.kind);
        model.input_model_layout = std::move(layout_map);
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::ov::backend(); }
    std::string   tag()     const { return Net::tag(); }
    cv::util::any params()  const { return { desc }; }
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

    /** @see ov::Params::cfgInputLayers. */
    Params& cfgInputLayers(const std::vector<std::string> &input_names) {
        m_desc.input_names = input_names;
        return *this;
    }

    /** @overload */
    Params& cfgOutputLayers(const std::vector<std::string> &output_names) {
        m_desc.output_names = output_names;
        return *this;
    }

    /** @see ov::Params::cfgPluginConfig. */
    Params& cfgPluginConfig(const detail::ParamDesc::PluginConfigT &config) {
        m_desc.config = config;
        return *this;
    }

    /** @see ov::Params::cfgOutTensorPrecision. */
    Params& cfgOutTensorPrecision(int precision) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(m_desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying output tensor precision isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(m_desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(m_desc.kind);
        model.output_precision = precision;
        return *this;
    }

    /** @overload */
    Params&
    cfgOutTensorPrecision(detail::ParamDesc::Model::Map<int> precision_map) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(m_desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying output tensor precision isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(m_desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(m_desc.kind);
        model.output_precision = precision_map;
        return *this;
    }

    /** @see ov::Params::cfgInTensorLayout. */
    Params& cfgInTensorLayout(std::string tensor_layout) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(m_desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input tensor layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(m_desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(m_desc.kind);
        model.input_tensor_layout = std::move(tensor_layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgInTensorLayout(detail::ParamDesc::Model::Map<std::string> layout_map) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(m_desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input tensor layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(m_desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(m_desc.kind);
        model.input_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @see ov::Params::cfgInModelLayout. */
    Params& cfgInModelLayout(std::string model_layout) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(m_desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input model layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(m_desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(m_desc.kind);
        model.input_model_layout = std::move(model_layout);
        return *this;
    }

    /** @overload */
    Params&
    cfgInModelLayout(detail::ParamDesc::Model::Map<std::string> layout_map) {
        if (cv::util::holds_alternative<detail::ParamDesc::CompiledModel>(m_desc.kind)) {
            cv::util::throw_error(
                    std::logic_error("Specifying input model layout isn't"
                                     " possible for ov::CompiledModel."));
        }
        GAPI_Assert(cv::util::holds_alternative<detail::ParamDesc::Model>(m_desc.kind));
        auto &model = cv::util::get<detail::ParamDesc::Model>(m_desc.kind);
        model.input_model_layout = std::move(layout_map);
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
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_OV_HPP
