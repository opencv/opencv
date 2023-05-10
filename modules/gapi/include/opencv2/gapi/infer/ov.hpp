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
    struct IR {
        std::string xml_path;
        std::string bin_path;
    };
    struct Blob {
        std::string blob_path;
    };
    using Kind = cv::util::variant<IR, Blob>;
    Kind kind;

    std::string device;

    bool is_generic;

    std::size_t num_in;
    std::size_t num_out;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    using PluginConfigT = std::map<std::string, std::string>;
    PluginConfigT config;

    template <typename T>
    using Map = std::unordered_map<std::string, T>;

    // NB: This type is supposed to be used to hold in/out layers
    // attributes such as precision, layout, shape etc.
    //
    // Because user can provide attributes either:
    // 1. cv::util::monostate - No value specified explicitly.
    // 2. T - value specified explicitly that should be broadcasted to all layers.
    // 3. Map[str->T] - map specifies value for particular layer.
    template <typename T>
    using VariantMapT = cv::util::variant<cv::util::monostate, Map<T>, T>;

    VariantMapT<std::string> input_tensor_layout;
    VariantMapT<std::string> input_model_layout;
    VariantMapT<int>         output_precision;
};

} // namespace detail

template<typename Net> class Params {
public:
    Params(const std::string &xml,
           const std::string &bin,
           const std::string &device)
        : desc{ detail::ParamDesc::Kind{detail::ParamDesc::IR{xml, bin}}
              , device
              , false /* is generic */
              , std::tuple_size<typename Net::InArgs>::value
              , std::tuple_size<typename Net::OutArgs>::value
              , {} /* output_names */
              , {} /* input_names */
              , {} /* config */
              , {} /* output_precision */
              , {} /* input_tensor_layout */
              , {} /* input_model_layout */ } {
    }

    Params(const std::string &blob,
           const std::string &device)
        : desc{ detail::ParamDesc::Kind{detail::ParamDesc::Blob{blob}}
              , device
              , false /* is generic */
              , std::tuple_size<typename Net::InArgs>::value
              , std::tuple_size<typename Net::OutArgs>::value
              , {} /* output_names */
              , {} /* input_names */
              , {} /* config */
              , {} /* output_precision */
              , {} /* input_tensor_layout */
              , {} /* input_model_layout */ } {
    }

    Params<Net>& cfgInputLayers(const std::vector<std::string> &input_names) {
        desc.input_names = input_names;
        return *this;
    }

    Params<Net>& cfgOutputLayers(const std::vector<std::string> &output_names) {
        desc.output_names = output_names;
        return *this;
    }

    Params<Net>& cfgPluginConfig(const detail::ParamDesc::PluginConfigT &config) {
        desc.config = config;
        return *this;
    }

    /** @brief Specifies the output precision for model.

    The function is used to set an output precision for model.

    @param precision Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgOutTensorPrecision(int precision) {
        desc.output_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutTensorPrecision(detail::ParamDesc::Map<int> precision_map) {
        desc.output_precision = std::move(precision_map);
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInTensorLayout(std::string layout) {
        desc.input_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInTensorLayout(detail::ParamDesc::Map<std::string> layout_map) {
        desc.input_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInModelLayout(const std::string &layout) {
        desc.input_model_layout = layout;
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInModelLayout(detail::ParamDesc::Map<std::string> layout_map) {
        desc.input_model_layout = std::move(layout_map);
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
    Params(const std::string &tag,
           const std::string &xml,
           const std::string &bin,
           const std::string &device)
        : m_tag(tag),
          m_desc{ detail::ParamDesc::Kind{detail::ParamDesc::IR{xml, bin}}
                , device
                , true /* is generic */
                , 0u
                , 0u
                , {} /* output_names */
                , {} /* input_names */
                , {} /* config */
                , {} /* output_precision */
                , {} /* input_tensor_layout */
                , {} /* input_model_layout */ } {
    }

    Params(const std::string &tag,
           const std::string &blob,
           const std::string &device)
        : m_tag(tag),
          m_desc{ detail::ParamDesc::Kind{detail::ParamDesc::Blob{blob}}
                , device
                , true /* is generic */
                , 0u
                , 0u
                , {} /* output_names */
                , {} /* input_names */
                , {} /* config */
                , {} /* output_precision */
                , {} /* input_tensor_layout */
                , {} /* input_model_layout */ } {
    }

    Params& cfgInputLayers(const std::vector<std::string> &input_names) {
        m_desc.input_names = input_names;
        return *this;
    }

    Params& cfgOutputLayers(const std::vector<std::string> &output_names) {
        m_desc.output_names = output_names;
        return *this;
    }

    Params& cfgPluginConfig(const detail::ParamDesc::PluginConfigT &config) {
        m_desc.config = config;
        return *this;
    }

    /** @brief Specifies the output precision for model.

    The function is used to set an output precision for model.

    @param precision Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params& cfgOutTensorPrecision(int precision) {
        m_desc.output_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params&
    cfgOutTensorPrecision(detail::ParamDesc::Map<int> precision_map) {
        m_desc.output_precision = std::move(precision_map);
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params& cfgInTensorLayout(std::string layout) {
        m_desc.input_tensor_layout = std::move(layout);
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params&
    cfgInTensorLayout(detail::ParamDesc::Map<std::string> layout_map) {
        m_desc.input_tensor_layout = std::move(layout_map);
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params& cfgInModelLayout(std::string layout) {
        m_desc.input_model_layout = std::move(layout);
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params&
    cfgInModelLayout(detail::ParamDesc::Map<std::string> layout_map) {
        m_desc.input_model_layout = std::move(layout_map);
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
