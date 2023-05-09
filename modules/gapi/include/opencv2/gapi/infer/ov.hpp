// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_INFER_OV_HPP
#define OPENCV_GAPI_INFER_OV_HPP

#include <string>

#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/own/exports.hpp>  // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp>   // GKernelType[M], GBackend

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

    using PrecisionT = int;
    using PrecisionMapT = std::unordered_map<std::string, PrecisionT>;
    // NB: This parameter can contain:
    // 1. cv::util::monostate - Don't specify precision, but use default from IR/Blob.
    // 2. PrecisionT (CV_8U, CV_32F, ...) - Specifies precision for all layers.
    // 3. PrecisionMapT ({{"layer0", CV_32F}, {"layer1", CV_16F}} - Specifies per-layer precision.
    // cv::util::monostate is default value that means precision wasn't specified.
    using PrecisionVariantT = cv::util::variant<cv::util::monostate,
                                                PrecisionT,
                                                PrecisionMapT>;
    using LayoutT = std::string;
    // NB: This parameter can contain:
    // 1. cv::util::monostate - Don't specify layout, but use default from IR/Blob.
    // 2. LayoutT ("NCHW", "NHWC", "?CHW", etc) - Specifies layout for all layers.
    // 3. LayoutMapT ({{"layer0", "NCHW"}, {"layer1", "NC"}} - Specifies per-layer layout.
    // cv::util::monostate is default value that means layout wasn't specified.
    using LayoutMapT = std::unordered_map<std::string, LayoutT>;
    using LayoutVariantT = cv::util::variant<cv::util::monostate,
                                             LayoutT,
                                             LayoutMapT>;

    PrecisionVariantT output_precision;
    LayoutVariantT    input_tensor_layout;
    LayoutVariantT    input_model_layout;
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
    Params<Net>& cfgOutTensorPrecision(detail::ParamDesc::PrecisionT precision) {
        desc.output_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgOutTensorPrecision(detail::ParamDesc::PrecisionMapT precision_map) {
        desc.output_precision = precision_map;
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInTensorLayout(detail::ParamDesc::LayoutT layout) {
        desc.input_tensor_layout = layout;
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInTensorLayout(detail::ParamDesc::LayoutMapT layout_map) {
        desc.input_tensor_layout = layout_map;
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params<Net>& cfgInModelLayout(detail::ParamDesc::LayoutT layout) {
        desc.input_model_layout = layout;
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params<Net>&
    cfgInModelLayout(detail::ParamDesc::LayoutMapT layout_map) {
        desc.input_model_layout = layout_map;
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
    Params& cfgOutTensorPrecision(detail::ParamDesc::PrecisionT precision) {
        m_desc.output_precision = precision;
        return *this;
    }

    /** @overload

    @param precision_map Map of pairs: name of corresponding output layer
    and its precision in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params&
    cfgOutTensorPrecision(detail::ParamDesc::PrecisionMapT precision_map) {
        m_desc.output_precision = precision_map;
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params& cfgInTensorLayout(detail::ParamDesc::LayoutT layout) {
        m_desc.input_tensor_layout = layout;
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params&
    cfgInTensorLayout(detail::ParamDesc::LayoutMapT layout_map) {
        m_desc.input_tensor_layout = layout_map;
        return *this;
    }

    /** @brief Specifies the output layout for model.

    The function is used to set an output layout for model.

    @param layout Precision in OpenCV format (CV_8U, CV_32F, ...)
    will be applied to all output layers.
    @return reference to this parameter structure.
    */
    Params& cfgInModelLayout(detail::ParamDesc::LayoutT layout) {
        m_desc.input_model_layout = layout;
        return *this;
    }

    /** @overload

    @param layout_map Map of pairs: name of corresponding output layer
    and its layout in OpenCV format (CV_8U, CV_32F, ...)
    @return reference to this parameter structure.
    */
    Params&
    cfgInModelLayout(detail::ParamDesc::LayoutMapT layout_map) {
        m_desc.input_model_layout = layout_map;
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
