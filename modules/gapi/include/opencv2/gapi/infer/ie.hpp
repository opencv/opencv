// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation

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

namespace cv {
namespace gapi {
// FIXME: introduce a new sub-namespace for NN?
namespace ie {

GAPI_EXPORTS cv::gapi::GBackend backend();

/**
 * Specify how G-API and IE should trait input data
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

namespace detail {
/**
* @brief This structure contains description of inference parameters
* which is specific to OpenVINO models.
*/
struct ParamDesc {
    /*@{*/
    std::string model_path; //!< Path to topology IR (.xml file)
    std::string weights_path; //!< Path to weights (.bin file).
    std::string device_id; //!< Device specifier.

    // NB: Here order follows the `Net` API
    std::vector<std::string> input_names; //!< Names of input CNN layers.
    std::vector<std::string> output_names; //!< Names of output CNN layers.

    using ConstInput = std::pair<cv::Mat, TraitAs>;
    std::unordered_map<std::string, ConstInput> const_inputs; //!< Map with pair of name of CNN layer and ConstInput which will be associated with this.

    // NB: nun_* may differ from topology's real input/output port numbers
    // (e.g. topology's partial execution)
    std::size_t num_in;  //!< How many inputs are defined in the operation
    std::size_t num_out; //!< How many outputs are defined in the operation

    enum class Kind { Load, Import };
    Kind kind; //!< What kind of model will be used: standard or pre-compiled.
    bool is_generic; //!< Enable generic version of inference.
    IEConfig config; //!< OpenVINO plugin configuration.

    std::map<std::string, std::vector<std::size_t>> reshape_table; //!< Map of pairs: name of corresponding data and its dimension.
    std::unordered_set<std::string> layer_names_to_reshape; //!< Set of names of CNN layers that will be used for CNN reshape.

    size_t nireq; //!< Number of asyncrhonious infer requests.
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
 * @brief This structure provides functions that fill inference parameters.
 */
template<typename Net> class Params {
public:
    /** @brief Class constructor.

    Constructs Params based on model information and sets default values for other
    inference description parameters. Model is loaded and compiled with OpenVINO.

    @param model path to topology IR (.xml file).
    @param weights path to weights (.bin file).
    @param device device specifier.
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
              , 1u} {
    };

    /** @overload
    This constructor for pre-compiled networks. Model is imported from pre-compiled
    blob.

    @param model path to model.
    @param device device specifier.
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
              , 1u} {
    };

    /** @brief Specifies sequence of CNN input layers names for inference.

    The function is used to associate data of graph inputs with input layers of
    CNN topology. Number of names has to match the number of CNN inputs. If a CNN
    has only one input layer, there is no need to call it as the layer is
    associated with input automatically but this doesn't prevent you from
    doing it yourself. Count of names has to match to number of CNN inputs.

    @param layer_names std::array<std::string, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Contains names of input layers.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputLayers(const typename PortCfg<Net>::In &layer_names) {
        desc.input_names.clear();
        desc.input_names.reserve(layer_names.size());
        std::copy(layer_names.begin(), layer_names.end(),
                  std::back_inserter(desc.input_names));
        return *this;
    }

    /** @brief Sets sequence of CNN output layers names for inference.

    The function is used to associate data of graph outputs with output layers of
    CNN topology. If a CNN has only one output layer, there is no need to call it
    as the layer is associated with ouput automatically but this doesn't prevent 
    you from doing it yourself. Count of names has to match to number of CNN
    outputs.

    @param layer_names std::array<std::string, N> where N is the number of outputs
    as defined in the @ref G_API_NET. Contains names of output layers.
    @return reference to object of class Params.
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
    a preprocessed tensor if its type is TENSOR. You should provide name of CNN layer
    which will receive provided data. For example, this can be useful for
    specifying size of image tensor defined in Faster-RCNN.

    @param layer_name name of CNN layer.
    @param data cv::Mat that contains data which will be associated with CNN layer.
    @param hint type of input (IMAGE or TENSOR).
    @return reference to object of class Params.
    */
    Params<Net>& constInput(const std::string &layer_name,
                            const cv::Mat &data,
                            TraitAs hint = TraitAs::TENSOR) {
        desc.const_inputs[layer_name] = {data, hint};
        return *this;
    }

    /** @brief Sets OpenVINO plugin configuration.

    The function is used to set configuration for OpenVINO plugin. Some parameters
    can be different for each plugin. Please follow https://docs.openvinotoolkit.org/latest/index.html
    for check information about your plugin.

    @param cfg map of pairs: (config parameter name, config parameter value).
    @return reference to object of class Params.
    */
       Params& pluginConfig(const IEConfig& cfg) {
        desc.config = cfg;
        return *this;
    }

    /** @overload
    Function with an rvalue parameter.

    @param cfg rvalue map of pairs: (config parameter name, config parameter value).
    @return reference to object of class Params.
    */
    Params& pluginConfig(IEConfig&& cfg) {
        desc.config = std::move(cfg);
        return *this;
    }

    /** @brief Specifies count of asynchronous inference requests.

    @param nireq count of inference asynchronous requests.
    @return reference to object of class Params.
    */
    Params& cfgNumRequests(size_t nireq) {
        GAPI_Assert(nireq > 0 && "Number of infer requests must be greater than zero!");
        desc.nireq = nireq;
        return *this;
    }

    /** @brief Specifies new input shapes for the CNN inputs.

    The function is used to set new input shapes for network. You can specify
    new dimensions for one or some or all of CNN input layers. CNN will be
    reshaped based on this information and your inputs will be preprocessed for
    new input sizes. Follow https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1CNNNetwork.html
    for additional information.

    @param reshape_table map of pairs: name of corresponding data and its dimension.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputReshape(const std::map<std::string, std::vector<std::size_t>>& reshape_table) {
        desc.reshape_table = reshape_table;
        return *this;
    }

    /** @overload
    Function with an rvalue parameters.

    @param reshape_table rvalue map of pairs: name of corresponding data and its
    dimension.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputReshape(std::map<std::string, std::vector<std::size_t>>&& reshape_table) {
        desc.reshape_table = std::move(reshape_table);
        return *this;
    }

    /** @overload
    Function for one input layer.

    @param layer_name name of layer.
    @param layer_dims new dimensions for this layer.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputReshape(const std::string& layer_name, const std::vector<size_t>& layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    /** @overload
    Function with an rvalue parameters for one input layer.

    @param layer_name rvalue name of layer.
    @param layer_dims rvalue new dimensions for this layer.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputReshape(std::string&& layer_name, std::vector<size_t>&& layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    /** @overload
    Function for reshape by image size.

    The function is used to set names of layers that will be used for CNN reshape.
    Dimensions will be constructed automatically by current CNN input and height and
    width of image.

    @param layer_names set of names of CNN layers that will be used for CNN reshape.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputReshape(const std::unordered_set<std::string>& layer_names) {
        desc.layer_names_to_reshape = layer_names;
        return *this;
    }

    /** @overload
    Function with an rvalue parameters for reshape by image size.

    @param layer_names rvalue set of the selected layers will be reshaped automatically
    its input image size.
    @return reference to object of class Params.
    */
    Params<Net>& cfgInputReshape(std::unordered_set<std::string>&& layer_names) {
        desc.layer_names_to_reshape = std::move(layer_names);
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

template<>
class Params<cv::gapi::Generic> {
public:
    Params(const std::string &tag,
           const std::string &model,
           const std::string &weights,
           const std::string &device)
        : desc{ model, weights, device, {}, {}, {}, 0u, 0u, detail::ParamDesc::Kind::Load, true, {}, {}, {}, 1u}, m_tag(tag) {
    };

    Params(const std::string &tag,
           const std::string &model,
           const std::string &device)
        : desc{ model, {}, device, {}, {}, {}, 0u, 0u, detail::ParamDesc::Kind::Import, true, {}, {}, {}, 1u}, m_tag(tag) {
    };

    Params& pluginConfig(IEConfig&& cfg) {
        desc.config = std::move(cfg);
        return *this;
    }

    Params& pluginConfig(const IEConfig& cfg) {
        desc.config = cfg;
        return *this;
    }

    Params& constInput(const std::string &layer_name,
                       const cv::Mat &data,
                       TraitAs hint = TraitAs::TENSOR) {
        desc.const_inputs[layer_name] = {data, hint};
        return *this;
    }

    Params& cfgNumRequests(size_t nireq) {
        GAPI_Assert(nireq > 0 && "Number of infer requests must be greater than zero!");
        desc.nireq = nireq;
        return *this;
    }

    Params& cfgInputReshape(std::map<std::string, std::vector<std::size_t>> && reshape_table) {
        desc.reshape_table = std::move(reshape_table);
        return *this;
    }

    Params& cfgInputReshape(const std::map<std::string, std::vector<std::size_t>>&reshape_table) {
        desc.reshape_table = reshape_table;
        return *this;
    }

    Params& cfgInputReshape(std::string && layer_name, std::vector<size_t> && layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    Params& cfgInputReshape(const std::string & layer_name, const std::vector<size_t>&layer_dims) {
        desc.reshape_table.emplace(layer_name, layer_dims);
        return *this;
    }

    Params& cfgInputReshape(std::unordered_set<std::string> && layer_names) {
        desc.layer_names_to_reshape = std::move(layer_names);
        return *this;
    }

    Params& cfgInputReshape(const std::unordered_set<std::string>&layer_names) {
        desc.layer_names_to_reshape = layer_names;
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
