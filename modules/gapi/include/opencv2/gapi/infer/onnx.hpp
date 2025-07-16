// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#ifndef OPENCV_GAPI_INFER_ONNX_HPP
#define OPENCV_GAPI_INFER_ONNX_HPP

#include <unordered_map>
#include <string>
#include <array>
#include <tuple> // tuple, tuple_size
#include <map>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/util/optional.hpp>

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage
#include <opencv2/gapi/infer.hpp>   // Generic

namespace cv {
namespace gapi {

/**
 * @brief This namespace contains G-API ONNX Runtime backend functions, structures, and symbols.
 */
namespace onnx {

/**
 * @brief This namespace contains Execution Providers structures for G-API ONNX Runtime backend.
 */
namespace ep {

/**
 * @brief This structure provides functions
 * that fill inference options for ONNX CoreML Execution Provider.
 * Please follow https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#coreml-execution-provider
 */
struct GAPI_EXPORTS_W_SIMPLE CoreML {
    /** @brief Class constructor.

    Constructs CoreML parameters.

    */
    GAPI_WRAP
    CoreML() = default;

    /** @brief Limit CoreML Execution Provider to run on CPU only.

    This function is used to limit CoreML to run on CPU only.
    Please follow: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#coreml_flag_use_cpu_only

    @return reference to this parameter structure.
    */
    GAPI_WRAP
    CoreML& cfgUseCPUOnly() {
        use_cpu_only = true;
        return *this;
    }

    /** @brief Enable CoreML EP to run on a subgraph in the body of a control flow ONNX operator (i.e. a Loop, Scan or If operator).

    This function is used to enable CoreML EP to run on
    a subgraph of a control flow of ONNX operation.
    Please follow: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#coreml_flag_enable_on_subgraph

    @return reference to this parameter structure.
    */
    GAPI_WRAP
    CoreML& cfgEnableOnSubgraph() {
        enable_on_subgraph = true;
        return *this;
    }

    /** @brief Enable CoreML EP to run only on Apple Neural Engine.

    This function is used to enable CoreML EP to run only on Apple Neural Engine.
    Please follow: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#coreml_flag_only_enable_device_with_ane

    @return reference to this parameter structure.
    */
    GAPI_WRAP
    CoreML& cfgEnableOnlyNeuralEngine() {
        enable_only_ane = true;
        return *this;
    }

    bool use_cpu_only = false;
    bool enable_on_subgraph = false;
    bool enable_only_ane = false;
};

/**
 * @brief This structure provides functions
 * that fill inference options for CUDA Execution Provider.
 * Please follow https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-execution-provider
 */
struct GAPI_EXPORTS_W_SIMPLE CUDA {
    // NB: Used from python.
    /// @private -- Exclude this constructor from OpenCV documentation
    GAPI_WRAP
    CUDA() = default;

    /** @brief Class constructor.

    Constructs CUDA parameters based on device type information.

    @param dev_id Target device id to use.
    */
    GAPI_WRAP
    explicit CUDA(const int dev_id)
        : device_id(dev_id) {
    }

    int device_id;
};

/**
 * @brief This structure provides functions
 * that fill inference options for TensorRT Execution Provider.
 * Please follow https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#tensorrt-execution-provider
 */
struct GAPI_EXPORTS_W_SIMPLE TensorRT {
    // NB: Used from python.
    /// @private -- Exclude this constructor from OpenCV documentation
    GAPI_WRAP
    TensorRT() = default;

    /** @brief Class constructor.

    Constructs TensorRT parameters based on device type information.

    @param dev_id Target device id to use.
    */
    GAPI_WRAP
    explicit TensorRT(const int dev_id)
        : device_id(dev_id) {
    }

    int device_id;
};

/**
 * @brief This structure provides functions
 * that fill inference options for ONNX OpenVINO Execution Provider.
 * Please follow https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options
 */
struct GAPI_EXPORTS_W_SIMPLE OpenVINO {
    // NB: Used from python.
    /// @private -- Exclude this constructor from OpenCV documentation
    GAPI_WRAP
    OpenVINO() = default;

    /** @brief Class constructor.

    Constructs OpenVINO parameters based on device type information.

    @param dev_type Target device type to use. ("CPU", "GPU", "GPU.0" etc)
    */
    GAPI_WRAP
    explicit OpenVINO(const std::string &dev_type)
        : device_type(dev_type) {
    }

    /** @brief Class constructor.

    Constructs OpenVINO parameters based on map of options passed.

    * @param params A map of parameter names and their corresponding string values.
    */
    GAPI_WRAP
    explicit OpenVINO(const std::map<std::string, std::string>& params)
        : params_map(params) {
    }

    /** @brief Specifies OpenVINO Execution Provider cache dir.

    This function is used to explicitly specify the path to save and load
    the blobs enabling model caching feature.

    @param dir Path to the directory what will be used as cache.
    @return reference to this parameter structure.
    */
    GAPI_WRAP
    OpenVINO& cfgCacheDir(const std::string &dir) {
        if (!params_map.empty()) {
            cv::util::throw_error(std::logic_error("ep::OpenVINO cannot be changed if"
                                                   "created from the parameters map."));
        }
        cache_dir = dir;
        return *this;
    }

    /** @brief Specifies OpenVINO Execution Provider number of threads.

    This function is used to override the accelerator default value
    of number of threads with this value at runtime.

    @param nthreads Number of threads.
    @return reference to this parameter structure.
    */
    GAPI_WRAP
    OpenVINO& cfgNumThreads(size_t nthreads) {
        if (!params_map.empty()) {
            cv::util::throw_error(std::logic_error("ep::OpenVINO cannot be changed if"
                                                   "created from the parameters map."));
        }
        num_of_threads = nthreads;
        return *this;
    }

    /** @brief Enables OpenVINO Execution Provider opencl throttling.

    This function is used to enable OpenCL queue throttling for GPU devices
    (reduces CPU utilization when using GPU).

    @return reference to this parameter structure.
    */
    GAPI_WRAP
    OpenVINO& cfgEnableOpenCLThrottling() {
        if (!params_map.empty()) {
            cv::util::throw_error(std::logic_error("ep::OpenVINO cannot be changed if"
                                                   "created from the parameters map."));
        }
        enable_opencl_throttling = true;
        return *this;
    }

    /** @brief Enables OpenVINO Execution Provider dynamic shapes.

    This function is used to enable OpenCL queue throttling for GPU devices
    (reduces CPU utilization when using GPU).
    This function is used to enable work with dynamic shaped models
    whose shape will be set dynamically based on the infer input
    image/data shape at run time in CPU.

    @return reference to this parameter structure.
    */
    GAPI_WRAP
    OpenVINO& cfgEnableDynamicShapes() {
        if (!params_map.empty()) {
            cv::util::throw_error(std::logic_error("ep::OpenVINO cannot be changed if"
                                                   "created from the parameters map."));
        }
        enable_dynamic_shapes = true;
        return *this;
    }

    std::string device_type;
    std::string cache_dir;
    size_t num_of_threads = 0;
    bool enable_opencl_throttling = false;
    bool enable_dynamic_shapes = false;
    std::map<std::string, std::string> params_map;
};

/**
 * @brief This structure provides functions
 * that fill inference options for ONNX DirectML Execution Provider.
 * Please follow https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#directml-execution-provider
 */
class GAPI_EXPORTS_W_SIMPLE DirectML {
public:
    // NB: Used from python.
    /// @private -- Exclude this constructor from OpenCV documentation
    GAPI_WRAP
    DirectML() = default;

    /** @brief Class constructor.

    Constructs DirectML parameters based on device id.

    @param device_id Target device id to use. ("0", "1", etc)
    */
    GAPI_WRAP
    explicit DirectML(const int device_id) : ddesc(device_id) { };

    /** @brief Class constructor.

    Constructs DirectML parameters based on adapter name.

    @param adapter_name Target adapter_name to use.
    */
    GAPI_WRAP
    explicit DirectML(const std::string &adapter_name) : ddesc(adapter_name) { };

    using DeviceDesc = cv::util::variant<int, std::string>;
    DeviceDesc ddesc;
};

using EP = cv::util::variant< cv::util::monostate
                            , OpenVINO
                            , DirectML
                            , CoreML
                            , CUDA
                            , TensorRT>;

} // namespace ep

GAPI_EXPORTS cv::gapi::GBackend backend();

enum class TraitAs: int {
    TENSOR, //!< G-API traits an associated cv::Mat as a raw tensor
            // and passes dimensions as-is
    IMAGE   //!< G-API traits an associated cv::Mat as an image so
            // creates an "image" blob (NCHW/NHWC, etc)
};

using PostProc = std::function<void(const std::unordered_map<std::string, cv::Mat> &,
                                          std::unordered_map<std::string, cv::Mat> &)>;

namespace detail {
/**
* @brief This structure contains description of inference parameters
* which is specific to ONNX models.
*/
struct ParamDesc {
    std::string model_path; //!< Path to model.

    // NB: nun_* may differ from topology's real input/output port numbers
    // (e.g. topology's partial execution)
    std::size_t num_in;  //!< How many inputs are defined in the operation
    std::size_t num_out; //!< How many outputs are defined in the operation

    // NB: Here order follows the `Net` API
    std::vector<std::string> input_names; //!< Names of input network layers.
    std::vector<std::string> output_names; //!< Names of output network layers.

    using ConstInput = std::pair<cv::Mat, TraitAs>;
    std::unordered_map<std::string, ConstInput> const_inputs; //!< Map with pair of name of network layer and ConstInput which will be associated with this.

    std::vector<cv::Scalar> mean; //!< Mean values for preprocessing.
    std::vector<cv::Scalar> stdev; //!< Standard deviation values for preprocessing.

    std::vector<cv::GMatDesc> out_metas; //!< Out meta information about your output (type, dimension).
    PostProc custom_post_proc; //!< Post processing function.

    std::vector<bool> normalize; //!< Vector of bool values that enabled or disabled normalize of input data.

    std::vector<std::string> names_to_remap; //!< Names of output layers that will be processed in PostProc function.

    bool is_generic;

    // TODO: Needs to modify the rest of ParamDesc accordingly to support
    // both generic and non-generic options without duplication
    // (as it was done for the OV IE backend)
    // These values are pushed into the respective vector<> fields above
    // when the generic infer parameters are unpacked (see GONNXBackendImpl::unpackKernel)
    std::unordered_map<std::string, std::pair<cv::Scalar, cv::Scalar> > generic_mstd;
    std::unordered_map<std::string, bool> generic_norm;

    std::map<std::string, std::string> session_options;
    std::vector<cv::gapi::onnx::ep::EP> execution_providers;
    bool disable_mem_pattern;
    cv::util::optional<int> opt_level;
};
} // namespace detail

template<typename Net>
struct PortCfg {
    using In = std::array
        < std::string
        , std::tuple_size<typename Net::InArgs>::value >;
    using Out = std::array
        < std::string
        , std::tuple_size<typename Net::OutArgs>::value >;
    using NormCoefs = std::array
        < cv::Scalar
        , std::tuple_size<typename Net::InArgs>::value >;
    using Normalize = std::array
        < bool
        , std::tuple_size<typename Net::InArgs>::value >;
};

/**
 * Contains description of inference parameters and kit of functions that
 * fill this parameters.
 */
template<typename Net> class Params {
public:
    /** @brief Class constructor.

    Constructs Params based on model information and sets default values for other
    inference description parameters.

    @param model Path to model (.onnx file).
    */
    Params(const std::string &model) {
        desc.model_path = model;
        desc.num_in  = std::tuple_size<typename Net::InArgs>::value;
        desc.num_out = std::tuple_size<typename Net::OutArgs>::value;
        desc.is_generic = false;
        desc.disable_mem_pattern = false;
    }

    /** @brief Specifies sequence of network input layers names for inference.

    The function is used to associate data of graph inputs with input layers of
    network topology. Number of names has to match the number of network inputs. If a network
    has only one input layer, there is no need to call it as the layer is
    associated with input automatically but this doesn't prevent you from
    doing it yourself. Count of names has to match to number of network inputs.

    @param layer_names std::array<std::string, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Contains names of input layers.
    @return the reference on modified object.
    */
    Params<Net>& cfgInputLayers(const typename PortCfg<Net>::In &layer_names) {
        desc.input_names.assign(layer_names.begin(), layer_names.end());
        return *this;
    }

    /** @brief Specifies sequence of output layers names for inference.

     The function is used to associate data of graph outputs with output layers of
    network topology. If a network has only one output layer, there is no need to call it
    as the layer is associated with output automatically but this doesn't prevent
    you from doing it yourself. Count of names has to match to number of network
    outputs or you can set your own output but for this case you have to
    additionally use @ref cfgPostProc function.

    @param layer_names std::array<std::string, N> where N is the number of outputs
    as defined in the @ref G_API_NET. Contains names of output layers.
    @return the reference on modified object.
    */
    Params<Net>& cfgOutputLayers(const typename PortCfg<Net>::Out &layer_names) {
        desc.output_names.assign(layer_names.begin(), layer_names.end());
        return *this;
    }

    /** @brief Sets a constant input.

    The function is used to set constant input. This input has to be
    a prepared tensor since preprocessing is disabled for this case. You should
    provide name of network layer which will receive provided data.

    @param layer_name Name of network layer.
    @param data cv::Mat that contains data which will be associated with network layer.
    @param hint Type of input (TENSOR).
    @return the reference on modified object.
    */
    Params<Net>& constInput(const std::string &layer_name,
                            const cv::Mat &data,
                            TraitAs hint = TraitAs::TENSOR) {
        desc.const_inputs[layer_name] = {data, hint};
        return *this;
    }

    /** @brief Specifies mean value and standard deviation for preprocessing.

    The function is used to set mean value and standard deviation for preprocessing
    of input data.

    @param m std::array<cv::Scalar, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Contains mean values.
    @param s std::array<cv::Scalar, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Contains standard deviation values.
    @return the reference on modified object.
    */
    Params<Net>& cfgMeanStd(const typename PortCfg<Net>::NormCoefs &m,
                            const typename PortCfg<Net>::NormCoefs &s) {
        desc.mean.assign(m.begin(), m.end());
        desc.stdev.assign(s.begin(), s.end());
        return *this;
    }

    /** @brief Configures graph output and provides the post processing function from user.

    The function is used when you work with networks with dynamic outputs.
    Since we can't know dimensions of inference result needs provide them for
    construction of graph output. This dimensions can differ from inference result.
    So you have to provide @ref PostProc function that gets information from inference
    result and fill output which is constructed by dimensions from out_metas.

    @param out_metas Out meta information about your output (type, dimension).
    @param remap_function Post processing function, which has two parameters. First is onnx
    result, second is graph output. Both parameters is std::map that contain pair of
    layer's name and cv::Mat.
    @return the reference on modified object.
    */
    Params<Net>& cfgPostProc(const std::vector<cv::GMatDesc> &out_metas,
                             const PostProc &remap_function) {
        desc.out_metas        = out_metas;
        desc.custom_post_proc = remap_function;
        return *this;
    }

    /** @overload
    Function with a rvalue parameters.

    @param out_metas rvalue out meta information about your output (type, dimension).
    @param remap_function rvalue post processing function, which has two parameters. First is onnx
    result, second is graph output. Both parameters is std::map that contain pair of
    layer's name and cv::Mat.
    @return the reference on modified object.
    */
    Params<Net>& cfgPostProc(std::vector<cv::GMatDesc> &&out_metas,
                             PostProc &&remap_function) {
        desc.out_metas        = std::move(out_metas);
        desc.custom_post_proc = std::move(remap_function);
        return *this;
    }

    /** @overload
    The function has additional parameter names_to_remap. This parameter provides
    information about output layers which will be used for inference and post
    processing function.

    @param out_metas Out meta information.
    @param remap_function Post processing function.
    @param names_to_remap Names of output layers. network's inference will
    be done on these layers. Inference's result will be processed in post processing
    function using these names.
    @return the reference on modified object.
    */
    Params<Net>& cfgPostProc(const std::vector<cv::GMatDesc> &out_metas,
                             const PostProc &remap_function,
                             const std::vector<std::string> &names_to_remap) {
        desc.out_metas        = out_metas;
        desc.custom_post_proc = remap_function;
        desc.names_to_remap   = names_to_remap;
        return *this;
    }

    /** @overload
    Function with a rvalue parameters and additional parameter names_to_remap.

    @param out_metas rvalue out meta information.
    @param remap_function rvalue post processing function.
    @param names_to_remap rvalue names of output layers. network's inference will
    be done on these layers. Inference's result will be processed in post processing
    function using these names.
    @return the reference on modified object.
    */
    Params<Net>& cfgPostProc(std::vector<cv::GMatDesc> &&out_metas,
                             PostProc &&remap_function,
                             std::vector<std::string> &&names_to_remap) {
        desc.out_metas        = std::move(out_metas);
        desc.custom_post_proc = std::move(remap_function);
        desc.names_to_remap   = std::move(names_to_remap);
        return *this;
    }

    /** @brief Specifies normalize parameter for preprocessing.

    The function is used to set normalize parameter for preprocessing of input data.

    @param normalizations std::array<cv::Scalar, N> where N is the number of inputs
    as defined in the @ref G_API_NET. Сontains bool values that enabled or disabled
    normalize of input data.
    @return the reference on modified object.
    */
    Params<Net>& cfgNormalize(const typename PortCfg<Net>::Normalize &normalizations) {
        desc.normalize.assign(normalizations.begin(), normalizations.end());
        return *this;
    }

    /** @brief Adds execution provider for runtime.

    The function is used to add ONNX Runtime OpenVINO Execution Provider options.

    @param ep OpenVINO Execution Provider options.
    @see cv::gapi::onnx::ep::OpenVINO.

    @return the reference on modified object.
    */
    Params<Net>& cfgAddExecutionProvider(ep::OpenVINO&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
        return *this;
    }

    /** @brief Adds execution provider for runtime.

    The function is used to add ONNX Runtime DirectML Execution Provider options.

    @param ep DirectML Execution Provider options.
    @see cv::gapi::onnx::ep::DirectML.

    @return the reference on modified object.
    */
    Params<Net>& cfgAddExecutionProvider(ep::DirectML&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
        return *this;
    }

    /** @brief Adds execution provider for runtime.

    The function is used to add ONNX Runtime CoreML Execution Provider options.

    @param ep CoreML Execution Provider options.
    @see cv::gapi::onnx::ep::CoreML.

    @return the reference on modified object.
    */
    Params<Net>& cfgAddExecutionProvider(ep::CoreML&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
        return *this;
    }

    /** @brief Adds execution provider for runtime.

    The function is used to add ONNX Runtime CUDA Execution Provider options.

    @param ep CUDA Execution Provider options.
    @see cv::gapi::onnx::ep::CUDA.

    @return the reference on modified object.
    */
    Params<Net>& cfgAddExecutionProvider(ep::CUDA&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
        return *this;
    }

    /** @brief Adds execution provider for runtime.

    The function is used to add ONNX Runtime TensorRT Execution Provider options.

    @param ep TensorRT Execution Provider options.
    @see cv::gapi::onnx::ep::TensorRT.

    @return the reference on modified object.
    */
    Params<Net>& cfgAddExecutionProvider(ep::TensorRT&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
        return *this;
    }

    /** @brief Disables the memory pattern optimization.

    @return the reference on modified object.
    */
    Params<Net>& cfgDisableMemPattern() {
        desc.disable_mem_pattern = true;
        return *this;
    }

    /** @brief Configures session options for ONNX Runtime.

    This function is used to set various session options for the ONNX Runtime
    session by accepting a map of key-value pairs.

    @param options A map of session option to be applied to the ONNX Runtime session.
    @return the reference on modified object.
    */
    Params<Net>& cfgSessionOptions(const std::map<std::string, std::string>& options) {
        desc.session_options.insert(options.begin(), options.end());
        return *this;
    }

    /** @brief Configures optimization level for ONNX Runtime.

    @param opt_level [optimization level]: Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).
    Please see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels.
    @return the reference on modified object.
    */
    Params<Net>& cfgOptLevel(const int opt_level) {
        desc.opt_level = cv::util::make_optional(opt_level);
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::onnx::backend(); }
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

    Constructs Params based on input information and sets default values for other
    inference description parameters.

    @param tag string tag of the network for which these parameters are intended.
    @param model_path path to model file (.onnx file).
    */
    Params(const std::string& tag, const std::string& model_path)
        : desc{ model_path, 0u, 0u, {}, {}, {}, {}, {}, {}, {}, {}, {}, true, {}, {}, {}, {}, false, {} }, m_tag(tag) {}

    /** @see onnx::Params::cfgMeanStdDev. */
    void cfgMeanStdDev(const std::string &layer,
                       const cv::Scalar &m,
                       const cv::Scalar &s) {
        desc.generic_mstd[layer] = std::make_pair(m, s);
    }

    /** @see onnx::Params::cfgNormalize. */
    void cfgNormalize(const std::string &layer, bool flag) {
        desc.generic_norm[layer] = flag;
    }

    /** @see onnx::Params::cfgAddExecutionProvider. */
    void cfgAddExecutionProvider(ep::OpenVINO&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
    }

    /** @see onnx::Params::cfgAddExecutionProvider. */
    void cfgAddExecutionProvider(ep::DirectML&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
    }

    /** @see onnx::Params::cfgAddExecutionProvider. */
    void cfgAddExecutionProvider(ep::CoreML&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
    }

    /** @see onnx::Params::cfgAddExecutionProvider. */
    void cfgAddExecutionProvider(ep::CUDA&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
    }

    /** @see onnx::Params::cfgAddExecutionProvider. */
    void cfgAddExecutionProvider(ep::TensorRT&& ep) {
        desc.execution_providers.emplace_back(std::move(ep));
    }

    /** @see onnx::Params::cfgDisableMemPattern. */
    void cfgDisableMemPattern() {
        desc.disable_mem_pattern = true;
    }

    /** @see onnx::Params::cfgSessionOptions. */
    void cfgSessionOptions(const std::map<std::string, std::string>& options) {
        desc.session_options.insert(options.begin(), options.end());
    }

/** @see onnx::Params::cfgOptLevel. */
    void cfgOptLevel(const int opt_level) {
        desc.opt_level = cv::util::make_optional(opt_level);
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::onnx::backend(); }
    std::string   tag()     const { return m_tag; }
    cv::util::any params()  const { return { desc }; }
    // END(G-API's network parametrization API)
protected:
    detail::ParamDesc desc;
    std::string m_tag;
};

class WorkloadTypeONNX : public WorkloadType {};
using WorkloadTypeOnnxPtr = std::shared_ptr<cv::gapi::onnx::WorkloadTypeONNX>;

} // namespace onnx
} // namespace gapi
namespace detail {
template<> struct CompileArgTag<std::shared_ptr<cv::gapi::onnx::WorkloadTypeONNX>> {
    static const char* tag() { return "gapi.wip.ov.workload_type_onnx_ptr"; }
};
} // namespace detail
} // namespace cv

#endif // OPENCV_GAPI_INFER_HPP
