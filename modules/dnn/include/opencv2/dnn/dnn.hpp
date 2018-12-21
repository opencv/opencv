/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_DNN_DNN_HPP
#define OPENCV_DNN_DNN_HPP

#include <vector>
#include <opencv2/core.hpp>

#if !defined CV_DOXYGEN && !defined CV_DNN_DONT_ADD_EXPERIMENTAL_NS
#define CV__DNN_EXPERIMENTAL_NS_BEGIN namespace experimental_dnn_34_v11 {
#define CV__DNN_EXPERIMENTAL_NS_END }
namespace cv { namespace dnn { namespace experimental_dnn_34_v11 { } using namespace experimental_dnn_34_v11; }}
#else
#define CV__DNN_EXPERIMENTAL_NS_BEGIN
#define CV__DNN_EXPERIMENTAL_NS_END
#endif

#include <opencv2/dnn/dict.hpp>

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN
//! @addtogroup dnn
//! @{

    typedef std::vector<int> MatShape;

    /**
     * @brief Enum of computation backends supported by layers.
     * @see Net::setPreferableBackend
     */
    enum Backend
    {
        //! DNN_BACKEND_DEFAULT equals to DNN_BACKEND_INFERENCE_ENGINE if
        //! OpenCV is built with Intel's Inference Engine library or
        //! DNN_BACKEND_OPENCV otherwise.
        DNN_BACKEND_DEFAULT,
        DNN_BACKEND_HALIDE,
        DNN_BACKEND_INFERENCE_ENGINE,
        DNN_BACKEND_OPENCV
    };

    /**
     * @brief Enum of target devices for computations.
     * @see Net::setPreferableTarget
     */
    enum Target
    {
        DNN_TARGET_CPU,
        DNN_TARGET_OPENCL,
        DNN_TARGET_OPENCL_FP16,
        DNN_TARGET_MYRIAD,
        //! FPGA device with CPU fallbacks using Inference Engine's Heterogeneous plugin.
        DNN_TARGET_FPGA
    };

    CV_EXPORTS std::vector< std::pair<Backend, Target> > getAvailableBackends();
    CV_EXPORTS std::vector<Target> getAvailableTargets(Backend be);

    /** @brief This class provides all data needed to initialize layer.
     *
     * It includes dictionary with scalar params (which can be read by using Dict interface),
     * blob params #blobs and optional meta information: #name and #type of layer instance.
    */
    class CV_EXPORTS LayerParams : public Dict
    {
    public:
        //TODO: Add ability to name blob params
        std::vector<Mat> blobs; //!< List of learned parameters stored as blobs.

        String name; //!< Name of the layer instance (optional, can be used internal purposes).
        String type; //!< Type name which was used for creating layer by layer factory (optional).
    };

   /**
    * @brief Derivatives of this class encapsulates functions of certain backends.
    */
    class BackendNode
    {
    public:
        BackendNode(int backendId);

        virtual ~BackendNode(); //!< Virtual destructor to make polymorphism.

        int backendId; //!< Backend identifier.
    };

    /**
     * @brief Derivatives of this class wraps cv::Mat for different backends and targets.
     */
    class BackendWrapper
    {
    public:
        BackendWrapper(int backendId, int targetId);

        /**
         * @brief Wrap cv::Mat for specific backend and target.
         * @param[in] targetId Target identifier.
         * @param[in] m cv::Mat for wrapping.
         *
         * Make CPU->GPU data transfer if it's require for the target.
         */
        BackendWrapper(int targetId, const cv::Mat& m);

        /**
         * @brief Make wrapper for reused cv::Mat.
         * @param[in] base Wrapper of cv::Mat that will be reused.
         * @param[in] shape Specific shape.
         *
         * Initialize wrapper from another one. It'll wrap the same host CPU
         * memory and mustn't allocate memory on device(i.e. GPU). It might
         * has different shape. Use in case of CPU memory reusing for reuse
         * associated memory on device too.
         */
        BackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape);

        virtual ~BackendWrapper(); //!< Virtual destructor to make polymorphism.

        /**
         * @brief Transfer data to CPU host memory.
         */
        virtual void copyToHost() = 0;

        /**
         * @brief Indicate that an actual data is on CPU.
         */
        virtual void setHostDirty() = 0;

        int backendId;  //!< Backend identifier.
        int targetId;   //!< Target identifier.
    };

    class CV_EXPORTS ActivationLayer;

    /** @brief This interface class allows to build new Layers - are building blocks of networks.
     *
     * Each class, derived from Layer, must implement allocate() methods to declare own outputs and forward() to compute outputs.
     * Also before using the new layer into networks you must register your layer by using one of @ref dnnLayerFactory "LayerFactory" macros.
     */
    class CV_EXPORTS_W Layer : public Algorithm
    {
    public:

        //! List of learned parameters must be stored here to allow read them by using Net::getParam().
        CV_PROP_RW std::vector<Mat> blobs;

        /** @brief Computes and sets internal parameters according to inputs, outputs and blobs.
         *  @deprecated Use Layer::finalize(InputArrayOfArrays, OutputArrayOfArrays) instead
         *  @param[in]  input  vector of already allocated input blobs
         *  @param[out] output vector of already allocated output blobs
         *
         * If this method is called after network has allocated all memory for input and output blobs
         * and before inferencing.
         */
        CV_DEPRECATED_EXTERNAL
        virtual void finalize(const std::vector<Mat*> &input, std::vector<Mat> &output);

        /** @brief Computes and sets internal parameters according to inputs, outputs and blobs.
         *  @param[in]  inputs  vector of already allocated input blobs
         *  @param[out] outputs vector of already allocated output blobs
         *
         * If this method is called after network has allocated all memory for input and output blobs
         * and before inferencing.
         */
        CV_WRAP virtual void finalize(InputArrayOfArrays inputs, OutputArrayOfArrays outputs);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @deprecated Use Layer::forward(InputArrayOfArrays, OutputArrayOfArrays, OutputArrayOfArrays) instead
         *  @param[in]  input  the input blobs.
         *  @param[out] output allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        CV_DEPRECATED_EXTERNAL
        virtual void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @param[in]  inputs  the input blobs.
         *  @param[out] outputs allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @param[in]  inputs  the input blobs.
         *  @param[out] outputs allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        void forward_fallback(InputArrayOfArrays inputs, OutputArrayOfArrays outputs, OutputArrayOfArrays internals);

        /** @brief
         * @overload
         * @deprecated Use Layer::finalize(InputArrayOfArrays, OutputArrayOfArrays) instead
         */
        CV_DEPRECATED_EXTERNAL
        void finalize(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs);

        /** @brief
         * @overload
         * @deprecated Use Layer::finalize(InputArrayOfArrays, OutputArrayOfArrays) instead
         */
        CV_DEPRECATED std::vector<Mat> finalize(const std::vector<Mat> &inputs);

        /** @brief Allocates layer and computes output.
         *  @deprecated This method will be removed in the future release.
         */
        CV_DEPRECATED CV_WRAP void run(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs,
                                       CV_IN_OUT std::vector<Mat> &internals);

        /** @brief Returns index of input blob into the input array.
         *  @param inputName label of input blob
         *
         * Each layer input and output can be labeled to easily identify them using "%<layer_name%>[.output_name]" notation.
         * This method maps label of input blob to its index into input vector.
         */
        virtual int inputNameToIndex(String inputName);
        /** @brief Returns index of output blob in output array.
         *  @see inputNameToIndex()
         */
        CV_WRAP virtual int outputNameToIndex(const String& outputName);

        /**
         * @brief Ask layer if it support specific backend for doing computations.
         * @param[in] backendId computation backend identifier.
         * @see Backend
         */
        virtual bool supportBackend(int backendId);

        /**
         * @brief Returns Halide backend node.
         * @param[in] inputs Input Halide buffers.
         * @see BackendNode, BackendWrapper
         *
         * Input buffers should be exactly the same that will be used in forward invocations.
         * Despite we can use Halide::ImageParam based on input shape only,
         * it helps prevent some memory management issues (if something wrong,
         * Halide tests will be failed).
         */
        virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs);

        virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> > &inputs);

       /**
        * @brief Automatic Halide scheduling based on layer hyper-parameters.
        * @param[in] node Backend node with Halide functions.
        * @param[in] inputs Blobs that will be used in forward invocations.
        * @param[in] outputs Blobs that will be used in forward invocations.
        * @param[in] targetId Target identifier
        * @see BackendNode, Target
        *
        * Layer don't use own Halide::Func members because we can have applied
        * layers fusing. In this way the fused function should be scheduled.
        */
        virtual void applyHalideScheduler(Ptr<BackendNode>& node,
                                          const std::vector<Mat*> &inputs,
                                          const std::vector<Mat> &outputs,
                                          int targetId) const;

        /**
         * @brief Implement layers fusing.
         * @param[in] node Backend node of bottom layer.
         * @see BackendNode
         *
         * Actual for graph-based backends. If layer attached successfully,
         * returns non-empty cv::Ptr to node of the same backend.
         * Fuse only over the last function.
         */
        virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node);

        /**
         * @brief Tries to attach to the layer the subsequent activation layer, i.e. do the layer fusion in a partial case.
         * @param[in] layer The subsequent activation layer.
         *
         * Returns true if the activation layer has been attached successfully.
         */
        virtual bool setActivation(const Ptr<ActivationLayer>& layer);

        /**
         * @brief Try to fuse current layer with a next one
         * @param[in] top Next layer to be fused.
         * @returns True if fusion was performed.
         */
        virtual bool tryFuse(Ptr<Layer>& top);

        /**
         * @brief Returns parameters of layers with channel-wise multiplication and addition.
         * @param[out] scale Channel-wise multipliers. Total number of values should
         *                   be equal to number of channels.
         * @param[out] shift Channel-wise offsets. Total number of values should
         *                   be equal to number of channels.
         *
         * Some layers can fuse their transformations with further layers.
         * In example, convolution + batch normalization. This way base layer
         * use weights from layer after it. Fused layer is skipped.
         * By default, @p scale and @p shift are empty that means layer has no
         * element-wise multiplications or additions.
         */
        virtual void getScaleShift(Mat& scale, Mat& shift) const;

        /**
         * @brief "Deattaches" all the layers, attached to particular layer.
         */
        virtual void unsetAttached();

        virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                     const int requiredOutputs,
                                     std::vector<MatShape> &outputs,
                                     std::vector<MatShape> &internals) const;
        virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                               const std::vector<MatShape> &outputs) const {CV_UNUSED(inputs); CV_UNUSED(outputs); return 0;}

        CV_PROP String name; //!< Name of the layer instance, can be used for logging or other internal purposes.
        CV_PROP String type; //!< Type name which was used for creating layer by layer factory.
        CV_PROP int preferableTarget; //!< prefer target for layer forwarding

        Layer();
        explicit Layer(const LayerParams &params);      //!< Initializes only #name, #type and #blobs fields.
        void setParamsFrom(const LayerParams &params);  //!< Initializes only #name, #type and #blobs fields.
        virtual ~Layer();
    };

    /** @brief This class allows to create and manipulate comprehensive artificial neural networks.
     *
     * Neural network is presented as directed acyclic graph (DAG), where vertices are Layer instances,
     * and edges specify relationships between layers inputs and outputs.
     *
     * Each network layer has unique integer id and unique string name inside its network.
     * LayerId can store either layer name or layer id.
     *
     * This class supports reference counting of its instances, i. e. copies point to the same instance.
     */
    class CV_EXPORTS_W_SIMPLE Net
    {
    public:

        CV_WRAP Net();  //!< Default constructor.
        CV_WRAP ~Net(); //!< Destructor frees the net only if there aren't references to the net anymore.

        /** @brief Create a network from Intel's Model Optimizer intermediate representation.
         *  @param[in] xml XML configuration file with network's topology.
         *  @param[in] bin Binary file with trained weights.
         *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
         *  backend.
         */
        CV_WRAP static Net readFromModelOptimizer(const String& xml, const String& bin);

        /** Returns true if there are no layers in the network. */
        CV_WRAP bool empty() const;

        /** @brief Adds new layer to the net.
         *  @param name   unique name of the adding layer.
         *  @param type   typename of the adding layer (type must be registered in LayerRegister).
         *  @param params parameters which will be used to initialize the creating layer.
         *  @returns unique identifier of created layer, or -1 if a failure will happen.
         */
        int addLayer(const String &name, const String &type, LayerParams &params);
        /** @brief Adds new layer and connects its first input to the first output of previously added layer.
         *  @see addLayer()
         */
        int addLayerToPrev(const String &name, const String &type, LayerParams &params);

        /** @brief Converts string name of the layer to the integer identifier.
         *  @returns id of the layer, or -1 if the layer wasn't found.
         */
        CV_WRAP int getLayerId(const String &layer);

        CV_WRAP std::vector<String> getLayerNames() const;

        /** @brief Container for strings and integers. */
        typedef DictValue LayerId;

        /** @brief Returns pointer to layer with specified id or name which the network use. */
        CV_WRAP Ptr<Layer> getLayer(LayerId layerId);

        /** @brief Returns pointers to input layers of specific layer. */
        std::vector<Ptr<Layer> > getLayerInputs(LayerId layerId); // FIXIT: CV_WRAP

        /** @brief Connects output of the first layer to input of the second layer.
         *  @param outPin descriptor of the first layer output.
         *  @param inpPin descriptor of the second layer input.
         *
         * Descriptors have the following template <DFN>&lt;layer_name&gt;[.input_number]</DFN>:
         * - the first part of the template <DFN>layer_name</DFN> is sting name of the added layer.
         *   If this part is empty then the network input pseudo layer will be used;
         * - the second optional part of the template <DFN>input_number</DFN>
         *   is either number of the layer input, either label one.
         *   If this part is omitted then the first layer input will be used.
         *
         *  @see setNetInputs(), Layer::inputNameToIndex(), Layer::outputNameToIndex()
         */
        CV_WRAP void connect(String outPin, String inpPin);

        /** @brief Connects #@p outNum output of the first layer to #@p inNum input of the second layer.
         *  @param outLayerId identifier of the first layer
         *  @param outNum number of the first layer output
         *  @param inpLayerId identifier of the second layer
         *  @param inpNum number of the second layer input
         */
        void connect(int outLayerId, int outNum, int inpLayerId, int inpNum);

        /** @brief Sets outputs names of the network input pseudo layer.
         *
         * Each net always has special own the network input pseudo layer with id=0.
         * This layer stores the user blobs only and don't make any computations.
         * In fact, this layer provides the only way to pass user data into the network.
         * As any other layer, this layer can label its outputs and this function provides an easy way to do this.
         */
        CV_WRAP void setInputsNames(const std::vector<String> &inputBlobNames);

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputName name for layer which output is needed to get
         *  @return blob for first output of specified layer.
         *  @details By default runs forward pass for the whole network.
         */
        CV_WRAP Mat forward(const String& outputName = String());

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputBlobs contains all output blobs for specified layer.
         *  @param outputName name for layer which output is needed to get
         *  @details If @p outputName is empty, runs forward pass for the whole network.
         */
        CV_WRAP void forward(OutputArrayOfArrays outputBlobs, const String& outputName = String());

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains blobs for first outputs of specified layers.
         *  @param outBlobNames names for layers which outputs are needed to get
         */
        CV_WRAP void forward(OutputArrayOfArrays outputBlobs,
                             const std::vector<String>& outBlobNames);

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains all output blobs for each layer specified in @p outBlobNames.
         *  @param outBlobNames names for layers which outputs are needed to get
         */
        CV_WRAP_AS(forwardAndRetrieve) void forward(CV_OUT std::vector<std::vector<Mat> >& outputBlobs,
                                                    const std::vector<String>& outBlobNames);

        /**
         * @brief Compile Halide layers.
         * @param[in] scheduler Path to YAML file with scheduling directives.
         * @see setPreferableBackend
         *
         * Schedule layers that support Halide backend. Then compile them for
         * specific target. For layers that not represented in scheduling file
         * or if no manual scheduling used at all, automatic scheduling will be applied.
         */
        CV_WRAP void setHalideScheduler(const String& scheduler);

        /**
         * @brief Ask network to use specific computation backend where it supported.
         * @param[in] backendId backend identifier.
         * @see Backend
         *
         * If OpenCV is compiled with Intel's Inference Engine library, DNN_BACKEND_DEFAULT
         * means DNN_BACKEND_INFERENCE_ENGINE. Otherwise it equals to DNN_BACKEND_OPENCV.
         */
        CV_WRAP void setPreferableBackend(int backendId);

        /**
         * @brief Ask network to make computations on specific target device.
         * @param[in] targetId target identifier.
         * @see Target
         *
         * List of supported combinations backend / target:
         * |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |
         * |------------------------|--------------------|------------------------------|--------------------|
         * | DNN_TARGET_CPU         |                  + |                            + |                  + |
         * | DNN_TARGET_OPENCL      |                  + |                            + |                  + |
         * | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |
         * | DNN_TARGET_MYRIAD      |                    |                            + |                    |
         * | DNN_TARGET_FPGA        |                    |                            + |                    |
         */
        CV_WRAP void setPreferableTarget(int targetId);

        /** @brief Sets the new input value for the network
         *  @param blob        A new blob. Should have CV_32F or CV_8U depth.
         *  @param name        A name of input layer.
         *  @param scalefactor An optional normalization scale.
         *  @param mean        An optional mean subtraction values.
         *  @see connect(String, String) to know format of the descriptor.
         *
         *  If scale or mean values are specified, a final input blob is computed
         *  as:
         * \f[input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\f]
         */
        CV_WRAP void setInput(InputArray blob, const String& name = "",
                              double scalefactor = 1.0, const Scalar& mean = Scalar());

        /** @brief Sets the new value for the learned param of the layer.
         *  @param layer name or id of the layer.
         *  @param numParam index of the layer parameter in the Layer::blobs array.
         *  @param blob the new value.
         *  @see Layer::blobs
         *  @note If shape of the new blob differs from the previous shape,
         *  then the following forward pass may fail.
        */
        CV_WRAP void setParam(LayerId layer, int numParam, const Mat &blob);

        /** @brief Returns parameter blob of the layer.
         *  @param layer name or id of the layer.
         *  @param numParam index of the layer parameter in the Layer::blobs array.
         *  @see Layer::blobs
         */
        CV_WRAP Mat getParam(LayerId layer, int numParam = 0);

        /** @brief Returns indexes of layers with unconnected outputs.
         */
        CV_WRAP std::vector<int> getUnconnectedOutLayers() const;

        /** @brief Returns names of layers with unconnected outputs.
         */
        CV_WRAP std::vector<String> getUnconnectedOutLayersNames() const;

        /** @brief Returns input and output shapes for all layers in loaded model;
         *  preliminary inferencing isn't necessary.
         *  @param netInputShapes shapes for all input blobs in net input layer.
         *  @param layersIds output parameter for layer IDs.
         *  @param inLayersShapes output parameter for input layers shapes;
         * order is the same as in layersIds
         *  @param outLayersShapes output parameter for output layers shapes;
         * order is the same as in layersIds
         */
        CV_WRAP void getLayersShapes(const std::vector<MatShape>& netInputShapes,
                                     CV_OUT std::vector<int>& layersIds,
                                     CV_OUT std::vector<std::vector<MatShape> >& inLayersShapes,
                                     CV_OUT std::vector<std::vector<MatShape> >& outLayersShapes) const;

        /** @overload */
        CV_WRAP void getLayersShapes(const MatShape& netInputShape,
                                     CV_OUT std::vector<int>& layersIds,
                                     CV_OUT std::vector<std::vector<MatShape> >& inLayersShapes,
                                     CV_OUT std::vector<std::vector<MatShape> >& outLayersShapes) const;

        /** @brief Returns input and output shapes for layer with specified
         * id in loaded model; preliminary inferencing isn't necessary.
         *  @param netInputShape shape input blob in net input layer.
         *  @param layerId id for layer.
         *  @param inLayerShapes output parameter for input layers shapes;
         * order is the same as in layersIds
         *  @param outLayerShapes output parameter for output layers shapes;
         * order is the same as in layersIds
         */
        void getLayerShapes(const MatShape& netInputShape,
                                    const int layerId,
                                    CV_OUT std::vector<MatShape>& inLayerShapes,
                                    CV_OUT std::vector<MatShape>& outLayerShapes) const; // FIXIT: CV_WRAP

        /** @overload */
        void getLayerShapes(const std::vector<MatShape>& netInputShapes,
                                    const int layerId,
                                    CV_OUT std::vector<MatShape>& inLayerShapes,
                                    CV_OUT std::vector<MatShape>& outLayerShapes) const; // FIXIT: CV_WRAP

        /** @brief Computes FLOP for whole loaded model with specified input shapes.
         * @param netInputShapes vector of shapes for all net inputs.
         * @returns computed FLOP.
         */
        CV_WRAP int64 getFLOPS(const std::vector<MatShape>& netInputShapes) const;
        /** @overload */
        CV_WRAP int64 getFLOPS(const MatShape& netInputShape) const;
        /** @overload */
        CV_WRAP int64 getFLOPS(const int layerId,
                               const std::vector<MatShape>& netInputShapes) const;
        /** @overload */
        CV_WRAP int64 getFLOPS(const int layerId,
                               const MatShape& netInputShape) const;

        /** @brief Returns list of types for layer used in model.
         * @param layersTypes output parameter for returning types.
         */
        CV_WRAP void getLayerTypes(CV_OUT std::vector<String>& layersTypes) const;

        /** @brief Returns count of layers of specified type.
         * @param layerType type.
         * @returns count of layers
         */
        CV_WRAP int getLayersCount(const String& layerType) const;

        /** @brief Computes bytes number which are required to store
         * all weights and intermediate blobs for model.
         * @param netInputShapes vector of shapes for all net inputs.
         * @param weights output parameter to store resulting bytes for weights.
         * @param blobs output parameter to store resulting bytes for intermediate blobs.
         */
        void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const; // FIXIT: CV_WRAP
        /** @overload */
        CV_WRAP void getMemoryConsumption(const MatShape& netInputShape,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const;
        /** @overload */
        CV_WRAP void getMemoryConsumption(const int layerId,
                                          const std::vector<MatShape>& netInputShapes,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const;
        /** @overload */
        CV_WRAP void getMemoryConsumption(const int layerId,
                                          const MatShape& netInputShape,
                                          CV_OUT size_t& weights, CV_OUT size_t& blobs) const;

        /** @brief Computes bytes number which are required to store
         * all weights and intermediate blobs for each layer.
         * @param netInputShapes vector of shapes for all net inputs.
         * @param layerIds output vector to save layer IDs.
         * @param weights output parameter to store resulting bytes for weights.
         * @param blobs output parameter to store resulting bytes for intermediate blobs.
         */
        void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                          CV_OUT std::vector<int>& layerIds,
                                          CV_OUT std::vector<size_t>& weights,
                                          CV_OUT std::vector<size_t>& blobs) const; // FIXIT: CV_WRAP
        /** @overload */
        void getMemoryConsumption(const MatShape& netInputShape,
                                          CV_OUT std::vector<int>& layerIds,
                                          CV_OUT std::vector<size_t>& weights,
                                          CV_OUT std::vector<size_t>& blobs) const; // FIXIT: CV_WRAP

        /** @brief Enables or disables layer fusion in the network.
         * @param fusion true to enable the fusion, false to disable. The fusion is enabled by default.
         */
        CV_WRAP void enableFusion(bool fusion);

        /** @brief Returns overall time for inference and timings (in ticks) for layers.
         * Indexes in returned vector correspond to layers ids. Some layers can be fused with others,
         * in this case zero ticks count will be return for that skipped layers.
         * @param timings vector for tick timings for all layers.
         * @return overall ticks for model inference.
         */
        CV_WRAP int64 getPerfProfile(CV_OUT std::vector<double>& timings);

    private:
        struct Impl;
        Ptr<Impl> impl;
    };

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
    *  @param cfgFile      path to the .cfg file with text description of the network architecture.
    *  @param darknetModel path to the .weights file with learned network.
    *  @returns Network object that ready to do forward, throw an exception in failure cases.
    *  @returns Net object.
    */
    CV_EXPORTS_W Net readNetFromDarknet(const String &cfgFile, const String &darknetModel = String());

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
     *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
     *  @param bufferModel A buffer contains a content of .weights file with learned network.
     *  @returns Net object.
     */
    CV_EXPORTS_W Net readNetFromDarknet(const std::vector<uchar>& bufferCfg,
                                        const std::vector<uchar>& bufferModel = std::vector<uchar>());

    /** @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
     *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
     *  @param lenCfg      Number of bytes to read from bufferCfg
     *  @param bufferModel A buffer contains a content of .weights file with learned network.
     *  @param lenModel    Number of bytes to read from bufferModel
     *  @returns Net object.
     */
    CV_EXPORTS Net readNetFromDarknet(const char *bufferCfg, size_t lenCfg,
                                      const char *bufferModel = NULL, size_t lenModel = 0);

    /** @brief Reads a network model stored in <a href="http://caffe.berkeleyvision.org">Caffe</a> framework's format.
      * @param prototxt   path to the .prototxt file with text description of the network architecture.
      * @param caffeModel path to the .caffemodel file with learned network.
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromCaffe(const String &prototxt, const String &caffeModel = String());

    /** @brief Reads a network model stored in Caffe model in memory.
      * @param bufferProto buffer containing the content of the .prototxt file
      * @param bufferModel buffer containing the content of the .caffemodel file
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromCaffe(const std::vector<uchar>& bufferProto,
                                      const std::vector<uchar>& bufferModel = std::vector<uchar>());

    /** @brief Reads a network model stored in Caffe model in memory.
      * @details This is an overloaded member function, provided for convenience.
      * It differs from the above function only in what argument(s) it accepts.
      * @param bufferProto buffer containing the content of the .prototxt file
      * @param lenProto length of bufferProto
      * @param bufferModel buffer containing the content of the .caffemodel file
      * @param lenModel length of bufferModel
      * @returns Net object.
      */
    CV_EXPORTS Net readNetFromCaffe(const char *bufferProto, size_t lenProto,
                                    const char *bufferModel = NULL, size_t lenModel = 0);

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
      * @param model  path to the .pb file with binary protobuf description of the network architecture
      * @param config path to the .pbtxt file that contains text graph definition in protobuf format.
      *               Resulting Net object is built by text graph using weights from a binary one that
      *               let us make it more flexible.
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromTensorflow(const String &model, const String &config = String());

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
      * @param bufferModel buffer containing the content of the pb file
      * @param bufferConfig buffer containing the content of the pbtxt file
      * @returns Net object.
      */
    CV_EXPORTS_W Net readNetFromTensorflow(const std::vector<uchar>& bufferModel,
                                           const std::vector<uchar>& bufferConfig = std::vector<uchar>());

    /** @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
      * @details This is an overloaded member function, provided for convenience.
      * It differs from the above function only in what argument(s) it accepts.
      * @param bufferModel buffer containing the content of the pb file
      * @param lenModel length of bufferModel
      * @param bufferConfig buffer containing the content of the pbtxt file
      * @param lenConfig length of bufferConfig
      */
    CV_EXPORTS Net readNetFromTensorflow(const char *bufferModel, size_t lenModel,
                                         const char *bufferConfig = NULL, size_t lenConfig = 0);

    /**
     *  @brief Reads a network model stored in <a href="http://torch.ch">Torch7</a> framework's format.
     *  @param model    path to the file, dumped from Torch by using torch.save() function.
     *  @param isBinary specifies whether the network was serialized in ascii mode or binary.
     *  @param evaluate specifies testing phase of network. If true, it's similar to evaluate() method in Torch.
     *  @returns Net object.
     *
     *  @note Ascii mode of Torch serializer is more preferable, because binary mode extensively use `long` type of C language,
     *  which has various bit-length on different systems.
     *
     * The loading file must contain serialized <a href="https://github.com/torch/nn/blob/master/doc/module.md">nn.Module</a> object
     * with importing network. Try to eliminate a custom objects from serialazing data to avoid importing errors.
     *
     * List of supported layers (i.e. object instances derived from Torch nn.Module class):
     * - nn.Sequential
     * - nn.Parallel
     * - nn.Concat
     * - nn.Linear
     * - nn.SpatialConvolution
     * - nn.SpatialMaxPooling, nn.SpatialAveragePooling
     * - nn.ReLU, nn.TanH, nn.Sigmoid
     * - nn.Reshape
     * - nn.SoftMax, nn.LogSoftMax
     *
     * Also some equivalents of these classes from cunn, cudnn, and fbcunn may be successfully imported.
     */
     CV_EXPORTS_W Net readNetFromTorch(const String &model, bool isBinary = true, bool evaluate = true);

     /**
      * @brief Read deep learning network represented in one of the supported formats.
      * @param[in] model Binary file contains trained weights. The following file
      *                  extensions are expected for models from different frameworks:
      *                  * `*.caffemodel` (Caffe, http://caffe.berkeleyvision.org/)
      *                  * `*.pb` (TensorFlow, https://www.tensorflow.org/)
      *                  * `*.t7` | `*.net` (Torch, http://torch.ch/)
      *                  * `*.weights` (Darknet, https://pjreddie.com/darknet/)
      *                  * `*.bin` (DLDT, https://software.intel.com/openvino-toolkit)
      * @param[in] config Text file contains network configuration. It could be a
      *                   file with the following extensions:
      *                  * `*.prototxt` (Caffe, http://caffe.berkeleyvision.org/)
      *                  * `*.pbtxt` (TensorFlow, https://www.tensorflow.org/)
      *                  * `*.cfg` (Darknet, https://pjreddie.com/darknet/)
      *                  * `*.xml` (DLDT, https://software.intel.com/openvino-toolkit)
      * @param[in] framework Explicit framework name tag to determine a format.
      * @returns Net object.
      *
      * This function automatically detects an origin framework of trained model
      * and calls an appropriate function such @ref readNetFromCaffe, @ref readNetFromTensorflow,
      * @ref readNetFromTorch or @ref readNetFromDarknet. An order of @p model and @p config
      * arguments does not matter.
      */
     CV_EXPORTS_W Net readNet(const String& model, const String& config = "", const String& framework = "");

     /**
      * @brief Read deep learning network represented in one of the supported formats.
      * @details This is an overloaded member function, provided for convenience.
      *          It differs from the above function only in what argument(s) it accepts.
      * @param[in] framework    Name of origin framework.
      * @param[in] bufferModel  A buffer with a content of binary file with weights
      * @param[in] bufferConfig A buffer with a content of text file contains network configuration.
      * @returns Net object.
      */
     CV_EXPORTS_W Net readNet(const String& framework, const std::vector<uchar>& bufferModel,
                              const std::vector<uchar>& bufferConfig = std::vector<uchar>());

    /** @brief Loads blob which was serialized as torch.Tensor object of Torch7 framework.
     *  @warning This function has the same limitations as readNetFromTorch().
     */
    CV_EXPORTS_W Mat readTorchBlob(const String &filename, bool isBinary = true);

    /** @brief Load a network from Intel's Model Optimizer intermediate representation.
     *  @param[in] xml XML configuration file with network's topology.
     *  @param[in] bin Binary file with trained weights.
     *  @returns Net object.
     *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
     *  backend.
     */
    CV_EXPORTS_W Net readNetFromModelOptimizer(const String &xml, const String &bin);

    /** @brief Reads a network model <a href="https://onnx.ai/">ONNX</a>.
     *  @param onnxFile path to the .onnx file with text description of the network architecture.
     *  @returns Network object that ready to do forward, throw an exception in failure cases.
     */
    CV_EXPORTS_W Net readNetFromONNX(const String &onnxFile);

    /** @brief Creates blob from .pb file.
     *  @param path to the .pb file with input tensor.
     *  @returns Mat.
     */
    CV_EXPORTS_W Mat readTensorFromONNX(const String& path);

    /** @brief Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,
     *  subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels.
     *  @param image input image (with 1-, 3- or 4-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p image values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @param crop flag which indicates whether image will be cropped after resize or not
     *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
     *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
     *  @returns 4-dimensional Mat with NCHW dimensions order.
     */
    CV_EXPORTS_W Mat blobFromImage(InputArray image, double scalefactor=1.0, const Size& size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                   int ddepth=CV_32F);

    /** @brief Creates 4-dimensional blob from image.
     *  @details This is an overloaded member function, provided for convenience.
     *           It differs from the above function only in what argument(s) it accepts.
     */
    CV_EXPORTS void blobFromImage(InputArray image, OutputArray blob, double scalefactor=1.0,
                                  const Size& size = Size(), const Scalar& mean = Scalar(),
                                  bool swapRB=false, bool crop=false, int ddepth=CV_32F);


    /** @brief Creates 4-dimensional blob from series of images. Optionally resizes and
     *  crops @p images from center, subtract @p mean values, scales values by @p scalefactor,
     *  swap Blue and Red channels.
     *  @param images input images (all with 1-, 3- or 4-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p images values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @param crop flag which indicates whether image will be cropped after resize or not
     *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
     *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
     *  @returns 4-dimensional Mat with NCHW dimensions order.
     */
    CV_EXPORTS_W Mat blobFromImages(InputArrayOfArrays images, double scalefactor=1.0,
                                    Size size = Size(), const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                    int ddepth=CV_32F);

    /** @brief Creates 4-dimensional blob from series of images.
     *  @details This is an overloaded member function, provided for convenience.
     *           It differs from the above function only in what argument(s) it accepts.
     */
    CV_EXPORTS void blobFromImages(InputArrayOfArrays images, OutputArray blob,
                                   double scalefactor=1.0, Size size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=false, bool crop=false,
                                   int ddepth=CV_32F);

    /** @brief Parse a 4D blob and output the images it contains as 2D arrays through a simpler data structure
     *  (std::vector<cv::Mat>).
     *  @param[in] blob_ 4 dimensional array (images, channels, height, width) in floating point precision (CV_32F) from
     *  which you would like to extract the images.
     *  @param[out] images_ array of 2D Mat containing the images extracted from the blob in floating point precision
     *  (CV_32F). They are non normalized neither mean added. The number of returned images equals the first dimension
     *  of the blob (batch size). Every image has a number of channels equals to the second dimension of the blob (depth).
     */
    CV_EXPORTS_W void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_);

    /** @brief Convert all weights of Caffe network to half precision floating point.
     * @param src Path to origin model from Caffe framework contains single
     *            precision floating point weights (usually has `.caffemodel` extension).
     * @param dst Path to destination model with updated weights.
     * @param layersTypes Set of layers types which parameters will be converted.
     *                    By default, converts only Convolutional and Fully-Connected layers'
     *                    weights.
     *
     * @note Shrinked model has no origin float32 weights so it can't be used
     *       in origin Caffe framework anymore. However the structure of data
     *       is taken from NVidia's Caffe fork: https://github.com/NVIDIA/caffe.
     *       So the resulting model may be used there.
     */
    CV_EXPORTS_W void shrinkCaffeModel(const String& src, const String& dst,
                                       const std::vector<String>& layersTypes = std::vector<String>());

    /** @brief Create a text representation for a binary network stored in protocol buffer format.
     *  @param[in] model  A path to binary network.
     *  @param[in] output A path to output text file to be created.
     *
     *  @note To reduce output file size, trained weights are not included.
     */
    CV_EXPORTS_W void writeTextGraph(const String& model, const String& output);

    /** @brief Performs non maximum suppression given boxes and corresponding scores.

     * @param bboxes a set of bounding boxes to apply NMS.
     * @param scores a set of corresponding confidences.
     * @param score_threshold a threshold used to filter boxes by score.
     * @param nms_threshold a threshold used in non maximum suppression.
     * @param indices the kept indices of bboxes after NMS.
     * @param eta a coefficient in adaptive threshold formula: \f$nms\_threshold_{i+1}=eta\cdot nms\_threshold_i\f$.
     * @param top_k if `>0`, keep at most @p top_k picked indices.
     */
    CV_EXPORTS_W void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                               const float score_threshold, const float nms_threshold,
                               CV_OUT std::vector<int>& indices,
                               const float eta = 1.f, const int top_k = 0);

    CV_EXPORTS_W void NMSBoxes(const std::vector<Rect2d>& bboxes, const std::vector<float>& scores,
                               const float score_threshold, const float nms_threshold,
                               CV_OUT std::vector<int>& indices,
                               const float eta = 1.f, const int top_k = 0);

    CV_EXPORTS_AS(NMSBoxesRotated) void NMSBoxes(const std::vector<RotatedRect>& bboxes, const std::vector<float>& scores,
                             const float score_threshold, const float nms_threshold,
                             CV_OUT std::vector<int>& indices,
                             const float eta = 1.f, const int top_k = 0);

    /** @brief Release a Myriad device is binded by OpenCV.
     *
     * Single Myriad device cannot be shared across multiple processes which uses
     * Inference Engine's Myriad plugin.
     */
    CV_EXPORTS_W void resetMyriadDevice();

//! @}
CV__DNN_EXPERIMENTAL_NS_END
}
}

#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/dnn.inl.hpp>

#endif  /* OPENCV_DNN_DNN_HPP */
