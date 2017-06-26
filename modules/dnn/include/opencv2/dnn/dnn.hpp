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
#include <opencv2/dnn/dict.hpp>

namespace cv
{
namespace dnn //! This namespace is used for dnn module functionlaity.
{
//! @addtogroup dnn
//! @{

    typedef std::vector<int> MatShape;

    /**
     * @brief Enum of computation backends supported by layers.
     */
    enum Backend
    {
        DNN_BACKEND_DEFAULT,
        DNN_BACKEND_HALIDE
    };

    /**
     * @brief Enum of target devices for computations.
     */
    enum Target
    {
        DNN_TARGET_CPU,
        DNN_TARGET_OPENCL
    };

    /** @brief Initialize dnn module and built-in layers.
     *
     * This function automatically called on most of OpenCV builds,
     * but you need to call it manually on some specific configurations (iOS for example).
     */
    CV_EXPORTS_W void initModule();

    /** @brief This class provides all data needed to initialize layer.
     *
     * It includes dictionary with scalar params (which can be readed by using Dict interface),
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
         * associented memory on device too.
         */
        BackendWrapper(const Ptr<BackendWrapper>& base, const MatShape& shape);

        virtual ~BackendWrapper(); //!< Virtual destructor to make polymorphism.

        /**
         * @brief Transfer data to CPU host memory.
         */
        virtual void copyToHost() = 0;

        int backendId;  //!< Backend identifier.
        int targetId;   //!< Target identifier.
    };

    /** @brief This interface class allows to build new Layers - are building blocks of networks.
     *
     * Each class, derived from Layer, must implement allocate() methods to declare own outputs and forward() to compute outputs.
     * Also before using the new layer into networks you must register your layer by using one of @ref dnnLayerFactory "LayerFactory" macros.
     */
    class CV_EXPORTS_W Layer
    {
    public:

        //! List of learned parameters must be stored here to allow read them by using Net::getParam().
        CV_PROP_RW std::vector<Mat> blobs;

        /** @brief Computes and sets internal parameters according to inputs, outputs and blobs.
         *  @param[in]  input  vector of already allocated input blobs
         *  @param[out] output vector of already allocated output blobs
         *
         * If this method is called after network has allocated all memory for input and output blobs
         * and before inferencing.
         */
        virtual void finalize(const std::vector<Mat*> &input, std::vector<Mat> &output);

        /** @brief Given the @p input blobs, computes the output @p blobs.
         *  @param[in]  input  the input blobs.
         *  @param[out] output allocated output blobs, which will store results of the computation.
         *  @param[out] internals allocated internal blobs
         */
        virtual void forward(std::vector<Mat*> &input, std::vector<Mat> &output, std::vector<Mat> &internals) = 0;

        /** @brief @overload */
        CV_WRAP void finalize(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs);

        /** @brief @overload */
        CV_WRAP std::vector<Mat> finalize(const std::vector<Mat> &inputs);

        /** @brief @overload */
        CV_WRAP void forward(const std::vector<Mat> &inputs, CV_IN_OUT std::vector<Mat> &outputs,
                             CV_IN_OUT std::vector<Mat> &internals);

        /** @brief Allocates layer and computes output. */
        CV_WRAP void run(const std::vector<Mat> &inputs, CV_OUT std::vector<Mat> &outputs,
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
        virtual int outputNameToIndex(String outputName);

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

        virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                     const int requiredOutputs,
                                     std::vector<MatShape> &outputs,
                                     std::vector<MatShape> &internals) const;
        virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                               const std::vector<MatShape> &outputs) const {(void)inputs; (void)outputs; return 0;}

        CV_PROP String name; //!< Name of the layer instance, can be used for logging or other internal purposes.
        CV_PROP String type; //!< Type name which was used for creating layer by layer factory.

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

        /** @brief Returns pointer to layer with specified name which the network use. */
        CV_WRAP Ptr<Layer> getLayer(LayerId layerId);

        /** @brief Returns pointers to input layers of specific layer. */
        CV_WRAP std::vector<Ptr<Layer> > getLayerInputs(LayerId layerId);

        /** @brief Delete layer for the network (not implemented yet) */
        CV_WRAP void deleteLayer(LayerId layer);

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
         *  @param inpLayerId identifier of the second layer
         *  @param outNum number of the first layer output
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
          * @details By default runs forward pass for the whole network.
          */
        CV_WRAP Mat forward(const String& outputName = String());

        /** @brief Runs forward pass to compute output of layer with name @p outputName.
         *  @param outputBlobs contains all output blobs for specified layer.
         *  @param outputName name for layer which output is needed to get
          * @details If @p outputName is empty, runs forward pass for the whole network.
          */
        CV_WRAP void forward(std::vector<Mat>& outputBlobs, const String& outputName = String());

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains blobs for first outputs of specified layers.
         *  @param outBlobNames names for layers which outputs are needed to get
          */
        CV_WRAP void forward(std::vector<Mat>& outputBlobs,
                             const std::vector<String>& outBlobNames);

        /** @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
         *  @param outputBlobs contains all output blobs for each layer specified in @p outBlobNames.
         *  @param outBlobNames names for layers which outputs are needed to get
          */
        CV_WRAP void forward(std::vector<std::vector<Mat> >& outputBlobs,
                             const std::vector<String>& outBlobNames);

        //TODO:
        /** @brief Optimized forward.
         *  @warning Not implemented yet.
         *  @details Makes forward only those layers which weren't changed after previous forward().
         */
        void forwardOpt(LayerId toLayer);
        /** @overload */
        void forwardOpt(const std::vector<LayerId> &toLayers);

        /**
         * @brief Compile Halide layers.
         * @param[in] scheduler Path to YAML file with scheduling directives.
         * @see setPreferableBackend
         *
         * Schedule layers that support Halide backend. Then compile them for
         * specific target. For layers that not represented in scheduling file
         * or if no manual scheduling used at all, automatic scheduling will be applied.
         */
        void setHalideScheduler(const String& scheduler);

        /**
         * @brief Ask network to use specific computation backend where it supported.
         * @param[in] backendId backend identifier.
         * @see Backend
         */
        void setPreferableBackend(int backendId);

        /**
         * @brief Ask network to make computations on specific target device.
         * @param[in] targetId target identifier.
         * @see Target
         */
        void setPreferableTarget(int targetId);

        /** @brief Sets the new value for the layer output blob
         *  @param name descriptor of the updating layer output blob.
         *  @param blob new blob.
         *  @see connect(String, String) to know format of the descriptor.
         *  @note If updating blob is not empty then @p blob must have the same shape,
         *  because network reshaping is not implemented yet.
         */
        CV_WRAP void setInput(const Mat &blob, const String& name = "");

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
                                      std::vector<int>* layersIds,
                                      std::vector<std::vector<MatShape> >* inLayersShapes,
                                      std::vector<std::vector<MatShape> >* outLayersShapes) const;

         /** @overload */
         CV_WRAP void getLayersShapes(const MatShape& netInputShape,
                                      std::vector<int>* layersIds,
                                      std::vector<std::vector<MatShape> >* inLayersShapes,
                                      std::vector<std::vector<MatShape> >* outLayersShapes) const;

         /** @brief Returns input and output shapes for layer with specified
          * id in loaded model; preliminary inferencing isn't necessary.
          *  @param netInputShape shape input blob in net input layer.
          *  @param layerId id for layer.
          *  @param inLayerShapes output parameter for input layers shapes;
          * order is the same as in layersIds
          *  @param outLayerShapes output parameter for output layers shapes;
          * order is the same as in layersIds
          */
         CV_WRAP void getLayerShapes(const MatShape& netInputShape,
                                     const int layerId,
                                     std::vector<MatShape>* inLayerShapes,
                                     std::vector<MatShape>* outLayerShapes) const;

         /** @overload */
         CV_WRAP void getLayerShapes(const std::vector<MatShape>& netInputShapes,
                                     const int layerId,
                                     std::vector<MatShape>* inLayerShapes,
                                     std::vector<MatShape>* outLayerShapes) const;
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
         CV_WRAP void getLayerTypes(std::vector<String>& layersTypes) const;

         /** @brief Returns count of layers of specified type.
          * @param layerType type.
          * @returns count of layers
          */
         CV_WRAP int getLayersCount(const String& layerType) const;

         /** @brief Computes bytes number which are requered to store
          * all weights and intermediate blobs for model.
          * @param netInputShapes vector of shapes for all net inputs.
          * @param weights output parameter to store resulting bytes for weights.
          * @param blobs output parameter to store resulting bytes for intermediate blobs.
          */
         CV_WRAP void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                           size_t& weights, size_t& blobs) const;
         /** @overload */
         CV_WRAP void getMemoryConsumption(const MatShape& netInputShape,
                                           size_t& weights, size_t& blobs) const;
         /** @overload */
         CV_WRAP void getMemoryConsumption(const int layerId,
                                           const std::vector<MatShape>& netInputShapes,
                                           size_t& weights, size_t& blobs) const;
         /** @overload */
         CV_WRAP void getMemoryConsumption(const int layerId,
                                           const MatShape& netInputShape,
                                           size_t& weights, size_t& blobs) const;

         /** @brief Computes bytes number which are requered to store
          * all weights and intermediate blobs for each layer.
          * @param netInputShapes vector of shapes for all net inputs.
          * @param layerIds output vector to save layer IDs.
          * @param weights output parameter to store resulting bytes for weights.
          * @param blobs output parameter to store resulting bytes for intermediate blobs.
          */
         CV_WRAP void getMemoryConsumption(const std::vector<MatShape>& netInputShapes,
                                           std::vector<int>& layerIds, std::vector<size_t>& weights,
                                           std::vector<size_t>& blobs) const;
         /** @overload */
         CV_WRAP void getMemoryConsumption(const MatShape& netInputShape,
                                           std::vector<int>& layerIds, std::vector<size_t>& weights,
                                           std::vector<size_t>& blobs) const;
    private:

        struct Impl;
        Ptr<Impl> impl;
    };

    /** @brief Small interface class for loading trained serialized models of different dnn-frameworks. */
    class CV_EXPORTS_W Importer
    {
    public:

        /** @brief Adds loaded layers into the @p net and sets connections between them. */
        CV_WRAP virtual void populateNet(Net net) = 0;

        virtual ~Importer();
    };

    /** @brief Creates the importer of <a href="http://caffe.berkeleyvision.org">Caffe</a> framework network.
     *  @param prototxt   path to the .prototxt file with text description of the network architecture.
     *  @param caffeModel path to the .caffemodel file with learned network.
     *  @returns Pointer to the created importer, NULL in failure cases.
     */
    CV_EXPORTS_W Ptr<Importer> createCaffeImporter(const String &prototxt, const String &caffeModel = String());

    /** @brief Reads a network model stored in Caffe model files.
      * @details This is shortcut consisting from createCaffeImporter and Net::populateNet calls.
      */
    CV_EXPORTS_W Net readNetFromCaffe(const String &prototxt, const String &caffeModel = String());

    /** @brief Reads a network model stored in Tensorflow model file.
      * @details This is shortcut consisting from createTensorflowImporter and Net::populateNet calls.
      */
    CV_EXPORTS_W Net readNetFromTensorflow(const String &model);

    /** @brief Reads a network model stored in Torch model file.
      * @details This is shortcut consisting from createTorchImporter and Net::populateNet calls.
      */
    CV_EXPORTS_W Net readNetFromTorch(const String &model, bool isBinary = true);

    /** @brief Creates the importer of <a href="http://www.tensorflow.org">TensorFlow</a> framework network.
     *  @param model   path to the .pb file with binary protobuf description of the network architecture.
     *  @returns Pointer to the created importer, NULL in failure cases.
     */
    CV_EXPORTS Ptr<Importer> createTensorflowImporter(const String &model);

    /** @brief Creates the importer of <a href="http://torch.ch">Torch7</a> framework network.
     *  @param filename path to the file, dumped from Torch by using torch.save() function.
     *  @param isBinary specifies whether the network was serialized in ascii mode or binary.
     *  @returns Pointer to the created importer, NULL in failure cases.
     *
     *  @warning Torch7 importer is experimental now, you need explicitly set CMake `opencv_dnn_BUILD_TORCH_IMPORTER` flag to compile its.
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
    CV_EXPORTS_W Ptr<Importer> createTorchImporter(const String &filename, bool isBinary = true);

    /** @brief Loads blob which was serialized as torch.Tensor object of Torch7 framework.
     *  @warning This function has the same limitations as createTorchImporter().
     */
    CV_EXPORTS_W Mat readTorchBlob(const String &filename, bool isBinary = true);
    /** @brief Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,
     *  subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels.
     *  @param image input image (with 1- or 3-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p image values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @details input image is resized so one side after resize is equal to corresponing
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  @returns 4-dimansional Mat with NCHW dimensions order.
     */
    CV_EXPORTS_W Mat blobFromImage(const Mat& image, double scalefactor=1.0, const Size& size = Size(),
                                   const Scalar& mean = Scalar(), bool swapRB=true);
    /** @brief Creates 4-dimensional blob from series of images. Optionally resizes and
     *  crops @p images from center, subtract @p mean values, scales values by @p scalefactor,
     *  swap Blue and Red channels.
     *  @param images input images (all with 1- or 3-channels).
     *  @param size spatial size for output image
     *  @param mean scalar with mean values which are subtracted from channels. Values are intended
     *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
     *  @param scalefactor multiplier for @p images values.
     *  @param swapRB flag which indicates that swap first and last channels
     *  in 3-channel image is necessary.
     *  @details input image is resized so one side after resize is equal to corresponing
     *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
     *  @returns 4-dimansional Mat with NCHW dimensions order.
     */
    CV_EXPORTS_W Mat blobFromImages(const std::vector<Mat>& images, double scalefactor=1.0,
                                    Size size = Size(), const Scalar& mean = Scalar(), bool swapRB=true);

//! @}
}
}

#include <opencv2/dnn/layer.hpp>
#include <opencv2/dnn/dnn.inl.hpp>

#endif  /* __OPENCV_DNN_DNN_HPP__ */
