// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
#ifndef OPENCV_DNN_LAYER_DETAILS_HPP
#define OPENCV_DNN_LAYER_DETAILS_HPP

#include <opencv2/dnn/layer.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

/** @brief Registers layer constructor in runtime.
*   @param type string, containing type name of the layer.
*   @param constructorFunc pointer to the function of type LayerRegister::Constructor, which creates the layer.
*   @details This macros must be placed inside the function code.
*/
#define CV_DNN_REGISTER_LAYER_FUNC(type, constructorFunc) \
    cv::dnn::LayerFactory::registerLayer(#type, constructorFunc);

/** @brief Registers layer class in runtime.
 *  @param type string, containing type name of the layer.
 *  @param class C++ class, derived from Layer.
 *  @details This macros must be placed inside the function code.
 */
#define CV_DNN_REGISTER_LAYER_CLASS(type, class) \
    cv::dnn::LayerFactory::registerLayer(#type, cv::dnn::details::_layerDynamicRegisterer<class>);

/** @brief Registers layer constructor on module load time.
*   @param type string, containing type name of the layer.
*   @param constructorFunc pointer to the function of type LayerRegister::Constructor, which creates the layer.
*   @details This macros must be placed outside the function code.
*/
#define CV_DNN_REGISTER_LAYER_FUNC_STATIC(type, constructorFunc) \
static cv::dnn::details::_LayerStaticRegisterer __LayerStaticRegisterer_##type(#type, constructorFunc);

/** @brief Registers layer class on module load time.
 *  @param type string, containing type name of the layer.
 *  @param class C++ class, derived from Layer.
 *  @details This macros must be placed outside the function code.
 */
#define CV_DNN_REGISTER_LAYER_CLASS_STATIC(type, class)                         \
Ptr<Layer> __LayerStaticRegisterer_func_##type(LayerParams &params) \
    { return Ptr<Layer>(new class(params)); }                       \
static cv::dnn::details::_LayerStaticRegisterer __LayerStaticRegisterer_##type(#type, __LayerStaticRegisterer_func_##type);

/** @brief Registers an LayerInfo (metadata node) class for the new graph engine.
 *  @param type string, containing the operation type name.
 *  @param class C++ class derived from LayerInfo, providing `static Ptr<LayerInfo> create(const LayerParams&)`.
 *  @details This macro must be placed inside the function code (e.g. initializeLayerFactory()).
 */
#define CV_DNN_REGISTER_OP_CLASS(type, class) \
    cv::dnn::LayerFactory::registerOp(#type, cv::dnn::details::_opDynamicRegisterer<class>);

/** @brief Registers a backend executor class for the new graph engine.
 *  @param type string, containing the operation type name.
 *  @param backendId backend id the executor targets (e.g. DNN_BACKEND_OPENCV, DNN_BACKEND_CUDA).
 *  @param class C++ class derived from Layer, providing
 *         `static Ptr<Layer> create(const Ptr<LayerInfo>&, void* backendCtx)` (null Ptr if unsupported).
 *  @details This macro must be placed inside the function code.
 */
#define CV_DNN_REGISTER_EXEC_CLASS(type, backendId, class) \
    cv::dnn::LayerFactory::registerExec(#type, backendId, cv::dnn::details::_execDynamicRegisterer<class>);

namespace details {

template<typename LayerClass>
Ptr<Layer> _layerDynamicRegisterer(LayerParams &params)
{
    return Ptr<Layer>(LayerClass::create(params));
}

template<typename OpClass>
Ptr<LayerInfo> _opDynamicRegisterer(const LayerParams &params)
{
    return Ptr<LayerInfo>(OpClass::create(params));
}

template<typename ExecClass>
Ptr<Layer> _execDynamicRegisterer(const Ptr<LayerInfo>& data, void* backendCtx)
{
    return Ptr<Layer>(ExecClass::create(data, backendCtx));
}

//allows automatically register created layer on module load time
class _LayerStaticRegisterer
{
    String type;
public:

    _LayerStaticRegisterer(const String &layerType, LayerFactory::Constructor layerConstructor)
    {
        this->type = layerType;
        LayerFactory::registerLayer(layerType, layerConstructor);
    }

    ~_LayerStaticRegisterer()
    {
        LayerFactory::unregisterLayer(type);
    }
};

} // namespace
CV__DNN_INLINE_NS_END
}} // namespace

#endif
