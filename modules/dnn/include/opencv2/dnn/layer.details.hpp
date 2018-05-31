// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
#ifndef OPENCV_DNN_LAYER_DETAILS_HPP
#define OPENCV_DNN_LAYER_DETAILS_HPP

#include <opencv2/dnn/layer.hpp>

namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

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

namespace details {

template<typename LayerClass>
Ptr<Layer> _layerDynamicRegisterer(LayerParams &params)
{
    return Ptr<Layer>(LayerClass::create(params));
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
CV__DNN_EXPERIMENTAL_NS_END
}} // namespace

#endif
