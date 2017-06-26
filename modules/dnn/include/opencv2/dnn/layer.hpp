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

#ifndef OPENCV_DNN_LAYER_HPP
#define OPENCV_DNN_LAYER_HPP
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{
//! @addtogroup dnn
//! @{
//!
//! @defgroup dnnLayerFactory Utilities for New Layers Registration
//! @{

/** @brief %Layer factory allows to create instances of registered layers. */
class CV_EXPORTS LayerFactory
{
public:

    //! Each Layer class must provide this function to the factory
    typedef Ptr<Layer>(*Constuctor)(LayerParams &params);

    //! Registers the layer class with typename @p type and specified @p constructor.
    static void registerLayer(const String &type, Constuctor constructor);

    //! Unregisters registered layer with specified type name.
    static void unregisterLayer(const String &type);

    /** @brief Creates instance of registered layer.
     *  @param type type name of creating layer.
     *  @param params parameters which will be used for layer initialization.
     */
    static Ptr<Layer> createLayerInstance(const String &type, LayerParams& params);

private:
    LayerFactory();

    struct Impl;
    static Ptr<Impl> impl();
};

/** @brief Registers layer constructor in runtime.
*   @param type string, containing type name of the layer.
*   @param constuctorFunc pointer to the function of type LayerRegister::Constuctor, which creates the layer.
*   @details This macros must be placed inside the function code.
*/
#define REG_RUNTIME_LAYER_FUNC(type, constuctorFunc) \
    cv::dnn::LayerFactory::registerLayer(#type, constuctorFunc);

/** @brief Registers layer class in runtime.
 *  @param type string, containing type name of the layer.
 *  @param class C++ class, derived from Layer.
 *  @details This macros must be placed inside the function code.
 */
#define REG_RUNTIME_LAYER_CLASS(type, class) \
    cv::dnn::LayerFactory::registerLayer(#type, _layerDynamicRegisterer<class>);

/** @brief Registers layer constructor on module load time.
*   @param type string, containing type name of the layer.
*   @param constuctorFunc pointer to the function of type LayerRegister::Constuctor, which creates the layer.
*   @details This macros must be placed outside the function code.
*/
#define REG_STATIC_LAYER_FUNC(type, constuctorFunc) \
static cv::dnn::_LayerStaticRegisterer __LayerStaticRegisterer_##type(#type, constuctorFunc);

/** @brief Registers layer class on module load time.
 *  @param type string, containing type name of the layer.
 *  @param class C++ class, derived from Layer.
 *  @details This macros must be placed outside the function code.
 */
#define REG_STATIC_LAYER_CLASS(type, class)                         \
Ptr<Layer> __LayerStaticRegisterer_func_##type(LayerParams &params) \
    { return Ptr<Layer>(new class(params)); }                       \
static _LayerStaticRegisterer __LayerStaticRegisterer_##type(#type, __LayerStaticRegisterer_func_##type);


//! @}
//! @}


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

    _LayerStaticRegisterer(const String &layerType, LayerFactory::Constuctor layerConstuctor)
    {
        this->type = layerType;
        LayerFactory::registerLayer(layerType, layerConstuctor);
    }

    ~_LayerStaticRegisterer()
    {
        LayerFactory::unregisterLayer(type);
    }
};

}
}
#endif
