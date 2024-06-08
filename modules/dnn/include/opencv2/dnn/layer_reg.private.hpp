// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_LAYER_REG_HPP
#define OPENCV_DNN_LAYER_REG_HPP
#include <opencv2/dnn.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN
//! @addtogroup dnn
//! @{

typedef std::map<std::string, std::vector<LayerFactory::Constructor> > LayerFactory_Impl;

//! Register layer types of DNN model.
//!
//! @note In order to thread-safely access the factory, see getLayerFactoryMutex() function.
LayerFactory_Impl& getLayerFactoryImpl();

//! Get the mutex guarding @ref LayerFactory_Impl, see getLayerFactoryImpl() function.
Mutex& getLayerFactoryMutex();

//! @}
CV__DNN_INLINE_NS_END
}
}
#endif
