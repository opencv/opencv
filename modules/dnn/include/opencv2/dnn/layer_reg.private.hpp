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

//! Register layer types of DNN model.
typedef std::map<std::string, std::vector<LayerFactory::Constructor> > LayerFactory_Impl;
LayerFactory_Impl& getLayerFactoryImpl();

//! @}
CV__DNN_INLINE_NS_END
}
}
#endif
