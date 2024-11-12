#ifndef OPENCV_FEATURE2D_HAL_INTERFACE_H
#define OPENCV_FEATURE2D_HAL_INTERFACE_H

#include "opencv2/core/cvdef.h"
//! @addtogroup features_hal_interface
//! @{

//! @name Fast feature detector types
//! @sa cv::FastFeatureDetector
//! @{
#define CV_HAL_TYPE_5_8  0
#define CV_HAL_TYPE_7_12 1
#define CV_HAL_TYPE_9_16 2
//! @}

//! @name Key point
//! @sa cv::KeyPoint
//! @{
struct CV_EXPORTS cvhalKeyPoint
{
    float x;
    float y;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
};
//! @}

//! @}

#endif
