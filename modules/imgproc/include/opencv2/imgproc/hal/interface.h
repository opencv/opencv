#ifndef OPENCV_IMGPROC_HAL_INTERFACE_H
#define OPENCV_IMGPROC_HAL_INTERFACE_H

//! @addtogroup imgproc_hal_interface
//! @{

//! @name Interpolation modes
//! @sa cv::InterpolationFlags
//! @{
#define CV_HAL_INTER_NEAREST 0
#define CV_HAL_INTER_LINEAR 1
#define CV_HAL_INTER_CUBIC 2
#define CV_HAL_INTER_AREA 3
#define CV_HAL_INTER_LANCZOS4 4
//! @}

//! @name Morphology operations
//! @sa cv::MorphTypes
//! @{
#define CV_HAL_MORPH_ERODE 0
#define CV_HAL_MORPH_DILATE 1
//! @}

//! @}

#endif
