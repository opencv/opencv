// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_OBJDETECT_YOLOX_HPP
#define OPENCV_OBJDETECT_YOLOX_HPP

#include <opencv2/core.hpp>
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv
{

//! @addtogroup objdetect_dnn_yolox
//! @{

/** @brief DNN-based object detector based on yolox model

model download link: https://github.com/opencv/opencv_zoo/tree/master/models/object_detection_yolox
 */
class CV_EXPORTS_W ObjectDetectorYX
{
public:
    virtual ~ObjectDetectorYX() {};

    /** @brief Get the size for the network input, which overwrites the input size of creating model. Call this method when the size of input image does not match the input size when creating model
     */

    CV_WRAP virtual Size getInputSize() = 0;

    /** @brief Set the confidence threshold to filter out bounding boxes of score less than the given value
     *
     * @param confThresh threshold for filtering out bounding boxes
     */
    CV_WRAP virtual void setConfThreshold(float confThresh) = 0;

    CV_WRAP virtual float getConfThreshold() = 0;

    /** @brief Set the Non-maximum-suppression threshold to suppress bounding boxes that have IoU greater than the given value
     * IoU Intersection surface Over Union surface
     * @param nmsThresh threshold for NMS operation
     */
    CV_WRAP virtual void setNMSThreshold(float nmsThresh) = 0;

    CV_WRAP virtual float getNMSThreshold() = 0;


    /** @brief Detects objectin the input image.
     */
    CV_WRAP virtual int detect(InputArray image, OutputArray faces) = 0;

    /** @brief Creates an instance of this class with given parameters
     *
     *  @param modelPath the path to the requested model
     *  @param confThresh the path to the config file for compability, which is not requested for ONNX models
     *  @param nmsThresh the size of the input image
     *  @param bId the id of backend
     *  @param tId the id of target device
     */
    CV_WRAP static Ptr<ObjectDetectorYX> create( std::string modelPath, float confThresh = 0.35f, float nmsThresh = 0.5f,
                                                 dnn::Backend bId = dnn::DNN_BACKEND_DEFAULT, dnn::Target tId = dnn::DNN_TARGET_CPU);
};


//! @}
} // namespace cv

#endif
