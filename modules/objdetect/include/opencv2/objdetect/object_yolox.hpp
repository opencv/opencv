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
     *
     * @param input_size the size of the input image
     */

    CV_WRAP virtual Size getInputSize() = 0;

    /** @brief Set the confidence threshold to filter out bounding boxes of score less than the given value
     *
     * @param conThreshold threshold for filtering out bounding boxes
     */
    CV_WRAP virtual void setConfThreshold(float confThresh) = 0;

    CV_WRAP virtual float getConfThreshold() = 0;

    /** @brief Set the Non-maximum-suppression threshold to suppress bounding boxes that have IoU greater than the given value
     * IoU Intersection surface Over Union surface
     * @param nmsThresh threshold for NMS operation
     */
    CV_WRAP virtual void setNMSThreshold(float nmsThresh) = 0;

    CV_WRAP virtual float getNMSThreshold() = 0;


    /** @brief Detects objectin the input image. Following is an example output.

     * ![image](pics/lena-face-detection.jpg)

     *  @param image an image to detect
     *  @param faces detection results stored in a 2D cv::Mat of shape [num_faces, 15]
     *  - 0-1: x, y of bbox top left corner
     *  - 2-3: width, height of bbox
     *  - 4-5: x, y of right eye (blue point in the example image)
     *  - 6-7: x, y of left eye (red point in the example image)
     *  - 8-9: x, y of nose tip (green point in the example image)
     *  - 10-11: x, y of right corner of mouth (pink point in the example image)
     *  - 12-13: x, y of left corner of mouth (yellow point in the example image)
     *  - 14: face score
     */
    CV_WRAP virtual int detect(InputArray image, OutputArray faces) = 0;

    /** @brief Creates an instance of this class with given parameters
     *
     *  @param modelPath the path to the requested model
     *  @param confThresh the path to the config file for compability, which is not requested for ONNX models
     *  @param nmsThresh the size of the input image
     *  @param objThresh the threshold to filter out bounding boxes of score smaller than the given value
     *  @param backend_id the id of backend
     *  @param target_id the id of target device
     */
    CV_WRAP static Ptr<ObjectDetectorYX> create( std::string modelPath, float confThresh = 0.35, float nmsThresh = 0.5,
                                                 dnn::Backend bId = dnn::DNN_BACKEND_DEFAULT, dnn::Target tId = dnn::DNN_TARGET_CPU);
};


//! @}
} // namespace cv

#endif
