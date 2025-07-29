// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_OBJDETECT_FACE_HPP
#define OPENCV_OBJDETECT_FACE_HPP

#include <opencv2/core.hpp>

namespace cv
{

//! @addtogroup objdetect_dnn_face
//! @{

/** @brief DNN-based face detector

model download link: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
 */
class CV_EXPORTS_W FaceDetectorYN
{
public:
    virtual ~FaceDetectorYN() {}

    /** @brief Set the size for the network input, which overwrites the input size of creating model. Call this method when the size of input image does not match the input size when creating model
     *
     * @param input_size the size of the input image
     */
    CV_WRAP virtual void setInputSize(const Size& input_size) = 0;

    CV_WRAP virtual Size getInputSize() = 0;

    /** @brief Set the score threshold to filter out bounding boxes of score less than the given value
     *
     * @param score_threshold threshold for filtering out bounding boxes
     */
    CV_WRAP virtual void setScoreThreshold(float score_threshold) = 0;

    CV_WRAP virtual float getScoreThreshold() = 0;

    /** @brief Set the Non-maximum-suppression threshold to suppress bounding boxes that have IoU greater than the given value
     *
     * @param nms_threshold threshold for NMS operation
     */
    CV_WRAP virtual void setNMSThreshold(float nms_threshold) = 0;

    CV_WRAP virtual float getNMSThreshold() = 0;

    /** @brief Set the number of bounding boxes preserved before NMS
     *
     * @param top_k the number of bounding boxes to preserve from top rank based on score
     */
    CV_WRAP virtual void setTopK(int top_k) = 0;

    CV_WRAP virtual int getTopK() = 0;

    /** @brief Detects faces in the input image. Following is an example output.

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

    /** @brief Creates an instance of face detector class with given parameters
     *
     *  @param model the path to the requested model
     *  @param config the path to the config file for compability, which is not requested for ONNX models
     *  @param input_size the size of the input image
     *  @param score_threshold the threshold to filter out bounding boxes of score smaller than the given value
     *  @param nms_threshold the threshold to suppress bounding boxes of IoU bigger than the given value
     *  @param top_k keep top K bboxes before NMS
     *  @param backend_id the id of backend
     *  @param target_id the id of target device
     */
    CV_WRAP static Ptr<FaceDetectorYN> create(CV_WRAP_FILE_PATH const String& model,
                                              CV_WRAP_FILE_PATH const String& config,
                                              const Size& input_size,
                                              float score_threshold = 0.9f,
                                              float nms_threshold = 0.3f,
                                              int top_k = 5000,
                                              int backend_id = 0,
                                              int target_id = 0);

    /** @overload
     *
     *  @param framework Name of origin framework
     *  @param bufferModel A buffer with a content of binary file with weights
     *  @param bufferConfig A buffer with a content of text file contains network configuration
     *  @param input_size the size of the input image
     *  @param score_threshold the threshold to filter out bounding boxes of score smaller than the given value
     *  @param nms_threshold the threshold to suppress bounding boxes of IoU bigger than the given value
     *  @param top_k keep top K bboxes before NMS
     *  @param backend_id the id of backend
     *  @param target_id the id of target device
     */
    CV_WRAP static Ptr<FaceDetectorYN> create(const String& framework,
                                              const std::vector<uchar>& bufferModel,
                                              const std::vector<uchar>& bufferConfig,
                                              const Size& input_size,
                                              float score_threshold = 0.9f,
                                              float nms_threshold = 0.3f,
                                              int top_k = 5000,
                                              int backend_id = 0,
                                              int target_id = 0);

};

/** @brief DNN-based face recognizer

model download link: https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface
 */
class CV_EXPORTS_W FaceRecognizerSF
{
public:
    virtual ~FaceRecognizerSF() {}

    /** @brief Definition of distance used for calculating the distance between two face features
     */
    enum DisType { FR_COSINE=0, FR_NORM_L2=1 };

    /** @brief Aligns detected face with the source input image and crops it
     *  @param src_img input image
     *  @param face_box the detected face result from the input image
     *  @param aligned_img output aligned image
     */
    CV_WRAP virtual void alignCrop(InputArray src_img, InputArray face_box, OutputArray aligned_img) const = 0;

    /** @brief Extracts face feature from aligned image
     *  @param aligned_img input aligned image
     *  @param face_feature output face feature
     */
    CV_WRAP virtual void feature(InputArray aligned_img, OutputArray face_feature) = 0;

    /** @brief Calculates the distance between two face features
     *  @param face_feature1 the first input feature
     *  @param face_feature2 the second input feature of the same size and the same type as face_feature1
     *  @param dis_type defines how to calculate the distance between two face features with optional values "FR_COSINE" or "FR_NORM_L2"
     */
    CV_WRAP virtual double match(InputArray face_feature1, InputArray face_feature2, int dis_type = FaceRecognizerSF::FR_COSINE) const = 0;

    /** @brief Creates an instance of this class with given parameters
     *  @param model the path of the onnx model used for face recognition
     *  @param config the path to the config file for compability, which is not requested for ONNX models
     *  @param backend_id the id of backend
     *  @param target_id the id of target device
     */
    CV_WRAP static Ptr<FaceRecognizerSF> create(CV_WRAP_FILE_PATH const String& model, CV_WRAP_FILE_PATH const String& config, int backend_id = 0, int target_id = 0);

    /**
     *  @brief Creates an instance of this class from a buffer containing the model weights and configuration.
     *  @param framework Name of the framework (ONNX, etc.)
     *  @param bufferModel A buffer containing the binary model weights.
     *  @param bufferConfig A buffer containing the network configuration.
     *  @param backend_id The id of the backend.
     *  @param target_id The id of the target device.
     *
     *  @return A pointer to the created instance of FaceRecognizerSF.
     */
    CV_WRAP static Ptr<FaceRecognizerSF> create(const String& framework,
                                                const std::vector<uchar>& bufferModel,
                                                const std::vector<uchar>& bufferConfig,
                                                int backend_id = 0,
                                                int target_id = 0);
};

//! @}
} // namespace cv

#endif
