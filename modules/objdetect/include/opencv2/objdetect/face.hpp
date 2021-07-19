// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_DNN_FACE_CORE_HPP_
#define _OPENCV_DNN_FACE_CORE_HPP_

#include <opencv2/core.hpp>

/** @defgroup dnn_face DNN-based face detection and recognition
 */

namespace cv
{

/** @brief DNN-based face detector, model download link: https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.
 */
class CV_EXPORTS_W FaceDetector
{
public:
    virtual ~FaceDetector() {};

    CV_WRAP virtual void setInputSize(const Size& input_size) = 0;

    CV_WRAP virtual Size getInputSize() = 0;

    CV_WRAP virtual void setScoreThreshold(float score_threshold) = 0;

    CV_WRAP virtual float getScoreThreshold() = 0;

    CV_WRAP virtual void setNMSThreshold(float nms_threshold) = 0;

    CV_WRAP virtual float getNMSThreshold() = 0;

    CV_WRAP virtual void setTopK(int top_k) = 0;

    CV_WRAP virtual int getTopK() = 0;

    /** @brief A simple interface to detect face from given image
     *
     *  @param image an image to detect
     *  @param faces detection results stored in a cv::Mat
     */
    CV_WRAP virtual int detect(InputArray image, OutputArray faces) = 0;

    /** @brief Creates an instance of this class with given parameters
     * 
     *  @param onnx_path the path to the downloaded ONNX model
     *  @param input_size the size of the input image
     *  @param score_threshold the threshold to filter out bounding boxes of score smaller than the given value
     *  @param nms_threshold the threshold to suppress bounding boxes of IoU bigger than the given value
     *  @param top_k keep top K bboxes before NMS
     *  @param backend_id the id of backend
     *  @param target_id the id of target device
     */
    CV_WRAP static Ptr<FaceDetector> create(const String& onnx_path,
                                            const Size& input_size,
                                            float score_threshold = 0.9,
                                            float nms_threshold = 0.3,
                                            int top_k = 5000,
                                            int backend_id = 0,
                                            int target_id = 0);
};

class CV_EXPORTS_W FaceRecognizer
{
public:
    virtual ~FaceRecognizer() {};

    /** @brief Aligning image to put face on the standard position
     *  @param src_img input image
     *  @param face_box the detection result used for indicate face in input image
     *  @param aligned_img output aligned image
     */
    CV_WRAP virtual void AlignCrop(InputArray src_img, InputArray face_box, OutputArray aligned_img) const = 0;

    /** @brief Extracting face feature from aligned image
     *  @param aligned_img input aligned image
     *  @param face_feature output face feature
     */
    CV_WRAP virtual void facefeature(InputArray aligned_img, OutputArray face_feature) = 0;

    /** @brief Calculating the distance between two face features
     *  @param _face_feature1 the first input feature
     *  @param _face_feature2 the second input feature of the same size and the same type as _face_feature1
     *  @param distype defining the similarity with optional values "norml2" or "cosine"
     */
    CV_WRAP virtual double facematch(InputArray _face_feature1, InputArray _face_feature2, const String& distype) const = 0;

    /** @brief Creates an instance of this class with given parameters
     *  @param onnx_path the path of the onnx model used for face recognition
     */
    CV_WRAP static Ptr<FaceRecognizer> create(const String& onnx_path);
};

} // namespace cv

#endif
