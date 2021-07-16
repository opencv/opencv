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
class CV_EXPORTS_W DNNFaceDetector
{
public:
    virtual ~DNNFaceDetector() {};

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
    CV_WRAP static Ptr<DNNFaceDetector> create(const String& onnx_path,
                                                const Size& input_size,
                                                float score_threshold = 0.9,
                                                float nms_threshold = 0.3,
                                                int top_k = 5000,
                                                int backend_id = 0,
                                                int target_id = 0);
};

class CV_EXPORTS_W DNNFaceRecognizer
{
public:
    virtual ~DNNFaceRecognizer() {};
    CV_WRAP virtual void AlignCrop(InputArray src_img, InputArray face_box, OutputArray aligned_img) const = 0;
    CV_WRAP virtual void facefeature(InputArray aligned_img, OutputArray face_feature) = 0;
    CV_WRAP virtual double facematch(InputArray _face_feature1, InputArray _face_feature2, const String& distype) const = 0;
    CV_WRAP static Ptr<DNNFaceRecognizer> create(const String& onnx_path);
};

} // namespace cv

#endif
