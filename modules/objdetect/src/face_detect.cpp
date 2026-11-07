// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

#include <algorithm>

namespace cv
{

#ifdef HAVE_OPENCV_DNN
class FaceDetectorYNImpl : public FaceDetectorYN
{
public:
    FaceDetectorYNImpl(const String& model,
                       const String& config,
                       const Size& input_size,
                       float score_threshold,
                       float nms_threshold,
                       int top_k,
                       int backend_id,
                       int target_id)
                       :divisor(32),
                       strides({8, 16, 32})
    {
        net = dnn::readNet(model, config);
        CV_Assert(!net.empty());

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);

        inputW = input_size.width;
        inputH = input_size.height;

        padW = (int((inputW - 1) / divisor) + 1) * divisor;
        padH = (int((inputH - 1) / divisor) + 1) * divisor;

        scoreThreshold = score_threshold;
        nmsThreshold = nms_threshold;
        topK = top_k;
    }

    FaceDetectorYNImpl(const String& framework,
                       const std::vector<uchar>& bufferModel,
                       const std::vector<uchar>& bufferConfig,
                       const Size& input_size,
                       float score_threshold,
                       float nms_threshold,
                       int top_k,
                       int backend_id,
                       int target_id)
                       :divisor(32),
                       strides({8, 16, 32})
    {
        net = dnn::readNet(framework, bufferModel, bufferConfig);
        CV_Assert(!net.empty());

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);

        inputW = input_size.width;
        inputH = input_size.height;

        padW = (int((inputW - 1) / divisor) + 1) * divisor;
        padH = (int((inputH - 1) / divisor) + 1) * divisor;

        scoreThreshold = score_threshold;
        nmsThreshold = nms_threshold;
        topK = top_k;
    }

    void setInputSize(const Size& input_size) override
    {
        inputW = input_size.width;
        inputH = input_size.height;
        padW = ((inputW - 1) / divisor + 1) * divisor;
        padH = ((inputH - 1) / divisor + 1) * divisor;
    }

    Size getInputSize() override
    {
        Size input_size;
        input_size.width = inputW;
        input_size.height = inputH;
        return input_size;
    }

    void setScoreThreshold(float score_threshold) override
    {
        scoreThreshold = score_threshold;
    }

    float getScoreThreshold() override
    {
        return scoreThreshold;
    }

    void setNMSThreshold(float nms_threshold) override
    {
        nmsThreshold = nms_threshold;
    }

    float getNMSThreshold() override
    {
        return nmsThreshold;
    }

    void setTopK(int top_k) override
    {
        topK = top_k;
    }

    int getTopK() override
    {
        return topK;
    }

    int detect(InputArray input_image, OutputArray faces) override
    {
        // TODO: more checkings should be done?
        if (input_image.empty())
        {
            return 0;
        }
        CV_CheckEQ(input_image.size(), Size(inputW, inputH), "Size does not match. Call setInputSize(size) if input size does not match the preset size");

        Mat input_blob;
        if(input_image.kind() == _InputArray::UMAT) {
            // Pad input_image with divisor 32
            UMat pad_image;
            padWithDivisor(input_image, pad_image);
            // Build blob from input image
            input_blob = dnn::blobFromImage(pad_image);
        } else {
            // Pad input_image with divisor 32
            Mat pad_image;
            padWithDivisor(input_image, pad_image);
            // Build blob from input image
            input_blob = dnn::blobFromImage(pad_image);
        }
        // Forward
        std::vector<String> output_names = { "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };
        std::vector<Mat> output_blobs;
        net.setInput(input_blob);
        net.forward(output_blobs, output_names);

        // Post process
        Mat results = postProcess(output_blobs);
        results.convertTo(faces, CV_32FC1);
        return 1;
    }
private:
    Mat postProcess(const std::vector<Mat>& output_blobs)
    {
        Mat faces;
        for (size_t i = 0; i < strides.size(); ++i) {
            int cols = int(padW / strides[i]);
            int rows = int(padH / strides[i]);

            // Extract from output_blobs
            Mat cls = output_blobs[i];
            Mat obj = output_blobs[i + strides.size() * 1];
            Mat bbox = output_blobs[i + strides.size() * 2];
            Mat kps = output_blobs[i + strides.size() * 3];

            // Decode from predictions
            float* cls_v = (float*)(cls.data);
            float* obj_v = (float*)(obj.data);
            float* bbox_v = (float*)(bbox.data);
            float* kps_v = (float*)(kps.data);

            // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
            // 'tl': top left point of the bounding box
            // 're': right eye, 'le': left eye
            // 'nt':  nose tip
            // 'rcm': right corner of mouth, 'lcm': left corner of mouth
            Mat face(1, 15, CV_32FC1);

            for(int r = 0; r < rows; ++r) {
                for(int c = 0; c < cols; ++c) {
                    size_t idx = r * cols + c;

                    // Get score
                    float cls_score = cls_v[idx];
                    float obj_score = obj_v[idx];

                    // Clamp
                    cls_score = MIN(cls_score, 1.f);
                    cls_score = MAX(cls_score, 0.f);
                    obj_score = MIN(obj_score, 1.f);
                    obj_score = MAX(obj_score, 0.f);
                    float score = std::sqrt(cls_score * obj_score);
                    face.at<float>(0, 14) = score;

                    // Checking if the score meets the threshold before adding the face
                    if (score < scoreThreshold)
                        continue;
                    // Get bounding box
                    float cx = ((c + bbox_v[idx * 4 + 0]) * strides[i]);
                    float cy = ((r + bbox_v[idx * 4 + 1]) * strides[i]);
                    float w = exp(bbox_v[idx * 4 + 2]) * strides[i];
                    float h = exp(bbox_v[idx * 4 + 3]) * strides[i];

                    float x1 = cx - w / 2.f;
                    float y1 = cy - h / 2.f;

                    face.at<float>(0, 0) = x1;
                    face.at<float>(0, 1) = y1;
                    face.at<float>(0, 2) = w;
                    face.at<float>(0, 3) = h;

                    // Get landmarks
                    for(int n = 0; n < 5; ++n) {
                        face.at<float>(0, 4 + 2 * n) = (kps_v[idx * 10 + 2 * n] + c) * strides[i];
                        face.at<float>(0, 4 + 2 * n + 1) = (kps_v[idx * 10 + 2 * n + 1]+ r) * strides[i];
                    }
                    faces.push_back(face);
                }
            }
        }

        if (faces.rows > 1)
        {
            // Retrieve boxes and scores
            std::vector<Rect2i> faceBoxes;
            std::vector<float> faceScores;
            for (int rIdx = 0; rIdx < faces.rows; rIdx++)
            {
                faceBoxes.push_back(Rect2i(int(faces.at<float>(rIdx, 0)),
                                           int(faces.at<float>(rIdx, 1)),
                                           int(faces.at<float>(rIdx, 2)),
                                           int(faces.at<float>(rIdx, 3))));
                faceScores.push_back(faces.at<float>(rIdx, 14));
            }

            std::vector<int> keepIdx;
            dnn::NMSBoxes(faceBoxes, faceScores, scoreThreshold, nmsThreshold, keepIdx, 1.f, topK);

            // Get NMS results
            Mat nms_faces;
            for (int idx: keepIdx)
            {
                nms_faces.push_back(faces.row(idx));
            }
            return nms_faces;
        }
        else
        {
            return faces;
        }
    }

    void padWithDivisor(InputArray input_image, OutputArray pad_image)
    {
        int bottom = padH - inputH;
        int right = padW - inputW;
        copyMakeBorder(input_image, pad_image, 0, bottom, 0, right, BORDER_CONSTANT, 0);
    }
private:
    dnn::Net net;

    int inputW;
    int inputH;
    int padW;
    int padH;
    const int divisor;
    int topK;
    float scoreThreshold;
    float nmsThreshold;
    const std::vector<int> strides;
};
#endif

Ptr<FaceDetectorYN> FaceDetectorYN::create(const String& model,
                                           const String& config,
                                           const Size& input_size,
                                           const float score_threshold,
                                           const float nms_threshold,
                                           const int top_k,
                                           const int backend_id,
                                           const int target_id)
{
#ifdef HAVE_OPENCV_DNN
    return makePtr<FaceDetectorYNImpl>(model, config, input_size, score_threshold, nms_threshold, top_k, backend_id, target_id);
#else
    CV_UNUSED(model); CV_UNUSED(config); CV_UNUSED(input_size); CV_UNUSED(score_threshold); CV_UNUSED(nms_threshold); CV_UNUSED(top_k); CV_UNUSED(backend_id); CV_UNUSED(target_id);
    CV_Error(cv::Error::StsNotImplemented, "cv::FaceDetectorYN requires enabled 'dnn' module.");
#endif
}

Ptr<FaceDetectorYN> FaceDetectorYN::create(const String& framework,
                                           const std::vector<uchar>& bufferModel,
                                           const std::vector<uchar>& bufferConfig,
                                           const Size& input_size,
                                           const float score_threshold,
                                           const float nms_threshold,
                                           const int top_k,
                                           const int backend_id,
                                           const int target_id)
{
#ifdef HAVE_OPENCV_DNN
    return makePtr<FaceDetectorYNImpl>(framework, bufferModel, bufferConfig, input_size, score_threshold, nms_threshold, top_k, backend_id, target_id);
#else
    CV_UNUSED(framework);  CV_UNUSED(bufferModel); CV_UNUSED(bufferConfig); CV_UNUSED(input_size); CV_UNUSED(score_threshold); CV_UNUSED(nms_threshold); CV_UNUSED(top_k); CV_UNUSED(backend_id); CV_UNUSED(target_id);
    CV_Error(cv::Error::StsNotImplemented, "cv::FaceDetectorYN requires enabled 'dnn' module.");
#endif
}

} // namespace cv
