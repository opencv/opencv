// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

#include <algorithm>

namespace cv
{

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
    {
        net = dnn::readNet(model, config);
        CV_Assert(!net.empty());

        net.setPreferableBackend(backend_id);
        net.setPreferableTarget(target_id);

        inputW = input_size.width;
        inputH = input_size.height;

        scoreThreshold = score_threshold;
        nmsThreshold = nms_threshold;
        topK = top_k;

        generatePriors();
    }

    void setInputSize(const Size& input_size) override
    {
        inputW = input_size.width;
        inputH = input_size.height;
        generatePriors();
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

        // Build blob from input image
        Mat input_blob = dnn::blobFromImage(input_image);

        // Forward
        std::vector<String> output_names = { "loc", "conf", "iou" };
        std::vector<Mat> output_blobs;
        net.setInput(input_blob);
        net.forward(output_blobs, output_names);

        // Post process
        Mat results = postProcess(output_blobs);
        results.convertTo(faces, CV_32FC1);
        return 1;
    }
private:
    void generatePriors()
    {
        // Calculate shapes of different scales according to the shape of input image
        Size feature_map_2nd = {
            int(int((inputW+1)/2)/2), int(int((inputH+1)/2)/2)
        };
        Size feature_map_3rd = {
            int(feature_map_2nd.width/2), int(feature_map_2nd.height/2)
        };
        Size feature_map_4th = {
            int(feature_map_3rd.width/2), int(feature_map_3rd.height/2)
        };
        Size feature_map_5th = {
            int(feature_map_4th.width/2), int(feature_map_4th.height/2)
        };
        Size feature_map_6th = {
            int(feature_map_5th.width/2), int(feature_map_5th.height/2)
        };

        std::vector<Size> feature_map_sizes;
        feature_map_sizes.push_back(feature_map_3rd);
        feature_map_sizes.push_back(feature_map_4th);
        feature_map_sizes.push_back(feature_map_5th);
        feature_map_sizes.push_back(feature_map_6th);

        // Fixed params for generating priors
        const std::vector<std::vector<float>> min_sizes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}
        };
        const std::vector<int> steps = { 8, 16, 32, 64 };

        // Generate priors
        priors.clear();
        for (size_t i = 0; i < feature_map_sizes.size(); ++i)
        {
            Size feature_map_size = feature_map_sizes[i];
            std::vector<float> min_size = min_sizes[i];

            for (int _h = 0; _h < feature_map_size.height; ++_h)
            {
                for (int _w = 0; _w < feature_map_size.width; ++_w)
                {
                    for (size_t j = 0; j < min_size.size(); ++j)
                    {
                        float s_kx = min_size[j] / inputW;
                        float s_ky = min_size[j] / inputH;

                        float cx = (_w + 0.5f) * steps[i] / inputW;
                        float cy = (_h + 0.5f) * steps[i] / inputH;

                        Rect2f prior = { cx, cy, s_kx, s_ky };
                        priors.push_back(prior);
                    }
                }
            }
        }
    }

    Mat postProcess(const std::vector<Mat>& output_blobs)
    {
        // Extract from output_blobs
        Mat loc = output_blobs[0];
        Mat conf = output_blobs[1];
        Mat iou = output_blobs[2];

        // Decode from deltas and priors
        const std::vector<float> variance = {0.1f, 0.2f};
        float* loc_v = (float*)(loc.data);
        float* conf_v = (float*)(conf.data);
        float* iou_v = (float*)(iou.data);
        Mat faces;
        // (tl_x, tl_y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score)
        // 'tl': top left point of the bounding box
        // 're': right eye, 'le': left eye
        // 'nt':  nose tip
        // 'rcm': right corner of mouth, 'lcm': left corner of mouth
        Mat face(1, 15, CV_32FC1);
        for (size_t i = 0; i < priors.size(); ++i) {
            // Get score
            float clsScore = conf_v[i*2+1];
            float iouScore = iou_v[i];
            // Clamp
            if (iouScore < 0.f) {
                iouScore = 0.f;
            }
            else if (iouScore > 1.f) {
                iouScore = 1.f;
            }
            float score = std::sqrt(clsScore * iouScore);
            face.at<float>(0, 14) = score;

            // Get bounding box
            float cx = (priors[i].x + loc_v[i*14+0] * variance[0] * priors[i].width)  * inputW;
            float cy = (priors[i].y + loc_v[i*14+1] * variance[0] * priors[i].height) * inputH;
            float w  = priors[i].width  * exp(loc_v[i*14+2] * variance[0]) * inputW;
            float h  = priors[i].height * exp(loc_v[i*14+3] * variance[1]) * inputH;
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            face.at<float>(0, 0) = x1;
            face.at<float>(0, 1) = y1;
            face.at<float>(0, 2) = w;
            face.at<float>(0, 3) = h;

            // Get landmarks
            face.at<float>(0, 4) = (priors[i].x + loc_v[i*14+ 4] * variance[0] * priors[i].width)  * inputW;  // right eye, x
            face.at<float>(0, 5) = (priors[i].y + loc_v[i*14+ 5] * variance[0] * priors[i].height) * inputH;  // right eye, y
            face.at<float>(0, 6) = (priors[i].x + loc_v[i*14+ 6] * variance[0] * priors[i].width)  * inputW;  // left eye, x
            face.at<float>(0, 7) = (priors[i].y + loc_v[i*14+ 7] * variance[0] * priors[i].height) * inputH;  // left eye, y
            face.at<float>(0, 8) = (priors[i].x + loc_v[i*14+ 8] * variance[0] * priors[i].width)  * inputW;  // nose tip, x
            face.at<float>(0, 9) = (priors[i].y + loc_v[i*14+ 9] * variance[0] * priors[i].height) * inputH;  // nose tip, y
            face.at<float>(0, 10) = (priors[i].x + loc_v[i*14+10] * variance[0] * priors[i].width)  * inputW; // right corner of mouth, x
            face.at<float>(0, 11) = (priors[i].y + loc_v[i*14+11] * variance[0] * priors[i].height) * inputH; // right corner of mouth, y
            face.at<float>(0, 12) = (priors[i].x + loc_v[i*14+12] * variance[0] * priors[i].width)  * inputW; // left corner of mouth, x
            face.at<float>(0, 13) = (priors[i].y + loc_v[i*14+13] * variance[0] * priors[i].height) * inputH; // left corner of mouth, y

            faces.push_back(face);
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
private:
    dnn::Net net;

    int inputW;
    int inputH;
    float scoreThreshold;
    float nmsThreshold;
    int topK;

    std::vector<Rect2f> priors;
};

Ptr<FaceDetectorYN> FaceDetectorYN::create(const String& model,
                                           const String& config,
                                           const Size& input_size,
                                           const float score_threshold,
                                           const float nms_threshold,
                                           const int top_k,
                                           const int backend_id,
                                           const int target_id)
{
    return makePtr<FaceDetectorYNImpl>(model, config, input_size, score_threshold, nms_threshold, top_k, backend_id, target_id);
}

} // namespace cv
