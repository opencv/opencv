#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
using namespace cv;

cv::Mat postprocess(const cv::Mat& blob, const float confidence_threshold = 0.5, const float nms_threshold = 0.5)
{
    std::vector<int> strides{8, 16, 32};
    std::vector<int> hsizes{80, 40, 20};
    std::vector<int> wsizes{80, 40, 20};

    std::vector<Point2f> grids(8400);
    std::vector<float> expanded_strides(8400);
    int i, j, k, l = 0, h, w;
    for (i = 0; i < hsizes.size(); i++)
    {
        h = hsizes[i];
        w = wsizes[i];
        for (j = 0; j < h; j++)
        {
            for (k = 0; k < w; k++)
            {
                Point2f grid{float(k), float(j)};
                grids[l] = grid;
                expanded_strides[l] = float(strides[i]);
                l++;
            }
        }
    }

    const float* p_delta = (const float*)blob.data;

    Mat outs;
    Mat out(1, 6, CV_32FC1);
    for (i = 0; i < 8400; i++)
    {
        j = i * 85;
        Point2f grid = grids[i];
        float expanded_stride = expanded_strides[i];

        // retrieve objectness score
        float objectness = p_delta[j + 4];

        // retrieve class scores
        float max_score = -1.f;
        float max_idx = -1.f;
        float this_score;
        for (k = 5; k < 85; k++)
        {
            this_score = p_delta[j + k] * objectness;
            if (this_score > max_score)
            {
                max_score = this_score;
                max_idx = k - 5;
            }
        }
        if (max_score < 0.5)
            continue;
        out.at<float>(0, 4) = max_score;
        out.at<float>(0, 5) = max_idx;

        // retrieve bbox
        float cx = (p_delta[j] + grid.x) * expanded_stride;
        float cy = (p_delta[j + 1] + grid.y) * expanded_stride;
        float width = std::exp(p_delta[j + 2]) * expanded_stride;
        float height = std::exp(p_delta[j + 3]) * expanded_stride;
        out.at<float>(0, 0) = cx - width / 2;
        out.at<float>(0, 1) = cy - height / 2;
        out.at<float>(0, 2) = cx + width / 2;
        out.at<float>(0, 3) = cy + height / 2;

        outs.push_back(out);
    }

    Mat dets = outs;
    if (dets.rows > 1)
    {
        // batched nms
        float max_coord = -1;
        for (i = 0; i < dets.rows; i++)
            for (j = 0; j < 4; j++)
                if (max_coord < dets.at<float>(i, j))
                    max_coord = dets.at<float>(i, j);

        std::vector<Rect2i> boxes;
        std::vector<float> scores;
        float offsets;
        for (i = 0; i < dets.rows; i++)
        {
            offsets = dets.at<float>(i, 5) * (max_coord + 1);
            boxes.push_back(
                Rect2i(int(dets.at<float>(i, 0) + offsets),
                       int(dets.at<float>(i, 1) + offsets),
                       int(dets.at<float>(i, 2) + offsets),
                       int(dets.at<float>(i, 3) + offsets))
            );
            scores.push_back(dets.at<float>(i, 4));
        }
        std::vector<int> keep;
        dnn::NMSBoxes(boxes, scores, 0.5, 0.5, keep);

        Mat dets_after_nms;
        for (auto idx : keep)
            dets_after_nms.push_back(dets.row(idx));

        dets = dets_after_nms;
    }
    return dets;
}

int main(int argc, char** argv)
{
    Mat image = imread("/path/to/image"); // replace with the path to your image
    Mat input_blob = dnn::blobFromImage(image, 1.0f, cv::Size(640, 640), cv::Scalar(), true);

    dnn::Net net = dnn::readNet("/path/to/object_detection_yolox_2022nov.onnx"); // replace with the path to the model
    net.setPreferableBackend(dnn::DNN_BACKEND_CANN);
    net.setPreferableTarget(dnn::DNN_TARGET_NPU);

    net.setInput(input_blob);
    Mat out = net.forward();

    Mat dets = postprocess(out);
    for (int i = 0; i < dets.rows; i++)
    {
        int x1 = int(dets.at<float>(i, 0));
        int y1 = int(dets.at<float>(i, 1));
        int x2 = int(dets.at<float>(i, 2));
        int y2 = int(dets.at<float>(i, 3));
        float score = dets.at<float>(i, 4);
        int cls = int(dets.at<float>(i, 5));
        std::cout << cv::format("box [%d, %d, %d, %d], score %f, class %d\n", x1, y1, x2, y2, score, cls);
    }

    dnn::Net::finalizeDevice();
    return 0;
}
