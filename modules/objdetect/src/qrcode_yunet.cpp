#include "qrcode_yunet.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

#include "qrcode_yunet_model.inc"

// Helper function: sigmoid.
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

YunetWrapper::YunetWrapper()
{
    try {
        net_ = cv::dnn::readNetFromONNX(reinterpret_cast<const char*>(kYunetOnnxModel),
                                        static_cast<size_t>(kYunetOnnxModel_len));
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading embedded Yunet model: " << e.what() << std::endl;
        return;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    out_names_ = net_.getUnconnectedOutLayersNames();
}

YunetWrapper::YunetWrapper(const std::string& model_path)
{
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading Yunet model: " << e.what() << std::endl;
        return;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    out_names_ = net_.getUnconnectedOutLayersNames();
}

std::vector<int> YunetWrapper::nms(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   float thresh)
{
    std::vector<int> idx;
    if (boxes.empty()) return idx;
    cv::dnn::NMSBoxes(boxes, scores, 0.0f, thresh, idx);
    return idx;
}

bool YunetWrapper::detect(const cv::Mat& img, cv::Rect& out_box)
{
    if (net_.empty() || img.empty()) return false;

    // 1. Preprocessing (letterbox).
    int w = img.cols;
    int h = img.rows;
    float scale = std::min((float)input_w_ / w, (float)input_h_ / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;

    cv::Mat resized;
    if (w != new_w || h != new_h) cv::resize(img, resized, cv::Size(new_w, new_h));
    else resized = img;

    cv::Mat input_blob_img;
    cv::copyMakeBorder(resized, input_blob_img, 
                       dh, input_h_ - new_h - dh, 
                       dw, input_w_ - new_w - dw, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Mat blob = cv::dnn::blobFromImage(input_blob_img, 1.0, cv::Size(), cv::Scalar(0, 0, 0), false, false);
    net_.setInput(blob);

    // 2. Inference.
    std::vector<cv::Mat> outs;
    net_.forward(outs, out_names_);

    std::map<std::string, cv::Mat> out_map;
    for (size_t i = 0; i < out_names_.size(); ++i) out_map[out_names_[i]] = outs[i];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> strides = {8, 16, 32};

    // 3. Decode (adapted for flattened output).
    for (int stride : strides)
    {
        std::string layer_box = "bbox_" + std::to_string(stride);
        std::string layer_cls = "cls_" + std::to_string(stride);
        std::string layer_obj = "obj_" + std::to_string(stride);

        if (out_map.find(layer_box) == out_map.end()) continue;

        const cv::Mat& box_mat = out_map[layer_box];
        const cv::Mat& cls_mat = out_map[layer_cls];
        const cv::Mat& obj_mat = out_map[layer_obj];

        // Compute the expected grid size for the current stride.
        int grid_w = input_w_ / stride; // e.g., 640/8 = 80
        int grid_h = input_h_ / stride; // e.g., 640/8 = 80
        int num_anchors = grid_w * grid_h;

        // Get data pointers (assume float and continuous storage).
        const float* ptr_box = (float*)box_mat.data;
        const float* ptr_cls = (float*)cls_mat.data;
        const float* ptr_obj = (float*)obj_mat.data;

        // Detect the tensor layout automatically.
        // If the layout is [1, C, H, W], the step is usually large.
        // If the layout is [1, N, C], the step is usually C.
        // Use the most robust approach: iterate by the total number of anchors
        // and derive the per-anchor stride from total_size / num_anchors.
        
        int total_elements_box = box_mat.total();
        int step_box = total_elements_box / num_anchors; // Should be 4.

        int total_elements_cls = cls_mat.total();
        int step_cls = total_elements_cls / num_anchors; // Should be the class count.

        int total_elements_obj = obj_mat.total();
        int step_obj = total_elements_obj / num_anchors; // Should be 1.

        for (int i = 0; i < num_anchors; i++)
        {
            // 1. Compute the anchor coordinates in the grid (critical fix).
            int grid_y = i / grid_w;
            int grid_x = i % grid_w;

            // 2. Read objectness.
            // If step_obj == 1, ptr_obj[i] is used directly. For NCHW layouts this
            // logic may differ, but the logs indicate a flattened layout (N,1) or (1,N,1).
            float obj_score = sigmoid(ptr_obj[i * step_obj]);
            if (obj_score < 0.1f) continue;

            // 3. Read class scores and keep only the QR Code class (cls_id == 3).
            float max_cls_score = 0.f;
            int argmax_cls = -1;
            for (int c = 0; c < step_cls; c++) {
                float s = sigmoid(ptr_cls[i * step_cls + c]);
                if (s > max_cls_score) { max_cls_score = s; argmax_cls = c; }
            }
            if (argmax_cls != 3) continue;  // Keep only the QR Code class.

            float final_score = max_cls_score * obj_score;
            if (final_score < 0.02f) continue;

            // 4. Decode the box.
            // Pointer offset: anchor i plus offsets 0..3.
            float r0 = ptr_box[i * step_box + 0]; // x
            float r1 = ptr_box[i * step_box + 1]; // y
            float r2 = ptr_box[i * step_box + 2]; // w
            float r3 = ptr_box[i * step_box + 3]; // h

            // Critical fix: use grid_x/grid_y here instead of the loop index i.
            float cx = (r0 * stride) + (grid_x * stride);
            float cy = (r1 * stride) + (grid_y * stride);

            float w_pred = std::exp(r2) * stride;
            float h_pred = std::exp(r3) * stride;

            float x1 = cx - w_pred / 2.0f;
            float y1 = cy - h_pred / 2.0f;

            if (w_pred > 0 && h_pred > 0) {
                boxes.emplace_back(x1, y1, w_pred, h_pred);
                scores.push_back(final_score);
            }
        }
    }

    if (boxes.empty()) return false;

    // 4. NMS
    std::vector<int> keep = nms(boxes, scores, 0.45f);
    if (keep.empty()) return false;

    cv::Rect best_box_net = boxes[keep[0]];

    // 5. Map coordinates back to the original image.
    float x_final = (best_box_net.x - dw) / scale;
    float y_final = (best_box_net.y - dh) / scale;
    float w_final = best_box_net.width / scale;
    float h_final = best_box_net.height / scale;

    int x1 = std::max(0, (int)x_final);
    int y1 = std::max(0, (int)y_final);
    int x2 = std::min(w, (int)(x_final + w_final));
    int y2 = std::min(h, (int)(y_final + h_final));

    out_box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    
    // Add a small robustness margin.
    return (out_box.width > 2 && out_box.height > 2);
}


// ============================================================
// detectMulti
// ------------------------------------------------------------
// Same preprocessing / decode / NMS as detect.
// The only difference is:
//   1. Keep all boxes after NMS.
//   2. Apply inverse letterbox to every box.
// ============================================================
bool YunetWrapper::detectMulti(
    const cv::Mat& img,
    std::vector<cv::Rect>& out_boxes)
{
    out_boxes.clear();
    if (net_.empty() || img.empty())
        return false;

    // ----------------------------
    // 1. Letterbox preprocessing (same as detect).
    // ----------------------------
    int w = img.cols;
    int h = img.rows;

    float scale = std::min((float)input_w_ / w, (float)input_h_ / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);

    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;

    cv::Mat resized;
    if (w != new_w || h != new_h)
        cv::resize(img, resized, cv::Size(new_w, new_h));
    else
        resized = img;

    cv::Mat input_blob_img;
    cv::copyMakeBorder(
        resized,
        input_blob_img,
        dh, input_h_ - new_h - dh,
        dw, input_w_ - new_w - dw,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    cv::Mat blob = cv::dnn::blobFromImage(
        input_blob_img,
        1.0,
        cv::Size(),
        cv::Scalar(0, 0, 0),
        false,
        false
    );

    net_.setInput(blob);

    // ----------------------------
    // 2. Inference.
    // ----------------------------
    std::vector<cv::Mat> outs;
    net_.forward(outs, out_names_);

    std::map<std::string, cv::Mat> out_map;
    for (size_t i = 0; i < out_names_.size(); ++i)
        out_map[out_names_[i]] = outs[i];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    std::vector<int> strides = {8, 16, 32};

    // ----------------------------
    // 3. Decode (same as detect).
    // ----------------------------
    for (int stride : strides)
    {
        std::string layer_box = "bbox_" + std::to_string(stride);
        std::string layer_cls = "cls_"  + std::to_string(stride);
        std::string layer_obj = "obj_"  + std::to_string(stride);

        if (out_map.find(layer_box) == out_map.end())
            continue;

        const cv::Mat& box_mat = out_map[layer_box];
        const cv::Mat& cls_mat = out_map[layer_cls];
        const cv::Mat& obj_mat = out_map[layer_obj];

        int grid_w = input_w_ / stride;
        int grid_h = input_h_ / stride;
        int num_anchors = grid_w * grid_h;

        const float* ptr_box = (float*)box_mat.data;
        const float* ptr_cls = (float*)cls_mat.data;
        const float* ptr_obj = (float*)obj_mat.data;

        int step_box = box_mat.total() / num_anchors;
        int step_cls = cls_mat.total() / num_anchors;
        int step_obj = obj_mat.total() / num_anchors;

        for (int i = 0; i < num_anchors; i++)
        {
            int grid_y = i / grid_w;
            int grid_x = i % grid_w;

            float obj_score = sigmoid(ptr_obj[i * step_obj]);
            if (obj_score < 0.02f) continue;

            // Keep only the QR Code class (cls_id == 3).
            float max_cls_score = 0.f;
            int argmax_cls = -1;
            for (int c = 0; c < step_cls; c++) {
                float s = sigmoid(ptr_cls[i * step_cls + c]);
                if (s > max_cls_score) { max_cls_score = s; argmax_cls = c; }
            }
            if (argmax_cls != 3) continue;

            float final_score = max_cls_score * obj_score;
            if (final_score < 0.1f) continue;

            float r0 = ptr_box[i * step_box + 0];
            float r1 = ptr_box[i * step_box + 1];
            float r2 = ptr_box[i * step_box + 2];
            float r3 = ptr_box[i * step_box + 3];

            float cx = r0 * stride + grid_x * stride;
            float cy = r1 * stride + grid_y * stride;

            float bw = std::exp(r2) * stride;
            float bh = std::exp(r3) * stride;

            boxes.emplace_back(
                cx - bw * 0.5f,
                cy - bh * 0.5f,
                bw,
                bh
            );
            scores.push_back(final_score);
        }
    }

    if (boxes.empty())
        return false;

    // ----------------------------
    // 4. NMS (keep all selected boxes).
    // ----------------------------
    std::vector<int> keep = nms(boxes, scores, 0.45f);
    if (keep.empty())
        return false;

    const int MAX_BOXES = 500;

    // Sort keep by score in descending order.
    std::sort(keep.begin(), keep.end(),
            [&](int a, int b) {
                return scores[a] > scores[b];
            });

    // Keep at most 500 boxes.
    if ((int)keep.size() > MAX_BOXES)
        keep.resize(MAX_BOXES);

    // ----------------------------
    // 5. Apply inverse letterbox to all boxes.
    // ----------------------------
    for (int idx : keep)
    {
        const cv::Rect& b = boxes[idx];

        float x = (b.x - dw) / scale;
        float y = (b.y - dh) / scale;
        float w0 = b.width  / scale;
        float h0 = b.height / scale;

        int x1 = std::max(0, (int)x);
        int y1 = std::max(0, (int)y);
        int x2 = std::min(w, (int)(x + w0));
        int y2 = std::min(h, (int)(y + h0));

        if (x2 > x1 && y2 > y1)
            out_boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
    }

    return !out_boxes.empty();
}
