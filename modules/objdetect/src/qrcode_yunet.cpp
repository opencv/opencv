#include "qrcode_yunet.hpp"
#include <algorithm>
#include <cmath>
#include <map>

// Sigmoid helper.
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

YunetWrapper::YunetWrapper(const std::string& model_path)
{
    try {
        net_ = cv::dnn::readNet(model_path);
    } catch (const cv::Exception&) {
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

    // 1. Letterbox preprocessing.
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

    // 3. Decode flattened outputs.
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

        // Get contiguous float output pointers.
        const float* ptr_box = (float*)box_mat.data;
        const float* ptr_cls = (float*)cls_mat.data;
        const float* ptr_obj = (float*)obj_mat.data;

        // Infer the output layout from the total element count.
        // For [1, C, H, W], the step is usually large; for [1, N, C], the step is C.
        // Iterate by anchor count and derive the per-anchor step from total_size / num_anchors.
        
        int total_elements_box = box_mat.total();
        int step_box = total_elements_box / num_anchors; // expected to be 4

        int total_elements_cls = cls_mat.total();
        int step_cls = total_elements_cls / num_anchors; // expected to be the class count

        int total_elements_obj = obj_mat.total();
        int step_obj = total_elements_obj / num_anchors; // expected to be 1

        for (int i = 0; i < num_anchors; i++)
        {
            // 1. Compute the current anchor's grid coordinates.
            int grid_y = i / grid_w;
            int grid_x = i % grid_w;

            // 2. Read objectness.
            // With step_obj == 1, ptr_obj[i] matches flattened (N,1) or (1,N,1) output.
            float obj_score = sigmoid(ptr_obj[i * step_obj]);
            if (obj_score < 0.02f) continue;

            // 3. Read the class score and keep only the QR Code class (cls_id == 3).
            float max_cls_score = 0.f;
            int argmax_cls = -1;
            for (int c = 0; c < step_cls; c++) {
                float s = sigmoid(ptr_cls[i * step_cls + c]);
                if (s > max_cls_score) { max_cls_score = s; argmax_cls = c; }
            }
            if (argmax_cls != 3) continue;

            float final_score = max_cls_score * obj_score;
            if (final_score < 0.02f) continue;

            // 4. Decode Box
            // Offset into the i-th anchor's four box values.
            float r0 = ptr_box[i * step_box + 0]; // x
            float r1 = ptr_box[i * step_box + 1]; // y
            float r2 = ptr_box[i * step_box + 2]; // w
            float r3 = ptr_box[i * step_box + 3]; // h

            // Use the grid coordinates rather than the flat loop index.
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

    // 5. Restore coordinates to the original image.
    float x_final = (best_box_net.x - dw) / scale;
    float y_final = (best_box_net.y - dh) / scale;
    float w_final = best_box_net.width / scale;
    float h_final = best_box_net.height / scale;

    int x1 = std::max(0, (int)x_final);
    int y1 = std::max(0, (int)y_final);
    int x2 = std::min(w, (int)(x_final + w_final));
    int y2 = std::min(h, (int)(y_final + h_final));

    out_box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
    
    // Reject tiny boxes for robustness.
    return (out_box.width > 2 && out_box.height > 2);
}


// ============================================================
// detectMulti
// ------------------------------------------------------------
// The preprocessing, decode, and NMS steps match detect().
// Differences:
//   1. Keep all boxes returned by NMS.
//   2. Apply inverse letterbox to each box.
// ============================================================
bool YunetWrapper::detectMulti(
    const cv::Mat& img,
    std::vector<cv::Rect>& out_boxes)
{
    const float confidence_threshold = 0.2f;
    const int max_boxes = 50;

    out_boxes.clear();
    if (net_.empty() || img.empty())
        return false;

    // ----------------------------
    // 1. Letterbox preprocessing, matching detect().
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
    // 3. Decode, matching detect().
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
            if (final_score < confidence_threshold) continue;

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
    // 4. NMS, preserving all kept indices.
    // ----------------------------
    std::vector<int> keep = nms(boxes, scores, 0.45f);
    if (keep.empty())
        return false;

    // Sort kept indices by score in descending order.
    std::sort(keep.begin(), keep.end(),
            [&](int a, int b) {
                return scores[a] > scores[b];
            });

    // Keep only the highest-scoring candidates when the limit is exceeded.
    if ((int)keep.size() > max_boxes)
        keep.resize(max_boxes);

    // ----------------------------
    // 5. Apply inverse letterbox to every kept box.
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