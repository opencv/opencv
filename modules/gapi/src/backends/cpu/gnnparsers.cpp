// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "gnnparsers.hpp"

namespace cv
{
namespace gapi
{
namespace nn
{
class YoloParser
{
public:
    YoloParser(const float* out, const int side, const int lcoords, const int lclasses)
        : m_out(out), m_side(side), m_lcoords(lcoords), m_lclasses(lclasses)
    {}

    float scale(const int i, const int b)
    {
        int obj_index = index(i, b, m_lcoords);
        return m_out[obj_index];
    }

    double x(const int i, const int b)
    {
        int box_index = index(i, b, 0);
        int col = i % m_side;
        return (col + m_out[box_index]) / m_side;
    }

    double y(const int i, const int b)
    {
        int box_index = index(i, b, 0);
        int row = i / m_side;
        return (row + m_out[box_index + m_side * m_side]) / m_side;
    }

    double width(const int i, const int b, const float anchor)
    {
        int box_index = index(i, b, 0);
        return std::exp(m_out[box_index + 2 * m_side * m_side]) * anchor / m_side;
    }

    double height(const int i, const int b, const float anchor)
    {
        int box_index = index(i, b, 0);
        return std::exp(m_out[box_index + 3 * m_side * m_side]) * anchor / m_side;
    }

    float classConf(const int i, const int b, const int label)
    {
         int class_index = index(i, b, m_lcoords + 1 + label);
         return m_out[class_index];
    }

    cv::Rect toBox(const double x, const double y, const double h, const double w, const cv::Size& in_sz)
    {
        auto h_scale = in_sz.height;
        auto w_scale = in_sz.width;
        cv::Rect r;
        r.x = static_cast<int>((x - w / 2) * w_scale);
        r.y = static_cast<int>((y - h / 2) * h_scale);
        r.width = static_cast<int>(w * w_scale);
        r.height = static_cast<int>(h * h_scale);
        return r;
    }

private:
    const float* m_out = nullptr;
    int m_side = 0, m_lcoords = 0, m_lclasses = 0;

    int index(const int i, const int b, const int entry)
    {
        return b * m_side * m_side * (m_lcoords + m_lclasses + 1) + entry * m_side * m_side + i;
    }
};

struct YoloParams
{
    int    num = 5;
    int coords = 4;
};

struct Detection
{
    Detection(const cv::Rect& in_rect, const float in_conf, const int in_label)
        : rect(in_rect), conf(in_conf), label(in_label)
    {}
    cv::Rect rect;
    float    conf = 0.0f;
    int      label = 0;
};

class SSDParser
{
public:
    SSDParser(const cv::MatSize& in_ssd_dims, const cv::Size& in_size, const float* data)
        : m_dims(in_ssd_dims), m_maxProp(in_ssd_dims[2]), m_objSize(in_ssd_dims[3]),
          m_data(data), m_surface(cv::Rect({0,0}, in_size)), m_size(in_size)
    {
        GAPI_Assert(in_ssd_dims.dims() == 4u); // Fixed output layout
        GAPI_Assert(m_objSize  == 7);          // Fixed SSD object size
    }

    void adjustBoundingBox(cv::Rect& boundingBox)
    {
        auto w = boundingBox.width;
        auto h = boundingBox.height;

        boundingBox.x -= static_cast<int>(0.067 * w);
        boundingBox.y -= static_cast<int>(0.028 * h);

        boundingBox.width += static_cast<int>(0.15 * w);
        boundingBox.height += static_cast<int>(0.13 * h);

        if (boundingBox.width < boundingBox.height)
        {
            auto dx = (boundingBox.height - boundingBox.width);
            boundingBox.x -= dx / 2;
            boundingBox.width += dx;
        }
        else
        {
            auto dy = (boundingBox.width - boundingBox.height);
            boundingBox.y -= dy / 2;
            boundingBox.height += dy;
        }
    }

    std::tuple<cv::Rect, float, float, int> extract(const size_t step)
    {
        const float* it = m_data + step * m_objSize;
        float image_id   = it[0];
        int   label      = static_cast<int>(it[1]);
        float confidence = it[2];
        float rc_left    = it[3];
        float rc_top     = it[4];
        float rc_right   = it[5];
        float rc_bottom  = it[6];

        cv::Rect rc;  // Map relative coordinates to the original image scale
        rc.x      = static_cast<int>(rc_left   * m_size.width);
        rc.y      = static_cast<int>(rc_top    * m_size.height);
        rc.width  = static_cast<int>(rc_right  * m_size.width)  - rc.x;
        rc.height = static_cast<int>(rc_bottom * m_size.height) - rc.y;
        return std::make_tuple(rc, image_id, confidence, label);
    }

    int getMaxProposals()
    {
        return m_maxProp;
    }

    cv::Rect getSurface()
    {
        return m_surface;
    }

private:
    const cv::MatSize m_dims;
    int m_maxProp = 0, m_objSize = 0;
    const float* m_data = nullptr;
    const cv::Rect m_surface;
    const cv::Size m_size;
};
} // namespace nn
} // namespace gapi

void ParseSSD(const cv::Mat&  in_ssd_result,
              const cv::Size& in_size,
              const float     confidence_threshold,
              const int       filter_label,
              const bool      alignment_to_square,
              const bool      filter_out_of_bounds,
              std::vector<cv::Rect>& out_boxes,
              std::vector<int>&      out_labels)
{
    cv::gapi::nn::SSDParser parser(in_ssd_result.size, in_size, in_ssd_result.ptr<float>());
    out_boxes.clear();
    out_labels.clear();
    cv::Rect rc;
    float image_id, confidence;
    int label;
    const size_t range = parser.getMaxProposals();
    for (size_t i = 0; i < range; ++i)
    {
        std::tie(rc, image_id, confidence, label) = parser.extract(i);

        if (image_id < 0.f)
        {
            break;    // marks end-of-detections
        }
        if (confidence < confidence_threshold)
        {
            continue; // skip objects with low confidence
        }
        if((filter_label != -1) && (label != filter_label))
        {
            continue; // filter out object classes if filter is specified
        }
        if (alignment_to_square)
        {
            parser.adjustBoundingBox(rc);
        }
        const auto clipped_rc = rc & parser.getSurface();
        if (filter_out_of_bounds)
        {
            if (clipped_rc.area() != rc.area())
            {
                continue;
            }
        }
        out_boxes.emplace_back(clipped_rc);
        out_labels.emplace_back(label);
    }
}

static void checkYoloDims(const MatSize& dims) {
    const auto d = dims.dims();
    // Accept 1x13x13xN and 13x13xN
    GAPI_Assert(d >= 2);
    if (d >= 3) {
        if (dims[d-2] == 13) {
            GAPI_Assert(dims[d-1]%5 == 0);
            GAPI_Assert(dims[d-2] == 13);
            GAPI_Assert(dims[d-3] == 13);
            for (int i = 0; i < d-3; i++) {
                GAPI_Assert(dims[i] == 1);
            }
            return;
        }
    }
    // Accept 1x1x1xN, 1x1xN, 1xN
    GAPI_Assert(dims[d-1]%(5*13*13) == 0);
    for (int i = 0; i < d-1; i++) {
        GAPI_Assert(dims[i] == 1);
    }
}

void parseYolo(const cv::Mat&  in_yolo_result,
               const cv::Size& in_size,
               const float     confidence_threshold,
               const float     nms_threshold,
               const std::vector<float>& anchors,
               std::vector<cv::Rect>& out_boxes,
               std::vector<int>&      out_labels)
{
    const auto& dims = in_yolo_result.size;
    checkYoloDims(dims);
    int acc = 1;
    for (int i = 0; i < dims.dims(); i++) {
        acc *= dims[i];
    }
    const auto num_classes = acc/(5*13*13)-5;
    GAPI_Assert(num_classes > 0);
    GAPI_Assert(0 < nms_threshold && nms_threshold <= 1);
    out_boxes.clear();
    out_labels.clear();
    gapi::nn::YoloParams params;
    constexpr auto side = 13;
    constexpr auto side_square = side * side;
    const auto output = in_yolo_result.ptr<float>();

    gapi::nn::YoloParser parser(output, side, params.coords, num_classes);

    std::vector<gapi::nn::Detection> detections;

    for (int i = 0; i < side_square; ++i)
    {
        for (int b = 0; b < params.num; ++b)
        {
            float scale = parser.scale(i, b);
            if (scale < confidence_threshold)
            {
                continue;
            }
            double x = parser.x(i, b);
            double y = parser.y(i, b);
            double height = parser.height(i, b, anchors[2 * b + 1]);
            double width = parser.width(i, b, anchors[2 * b]);

            for (int label = 0; label < num_classes; ++label)
            {
                float prob = scale * parser.classConf(i,b,label);
                if (prob < confidence_threshold)
                {
                    continue;
                }
                auto box = parser.toBox(x, y, height, width, in_size);
                detections.emplace_back(gapi::nn::Detection(box, prob, label));
            }
        }
    }
    std::stable_sort(std::begin(detections), std::end(detections),
                     [](const gapi::nn::Detection& a, const gapi::nn::Detection& b)
                     {
                         return a.conf > b.conf;
                     });

    if (nms_threshold < 1.0f)
    {
        for (const auto& d : detections)
        {
            // Reject boxes which overlap with previously pushed ones
            // (They are sorted by confidence, so rejected box
            // always has a smaller confidence
            if (std::end(out_boxes) ==
                std::find_if(std::begin(out_boxes), std::end(out_boxes),
                             [&d, nms_threshold](const cv::Rect& r)
                             {
                                 float rectOverlap = 1.f - static_cast<float>(jaccardDistance(r, d.rect));
                                 return rectOverlap > nms_threshold;
                             }))
            {
                out_boxes. emplace_back(d.rect);
                out_labels.emplace_back(d.label);
            }
        }
    }
    else
    {
        for (const auto& d: detections)
        {
            out_boxes. emplace_back(d.rect);
            out_labels.emplace_back(d.label);
        }
    }
}
} // namespace cv
