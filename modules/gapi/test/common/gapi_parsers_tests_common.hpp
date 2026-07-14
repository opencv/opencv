// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_PARSERS_TESTS_COMMON_HPP
#define OPENCV_GAPI_PARSERS_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/infer/parsers.hpp"

namespace opencv_test
{
class ParserSSDTest
{
public:
    cv::Mat generateSSDoutput(const cv::Size& in_sz)
    {
        constexpr int maxN = 200;
        constexpr int objSize = 7;
        std::vector<int> dims{ 1, 1, maxN, objSize };
        cv::Mat mat(dims, CV_32FC1);
        auto data = mat.ptr<float>();

        for (int i = 0; i < maxN; ++i)
        {
            float* it = data + i * objSize;
            auto ssdIt = generateItem(i, in_sz);
            it[0] = ssdIt.image_id;
            it[1] = ssdIt.label;
            it[2] = ssdIt.confidence;
            it[3] = ssdIt.rc_left;
            it[4] = ssdIt.rc_top;
            it[5] = ssdIt.rc_right;
            it[6] = ssdIt.rc_bottom;
        }
        return mat;
    }

    void parseSSDref(const cv::Mat& in_ssd_result,
                     const cv::Size& in_size,
                     const float confidence_threshold,
                     const bool alignment_to_square,
                     const bool filter_out_of_bounds,
                     std::vector<cv::Rect>& out_boxes)
    {
        out_boxes.clear();
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE  == 7); // fixed SSD object size

        const float *data = in_ssd_result.ptr<float>();
        cv::Rect surface({0,0}, in_size), rc;
        float image_id, confidence;
        int label;
        for (int i = 0; i < MAX_PROPOSALS; ++i)
        {
            std::tie(rc, image_id, confidence, label)
                = extract(data + i*OBJECT_SIZE, in_size);
            if (image_id < 0.f)
            {
                break;    // marks end-of-detections
            }

            if (confidence < confidence_threshold)
            {
                continue; // skip objects with low confidence
            }

            if (alignment_to_square)
            {
                adjustBoundingBox(rc);
            }

            const auto clipped_rc = rc & surface;
            if (filter_out_of_bounds)
            {
                if (clipped_rc.area() != rc.area())
                {
                    continue;
                }
            }
            out_boxes.emplace_back(clipped_rc);
        }
    }

    void parseSSDBLref(const cv::Mat& in_ssd_result,
                       const cv::Size& in_size,
                       const float confidence_threshold,
                       const int filter_label,
                       std::vector<cv::Rect>& out_boxes,
                       std::vector<int>& out_labels)
    {
        out_boxes.clear();
        out_labels.clear();
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE  == 7); // fixed SSD object size
        cv::Rect surface({0,0}, in_size), rc;
        float image_id, confidence;
        int label;
        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++)
        {
            std::tie(rc, image_id, confidence, label)
                = extract(data + i*OBJECT_SIZE, in_size);
            if (image_id < 0.f)
            {
                break;    // marks end-of-detections
            }

            if (confidence < confidence_threshold ||
                (filter_label != -1 && label != filter_label))
            {
                continue; // filter out object classes if filter is specified
            }

            out_boxes.emplace_back(rc & surface);
            out_labels.emplace_back(label);
        }
    }

private:
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

    std::tuple<cv::Rect, float, float, int> extract(const float* it,
                                                    const cv::Size& in_size)
    {
        float image_id   = it[0];
        int   label      = static_cast<int>(it[1]);
        float confidence = it[2];
        float rc_left    = it[3];
        float rc_top     = it[4];
        float rc_right   = it[5];
        float rc_bottom  = it[6];

        cv::Rect rc;  // map relative coordinates to the original image scale
        rc.x      = static_cast<int>(rc_left   * in_size.width);
        rc.y      = static_cast<int>(rc_top    * in_size.height);
        rc.width  = static_cast<int>(rc_right  * in_size.width)  - rc.x;
        rc.height = static_cast<int>(rc_bottom * in_size.height) - rc.y;
        return std::make_tuple(rc, image_id, confidence, label);
    }

    int randInRange(const int start, const int end)
    {
        GAPI_Assert(start <= end);
        return theRNG().uniform(start, end);
    }

    cv::Rect generateBox(const cv::Size& in_sz)
    {
        // Generated rectangle can reside outside of the initial image by border pixels
        constexpr int border = 10;
        constexpr int minW = 16;
        constexpr int minH = 16;
        cv::Rect box;
        box.width  = randInRange(minW, in_sz.width  + 2*border);
        box.height = randInRange(minH, in_sz.height + 2*border);
        box.x = randInRange(-border, in_sz.width  + border - box.width);
        box.y = randInRange(-border, in_sz.height + border - box.height);
        return box;
    }

    struct SSDitem
    {
        float image_id = 0.0f;
        float label = 0.0f;
        float confidence = 0.0f;
        float rc_left = 0.0f;
        float rc_top = 0.0f;
        float rc_right = 0.0f;
        float rc_bottom = 0.0f;
    };

    SSDitem generateItem(const int i, const cv::Size& in_sz)
    {
        const auto normalize = [](int v, int range) { return static_cast<float>(v) / range; };

        SSDitem it;
        it.image_id = static_cast<float>(i);
        it.label = static_cast<float>(randInRange(0, 9));
        it.confidence = theRNG().uniform(0.f, 1.f);
        auto box = generateBox(in_sz);
        it.rc_left   = normalize(box.x, in_sz.width);
        it.rc_right  = normalize(box.x + box.width, in_sz.width);
        it.rc_top    = normalize(box.y, in_sz.height);
        it.rc_bottom = normalize(box.y + box.height, in_sz.height);

        return it;
    }
};

class ParserYoloTest
{
public:
    cv::Mat generateYoloOutput(const int num_classes, std::pair<bool,int> dims_config = {false, 4})
    {
        bool one_dim = false;
        int num_dims = 0;
        std::tie(one_dim, num_dims) = dims_config;
        GAPI_Assert(num_dims <= 4);
        GAPI_Assert((!one_dim && num_dims >= 3) ||
                    ( one_dim && num_dims >= 1));
        std::vector<int> dims(num_dims, 1);
        if (one_dim) {
            dims.back() = (num_classes+5)*5*13*13;
        } else {
            dims.back() = (num_classes+5)*5;
            dims[num_dims-2] = 13;
            dims[num_dims-3] = 13;
        }
        cv::Mat mat(dims, CV_32FC1);
        auto data = mat.ptr<float>();

        const size_t range = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
        cv::RNG& rng = theRNG();
        for (size_t i = 0; i < range; ++i)
        {
            data[i] = rng.uniform(0.f, 1.f);
        }
        return mat;
    }

    void parseYoloRef(const cv::Mat&  in_yolo_result,
                      const cv::Size& in_size,
                      const float confidence_threshold,
                      const float nms_threshold,
                      const int num_classes,
                      const std::vector<float>& anchors,
                      std::vector<cv::Rect>& out_boxes,
                      std::vector<int>& out_labels)
    {
        YoloParams params;
        constexpr auto side_square = 13 * 13;
        this->m_out = in_yolo_result.ptr<float>();
        this->m_side = 13;
        this->m_lcoords = params.coords;
        this->m_lclasses = num_classes;

        std::vector<Detection> detections;

        for (int i = 0; i < side_square; ++i)
        {
            for (int b = 0; b < params.num; ++b)
            {
                float scale = this->scale(i, b);
                if (scale < confidence_threshold)
                {
                    continue;
                }
                double x = this->x(i, b);
                double y = this->y(i, b);
                double height = this->height(i, b, anchors[2 * b + 1]);
                double width = this->width(i, b, anchors[2 * b]);

                for (int label = 0; label < num_classes; ++label)
                {
                    float prob = scale * classConf(i,b,label);
                    if (prob < confidence_threshold)
                    {
                        continue;
                    }
                    auto box = toBox(x, y, height, width, in_size);
                    detections.emplace_back(Detection(box, prob, label));
                }
            }
        }
        std::stable_sort(std::begin(detections), std::end(detections),
                         [](const Detection& a, const Detection& b)
                         {
                             return a.conf > b.conf;
                         });

        if (nms_threshold < 1.0f)
        {
            for (const auto& d : detections)
            {
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

private:
    struct Detection
    {
        Detection(const cv::Rect& in_rect, const float in_conf, const int in_label)
            : rect(in_rect), conf(in_conf), label(in_label)
        {}
        cv::Rect rect;
        float    conf = 0.0f;
        int      label = 0;
    };

    struct YoloParams
    {
        int    num = 5;
        int coords = 4;
    };

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

    int index(const int i, const int b, const int entry)
    {
        return b * m_side * m_side * (m_lcoords + m_lclasses + 1) + entry * m_side * m_side + i;
    }

    const float* m_out = nullptr;
    int m_side = 0, m_lcoords = 0, m_lclasses = 0;
};

} // namespace opencv_test

#endif // OPENCV_GAPI_PARSERS_TESTS_COMMON_HPP
