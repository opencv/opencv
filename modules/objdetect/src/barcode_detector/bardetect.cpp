// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "../precomp.hpp"
#include "bardetect.hpp"


namespace cv {
namespace barcode {
static constexpr float PI = static_cast<float>(CV_PI);
static constexpr float HALF_PI = static_cast<float>(CV_PI / 2);

#define CALCULATE_SUM(ptr, result) \
    top_left = static_cast<float>(*((ptr) + left_col + integral_cols * top_row));\
    top_right = static_cast<float>(*((ptr) + integral_cols * top_row + right_col));\
    bottom_right = static_cast<float>(*((ptr) + right_col + bottom_row * integral_cols));\
    bottom_left = static_cast<float>(*((ptr) + bottom_row * integral_cols + left_col));\
    (result) = (bottom_right - bottom_left - top_right + top_left);


inline bool Detect::isValidCoord(const Point &coord, const Size &limit)
{
    if ((coord.x < 0) || (coord.y < 0))
    {
        return false;
    }

    if ((unsigned) coord.x > (unsigned) (limit.width - 1) || ((unsigned) coord.y > (unsigned) (limit.height - 1)))
    {
        return false;
    }

    return true;
}

//==============================================================================
// NMSBoxes copied from modules/dnn/src/nms.inl.hpp
// TODO: move NMSBoxes outside the dnn module to allow other modules use it

namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    CV_DbgAssert(score_index_vec.empty());
    // Generate index score pairs.
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], (int)i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);

    // Keep top_k scores if needed.
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices,
      float (*computeOverlap)(const BoxType&, const BoxType&),
      size_t limit = std::numeric_limits<int>::max())
{
    CV_Assert(bboxes.size() == scores.size());

    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep) {
            indices.push_back(idx);
            if (indices.size() >= limit) {
                break;
            }
        }
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}

static inline float rotatedRectIOU(const RotatedRect& a, const RotatedRect& b)
{
    std::vector<Point2f> inter;
    int res = rotatedRectangleIntersection(a, b, inter);
    if (inter.empty() || res == INTERSECT_NONE)
        return 0.0f;
    if (res == INTERSECT_FULL)
        return 1.0f;
    float interArea = (float)contourArea(inter);
    return interArea / (a.size.area() + b.size.area() - interArea);
}

static void NMSBoxes(const std::vector<RotatedRect>& bboxes, const std::vector<float>& scores,
              const float score_threshold, const float nms_threshold,
              std::vector<int>& indices, const float eta = 1.f, const int top_k = 0)
{
    CV_Assert_N(bboxes.size() == scores.size(), score_threshold >= 0,
        nms_threshold >= 0, eta > 0);
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rotatedRectIOU);
}

} // namespace <anonymous>::


//==============================================================================

void Detect::init(const Mat &src)
{
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        width = cvRound(src.size().width / coeff_expansion);
        height = cvRound(src.size().height / coeff_expansion);
        Size new_size(width, height);
        resize(src, resized_barcode, new_size, 0, 0, INTER_AREA);
    }
//    else if (min_side < 512.0)
//    {
//        purpose = ZOOMING;
//        coeff_expansion = 512.0 / min_side;
//        width = cvRound(src.size().width * coeff_expansion);
//        height = cvRound(src.size().height * coeff_expansion);
//        Size new_size(width, height);
//        resize(src, resized_barcode, new_size, 0, 0, INTER_CUBIC);
//    }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
        width = src.size().width;
        height = src.size().height;
        resized_barcode = src.clone();
    }
    // median blur: sometimes it reduces the noise, but also reduces the recall
    // medianBlur(resized_barcode, resized_barcode, 3);

}


void Detect::localization()
{

    localization_bbox.clear();
    bbox_scores.clear();

    // get integral image
    preprocess();
    // empirical setting
    static constexpr float SCALE_LIST[] = {0.01f, 0.03f, 0.06f, 0.08f};
    const auto min_side = static_cast<float>(std::min(width, height));
    int window_size;
    for (const float scale:SCALE_LIST)
    {
        window_size = cvRound(min_side * scale);
        if(window_size == 0) {
            window_size = 1;
        }
        calCoherence(window_size);
        barcodeErode();
        regionGrowing(window_size);
    }

}


bool Detect::computeTransformationPoints()
{

    bbox_indices.clear();
    transformation_points.clear();
    transformation_points.reserve(bbox_indices.size());
    RotatedRect rect;
    Point2f temp[4];
    const float THRESHOLD_SCORE = float(width * height) / 300.f;
    NMSBoxes(localization_bbox, bbox_scores, THRESHOLD_SCORE, 0.1f, bbox_indices);

    for (const auto &bbox_index : bbox_indices)
    {
        rect = localization_bbox[bbox_index];
        if (purpose == ZOOMING)
        {
            rect.center /= coeff_expansion;
            rect.size.height /= static_cast<float>(coeff_expansion);
            rect.size.width /= static_cast<float>(coeff_expansion);
        }
        else if (purpose == SHRINKING)
        {
            rect.center *= coeff_expansion;
            rect.size.height *= static_cast<float>(coeff_expansion);
            rect.size.width *= static_cast<float>(coeff_expansion);
        }
        rect.points(temp);
        transformation_points.emplace_back(vector<Point2f>{temp[0], temp[1], temp[2], temp[3]});
    }

    return !transformation_points.empty();
}


void Detect::preprocess()
{
    Mat scharr_x, scharr_y, temp;
    static constexpr double THRESHOLD_MAGNITUDE = 64.;
    Scharr(resized_barcode, scharr_x, CV_32F, 1, 0);
    Scharr(resized_barcode, scharr_y, CV_32F, 0, 1);
    // calculate magnitude of gradient and truncate
    magnitude(scharr_x, scharr_y, temp);
    threshold(temp, temp, THRESHOLD_MAGNITUDE, 1, THRESH_BINARY);
    temp.convertTo(gradient_magnitude, CV_8U);
    integral(gradient_magnitude, integral_edges, CV_32F);


    for (int y = 0; y < height; y++)
    {
        auto *const x_row = scharr_x.ptr<float_t>(y);
        auto *const y_row = scharr_y.ptr<float_t>(y);
        auto *const magnitude_row = gradient_magnitude.ptr<uint8_t>(y);
        for (int pos = 0; pos < width; pos++)
        {
            if (magnitude_row[pos] == 0)
            {
                x_row[pos] = 0;
                y_row[pos] = 0;
                continue;
            }
            if (x_row[pos] < 0)
            {
                x_row[pos] *= -1;
                y_row[pos] *= -1;
            }
        }
    }
    integral(scharr_x, temp, integral_x_sq, CV_32F, CV_32F);
    integral(scharr_y, temp, integral_y_sq, CV_32F, CV_32F);
    integral(scharr_x.mul(scharr_y), integral_xy, temp, CV_32F, CV_32F);
}


// Change coherence orientation edge_nums
// depend on width height integral_edges integral_x_sq integral_y_sq integral_xy
void Detect::calCoherence(int window_size)
{
    static constexpr float THRESHOLD_COHERENCE = 0.9f;
    int right_col, left_col, top_row, bottom_row;
    float xy, x_sq, y_sq, d, rect_area;
    const float THRESHOLD_AREA = float(window_size * window_size) * 0.42f;
    Size new_size(width / window_size, height / window_size);
    coherence = Mat(new_size, CV_8U), orientation = Mat(new_size, CV_32F), edge_nums = Mat(new_size, CV_32F);

    float top_left, top_right, bottom_left, bottom_right;
    int integral_cols = width + 1;
    const auto *edges_ptr = integral_edges.ptr<float_t>(), *x_sq_ptr = integral_x_sq.ptr<float_t>(), *y_sq_ptr = integral_y_sq.ptr<float_t>(), *xy_ptr = integral_xy.ptr<float_t>();
    for (int y = 0; y < new_size.height; y++)
    {
        auto *coherence_row = coherence.ptr<uint8_t>(y);
        auto *orientation_row = orientation.ptr<float_t>(y);
        auto *edge_nums_row = edge_nums.ptr<float_t>(y);
        if (y * window_size >= height)
        {
            continue;
        }
        top_row = y * window_size;
        bottom_row = min(height, (y + 1) * window_size);

        for (int pos = 0; pos < new_size.width; pos++)
        {

            // then calculate the column locations of the rectangle and set them to -1
            // if they are outside the matrix bounds
            if (pos * window_size >= width)
            {
                continue;
            }
            left_col = pos * window_size;
            right_col = min(width, (pos + 1) * window_size);

            //we had an integral image to count non-zero elements
            CALCULATE_SUM(edges_ptr, rect_area)
            if (rect_area < THRESHOLD_AREA)
            {
                // smooth region
                coherence_row[pos] = 0;
                continue;
            }

            CALCULATE_SUM(x_sq_ptr, x_sq)
            CALCULATE_SUM(y_sq_ptr, y_sq)
            CALCULATE_SUM(xy_ptr, xy)

            // get the values of the rectangle corners from the integral image - 0 if outside bounds
            d = sqrt((x_sq - y_sq) * (x_sq - y_sq) + 4 * xy * xy) / (x_sq + y_sq);
            if (d > THRESHOLD_COHERENCE)
            {
                coherence_row[pos] = 255;
                orientation_row[pos] = atan2(x_sq - y_sq, 2 * xy) / 2.0f;
                edge_nums_row[pos] = rect_area;
            }
            else
            {
                coherence_row[pos] = 0;
            }

        }

    }
}

// will change localization_bbox bbox_scores
// will change coherence,
// depend on coherence orientation edge_nums
void Detect::regionGrowing(int window_size)
{
    static constexpr float LOCAL_THRESHOLD_COHERENCE = 0.95f, THRESHOLD_RADIAN =
            PI / 30, LOCAL_RATIO = 0.5f, EXPANSION_FACTOR = 1.2f;
    static constexpr uint THRESHOLD_BLOCK_NUM = 35;
    Point pt_to_grow, pt;                       //point to grow

    float src_value;
    float cur_value;
    float edge_num;
    float rect_orientation;
    float sin_sum, cos_sum;
    uint counter;
    //grow direction
    static constexpr int DIR[8][2] = {{-1, -1},
                                      {0,  -1},
                                      {1,  -1},
                                      {1,  0},
                                      {1,  1},
                                      {0,  1},
                                      {-1, 1},
                                      {-1, 0}};
    vector<Point2f> growingPoints, growingImgPoints;
    for (int y = 0; y < coherence.rows; y++)
    {
        auto *coherence_row = coherence.ptr<uint8_t>(y);

        for (int x = 0; x < coherence.cols; x++)
        {
            if (coherence_row[x] == 0)
            {
                continue;
            }
            // flag
            coherence_row[x] = 0;
            growingPoints.clear();
            growingImgPoints.clear();

            pt = Point(x, y);
            cur_value = orientation.at<float_t>(pt);
            sin_sum = sin(2 * cur_value);
            cos_sum = cos(2 * cur_value);
            counter = 1;
            edge_num = edge_nums.at<float_t>(pt);
            growingPoints.push_back(pt);
            growingImgPoints.push_back(Point(pt));
            while (!growingPoints.empty())
            {
                pt = growingPoints.back();
                growingPoints.pop_back();
                src_value = orientation.at<float_t>(pt);

                //growing in eight directions
                for (auto i : DIR)
                {
                    pt_to_grow = Point(pt.x + i[0], pt.y + i[1]);

                    //check if out of boundary
                    if (!isValidCoord(pt_to_grow, coherence.size()))
                    {
                        continue;
                    }

                    if (coherence.at<uint8_t>(pt_to_grow) == 0)
                    {
                        continue;
                    }
                    cur_value = orientation.at<float_t>(pt_to_grow);
                    if (abs(cur_value - src_value) < THRESHOLD_RADIAN ||
                        abs(cur_value - src_value) > PI - THRESHOLD_RADIAN)
                    {
                        coherence.at<uint8_t>(pt_to_grow) = 0;
                        sin_sum += sin(2 * cur_value);
                        cos_sum += cos(2 * cur_value);
                        counter += 1;
                        edge_num += edge_nums.at<float_t>(pt_to_grow);
                        growingPoints.push_back(pt_to_grow);                 //push next point to grow back to stack
                        growingImgPoints.push_back(pt_to_grow);
                    }
                }
            }
            //minimum block num
            if (counter < THRESHOLD_BLOCK_NUM)
            {
                continue;
            }
            float local_coherence = (sin_sum * sin_sum + cos_sum * cos_sum) / static_cast<float>(counter * counter);
            // minimum local gradient orientation_arg coherence_arg
            if (local_coherence < LOCAL_THRESHOLD_COHERENCE)
            {
                continue;
            }
            RotatedRect minRect = minAreaRect(growingImgPoints);
            if (edge_num < minRect.size.area() * float(window_size * window_size) * LOCAL_RATIO ||
                static_cast<float>(counter) < minRect.size.area() * LOCAL_RATIO)
            {
                continue;
            }
            const float local_orientation = atan2(cos_sum, sin_sum) / 2.0f;
            // only orientation_arg is approximately equal to the rectangle orientation_arg
            rect_orientation = (minRect.angle) * PI / 180.f;
            if (minRect.size.width < minRect.size.height)
            {
                rect_orientation += (rect_orientation <= 0.f ? HALF_PI : -HALF_PI);
                std::swap(minRect.size.width, minRect.size.height);
            }
            if (abs(local_orientation - rect_orientation) > THRESHOLD_RADIAN &&
                abs(local_orientation - rect_orientation) < PI - THRESHOLD_RADIAN)
            {
                continue;
            }
            minRect.angle = local_orientation * 180.f / PI;
            minRect.size.width *= static_cast<float>(window_size) * EXPANSION_FACTOR;
            minRect.size.height *= static_cast<float>(window_size);
            minRect.center.x = (minRect.center.x + 0.5f) * static_cast<float>(window_size);
            minRect.center.y = (minRect.center.y + 0.5f) * static_cast<float>(window_size);
            localization_bbox.push_back(minRect);
            bbox_scores.push_back(edge_num);

        }
    }
}

inline const std::array<Mat, 4> &getStructuringElement()
{
    static const std::array<Mat, 4> structuringElement{
            Mat_<uint8_t>{{3,   3},
                          {255, 0, 0, 0, 0, 0, 0, 0, 255}}, Mat_<uint8_t>{{3, 3},
                                                                          {0, 0, 255, 0, 0, 0, 255, 0, 0}},
            Mat_<uint8_t>{{3, 3},
                          {0, 0, 0, 255, 0, 255, 0, 0, 0}}, Mat_<uint8_t>{{3, 3},
                                                                          {0, 255, 0, 0, 0, 0, 0, 255, 0}}};
    return structuringElement;
}

// Change mat
void Detect::barcodeErode()
{
    static const std::array<Mat, 4> &structuringElement = getStructuringElement();
    Mat m0, m1, m2, m3;
    dilate(coherence, m0, structuringElement[0]);
    dilate(coherence, m1, structuringElement[1]);
    dilate(coherence, m2, structuringElement[2]);
    dilate(coherence, m3, structuringElement[3]);
    int sum;
    for (int y = 0; y < coherence.rows; y++)
    {
        auto coherence_row = coherence.ptr<uint8_t>(y);
        auto m0_row = m0.ptr<uint8_t>(y);
        auto m1_row = m1.ptr<uint8_t>(y);
        auto m2_row = m2.ptr<uint8_t>(y);
        auto m3_row = m3.ptr<uint8_t>(y);

        for (int pos = 0; pos < coherence.cols; pos++)
        {
            if (coherence_row[pos] != 0)
            {
                sum = m0_row[pos] + m1_row[pos] + m2_row[pos] + m3_row[pos];
                //more than 2 group
                coherence_row[pos] = sum > 600 ? 255 : 0;
            }
        }
    }
}
}
}
