// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRACKER_CSRT_SCALE_ESTIMATION
#define OPENCV_TRACKER_CSRT_SCALE_ESTIMATION

#include "opencv2/core/mat.hpp"

namespace cv
{

class DSST {
public:
    DSST() {};
    DSST(const Mat &image, Rect2f bounding_box, Size2f template_size, int numberOfScales,
            float scaleStep, float maxModelArea, float sigmaFactor, float scaleLearnRate);
    ~DSST();
    void update(const Mat &image, const Point2f objectCenter);
    float getScale(const Mat &image, const Point2f objecCenter);
private:
    Mat get_scale_features(Mat img, Point2f pos, Size2f base_target_sz, float current_scale);

    Size scale_model_sz;
    Mat ys;
    Mat ysf;
    Mat scale_window;
    std::vector<float> scale_factors;
    Mat sf_num;
    Mat sf_den;
    float scale_sigma;
    float min_scale_factor;
    float max_scale_factor;
    float current_scale_factor;
    int scales_count;
    float scale_step;
    float max_model_area;
    float sigma_factor;
    float learn_rate;

    Size original_targ_sz;
};

} /* namespace cv */

#endif
