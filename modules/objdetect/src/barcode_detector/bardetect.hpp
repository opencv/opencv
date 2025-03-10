// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef OPENCV_BARCODE_BARDETECT_HPP
#define OPENCV_BARCODE_BARDETECT_HPP


#include <opencv2/core.hpp>

namespace cv {
namespace barcode {
using std::vector;

class Detect
{
private:
    vector<RotatedRect> localization_rects;
    vector<RotatedRect> localization_bbox;
    vector<float> bbox_scores;
    vector<int> bbox_indices;
    vector<vector<Point2f>> transformation_points;


public:
    void init(const Mat &src, double detectorThreshDownSamplingLimit);

    void localization(const vector<float>& detectorWindowSizes, double detectorGradientMagnitudeThresh);

    vector<vector<Point2f>> getTransformationPoints()
    { return transformation_points; }

    bool computeTransformationPoints();

protected:
    enum resize_direction
    {
        ZOOMING, SHRINKING, UNCHANGED
    } purpose = UNCHANGED;


    double coeff_expansion = 1.0;
    int height, width;
    Mat resized_barcode, gradient_magnitude, coherence, orientation, edge_nums, integral_x_sq, integral_y_sq, integral_xy, integral_edges;

    void preprocess(double detectorThreshGradientMagnitude);

    void calCoherence(int window_size);

    static inline bool isValidCoord(const Point &coord, const Size &limit);

    void regionGrowing(int window_size);

    void barcodeErode();


};
}
}

#endif // OPENCV_BARCODE_BARDETECT_HPP
