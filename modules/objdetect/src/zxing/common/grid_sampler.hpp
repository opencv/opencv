#ifndef __GRID_SAMPLER_H__
#define __GRID_SAMPLER_H__

/*
 *  GridSampler.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "counted.hpp"
#include "bit_matrix.hpp"
#include "perspective_transform.hpp"
#include "byte_matrix.hpp"

#include "opencv2/core.hpp"

namespace zxing {
class GridSampler {
private:
    static GridSampler gridSampler;
    GridSampler();
    
public:
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler);
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimensionX, int dimensionY, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler);
    Ref<ByteMatrix> sampleGrid(Ref<ByteMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler);
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, cv::Mat& transform, ErrorHandler &err_handler);
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension,
                              float ax, float bx, float cx, float dx, float ex, float fx,
                              float ay, float by, float cy, float dy, float ey, float fy,
                              ErrorHandler &err_handler);
    
    
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, float p1ToX, float p1ToY, float p2ToX, float p2ToY,
                              float p3ToX, float p3ToY, float p4ToX, float p4ToY, float p1FromX, float p1FromY, float p2FromX,
                              float p2FromY, float p3FromX, float p3FromY, float p4FromX, float p4FromY, ErrorHandler & err_handler);
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimensionX, int dimensionY, float p1ToX, float p1ToY, float p2ToX, float p2ToY,
                              float p3ToX, float p3ToY, float p4ToX, float p4ToY, float p1FromX, float p1FromY, float p2FromX,
                              float p2FromY, float p3FromX, float p3FromY, float p4FromX, float p4FromY, ErrorHandler & err_handler);
    static int checkAndNudgePoints(int width, int height, std::vector<float> &points, ErrorHandler & err_handler);
    static int checkAndNudgePoints(int width, int height, std::vector<cv::Point2f> &points, ErrorHandler & err_handler);
    static int checkAndNudgePoints_new(int width, int height, std::vector<float> &points, ErrorHandler &err_handler);
    static GridSampler &getInstance();
};
}  // namespace zxing

#endif  // __GRID_SAMPLER_H__
