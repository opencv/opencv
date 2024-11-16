/*
 *  GridSampler.cpp
 *  zxing
 *
 *  Created by Christian Brunschen on 18/05/2008.
 *  Copyright 2008 ZXing authors All rights reserved.
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

#include "grid_sampler.hpp"
#include "perspective_transform.hpp"
#include "../reader_exception.hpp"
#include <iostream>
#include <sstream>


namespace zxing {

GridSampler GridSampler::gridSampler;

GridSampler::GridSampler() {
}

// Samples an image for a rectangular matrix of bits of the given dimension.
Ref<ByteMatrix> GridSampler::sampleGrid(Ref<ByteMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler) {
    Ref<ByteMatrix> bits(new ByteMatrix(dimension));
    std::vector<float> points(dimension << 1, (const float)0.0f);
    
    int outlier = 0;
    int maxOutlier = dimension*dimension * 3 / 10 - 1;
    
    for (int y = 0; y < dimension; y++) {
        int max = points.size();
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < max; x += 2) {
            points[x] = static_cast<float>(x >> 1) + 0.5f;
            points[x + 1] = yValue;
        }
        transform->transformPoints(points);
        // Quick check to see if points transformed to something inside the image;
        // sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode())   return Ref<ByteMatrix>();
        
        if (outlier >= maxOutlier)
        {
            std::ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            return Ref<ByteMatrix>();
        }
        
        for (int x = 0; x < max; x += 2) {
            bits->set(x >> 1, y, image->get(static_cast<int>(points[x]), static_cast<int>(points[x + 1])));
        }
    }
    return bits;
}

// Samples an image for a rectangular matrix of bits of the given dimension.
Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler) {
    Ref<BitMatrix> bits(new BitMatrix(dimension, err_handler));
    
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    
    std::vector<float> points(dimension << 1, (const float)0.0f);
    
    int outlier = 0;
    int maxOutlier = dimension*dimension * 3 / 10 - 1;
    
    for (int y = 0; y < dimension; y++) {
        int max = points.size();
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < max; x += 2) {
            points[x] = static_cast<float>(x >> 1) + 0.5f;
            points[x + 1] = yValue;
        }
        transform->transformPoints(points);
        
        // Quick check to see if points transformed to something inside the image;
        // sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode())   return Ref<BitMatrix>();
        
        if (outlier >= maxOutlier)
        {
            std::ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            return Ref<BitMatrix>();
        }
        
        for (int x = 0; x < max; x += 2) {
            {
                int r_x = (x >> 1);
                if (r_x == 0 && y >= 0 && y < 7) {bits->set(r_x, y); continue;}
                if (y == 0 && r_x >= 0 && r_x < 7) {bits->set(r_x, y); continue;}
                if (r_x == 6 && y >= 0 && y < 7) {bits->set(r_x, y); continue;}
                if (y == 6 && r_x >= 0 && r_x < 7) {bits->set(r_x, y); continue;}
                if (y >= 2 && y <= 4 && r_x >= 2 && r_x <= 4) {bits->set(r_x, y); continue;}
                
                if (r_x == 1 && y >= 1 && y <= 5) continue;
                if (r_x == 5 && y >= 1 && y <= 5) continue;
                if (y == 1 && r_x >= 1 && r_x <= 5) continue;
                if (y == 5 && r_x >= 1 && r_x <= 5) continue;
                if (r_x == 7 && y >= 0 && y <= 7) continue;
                if (y == 7 && r_x >= 0 && r_x <= 7) continue;
                
                // bottom left
                if (r_x == 0 && y >= dimension - 7 && y < dimension) {bits->set(r_x, y); continue;}
                if (y == dimension - 1 && r_x >= 0 && r_x < 7) {bits->set(r_x, y); continue;}
                if (r_x == 6 && y >= dimension - 7 && y < dimension) {bits->set(r_x, y); continue;}
                if (y == dimension - 7 && r_x >= 0 && r_x < 7) {bits->set(r_x, y); continue;}
                if (y >= dimension - 5 && y <= dimension - 3 && r_x >= 2 && r_x <= 4) {bits->set(r_x, y); continue;}
                
                if (r_x == 1 && y >= dimension - 6 && y < dimension - 1) continue;
                if (r_x == 5 && y >= dimension - 6 && y < dimension - 1) continue;
                if (y == dimension - 2 && r_x >= 1 && r_x <= 5) continue;
                if (y == dimension - 6 && r_x >= 1 && r_x <= 5) continue;
                if (r_x == 7 && y >= dimension - 8 && y < dimension) continue;
                if (y == dimension - 8 && r_x >= 0 && r_x <= 7) continue;
                
                // top right
                if (y == 0 && r_x >= dimension - 7 && r_x < dimension) {bits->set(r_x, y); continue;}
                if (r_x == dimension - 1 && y >= 0 && y < 7) {bits->set(r_x, y); continue;}
                if (y == 6 && r_x >= dimension - 7 && r_x < dimension) {bits->set(r_x, y); continue;}
                if (r_x == dimension - 7 && y >= 0 && y < 7) {bits->set(r_x, y); continue;}
                if (r_x >= dimension - 5 && r_x < dimension - 2 && y >= 2 && y <= 4) {bits->set(r_x, y); continue;}
                
                if (y == 1 && r_x >= dimension - 6 && r_x < dimension - 1) continue;
                if (y == 5 && r_x >= dimension - 6 && r_x < dimension - 1) continue;
                if (r_x == dimension - 2 && y >= 1 && y <= 5) continue;
                if (r_x == dimension - 6 && y >= 1 && y <= 5) continue;
                if (y == 7 && r_x >= dimension - 8 && r_x < dimension) continue;
                if (r_x == dimension - 8 && y >= 0 && y <= 7) continue;
            }
            
            if (image->get(static_cast<int>(points[x]), static_cast<int>(points[x + 1]))) {
                // Black (-ish) pixel
                bits->set(x >> 1, y);
            }
        }
    }
    
    return bits;
}

Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimension, cv::Mat& transform, ErrorHandler &err_handler) {
    Ref<BitMatrix> bits(new BitMatrix(dimension, err_handler));
    
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    
    std::vector<cv::Point2f> points(dimension);
    
    int outlier = 0;
    int maxOutlier = dimension * dimension * 3 / 10 - 1;
    
    for (int y = 0; y < dimension; y++) {
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < dimension; x ++) {
            points[x].x = x + 0.5f;
            points[x].y = yValue;
        }
        cv::perspectiveTransform(points, points, transform);
        
        // Quick check to see if points transformed to something inside the image;
        // sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode())   return Ref<BitMatrix>();
        
        if (outlier >= maxOutlier)
        {
            std::ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            return Ref<BitMatrix>();
        }
        
        for (int x = 0; x < dimension; x++) {
            {
                if (x == 0 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (y == 0 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (x == 6 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (y == 6 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (y >= 2 && y <= 4 && x >= 2 && x <= 4) {bits->set(x, y); continue;}
                
                if (x == 1 && y >= 1 && y <= 5) continue;
                if (x == 5 && y >= 1 && y <= 5) continue;
                if (y == 1 && x >= 1 && x <= 5) continue;
                if (y == 5 && x >= 1 && x <= 5) continue;
                if (x == 7 && y >= 0 && y <= 7) continue;
                if (y == 7 && x >= 0 && x <= 7) continue;
                
                // bottom left
                if (x == 0 && y >= dimension - 7 && y < dimension) {bits->set(x, y); continue;}
                if (y == dimension - 1 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (x == 6 && y >= dimension - 7 && y < dimension) {bits->set(x, y); continue;}
                if (y == dimension - 7 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (y >= dimension - 5 && y <= dimension - 3 && x >= 2 && x <= 4) {bits->set(x, y); continue;}
                
                if (x == 1 && y >= dimension - 6 && y < dimension - 1) continue;
                if (x == 5 && y >= dimension - 6 && y < dimension - 1) continue;
                if (y == dimension - 2 && x >= 1 && x <= 5) continue;
                if (y == dimension - 6 && x >= 1 && x <= 5) continue;
                if (x == 7 && y >= dimension - 8 && y < dimension) continue;
                if (y == dimension - 8 && x >= 0 && x <= 7) continue;
                
                // top right
                if (y == 0 && x >= dimension - 7 && x < dimension) {bits->set(x, y); continue;}
                if (x == dimension - 1 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (y == 6 && x >= dimension - 7 && x < dimension) {bits->set(x, y); continue;}
                if (x == dimension - 7 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (x >= dimension - 5 && x < dimension - 2 && y >= 2 && y <= 4) {bits->set(x, y); continue;}
                
                if (y == 1 && x >= dimension - 6 && x < dimension - 1) continue;
                if (y == 5 && x >= dimension - 6 && x < dimension - 1) continue;
                if (x == dimension - 2 && y >= 1 && y <= 5) continue;
                if (x == dimension - 6 && y >= 1 && y <= 5) continue;
                if (y == 7 && x >= dimension - 8 && x < dimension) continue;
                if (x == dimension - 8 && y >= 0 && y <= 7) continue;
            }
            
            if (image->get(static_cast<int>(points[x].x), static_cast<int>(points[x].y)))
            {
                // Black (-ish) pixel
                bits->set(x, y);
            }
        }
    }
    
    return bits;
}

Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimension,
                                       float ax, float bx, float cx, float dx, float ex, float fx,
                                       float ay, float by, float cy, float dy, float ey, float fy,
                                       ErrorHandler &err_handler)
{
    Ref<BitMatrix> bits(new BitMatrix(dimension, err_handler));
    
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    
    std::vector<cv::Point2f> points(dimension);
    
    int outlier = 0;
    int maxOutlier = dimension * dimension * 3 / 10 - 1;
    
    for (int y = 0; y < dimension; y++)
    {
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < dimension; x ++)
        {
            float xValue = x + 0.5f;
            
            points[x].x = ax + bx * xValue + cx * yValue + dx * xValue * xValue + ex * yValue * yValue + fx * xValue * yValue;
            points[x].y = ay + by * xValue + cy * yValue + dy * xValue * xValue + ey * yValue * yValue + fy * xValue * yValue;
        }
        
        // Quick check to see if points transformed to something inside the image;
        // sufficient to check the endpoings
        outlier += checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode())   return Ref<BitMatrix>();
        
        if (outlier >= maxOutlier)
        {
            std::ostringstream s;
            s << "Over 30% points out of bounds.";
            err_handler = ReaderErrorHandler(s.str().c_str());
            return Ref<BitMatrix>();
        }
        
        for (int x = 0; x < dimension; x++) {
            {
                if (x == 0 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (y == 0 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (x == 6 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (y == 6 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (y >= 2 && y <= 4 && x >= 2 && x <= 4) {bits->set(x, y); continue;}
                
                if (x == 1 && y >= 1 && y <= 5) continue;
                if (x == 5 && y >= 1 && y <= 5) continue;
                if (y == 1 && x >= 1 && x <= 5) continue;
                if (y == 5 && x >= 1 && x <= 5) continue;
                if (x == 7 && y >= 0 && y <= 7) continue;
                if (y == 7 && x >= 0 && x <= 7) continue;
                
                // bottom left
                if (x == 0 && y >= dimension - 7 && y < dimension) {bits->set(x, y); continue;}
                if (y == dimension - 1 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (x == 6 && y >= dimension - 7 && y < dimension) {bits->set(x, y); continue;}
                if (y == dimension - 7 && x >= 0 && x < 7) {bits->set(x, y); continue;}
                if (y >= dimension - 5 && y <= dimension - 3 && x >= 2 && x <= 4) {bits->set(x, y); continue;}
                
                if (x == 1 && y >= dimension - 6 && y < dimension - 1) continue;
                if (x == 5 && y >= dimension - 6 && y < dimension - 1) continue;
                if (y == dimension - 2 && x >= 1 && x <= 5) continue;
                if (y == dimension - 6 && x >= 1 && x <= 5) continue;
                if (x == 7 && y >= dimension - 8 && y < dimension) continue;
                if (y == dimension - 8 && x >= 0 && x <= 7) continue;
                
                // top right
                if (y == 0 && x >= dimension - 7 && x < dimension) {bits->set(x, y); continue;}
                if (x == dimension - 1 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (y == 6 && x >= dimension - 7 && x < dimension) {bits->set(x, y); continue;}
                if (x == dimension - 7 && y >= 0 && y < 7) {bits->set(x, y); continue;}
                if (x >= dimension - 5 && x < dimension - 2 && y >= 2 && y <= 4) {bits->set(x, y); continue;}
                
                if (y == 1 && x >= dimension - 6 && x < dimension - 1) continue;
                if (y == 5 && x >= dimension - 6 && x < dimension - 1) continue;
                if (x == dimension - 2 && y >= 1 && y <= 5) continue;
                if (x == dimension - 6 && y >= 1 && y <= 5) continue;
                if (y == 7 && x >= dimension - 8 && x < dimension) continue;
                if (x == dimension - 8 && y >= 0 && y <= 7) continue;
            }
            
            if (image->get(static_cast<int>(points[x].x), static_cast<int>(points[x].y))) {
                // Black (-ish) pixel
                bits->set(x, y);
            }
        }
    }
    
    return bits;
}

Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimensionX, int dimensionY, Ref<PerspectiveTransform> transform, ErrorHandler &err_handler)
{
    Ref<BitMatrix> bits(new BitMatrix(dimensionX, dimensionY, err_handler));
    if (err_handler.ErrCode()) return Ref<BitMatrix>();
    std::vector<float> points(dimensionX << 1, (const float)0.0f);
    for (int y = 0; y < dimensionY; y++)
    {
        int max = points.size();
        float yValue = static_cast<float>(y) + 0.5f;
        for (int x = 0; x < max; x += 2)
        {
            points[x] = static_cast<float>(x >> 1) + 0.5f;
            points[x + 1] = yValue;
        }
        transform->transformPoints(points);
        checkAndNudgePoints(image->getWidth(), image->getHeight(), points, err_handler);
        if (err_handler.ErrCode())   return Ref<BitMatrix>();
        for (int x = 0; x < max; x += 2)
        {
            if (image->get(static_cast<int>(points[x]), static_cast<int>(points[x + 1])))
            {
                bits->set(x >> 1, y);
            }
        }
    }
    return bits;
}

Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimension, float p1ToX, float p1ToY, float p2ToX,
                                       float p2ToY, float p3ToX, float p3ToY, float p4ToX, float p4ToY, float p1FromX, float p1FromY, float p2FromX,
                                       float p2FromY, float p3FromX, float p3FromY, float p4FromX, float p4FromY, ErrorHandler &err_handler)
{
    Ref<PerspectiveTransform> transform(PerspectiveTransform::quadrilateralToQuadrilateral(p1ToX, p1ToY, p2ToX, p2ToY,
                                                                                           p3ToX, p3ToY, p4ToX, p4ToY, p1FromX, p1FromY, p2FromX, p2FromY, p3FromX, p3FromY, p4FromX, p4FromY));
    
    Ref<BitMatrix> rst = sampleGrid(image, dimension, transform, err_handler);
    if (err_handler.ErrCode())   return Ref<BitMatrix>();
    return rst;
}

Ref<BitMatrix> GridSampler::sampleGrid(Ref<BitMatrix> image, int dimensionX, int dimensionY, float p1ToX, float p1ToY, float p2ToX,
                                       float p2ToY, float p3ToX, float p3ToY, float p4ToX, float p4ToY, float p1FromX, float p1FromY, float p2FromX,
                                       float p2FromY, float p3FromX, float p3FromY, float p4FromX, float p4FromY, ErrorHandler &err_handler)
{
    Ref<PerspectiveTransform> transform(PerspectiveTransform::quadrilateralToQuadrilateral(p1ToX, p1ToY, p2ToX, p2ToY,
                                                                                           p3ToX, p3ToY, p4ToX, p4ToY, p1FromX, p1FromY, p2FromX, p2FromY, p3FromX, p3FromY, p4FromX, p4FromY));
    
    Ref<BitMatrix> rst = sampleGrid(image, dimensionX, dimensionY, transform, err_handler);
    if (err_handler.ErrCode())   return Ref<BitMatrix>();
    return rst;
    
}

int GridSampler::checkAndNudgePoints(int width, int height, std::vector<cv::Point2f> &points, ErrorHandler &err_handler)
{
    if (points.size() == 0)
    {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        return -1;
    }
    
    size_t size = points.size();
    
    int outCount = 0;
    
    float maxborder = width / size * 3;
    
    for (size_t offset = 0; offset < points.size(); offset++)
    {
        int x = static_cast<int>(points[offset].x);
        int y = static_cast<int>(points[offset].y);
        if (x < -1 || x > width || y < -1 || y > height)
        {
            outCount++;
            if (x > width + maxborder || y > height + maxborder || x < -maxborder || y < - maxborder)
            {
                err_handler = ReaderErrorHandler("checkAndNudgePoints::Out of bounds!");
                return -1;
            }
        }
        
        if (x <= -1)
        {
            points[offset].x = 1.0f;
        }
        else if (x >= width)
        {
            points[offset].x = static_cast<float>(width - 5);
        }
        if (y <= -1)
        {
            points[offset].y = 1.0f;
        }
        else if (y >= height)
        {
            points[offset].y = static_cast<float>(height - 5);
        }
    }
    
    return outCount;
}

int GridSampler::checkAndNudgePoints(int width, int height, std::vector<float> &points, ErrorHandler &err_handler) {
    // Modified to support stlport : valiantliu
    float* pts = NULL;
    
    if (points.size() > 0)
    {
        pts = &points[0];
    }
    else
    {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        return -1;
    }
    
    int size = static_cast<int>(points.size() / 2);
    
    // The Java code assumes that if the start and end points are in bounds, the rest will also be.
    // However, in some unusual cases points in the middle may also be out of bounds.
    // Since we can't rely on an ArrayIndexOutOfBoundsException like Java, we check every point.
    
    int outCount = 0;
    
    float maxborder = width/size*3;
    
    if (pts == NULL) {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        return -1;
    }
    
    for (size_t offset = 0; offset < points.size(); offset += 2) {
        int x = static_cast<int>(pts[offset]);
        int y = static_cast<int>(pts[offset + 1]);
        
        if (x < -1 || x > width || y < -1 || y > height)
        {
            outCount++;
            
            if (x > width + maxborder || y > height + maxborder || x < -maxborder || y < - maxborder)
            {
                err_handler = ReaderErrorHandler("checkAndNudgePoints::Out of bounds!");
                return -1;
            }
        }
        
        
        if (x <= -1)
        {
            points[offset] = 0.0f;
        }
        else if (x >= width)
        {
            points[offset] = static_cast<float>(width - 1);
        }
        if (y <= -1)
        {
            points[offset + 1] = 0.0f;
        }
        else if (y >= height)
        {
            points[offset + 1] = static_cast<float>(height - 1);
        }
    }
    
    return outCount;
}

int GridSampler::checkAndNudgePoints_new(int width, int height, std::vector<float> &points, ErrorHandler &err_handler) {
    // Modified to support stlport : valiantliu
    // float* pts = NULL;
    
    if (points.size() > 0)
    {
        // pts = &points[0];
    }
    else
    {
        err_handler = ReaderErrorHandler("checkAndNudgePoints:: no points!");
        return -1;
    }
    
    // Check and nudge points from start until we see some that are OK:
    bool nudged = true;
    int maxOffset = points.size() - 1;  // points.length must be even
    for (int offset = 0; offset < maxOffset && nudged; offset += 2)
    {
        int x = static_cast<int>(points[offset]);
        int y = static_cast<int>(points[offset + 1]);
        if (x < -1 || x > width || y < -1 || y > height)
        {
            err_handler = ReaderErrorHandler("not found instance!");
            return -1;
        }
        nudged = false;
        if (x == -1)
        {
            points[offset] = 0.0f;
            nudged = true;
        }
        else if (x == width)
        {
            points[offset] = width - 1;
            nudged = true;
        }
        if (y == -1)
        {
            points[offset + 1] = 0.0f;
            nudged = true;
        }
        else if (y == height)
        {
            points[offset + 1] = height - 1;
            nudged = true;
        }
    }
    
    // Check and nudge points from end:
    nudged = true;
    for (int offset = points.size() - 2; offset >= 0 && nudged; offset -= 2) {
        int x = static_cast<int>(points[offset]);
        int y = static_cast<int>(points[offset + 1]);
        if (x < -1 || x > width || y < -1 || y > height)
        {
            err_handler = ReaderErrorHandler("not found instance!");
            return -1;
        }
        nudged = false;
        if (x == -1)
        {
            points[offset] = 0.0f;
            nudged = true;
        }
        else if (x == width)
        {
            points[offset] = width - 1;
            nudged = true;
        }
        if (y == -1)
        {
            points[offset + 1] = 0.0f;
            nudged = true;
        }
        else if (y == height)
        {
            points[offset + 1] = height - 1;
            nudged = true;
        }
    }
    
    return 0;
}

GridSampler &GridSampler::getInstance()
{
    return gridSampler;
}

}  // namespace zxing
