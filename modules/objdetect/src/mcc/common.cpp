// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "precomp.hpp"
#include "common.hpp"

namespace cv
{
namespace mcc
{
Rect poly2mask(const std::vector<Point2f> &poly, Size size, InputOutputArray mask)
{
    // Create Polygon from vertices
    std::vector<Point> roi_poly;
    approxPolyDP(poly, roi_poly, 1.0, true);

    Rect roi = boundingRect(roi_poly);

    // Fill polygon white
    fillConvexPoly(mask, &roi_poly[0], (int)roi_poly.size(), 1, 8, 0);

    roi &= Rect(0, 0, size.width, size.height);
    if (roi.empty())
        roi = Rect(0, 0, 1, 1);
    return roi;
}

float perimeter(const std::vector<Point2f> &ps)
{
    float sum = 0, dx, dy;

    for (size_t i = 0; i < ps.size(); i++)
    {
        int i2 = (i + 1) % (int)ps.size();

        dx = ps[i].x - ps[i2].x;
        dy = ps[i].y - ps[i2].y;

        sum += sqrt(dx * dx + dy * dy);
    }

    return sum;
}

Point2f
mace_center(const std::vector<Point2f> &ps)
{
    Point2f center;
    int n;

    center = Point2f(0);
    n = (int)ps.size();
    for (int i = 0; i < n; i++)
        center += ps[i];
    center /= n;

    return center;
}

void polyanticlockwise(std::vector<Point2f> &points)
{
    // Sort the points in anti-clockwise order
    // Trace a line between the first and second point.
    // If the third point is at the right side, then the points are anti-clockwise
    Point2f v1 = points[1] - points[0];
    Point2f v2 = points[2] - points[0];

    //if the third point is in the left side, then sort in anti-clockwise order
    if ((v1.x * v2.y) - (v1.y * v2.x) < 0.0)
        std::swap(points[1], points[3]);
}
void polyclockwise(std::vector<Point2f> &points)
{
    // Sort the points in clockwise order
    // Trace a line between the first and second point.
    // If the third point is at the right side, then the points are clockwise
    Point2f v1 = points[1] - points[0];
    Point2f v2 = points[2] - points[0];

    //if the third point is in the left side, then sort in clockwise order
    if ((v1.x * v2.y) - (v1.y * v2.x) > 0.0)
        std::swap(points[1], points[3]);
}

} // namespace mcc
} // namespace cv
