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
#include "charts.hpp"

namespace cv
{
namespace mcc
{
CChart::CChart()
    : perimetro(0), area(0), large_side(0)
{
}

CChart::~CChart()
{
}

void CChart::
    setCorners(std::vector<Point2f> p)
{
    Point v1, v2;
    if (p.empty())
        return;

    // copy
    corners = p;

    // Sort the corners in anti-clockwise order
    polyanticlockwise(corners);

    // Properties
    area = contourArea(corners);
    perimetro = perimeter(corners);
    center = mace_center(corners);

    v1 = corners[2] - corners[0];
    v2 = corners[3] - corners[1];
    large_side = std::max(norm(v1), norm(v2));
}

//////////////////////////////////////////////////////////////////////////////////////////////

CChartDraw::
    CChartDraw(CChart &pChart, InputOutputArray image)
    : m_pChart(pChart), m_image(image.getMat())
{
}

void CChartDraw::
    drawContour(Scalar color /*= CV_RGB(0, 250, 0)*/) const
{

    //Draw lines
    int thickness = 2;
    line(m_image, (m_pChart).corners[0], (m_pChart).corners[1], color, thickness, LINE_AA);
    line(m_image, (m_pChart).corners[1], (m_pChart).corners[2], color, thickness, LINE_AA);
    line(m_image, (m_pChart).corners[2], (m_pChart).corners[3], color, thickness, LINE_AA);
    line(m_image, (m_pChart).corners[3], (m_pChart).corners[0], color, thickness, LINE_AA);
}

void CChartDraw::
    drawCenter(Scalar color /*= CV_RGB(0, 0, 255)*/) const
{
    int radius = 3;
    int thickness = 2;
    circle(m_image, (m_pChart).center, radius, color, thickness);
}
} // namespace mcc
} // namespace cv
