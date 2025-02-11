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

#ifndef _MCC_CHARTS_HPP
#define _MCC_CHARTS_HPP

namespace cv
{
namespace mcc
{

/** \brief Chart model
      *
      * .--------. <- px,py
      * |         | one chart for checker color
      * |  RGB   |
      * |  Lab   |
      * |        |
      * .--------.
      *
      * \author Pedro Marrero Fernndez
      */

class CChart
{

public:
    CChart();
    ~CChart();

    /**\brief set corners
          *\param p[in] new corners
          */
    void setCorners(std::vector<cv::Point2f> p);

public:
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    double perimetro;
    double area;
    double large_side;
};

/** \brief Chart draw */
class CChartDraw
{
public:
    /**\brief contructor */
    CChartDraw(CChart &pChart, InputOutputArray image);

    /**\brief draw the chart contour over the image */
    void drawContour(cv::Scalar color = CV_RGB(0, 250, 0)) const;

    /**\brief draw the chart center over the image */
    void drawCenter(cv::Scalar color = CV_RGB(0, 0, 255)) const;

private:
    CChart &m_pChart;
    cv::Mat m_image;
};

} // namespace mcc
} // namespace cv

#endif //_MCC_CHARTS_HPP
