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

#ifndef _MCC_GRAPH_CLUSTERS_HPP
#define _MCC_GRAPH_CLUSTERS_HPP

namespace cv
{
namespace mcc
{

class CB0cluster
{
public:
    CB0cluster();
    ~CB0cluster();

    inline void setVertex(const std::vector<cv::Point> &V) { X = V; }
    inline void setB0(const std::vector<double> &b0) { B0 = b0; }
    inline void setWeight(const std::vector<double> &Weight) { W = Weight; }

    void group();

    void getGroup(std::vector<int> &g) { g = G; }

private:
    //entrada
    std::vector<cv::Point> X;
    std::vector<double> B0;
    std::vector<double> W;

    //salida
    std::vector<int> G;

private:
    template <typename T>
    void find(const std::vector<T> &A, std::vector<int> &indx)
    {
        indx.clear();
        for (int i = 0; i < (int)A.size(); i++)
            if (A[i])
                indx.push_back(i);
    }
};

} // namespace mcc
} // namespace cv

#endif //_MCC_GRAPH_CLUSTERS_HPP
