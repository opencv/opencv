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

#ifndef _MCC_COMMON_HPP
#define _MCC_COMMON_HPP

namespace cv
{
namespace mcc
{

Rect poly2mask(const std::vector<Point2f> &poly, Size size, InputOutputArray mask);

template <typename T>
void circshift(std::vector<T> &A, int shiff)
{
    if (A.empty() || shiff < 1)
        return;
    int n = A.size();

    if (shiff >= n)
        return;

    std::vector<T> Tmp(n);
    for (int i = shiff; i < n + shiff; i++)
        Tmp[(i % n)] = A[i - shiff];

    A = Tmp;
}

float perimeter(const std::vector<cv::Point2f> &ps);

cv::Point2f mace_center(const std::vector<cv::Point2f> &ps);

template <typename T>
void unique(const std::vector<T> &A, std::vector<T> &U)
{

    int n = (int)A.size();
    std::vector<T> Tm = A;

    std::sort(Tm.begin(), Tm.end());

    U.clear();
    U.push_back(Tm[0]);
    for (int i = 1; i < n; i++)
        if (Tm[i] != Tm[i - 1])
            U.push_back(Tm[i]);
}

void polyanticlockwise(std::vector<cv::Point2f> &points);
void polyclockwise(std::vector<cv::Point2f> &points);

// Does lexical cast of the input argument to string
template <typename T>
std::string ToString(const T &value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template <typename T>
void change(T &a, T &b)
{
    T c = a;
    a = b;
    b = c;
}

template <typename T>
void sort(std::vector<T> &A, std::vector<int> &idx, bool ord = true)
{
    size_t N = A.size();
    if (N == 0)
        return;

    idx.clear();
    idx.resize(N);
    for (size_t i = 0; i < N; i++)
        idx[i] = (int)i;

    for (size_t i = 0; i < N - 1; i++)
    {

        size_t k = i;
        T valor = A[i];
        for (size_t j = i + 1; j < N; j++)
        {
            if ((A[j] < valor && ord) || (A[j] > valor && !ord))
            {
                valor = A[j];
                k = j;
            }
        }

        if (k == i)
            continue;

        change(A[i], A[k]);
        change(idx[i], idx[k]);
    }
}

} // namespace mcc
} // namespace cv

#endif //_MCC_COMMON_HPP
