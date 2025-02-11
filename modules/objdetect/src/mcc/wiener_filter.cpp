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

#include "wiener_filter.hpp"

namespace cv
{
namespace mcc
{
CWienerFilter::CWienerFilter()
{
}

CWienerFilter::~CWienerFilter()
{
}

void CWienerFilter::
    wiener2(InputArray _src, OutputArray _dst, int szWindowX, int szWindowY)
{

    CV_Assert(szWindowX > 0 && szWindowY > 0);

    Mat src = _src.getMat();

    int nRows;
    int nCols;
    Scalar v = 0;
    Mat p_kernel;
    Mat srcStub;
    //Now create a temporary holding matrix
    Mat p_tmpMat1, p_tmpMat2, p_tmpMat3, p_tmpMat4;
    double noise_power;

    nRows = szWindowY;
    nCols = szWindowX;

    p_kernel = Mat(nRows, nCols, CV_32F, Scalar(1.0 / (double)(nRows * nCols)));

    //Local mean of input
    filter2D(src, p_tmpMat1, -1, p_kernel, Point(nCols / 2, nRows / 2)); //localMean

    //Local variance of input
    p_tmpMat2 = src.mul(src);
    filter2D(p_tmpMat2, p_tmpMat3, -1, p_kernel, Point(nCols / 2, nRows / 2));

    //Subtract off local_mean^2 from local variance
    p_tmpMat4 = p_tmpMat1.mul(p_tmpMat1); //localMean^2
    p_tmpMat3 = p_tmpMat3 - p_tmpMat4;
    // Sub(p_tmpMat3, p_tmpMat4, p_tmpMat3); //filter(in^2) - localMean^2 ==> localVariance

    //Estimate noise power
    v = mean(p_tmpMat3);
    noise_power = v.val[0];
    // result = local_mean  + ( max(0, localVar - noise) ./ max(localVar, noise)) .* (in - local_mean)

    p_tmpMat4 = src - p_tmpMat1; //in - local_mean

    p_tmpMat2 = max(p_tmpMat3, noise_power); //max(localVar, noise)

    add(p_tmpMat3, Scalar(-noise_power), p_tmpMat3); //localVar - noise
    p_tmpMat3 = max(p_tmpMat3, 0);                     // max(0, localVar - noise)

    p_tmpMat3 = (p_tmpMat3 / p_tmpMat2); //max(0, localVar-noise) / max(localVar, noise)

    Mat dst = p_tmpMat3.mul(p_tmpMat4);
    dst = dst + p_tmpMat1;

    _dst.assign(dst);
}

} // namespace mcc
} // namespace cv
