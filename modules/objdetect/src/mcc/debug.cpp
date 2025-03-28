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
#include "debug.hpp"

#ifdef MCC_DEBUG

#include <opencv2/highgui.hpp>

namespace cv
{
namespace mcc
{
Scalar randomcolor(RNG &rng)
{
    int icolor = (unsigned)rng;
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void imshow_250xN(const std::string &name_, InputArray patch)
{

    Mat bigpatch;
    Size size = patch.size();
    float asp = (float)size.height / size.width;
    int new_size = 550;
    resize(patch, bigpatch, Size((int)new_size, int(new_size * asp)));
    imshow(name_, bigpatch);
}

void showAndSave(std::string name, InputArray m, std::string path)
{
    imshow_250xN(name, m);
    imwrite(path + "/" + name + ".png", m);
    if (waitKey(0) == 'q')
        return;
}

} // namespace mcc
} // namespace cv

#endif
