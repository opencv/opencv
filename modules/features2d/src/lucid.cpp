// Ziegler, Andrew, Eric Christiansen, David Kriegman, and Serge J. Belongie.
// "Locally uniform comparison image descriptor." In Advances in Neural Information
// Processing Systems, pp. 1-9. 2012.

// This implementation of, and any deviation from, the original algorithm as
// proposed by Ziegler et al. is not endorsed by Ziegler et al. nor does it
// claim to represent their definition of locally uniform comparison image
// descriptor. The original LUCID algorithm as proposed by Ziegler et al. remains
// the property of its respective authors. This implementation is an adaptation of
// said algorithm and contributed to OpenCV by Str3iber.

/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "precomp.hpp"

namespace cv {
    static void radix_sort(std::vector<std::size_t> *x, const std::size_t k) {
        std::vector<std::size_t> p(k);

        std::ptrdiff_t l;

        std::size_t i, e = 1, h = (*x)[0];

        for (i = 0; i < k; ++i) {
            if ((*x)[i] > h)
                h = (*x)[i];
        }

        while (h/e > 0) {
            std::ptrdiff_t b[10] = {0};

            for (i = 0; i < k; ++i)
                ++b[(*x)[i]/e%10];
            for (i = 1; i < 10; ++i)
                b[i] += b[i-1];
            for (l = k-1; l >= 0; --l)
                p[--b[(*x)[l]/e%10]] = (*x)[l];
            for (i = 0; i < k; ++i)
                (*x)[i] = p[i];

            e *= 10;
        }
    }

    void separable_blur(const InputArray _src, OutputArray _dst, const std::ptrdiff_t kernel) {
        std::ptrdiff_t z, p, r, g, b, m = kernel*2+1, width, height;

        Point3_<uchar> *pnt;

        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        width = dst.cols, height = dst.rows;

        for (std::ptrdiff_t y = 0; y < height; ++y) {
            for (std::ptrdiff_t x = 0; x < width; ++x) {
                z = kernel*-1;

                if (!x) {
                    r = 0, g = 0, b = 0;

                    for (p = x+z; z <= kernel; ++z, p=x+z) {
                        pnt = src.ptr<Point3_<uchar> >(y, (p < 0 ? width+p : p >= width ? p-width : p));
                        r += pnt->z;
                        g += pnt->y;
                        b += pnt->x;
                    }
                }
                else {
                    p = x+z-1;

                    pnt = src.ptr<Point3_<uchar> >(y, (p < 0 ? width+p : p >= width ? p-width : p));
                    r -= pnt->z;
                    g -= pnt->y;
                    b -= pnt->x;

                    p = x+kernel;

                    pnt = src.ptr<Point3_<uchar> >(y, (p < 0 ? width+p : p >= width ? p-width : p));
                    r += pnt->z;
                    g += pnt->y;
                    b += pnt->x;
                }

                pnt = dst.ptr<Point3_<uchar> >(y, x);
                pnt->z = r/m;
                pnt->y = g/m;
                pnt->x = b/m;
            }
        }

        for (std::ptrdiff_t x = 0, rl = 0, gl = 0, bl = 0, rn = 0, gn = 0, bn = 0; x < width; ++x) {
            for (std::ptrdiff_t y = 0; y < height; ++y) {
                z = kernel*-1;

                if (!y) {
                    r = 0, g = 0, b = 0;

                    for (p = y+z; z <= kernel; ++z, p=y+z) {
                        pnt = dst.ptr<Point3_<uchar> >((p < 0 ? height+p : p >= height ? p-height : p), x);
                        r += pnt->z;
                        g += pnt->y;
                        b += pnt->x;
                    }
                }
                else {
                    p = y+z-1;

                    pnt = dst.ptr<Point3_<uchar> >((p < 0 ? height+p : p >= height ? p-height : p), x);
                    r -= pnt->z, r -= rl;
                    g -= pnt->y, g -= gl;
                    b -= pnt->x, b -= bl;

                    p = y+kernel;

                    pnt = dst.ptr<Point3_<uchar> >((p < 0 ? height+p : p >= height ? p-height : p), x);
                    r += pnt->z, r += rn;
                    g += pnt->y, g += gn;
                    b += pnt->x, b += bn;
                }

                pnt = dst.ptr<Point3_<uchar> >(y, x);
                rl = pnt->z;
                gl = pnt->y;
                bl = pnt->x;
                rn = r/m;
                gn = g/m;
                bn = b/m;
                pnt->z = rn;
                pnt->y = gn;
                pnt->x = bn;
            }
        }
    }

    void LUCID(const InputArray _src, const std::vector<KeyPoint> &keypoints, std::vector<std::vector<std::size_t> > &descriptors, const std::ptrdiff_t lucid_kernel, const std::ptrdiff_t blur_kernel) {
        Mat src;

        separable_blur(_src.getMat(), src, blur_kernel);

        Point3_<uchar> *pnt;

        std::ptrdiff_t x, y, j, d, p, m = static_cast<std::ptrdiff_t>(std::pow(lucid_kernel*2+1, 2)*3), width = src.cols, height = src.rows;

        std::vector<Point2i> corners;

        corners.reserve(keypoints.size());
        for (std::size_t i = 0; i < keypoints.size(); ++i)
            corners.push_back(keypoints[i].pt);

        descriptors.clear();
        descriptors.reserve(corners.size());

        for (std::size_t i = 0; i < corners.size(); ++i) {
            x = corners[i].x-lucid_kernel, y = corners[i].y-lucid_kernel, d = x+2*lucid_kernel, p = y+2*lucid_kernel, j = x;

            std::vector<std::size_t> buf;
            buf.reserve(m);

            while (x <= d) {
                pnt = src.ptr<Point3_<uchar> >((y < 0 ? height+y : y >= height ? y-height : y), (x < 0 ? width+x : x >= width ? x-width : x));

                buf.push_back(pnt->x);
                buf.push_back(pnt->y);
                buf.push_back(pnt->z);

                ++x;
                if (x > d) {
                    if (y < p) {
                        ++y;

                        x = j;
                    }
                    else
                        break;
                }
            }

            radix_sort(&buf, m);
            descriptors.push_back(buf);
        }

        descriptors.swap(descriptors);
    }
}
