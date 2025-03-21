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
#include "bound_min.hpp"

namespace cv
{
namespace mcc
{
CBoundMin::CBoundMin()

{
}

CBoundMin::~CBoundMin()
{
}

void CBoundMin::calculate()
{

    corners.clear();
    size_t N = chart.size();
    if (!N)
        return;

    std::vector<Point2f> X(4 * N);
    for (size_t i = 0; i < N; i++)
    {
        mcc::CChart cc = chart[i];
        for (size_t j = 0; j < 4; j++)
        {
            X[i * 4 + j] = cc.corners[j];
        }
    }

    // media
    Point2f mu(0, 0);
    for (size_t i = 0; i < 4 * N; i++)
        mu += X[i];
    mu /= (4 * (int)N);

    for (size_t i = 0; i < 4 * N; i++)
        X[i] -= mu;

    // calculate all line
    std::vector<Point3f> L;
    L.resize(4 * N);
    for (size_t i = 0; i < N; i++)
    {
        Point3f v0, v1, v2, v3;
        v0.x = X[4 * i + 0].x;
        v0.y = X[4 * i + 0].y;
        v0.z = 1;
        v1.x = X[4 * i + 1].x;
        v1.y = X[4 * i + 1].y;
        v1.z = 1;
        v2.x = X[4 * i + 2].x;
        v2.y = X[4 * i + 2].y;
        v2.z = 1;
        v3.x = X[4 * i + 3].x;
        v3.y = X[4 * i + 3].y;
        v3.z = 1;

        L[4 * i + 0] = v0.cross(v1);
        L[4 * i + 1] = v1.cross(v2);
        L[4 * i + 2] = v2.cross(v3);
        L[4 * i + 3] = v3.cross(v0);
    }

    // line convex hull
    std::vector<int> dist;
    dist.resize(4 * N);
    Point2f n;
    float d;

    for (size_t i = 0; i < 4 * N; i++)
    {
        n.x = L[i].x;
        n.y = L[i].y;
        d = L[i].z;

        int s = 0;
        for (size_t j = 0; j < N; j++)
            s += (X[j].dot(n) + d) <= 0;
        dist[i] = s;
    }

    // sort
    std::vector<int> idx;
    std::vector<Point3f> Ls;
    Ls.resize(4 * N);
    mcc::sort(dist, idx);
    for (size_t i = 0; i < 4 * N; i++)
        Ls[i] = L[idx[i]];

    std::vector<Point3f> Lc;
    Lc.resize(4 * N);
    Lc[0] = Ls[0];
    Point3f ln;

    int j, k = 0;
    for (size_t i = 0; i < 4 * N; i++)
    {

        ln = Ls[i]; //current line
        if (!validateLine(Lc, ln, k, j))
        {

            Lc[k] = ln;
            k++;
        }
        else if ((abs(Lc[j].z) < abs(ln.z)) && (abs(dist[i] - dist[j]) < 2))
        {
            Lc[j] = ln;
        }
        if (k == 4 && abs(dist[i] - dist[k - 1]) > 2)
            break;
    }
    if (k < 4)
        return;

    std::vector<float> thetas;
    thetas.resize(4);
    for (size_t i = 0; i < 4; i++)
        thetas[i] = atan2(Lc[i].y / Lc[i].z, Lc[i].x / Lc[i].z);

    sort(thetas, idx, false);
    std::vector<Point3f> lines;
    lines.resize(4);
    for (size_t i = 0; i < 4; i++)
        lines[i] = Lc[idx[i]];

    Point3f Vcart;
    Point2f Vhom;
    std::vector<Point2f> V;
    V.resize(4);

    for (size_t i = 0; i < 4; i++)
    {
        j = (i + 1) % 4;
        Vcart = lines[i].cross(lines[j]);
        Vhom.x = Vcart.x / Vcart.z;
        Vhom.y = Vcart.y / Vcart.z;
        V[i] = Vhom + mu;
    }

    corners = V;
}
} // namespace mcc
} // namespace cv
