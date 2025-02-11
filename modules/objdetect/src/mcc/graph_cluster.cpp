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
#include "graph_cluster.hpp"

namespace cv
{
namespace mcc
{
CB0cluster::CB0cluster()
{
}

CB0cluster::~CB0cluster()
{
}

void CB0cluster::
    group()
{

    size_t n = X.size();
    G.clear();
    G.resize(n);

    for (int i = 0; i < (int)n - 1; i++)
    {
        std::vector<double> Y;
        Y.clear();
        Y.resize(n - i);
        Y[0] = 0;

        // 1. group similar blobs
        double dist, w, y;
        for (int j = i + 1, k = 1; j < (int)n; j++, k++)
        {
            //dist(X_i,X_j)
            dist = norm(X[i] - X[j]);

            //heuristic to combine two sub charts, This is pretty ugly at the moment,
            //Looking for somthing better
            w = min(abs(W[i] - W[j]) / (W[i] + W[j]),
                    abs(max(W[i], W[j]) - 24 / 11.0f * min(W[i], W[j])) / (max(W[i], W[j]) + 24 / 11.0f * min(W[i], W[j])));
            w = (w < 0.1);

            y = w * dist;
            Y[k] = (y < B0[i]) * y;
            ;
        }

        if (!G[i])
            G[i] = i + 1;

        std::vector<int> pos_b0;
        find(Y, pos_b0);

        size_t m = pos_b0.size();
        if (!m)
            continue;

        std::vector<int> pos_nz, pos_z;
        for (int j = 0; j < (int)m; j++)
        {
            pos_b0[j] = pos_b0[j] + i;
            if (G[pos_b0[j]])
                pos_nz.push_back(j);
            else
                pos_z.push_back(j);
        }

        for (int j = 0; j < (int)pos_z.size(); j++)
        {
            pos_z[j] = pos_b0[pos_z[j]];
            G[pos_z[j]] = G[i];
        }

        if (!pos_nz.size())
            continue;

        std::vector<int> g;
        for (size_t j = 0; j < pos_nz.size(); j++)
        {
            pos_nz[j] = pos_b0[pos_nz[j]];
            g.push_back(G[pos_nz[j]]);
        }

        unique(g, g);
        for (size_t k = 0; k < g.size(); k++)
        {
            int gk = g[k];
            for (size_t j = 0; j < G.size(); j++)
                if (G[j] == gk)
                    G[j] = G[i];
        }
    }

    if (!G[n - 1])
        G[n - 1] = (int)n;

    std::vector<int> S;
    S = G;
    unique(S, S);
    for (int k = 0; k < (int)S.size(); k++)
    {
        int gk = S[k];
        for (int j = 0; j < (int)G.size(); j++)
            if (G[j] == gk)
                G[j] = k;
    }
}

} // namespace mcc
} // namespace cv
