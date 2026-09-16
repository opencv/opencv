// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Jonker-Volgenant rectangular assignment, implemented from:
//   D. F. Crouse, "On implementing 2D rectangular assignment algorithms",
//   IEEE Trans. Aerospace and Electronic Systems 52(4), 2016.
//   R. Jonker, A. Volgenant, "A shortest augmenting path algorithm for dense and sparse
//   linear assignment problems", Computing 38, 1987.

#include "precomp.hpp"

#include <algorithm>
#include <limits>

namespace cv {

// cvIsNaN/cvIsInf inspect the bit pattern, so unlike std::isfinite they still work under
// -ffast-math.
static inline bool isFiniteVal(double x)
{
    return !cvIsNaN(x) && !cvIsInf(x);
}

// Assigns every row of `work` to a distinct column, minimising the total. `work` needs
// rows <= cols and finite values only. Each pass adds one row by a Dijkstra search over the
// reduced costs work(i,j) - u[i] - v[j], which the duals keep non-negative.
static void solveJV(const Mat& work, std::vector<int>& colOfRow)
{
    const int nrows = work.rows;
    const int ncols = work.cols;
    CV_DbgAssert(nrows <= ncols);

    std::vector<double> u((size_t)nrows, 0.0);
    std::vector<double> v((size_t)ncols, 0.0);
    std::vector<int> rowOfCol((size_t)ncols, -1);

    std::vector<double> dist((size_t)ncols);
    std::vector<int> prevRow((size_t)ncols);
    std::vector<uchar> labelled((size_t)ncols);
    std::vector<int> remaining((size_t)ncols);   // columns not yet labelled, in no order

    colOfRow.assign((size_t)nrows, -1);

    for (int freeRow = 0; freeRow < nrows; freeRow++)
    {
        std::fill(dist.begin(), dist.end(), std::numeric_limits<double>::max());
        std::fill(labelled.begin(), labelled.end(), (uchar)0);
        for (int j = 0; j < ncols; j++)
            remaining[j] = j;
        int nremaining = ncols;

        int row = freeRow;      // row currently being expanded
        double delta = 0.0;     // length of the shortest path found so far
        int sink = -1;          // free column that ends the augmenting path

        while (sink < 0)
        {
            const double* rowPtr = work.ptr<double>(row);
            const double urow = u[row];
            // One pass over the unlabelled columns, relaxing and picking the nearest together.
            // A settled column is swapped off the tail so it is never re-read.
            int best = 0;
            double bestDist = std::numeric_limits<double>::max();
            for (int k = 0; k < nremaining; k++)
            {
                const int j = remaining[k];
                const double cand = delta + rowPtr[j] - urow - v[j];
                if (cand < dist[j])
                {
                    dist[j] = cand;
                    prevRow[j] = row;
                }
                if (dist[j] < bestDist)
                {
                    bestDist = dist[j];
                    best = k;
                }
            }
            // ncols >= nrows leaves at least one column free, and every cost is finite, so the
            // search can always reach it.
            CV_Assert(bestDist < std::numeric_limits<double>::max());

            const int next = remaining[best];
            remaining[best] = remaining[--nremaining];
            labelled[next] = 1;    // still needed by the dual update below
            delta = bestDist;

            if (rowOfCol[next] < 0)
                sink = next;
            else
                row = rowOfCol[next];   // occupied, so keep going through its row
        }

        // Shift the duals so the path edges land at zero without disturbing the matched pairs.
        // Must run before the matching below, which it reads.
        u[freeRow] += delta;
        for (int j = 0; j < ncols; j++)
        {
            if (labelled[j] && j != sink)
            {
                const double shift = delta - dist[j];
                v[j] -= shift;
                u[rowOfCol[j]] += shift;
            }
        }

        // Walk the path back from the free column, flipping each edge.
        for (int j = sink;;)
        {
            const int i = prevRow[j];
            rowOfCol[j] = i;
            const int jPrev = colOfRow[i];
            colOfRow[i] = j;
            if (i == freeRow)
                break;
            j = jPrev;
        }
    }
}

double linearAssignment(InputArray _cost, std::vector<int>& assignment, double costThreshold)
{
    CV_INSTRUMENT_REGION();

    Mat cost = _cost.getMat();
    CV_Assert(cost.dims <= 2);

    assignment.assign((size_t)cost.rows, -1);
    if (cost.empty())
        return 0.0;

    CV_CheckType(cost.type(), cost.type() == CV_32FC1 || cost.type() == CV_64FC1,
                 "cost must be a single-channel floating point matrix");
    CV_Assert(!cvIsNaN(costThreshold));

    // The solver assigns every row, so it needs rows <= cols. It also keeps A and A transposed
    // in agreement, since the unassignment term is charged min(M,N) - matched either way.
    const bool transposed = cost.rows > cost.cols;
    Mat src;
    if (transposed)
        cv::transpose(cost, src);
    else
        src = cost;

    Mat orig;
    src.convertTo(orig, CV_64F);
    const int nrows = orig.rows;
    const int nreal = orig.cols;

    // Test finiteness FIRST: NaN > costThreshold is false, so a bare comparison would let NaN
    // through as allowed and poison every sum below.
    Mat allowed(nrows, nreal, CV_8U);
    double absSum = 0.0;
    double maxCost = -std::numeric_limits<double>::max();
    bool anyForbidden = false;
    for (int i = 0; i < nrows; i++)
    {
        const double* o = orig.ptr<double>(i);
        uchar* a = allowed.ptr<uchar>(i);
        for (int j = 0; j < nreal; j++)
        {
            const bool ok = isFiniteVal(o[j]) && o[j] <= costThreshold;
            a[j] = ok ? 1 : 0;
            if (ok)
            {
                absSum += std::abs(o[j]);
                maxCost = std::max(maxCost, o[j]);
            }
            else
                anyForbidden = true;
        }
    }

    // Price of leaving a row unmatched, clamped so the DBL_MAX default stays finite.
    // The 2 is required: k -> k+1 augments a path, so the delta is 2*absSum, not one edge.
    double dummy = 2.0 * absSum + 1.0;
    if (costThreshold < dummy)
        dummy = costThreshold;
    CV_Assert(isFiniteVal(dummy));

    // A forbidden cell only has to lose to a dummy, and there is one per row so they are never
    // scarce. The margin scales because dummy + 1.0 is dummy again once |dummy| passes 2^53.
    const double forbidden = dummy + std::max(1.0, std::abs(dummy) * 1e-12);
    CV_Assert(isFiniteVal(forbidden) && forbidden > dummy);

    // Skip the dummies when no row can go unmatched: nothing forbidden, every cost under the
    // no-pair price. Halves the columns on square input. maxCost == dummy is the tie, so excluded.
    const int ndummy = (anyForbidden || !(maxCost < dummy)) ? nrows : 0;
    const int ncols = nreal + ndummy;

    Mat work(nrows, ncols, CV_64F);
    for (int i = 0; i < nrows; i++)
    {
        const double* o = orig.ptr<double>(i);
        const uchar* a = allowed.ptr<uchar>(i);
        double* w = work.ptr<double>(i);
        for (int j = 0; j < nreal; j++)
            w[j] = a[j] ? o[j] : forbidden;
        for (int j = nreal; j < ncols; j++)
            w[j] = dummy;
    }

    std::vector<int> colOfRow;
    solveJV(work, colOfRow);

    double total = 0.0;
    for (int i = 0; i < nrows; i++)
    {
        const int j = colOfRow[i];
        if (j < 0 || j >= nreal || !allowed.at<uchar>(i, j))
            continue;   // landed on a dummy column or a forbidden cell, so it stays unmatched
        total += orig.at<double>(i, j);
        if (transposed)
            assignment[(size_t)j] = i;
        else
            assignment[(size_t)i] = j;
    }
    return total;
}

} // namespace cv
