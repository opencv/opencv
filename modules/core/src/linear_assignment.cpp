// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Jonker-Volgenant rectangular assignment. See @cite Crouse2016 and @cite Jonker1987.

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

// Type the duals and the running sums are kept in. double suits all three inputs: it holds every
// int and float value exactly, so the integer path needs no arithmetic of its own.
template <typename T> struct LapAccumulator { typedef double type; };

// A cell is usable only if it is a real cost within the threshold. Check finiteness first:
// NaN > costThreshold is false, so a bare comparison would let NaN through as allowed.
template <typename T>
static inline bool cellAllowed(T value, double costThreshold)
{
    const double v = (double)value;
    return isFiniteVal(v) && v <= costThreshold;
}

// Assigns every row to its own column at the lowest total. `work` is nrows x ncols, rows <= cols,
// all finite. Each pass adds one row by a Dijkstra search over the reduced costs.
template <typename Acc>
static void solveJV(const std::vector<Acc>& work, int nrows, int ncols, std::vector<int>& colOfRow)
{
    CV_DbgAssert(nrows <= ncols);

    std::vector<Acc> u((size_t)nrows, Acc(0));
    std::vector<Acc> v((size_t)ncols, Acc(0));
    std::vector<Acc> dist((size_t)ncols);
    std::vector<int> rowOfCol((size_t)ncols, -1);
    std::vector<int> prevRow((size_t)ncols);
    std::vector<uchar> labelled((size_t)ncols);
    std::vector<int> remaining((size_t)ncols);   // columns not yet labelled, in no order

    colOfRow.assign((size_t)nrows, -1);

    for (int freeRow = 0; freeRow < nrows; freeRow++)
    {
        std::fill(dist.begin(), dist.end(), std::numeric_limits<Acc>::max());
        std::fill(labelled.begin(), labelled.end(), (uchar)0);
        for (int j = 0; j < ncols; j++)
            remaining[j] = j;
        int nremaining = ncols;

        int row = freeRow;      // row currently being expanded
        Acc delta = Acc(0);     // length of the shortest path found so far
        int sink = -1;          // free column that ends the augmenting path

        while (sink < 0)
        {
            const Acc* rowPtr = &work[(size_t)row * ncols];
            const Acc urow = u[row];
            // One pass over the unlabelled columns relaxes them and picks the nearest. A settled
            // column is swapped off the tail so it is never read again.
            int best = 0;
            Acc bestDist = std::numeric_limits<Acc>::max();
            for (int k = 0; k < nremaining; k++)
            {
                const int j = remaining[k];
                const Acc cand = delta + rowPtr[j] - urow - v[j];
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
            CV_Assert(bestDist < std::numeric_limits<Acc>::max());

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
                const Acc shift = delta - dist[j];
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

// Solves for one input type. `src` is read in its own type, so the padded working matrix is the
// only copy made and no promoted duplicate of the input is allocated.
template <typename T>
static double solveTyped(const Mat& src, bool transposed, std::vector<int>& assignment,
                         double costThreshold)
{
    typedef typename LapAccumulator<T>::type Acc;

    const int nrows = src.rows;
    const int nreal = src.cols;

    Mat allowed(nrows, nreal, CV_8U);
    Acc absSum = Acc(0);
    double maxCost = -std::numeric_limits<double>::max();
    bool anyForbidden = false;
    for (int i = 0; i < nrows; i++)
    {
        const T* s = src.ptr<T>(i);
        uchar* a = allowed.ptr<uchar>(i);
        for (int j = 0; j < nreal; j++)
        {
            const bool ok = cellAllowed<T>(s[j], costThreshold);
            a[j] = ok ? 1 : 0;
            if (ok)
            {
                absSum += (Acc)std::abs((double)s[j]);
                maxCost = std::max(maxCost, (double)s[j]);
            }
            else
                anyForbidden = true;
        }
    }

    // Price of leaving a row unmatched, clamped so the DBL_MAX default stays finite.
    // The 2 is required: k -> k+1 augments a path, so the delta is 2*absSum, not one edge.
    Acc dummy = Acc(2) * absSum + Acc(1);
    if (costThreshold < (double)dummy)
        dummy = (Acc)costThreshold;
    CV_Assert(isFiniteVal((double)dummy));

    // A forbidden cell only has to lose to a dummy, and every row has one. The margin scales
    // because adding 1 to a very large dummy changes nothing.
    const Acc forbidden = dummy + (Acc)std::max(1.0, std::abs((double)dummy) * 1e-12);
    CV_Assert(isFiniteVal((double)forbidden) && forbidden > dummy);

    // Drop the dummy columns when no row can go unmatched: nothing forbidden, every cost below the
    // no-pair price. Halves the columns on square input; a cost equal to that price is a tie.
    const int ndummy = (anyForbidden || !(maxCost < (double)dummy)) ? nrows : 0;
    const int ncols = nreal + ndummy;

    std::vector<Acc> work((size_t)nrows * ncols);
    for (int i = 0; i < nrows; i++)
    {
        const T* s = src.ptr<T>(i);
        const uchar* a = allowed.ptr<uchar>(i);
        Acc* w = &work[(size_t)i * ncols];
        for (int j = 0; j < nreal; j++)
            w[j] = a[j] ? (Acc)s[j] : forbidden;
        for (int j = nreal; j < ncols; j++)
            w[j] = dummy;
    }

    std::vector<int> colOfRow;
    solveJV<Acc>(work, nrows, ncols, colOfRow);

    double total = 0.0;
    for (int i = 0; i < nrows; i++)
    {
        const int j = colOfRow[i];
        CV_DbgAssert(j >= 0 && j < ncols);      // solveJV gives every row a column
        if (j >= nreal || !allowed.at<uchar>(i, j))
            continue;   // landed on a dummy column or a forbidden cell, so it stays unmatched

        total += (double)src.ptr<T>(i)[j];
        const int row = transposed ? j : i;
        CV_DbgAssert(row >= 0 && (size_t)row < assignment.size());
        assignment[(size_t)row] = transposed ? i : j;
    }
    return total;
}

double linearAssignment(InputArray _cost, std::vector<int>& assignment, double costThreshold)
{
    CV_INSTRUMENT_REGION();

    Mat cost = _cost.getMat();
    CV_Assert(cost.dims <= 2);
    CV_Assert(!cost.empty());

    assignment.assign((size_t)cost.rows, -1);

    CV_CheckType(cost.type(),
                 cost.type() == CV_32FC1 || cost.type() == CV_64FC1 || cost.type() == CV_32SC1,
                 "cost must be a single-channel CV_32F, CV_64F or CV_32S matrix");
    CV_Assert(!cvIsNaN(costThreshold));

    // The solver assigns every row, so it needs rows <= cols. Transposing also makes A and its
    // transpose give the same answer, since min(M,N) is the same either way.
    const bool transposed = cost.rows > cost.cols;
    Mat src;
    if (transposed)
        cv::transpose(cost, src);
    else
        src = cost;

    switch (cost.depth())
    {
    case CV_32F: return solveTyped<float>(src, transposed, assignment, costThreshold);
    case CV_64F: return solveTyped<double>(src, transposed, assignment, costThreshold);
    case CV_32S: return solveTyped<int>(src, transposed, assignment, costThreshold);
    default:     CV_Error(Error::StsUnsupportedFormat, "unsupported cost matrix type");
    }
}

} // namespace cv
