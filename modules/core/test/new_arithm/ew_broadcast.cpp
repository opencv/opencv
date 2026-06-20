// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Implementation of broadcastOp (see ew_broadcast.hpp).

#include "ew_broadcast.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cv { namespace ew {

// ---------------------------------------------------------------------------
// Geometry helpers. (Mirrors the ones currently in ew_exec.cpp; they will be unified once
// exec() is moved on top of broadcastOp.)
// ---------------------------------------------------------------------------

// How a Mat's channels are mapped into the logical (shape, step, esz1) handed to the geometry:
//   CH_FOLD : channels stay scalar-wise, folded into the innermost dim (back() *= cn, step 1).
//             Used (expandChannels=true) when no channel broadcast is needed (all single-channel,
//             or all same-cn with equal back()) - the body then sees single-channel data.
//   CH_DIM  : channels become an explicit innermost iteration dim (cn, step 1); single-channel
//             operands get a size-1 channel that broadcasts 1->N. Used when channel broadcast is
//             needed (mixed channel counts, or multichannel with differing back()).
//   CH_ELEM : channels stay inside the element (esz = full elemSize, no channel dim). Used with
//             expandChannels=false; the body handles channels itself (deinterleave fast path).
enum ChMode { CH_FOLD, CH_DIM, CH_ELEM };

static void matLayout(const Mat& m, ChMode mode, MatShape& shp, EwSteps& step, int& esz1)
{
    const int nd = m.dims, cn = m.channels();
    if (mode == CH_ELEM)
    {
        esz1 = (int)m.elemSize();                       // one full (cn-channel) pixel
        shp.resize(nd);
        for (int i = 0; i < nd; i++) { shp[i] = m.size[i]; step[i] = m.step[i] / esz1; }
        return;
    }
    esz1 = (int)m.elemSize1();                           // one scalar (channel value)
    if (mode == CH_DIM)
    {
        shp.resize(nd + 1);
        for (int i = 0; i < nd; i++) { shp[i] = m.size[i]; step[i] = m.step[i] / esz1; }
        shp[nd] = cn; step[nd] = 1;                      // channels = explicit innermost dim
    }
    else // CH_FOLD
    {
        shp.resize(nd);
        for (int i = 0; i < nd; i++) { shp[i] = m.size[i]; step[i] = m.step[i] / esz1; }
        shp[nd - 1] *= cn;                               // fold channels into the innermost dim
        step[nd - 1] = 1;                                // scalars are contiguous there
    }
}

// A single-channel scalar (one value, cn==1, total()==1) broadcasts into everything trivially
// (step 0 on every axis incl. channels), so it must NOT force CH_DIM - it is excluded from the
// channel-mode decision entirely.
static bool isSingleChannelScalar(const Mat& m)
{
    return m.channels() == 1 && m.total() == 1;
}

// Decide (globally, across all operands) how channels are presented for expandChannels=true:
// CH_FOLD when no channel broadcast is needed, CH_DIM when it is. Single-channel scalars are
// excluded first; among the rest, multichannel operands must all share the same cn (an (n,m) mix
// with both > 1 is an error).
static ChMode decideChannelMode(const Mat* arrays, int K)
{
    int N = 1;                                           // the single multichannel count, if any
    for (int k = 0; k < K; k++)
    {
        if (isSingleChannelScalar(arrays[k])) continue;
        int c = arrays[k].channels();
        if (c > 1) { if (N == 1) N = c; else CV_Assert(N == c && "ew: (n,m) channel mix unsupported"); }
    }
    if (N == 1) return CH_FOLD;                          // all single-channel -> fold (a no-op)

    bool allMulti = true, sameBack = true;
    int back = -1;
    for (int k = 0; k < K; k++)
    {
        if (isSingleChannelScalar(arrays[k])) continue;
        if (arrays[k].channels() != N) allMulti = false;
        int b = arrays[k].size[arrays[k].dims - 1];
        if (back < 0) back = b; else if (b != back) sameBack = false;
    }
    return (allMulti && sameBack) ? CH_FOLD : CH_DIM;    // fold only if no channel broadcast
}

// numpy-style broadcast of several right-aligned shapes.
static bool broadcastShape(const std::vector<MatShape>& shps, MatShape& out)
{
    size_t nd = 0;
    for (size_t k = 0; k < shps.size(); k++) nd = std::max(nd, shps[k].size());
    out.assign(nd, 1);
    for (size_t k = 0; k < shps.size(); k++)
    {
        const MatShape& s = shps[k];
        size_t off = nd - s.size();
        for (size_t i = 0; i < s.size(); i++)
        {
            int d = s[i], &o = out[off + i];
            if (o == 1) o = d;
            else if (d != 1 && d != o) return false;
        }
    }
    return true;
}

// Right-align an arg's own (shp,step) to nd dims; broadcast dims get step 0.
static void alignArg(const MatShape& shp, const EwSteps& step, int nd,
                     EwSteps& as, MatShape& ash)
{
    as.fill(0);
    ash.assign(nd, 1);
    int off = nd - (int)shp.size();
    for (int i = 0; i < (int)shp.size(); i++)
    {
        int d = shp[i];
        ash[off + i] = d;
        as[off + i] = (d == 1) ? 0 : step[i];
    }
}

// Collapse adjacent dims that are contiguous (and broadcast-consistent) across all args.
static int collapseDims(std::vector<EwSteps>& S, std::vector<MatShape>& H, MatShape& D)
{
    int K = (int)S.size();
    int nd = (int)D.size();
    if (nd <= 1) return nd;

    int j = nd - 1;
    for (int i = j - 1; i >= 0; i--)
    {
        bool contig = true, scalar = true, consist = true;
        for (int k = 0; k < K; k++)
        {
            size_t st = S[k][j] * (size_t)H[k][j];
            bool prevScalar = H[k][j] == 1;
            bool curScalar = H[k][i] == 1;
            contig  = contig  && (st == S[k][i]);
            scalar  = scalar  && curScalar;
            consist = consist && (curScalar == prevScalar);
        }
        if (contig && (consist || scalar))
        {
            for (int k = 0; k < K; k++) H[k][j] *= H[k][i];
            D[j] *= D[i];
        }
        else
        {
            j--;
            if (i < j)
            {
                for (int k = 0; k < K; k++) { H[k][j] = H[k][i]; S[k][j] = S[k][i]; }
                D[j] = D[i];
            }
        }
    }

    int m = nd - j;
    for (int d = 0; d < m; d++)
    {
        D[d] = D[j + d];
        for (int k = 0; k < K; k++) { S[k][d] = S[k][j + d]; H[k][d] = H[k][j + d]; }
    }
    D.resize(m);
    for (int k = 0; k < K; k++) H[k].resize(m);

    // Zero out steps of broadcast (size-1) dims (numpy step==0 trick).
    for (int d = 0; d < m; d++)
        for (int k = 0; k < K; k++)
            if (H[k][d] == 1) S[k][d] = 0;

    return m;
}

// ---------------------------------------------------------------------------
// broadcastOp
// ---------------------------------------------------------------------------
void broadcastOp(Mat* arrays, size_t narrays,
                 const std::function<void(const EwTile&)>& body,
                 bool expandChannels,
                 double nstripes)
{
    const int K = (int)narrays;
    CV_Assert(K >= 1 && arrays != nullptr);

    // ---- 1. per-operand logical shape/step/esz1 + base pointer ----
    // Channel handling: expandChannels=true => present single-channel data to the body (fold the
    // channels into the innermost dim, or make them an explicit dim when channel broadcast is
    // needed); expandChannels=false => leave channels inside the element for the body to handle.
    const ChMode mode = expandChannels ? decideChannelMode(arrays, K) : CH_ELEM;
    std::vector<MatShape> shp(K);
    std::vector<EwSteps>  stp(K);
    std::vector<int>      esz1(K);
    std::vector<uchar*>   base(K);
    for (int k = 0; k < K; k++)
    {
        matLayout(arrays[k], mode, shp[k], stp[k], esz1[k]);
        base[k] = (uchar*)arrays[k].data;
    }

    // ---- 2. common broadcast shape (channel dim last) ----
    MatShape full;
    CV_Assert(broadcastShape(shp, full) && "ew: operands are not broadcast-compatible");
    const int nd = (int)full.size();

    // ---- 3. right-align + collapse across all operands ----
    std::vector<EwSteps>  S(K);
    std::vector<MatShape> H(K);
    for (int k = 0; k < K; k++) alignArg(shp[k], stp[k], nd, S[k], H[k]);
    MatShape D = full;
    const int m = collapseDims(S, H, D);

    // For a cv::Mat the innermost (channel/last) axis is contiguous, so after collapse the
    // innermost stride is always in {0,1}. No gather, no materialization.
    for (int k = 0; k < K; k++)
        CV_Assert(S[k][m - 1] <= 1 && "ew: unexpected innermost stride > 1");

    // ---- 4. inner 2D tile axes: width = D[m-1], height = D[m-2] (if any) ----
    const int wAxis = m - 1;
    const int hAxis = (m >= 2) ? m - 2 : -1;
    const int W   = D[wAxis];
    const int Hgt = (hAxis >= 0) ? D[hAxis] : 1;
    const int nOuter = (hAxis >= 0) ? m - 2 : m - 1;   // outer ("plane") axes = [0 .. nOuter)
    long long nplanes = 1;
    for (int d = 0; d < nOuter; d++) nplanes *= D[d];

    // ---- 5. desired parallel stripe count (work hint) ----
    const long long total = nplanes * (long long)Hgt * (long long)W;
    double stripes = nstripes;
    if (stripes <= 0)                                  // broadcastOp can't see the body's cost;
        stripes = (double)total * 100.0 / (double)(1 << 18);  // assume ~100 cycles/element
    const int wantTiles = std::max(1, (int)std::lround(stripes));

    // ---- 6. tile only for PARALLELISM. broadcastOp is op-agnostic: it does not know the body's
    //         temp-buffer footprint, so it does NOT tile for L1 - that is the body's job (it
    //         re-fragments a tile's width into L1-sized chunks for the fused intermediates).
    //         Start with the largest tile (one 2D block per plane) and split (height first, then
    //         width) only until there are at least `wantTiles` tiles. Bigger tiles => fewer
    //         body/decode calls. Width is G-aligned only in the fully-contiguous (1D) case. ----
    const int G = 16;                  // SIMD/cacheline granule
    int tw = W, th = Hgt;

    auto ntilesOf = [&](int tw_, int th_) {
        long long nw = (W + tw_ - 1) / tw_, nh = (Hgt + th_ - 1) / th_;
        return nplanes * nh * nw;
    };
    long long ntiles = ntilesOf(tw, th);
    while (ntiles < wantTiles && th > 1)               // split height for parallelism
    {
        th = (th + 1) / 2;
        ntiles = ntilesOf(tw, th);
    }
    while (ntiles < wantTiles && tw > G)               // then split width
    {
        tw = std::max(G, tw / 2);
        if (hAxis < 0 && tw > G) tw -= tw % G;          // keep width aligned in the 1D case
        ntiles = ntilesOf(tw, th);
    }
    CV_Assert(ntiles <= (long long)INT_MAX);

    const int ntilesW = (W + tw - 1) / tw;
    const int ntilesH = (Hgt + th - 1) / th;

    // ---- 7. parallel execution; O(ndims) decode of tile index -> per-operand slices ----
    parallel_for_(Range(0, (int)ntiles), [&](const Range& r)
    {
        AutoBuffer<EwSlice> slices(K);
        AutoBuffer<int> idx(std::max(1, nOuter));

        for (int t = r.start; t < r.end; t++)
        {
            int wt = t % ntilesW;
            int rest = t / ntilesW;
            int ht = rest % ntilesH;
            int plane = rest / ntilesH;

            const int wofs = wt * tw, ww = std::min(tw, W - wofs);
            const int hofs = ht * th, hh = std::min(th, Hgt - hofs);

            int p = plane;                              // decode plane -> outer multi-index
            for (int d = nOuter - 1; d >= 0; d--) { idx[d] = p % D[d]; p /= D[d]; }

            for (int k = 0; k < K; k++)
            {
                size_t off = 0;
                for (int d = 0; d < nOuter; d++) off += (size_t)idx[d] * S[k][d];
                if (hAxis >= 0) off += (size_t)hofs * S[k][hAxis];
                off += (size_t)wofs * S[k][wAxis];

                EwSlice& sl = slices[k];
                sl.ptr   = base[k] + off * (size_t)esz1[k];
                sl.stepy = (hAxis >= 0) ? S[k][hAxis] : 0;
                sl.stepx = S[k][wAxis];
            }

            EwTile tile;
            tile.width = ww; tile.height = hh; tile.narrays = K; tile.slices = slices.data();
            body(tile);
        }
    }, stripes);
}

}} // namespace cv::ew
