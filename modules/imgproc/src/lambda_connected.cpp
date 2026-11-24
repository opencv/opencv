// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/********************************* ALGORITHM NOTE *******************************\
  lambda-connected segmentation, following the fuzzy-connectedness formulation of:
    L. Chen, H.D. Cheng, J. Zhang, "Fuzzy subfiber and its application to seismic
    lithology classification", Information Sciences-Applications, 1(2), 1994.
    L. Chen, "The lambda-connected segmentation and the optimal algorithm for
    split-and-merge segmentation", Chinese J. Computers, Vol. 14, 1991.

  cv::lambdaConnected computes, for every pixel p, the widest-path (a.k.a. bottleneck)
  distance from a seed:

      D(seed, p) = min over every path seed -> p of (max pairwise pixel distance
                   along that path)

  (grayscale: absolute intensity difference; BGR: Euclidean color distance). This is the
  dual of the lambda-connectedness "degree of connectedness" B(seed, p) from Chen et al.:
  B = max(0, 1 - D/maxDist) for a chosen normalization distance maxDist, and thresholding
  B at lambda is exactly thresholding D at maxDist*(1-lambda) -- see the Doxygen comment
  on cv::lambdaConnected in modules/imgproc/include/opencv2/imgproc.hpp for the full
  equivalence and why this function returns the raw distance map rather than a
  lambda/maxDist-parameterized mask.

  D(seed, .) is the classical single-source maximum-bottleneck-spanning-tree ("widest
  path") problem. The per-edge keys are bounded, non-negative integers for 8-bit input:
  0..255 for grayscale absolute difference, 0..3*255*255 for *squared* Euclidean color
  distance (the square avoids a sqrt on every graph edge; the one sqrt needed to turn a
  pixel's final squared-distance key back into a true Euclidean distance is done once per
  pixel at the very end, never per edge). Because the keys are bounded integers, this is
  solved with a monotone integer bucket queue -- the same structure cv::watershed uses in
  modules/imgproc/src/segmentation.cpp for its own bounded-integer-key priority queue --
  instead of a float binary heap, giving O(N*connectivity + NQ) instead of O(N log N).
\**********************************************************************************/

#include "precomp.hpp"

namespace cv
{

namespace
{

// Integer edge-key traits: each returns a bounded, non-negative integer distance between
// two adjacent pixels (NQ = number of distinct key values, i.e. bucket-queue size), and a
// way to turn a pixel's final key back into a true distance (identity for grayscale,
// sqrt for color -- applied once per pixel, never per edge).

struct KeyIntensityDiff
{
    enum { NQ = 256 }; // abs diff of two uchars is in [0, 255]

    static inline int edgeKey(const uchar* a, const uchar* b)
    {
        return std::abs((int)a[0] - (int)b[0]);
    }

    static inline float toDistance(int key) { return (float)key; }
};

struct KeyColorSqEuclid
{
    enum { NQ = 3 * 255 * 255 + 1 }; // squared Euclidean distance of two BGR triples

    static inline int edgeKey(const uchar* a, const uchar* b)
    {
        int d0 = (int)a[0] - b[0];
        int d1 = (int)a[1] - b[1];
        int d2 = (int)a[2] - b[2];
        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    static inline float toDistance(int key) { return (float)std::sqrt((double)key); }
};

// A node in the bucket queue's per-bucket singly-linked FIFO list -- the exact structure
// cv::watershed uses for its own monotone integer-keyed queue (see allocWSNodes/WSQueue in
// modules/imgproc/src/segmentation.cpp): nodes are allocated from a growable free list that
// is decoupled from pixel identity, since the same pixel can be queued more than once (at a
// strictly decreasing sequence of keys) before its best entry is popped. Stale, superseded
// entries are discarded lazily in O(1) when popped.
struct QNode
{
    int next;
    int pixelIdx;
};

static int allocQNodes(std::vector<QNode>& storage)
{
    int sz = (int)storage.size();
    int newsz = std::max(128, sz * 3 / 2);
    storage.resize(newsz);
    if (sz == 0)
    {
        storage[0].next = 0;
        sz = 1;
    }
    for (int i = sz; i < newsz - 1; i++)
        storage[i].next = i + 1;
    storage[newsz - 1].next = 0;
    return sz;
}

// Prim's-algorithm-style widest/bottleneck-path computation using a monotone integer
// bucket queue (cv::watershed's pattern). Maintains, for every pixel, the best (smallest)
// bottleneck distance found so far to reach it from the seed, and repeatedly expands the
// not-yet-finalized pixel with the smallest current bottleneck distance. Relaxation
// combines values with max() (the "widest path" / minimax combine, the dual of a
// conventional shortest-path sum), and the queue hands out pixels in non-decreasing order
// of that value -- a monotone invariant that lets a flat array of FIFO buckets replace a
// heap, with total work O(N*connectivity + NQ) instead of O(N log N).
template<typename KeyFn>
static void bottleneckDistance(const Mat& img, Mat& dstMap, Point seed, int connectivity)
{
    const int rows = img.rows, cols = img.cols;
    const size_t elemSize = img.elemSize();
    const size_t step = img.step;
    const uchar* imgData = img.ptr<uchar>();
    const int n = rows * cols;
    const int NQ = (int)KeyFn::NQ;

    std::vector<int> dist(n, NQ); // best known bottleneck distance per pixel; NQ == "infinity"
    std::vector<uchar> finalized(n, 0);

    std::vector<QNode> storage;
    int freeNode = 0;
    std::vector<int> qfirst(NQ, 0), qlast(NQ, 0); // 0 means empty (node 0 is an unused sentinel)

    #define CV_LC_PUSH(k, pidx)                           \
    {                                                      \
        if (!freeNode) freeNode = allocQNodes(storage);     \
        int node_ = freeNode;                                \
        freeNode = storage[freeNode].next;                    \
        storage[node_].next = 0;                               \
        storage[node_].pixelIdx = (pidx);                       \
        if (qlast[k]) storage[qlast[k]].next = node_;            \
        else qfirst[k] = node_;                                   \
        qlast[k] = node_;                                           \
    }

    const int seedIdx = seed.y * cols + seed.x;
    dist[seedIdx] = 0;
    CV_LC_PUSH(0, seedIdx);

    static const int dx4[4] = { 1, -1, 0, 0 };
    static const int dy4[4] = { 0, 0, 1, -1 };
    static const int dx8[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static const int dy8[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
    const int* dx = connectivity == 8 ? dx8 : dx4;
    const int* dy = connectivity == 8 ? dy8 : dy4;
    const int nnb = connectivity;

    int activeQueue = 0;
    for (;;)
    {
        while (activeQueue < NQ && !qfirst[activeQueue])
            activeQueue++;
        if (activeQueue >= NQ)
            break;

        const int node = qfirst[activeQueue];
        const int idx = storage[node].pixelIdx;
        qfirst[activeQueue] = storage[node].next;
        if (!qfirst[activeQueue])
            qlast[activeQueue] = 0;
        storage[node].next = freeNode;
        freeNode = node;

        if (finalized[idx] || dist[idx] != activeQueue)
            continue; // stale duplicate, already superseded by a smaller distance

        finalized[idx] = 1;
        const int y = idx / cols, x = idx % cols;
        const uchar* pCur = imgData + (size_t)y * step + x * elemSize;

        for (int k = 0; k < nnb; k++)
        {
            const int ny = y + dy[k], nx = x + dx[k];
            if ((unsigned)ny >= (unsigned)rows || (unsigned)nx >= (unsigned)cols)
                continue;
            const int nIdx = ny * cols + nx;
            if (finalized[nIdx])
                continue;

            const uchar* pNb = imgData + (size_t)ny * step + nx * elemSize;
            const int edgeKey = KeyFn::edgeKey(pCur, pNb);
            const int cand = std::max(activeQueue, edgeKey); // minimax relaxation

            if (cand < dist[nIdx])
            {
                dist[nIdx] = cand;
                CV_LC_PUSH(cand, nIdx);
            }
        }
    }

    #undef CV_LC_PUSH

    dstMap.create(rows, cols, CV_32F);
    float* mapData = dstMap.ptr<float>();
    for (int i = 0; i < n; i++)
        mapData[i] = KeyFn::toDistance(dist[i]);
}

} // namespace

void lambdaConnected( InputArray _image, OutputArray _connectivityMap,
                       Point seed, int connectivity )
{
    CV_INSTRUMENT_REGION();

    Mat img = _image.getMat();
    CV_Assert(!img.empty());

    const int type = img.type();
    CV_CheckType(type, type == CV_8UC1 || type == CV_8UC3,
                 "lambdaConnected supports CV_8UC1 (grayscale) or CV_8UC3 (BGR) images only");

    if (connectivity != 4 && connectivity != 8)
        CV_Error(cv::Error::StsBadFlag, "Connectivity must be 4 or 8");

    if ((unsigned)seed.x >= (unsigned)img.cols || (unsigned)seed.y >= (unsigned)img.rows)
        CV_Error(cv::Error::StsOutOfRange, "Seed point is outside of image");

    Mat dstMap;
    if (type == CV_8UC1)
        bottleneckDistance<KeyIntensityDiff>(img, dstMap, seed, connectivity);
    else
        bottleneckDistance<KeyColorSqEuclid>(img, dstMap, seed, connectivity);

    dstMap.copyTo(_connectivityMap);
}

} // namespace cv
