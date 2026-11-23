// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

/********************************* ALGORITHM NOTE *******************************\
  lambda-connected segmentation, following the fuzzy-connectedness formulation of:
    L. Chen, H.D. Cheng, J. Zhang, "Fuzzy subfiber and its application to seismic
    lithology classification", Information Sciences-Applications, 1(2), 1994.
    L. Chen, "The lambda-connected segmentation and the optimal algorithm for
    split-and-merge segmentation", Chinese J. Computers, Vol. 14, 1991.

  Connectivity between the seed and a pixel p is the maximum, over all paths from
  the seed to p, of the minimum pairwise "degree of connectedness" along the path
  (a max-min / bottleneck-path value). This is the classical single-source maximum
  bottleneck spanning tree problem and is solved below with a Prim's-algorithm-style
  procedure using a binary max-heap, giving O(N log N) for an N-pixel image.
\**********************************************************************************/

#include "precomp.hpp"
#include <queue>

namespace cv
{

namespace
{

// Continuous degree-of-connectedness functors. Each returns a value in [0, 1]:
// 1 when the two pixels are identical, decaying linearly to 0 once the raw
// per-pixel distance reaches maxDist. These are intentionally simple, closed-form
// functions (plain intensity difference for grayscale, Euclidean color distance
// for BGR) -- see the PR description / doc comment on cv::lambdaConnected for why
// a fuller pluggable-functor API was not attempted in this first version.

struct DegreeIntensityDiff
{
    explicit DegreeIntensityDiff(double maxDist)
        : invMaxDist(maxDist > 0 ? 1.0 / maxDist : 0.0) {}

    inline float operator()(const uchar* a, const uchar* b) const
    {
        int d = std::abs((int)a[0] - (int)b[0]);
        double deg = 1.0 - d * invMaxDist;
        return deg > 0.0 ? (float)deg : 0.f;
    }

    double invMaxDist;
};

struct DegreeColorEuclid
{
    explicit DegreeColorEuclid(double maxDist)
        : invMaxDist(maxDist > 0 ? 1.0 / maxDist : 0.0) {}

    inline float operator()(const uchar* a, const uchar* b) const
    {
        double d0 = (double)a[0] - b[0];
        double d1 = (double)a[1] - b[1];
        double d2 = (double)a[2] - b[2];
        double dist = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
        double deg = 1.0 - dist * invMaxDist;
        return deg > 0.0 ? (float)deg : 0.f;
    }

    double invMaxDist;
};

// Prim's-algorithm-style widest-path computation: maintains, for every pixel, the
// best bottleneck value found so far to reach it from the seed, and repeatedly
// expands the not-yet-finalized pixel with the highest current bottleneck value
// using a binary max-heap. Standard Dijkstra-like structure, but relaxation
// combines values with min() and the heap picks the maximum (instead of summing
// distances and picking the minimum).
//
// Complexity: the grid graph has E = O(connectivity * N) edges. Every successful
// relaxation pushes exactly one heap entry, so the heap performs O(E) push/pop
// operations in total (stale duplicate entries left behind by relaxations that
// later get beaten are discarded in O(1) via the `visited` check), each costing
// O(log E) = O(log N). Overall: O(N log N).
template<typename Degree>
static void bottleneckConnectivity(const Mat& img, Mat& connectivityMap,
                                    Point seed, int connectivity, const Degree& degree)
{
    const int rows = img.rows, cols = img.cols;
    const size_t elemSize = img.elemSize();

    connectivityMap.create(rows, cols, CV_32F);
    connectivityMap.setTo(Scalar(0));

    Mat visited = Mat::zeros(rows, cols, CV_8U);
    uchar* visitedData = visited.ptr<uchar>();
    float* mapData = connectivityMap.ptr<float>();
    const uchar* imgData = img.ptr<uchar>();
    const size_t step = img.step;

    static const int dx4[4] = { 1, -1, 0, 0 };
    static const int dy4[4] = { 0, 0, 1, -1 };
    static const int dx8[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
    static const int dy8[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };
    const int* dx = connectivity == 8 ? dx8 : dx4;
    const int* dy = connectivity == 8 ? dy8 : dy4;
    const int nnb = connectivity;

    typedef std::pair<float, int> Entry; // (bottleneck value, flat pixel index)
    std::priority_queue<Entry> heap;     // std::priority_queue is a max-heap by default

    const int seedIdx = seed.y * cols + seed.x;
    mapData[seedIdx] = 1.f;
    heap.push(Entry(1.f, seedIdx));

    while (!heap.empty())
    {
        Entry e = heap.top();
        heap.pop();
        const float b = e.first;
        const int idx = e.second;

        if (visitedData[idx])
            continue; // stale duplicate from an earlier, weaker relaxation
        if (b < mapData[idx])
            continue; // superseded by a better value already recorded

        visitedData[idx] = 1;
        const int y = idx / cols, x = idx % cols;
        const uchar* pCur = imgData + (size_t)y * step + x * elemSize;

        for (int k = 0; k < nnb; k++)
        {
            const int ny = y + dy[k], nx = x + dx[k];
            if ((unsigned)ny >= (unsigned)rows || (unsigned)nx >= (unsigned)cols)
                continue;
            const int nIdx = ny * cols + nx;
            if (visitedData[nIdx])
                continue;

            const uchar* pNb = imgData + (size_t)ny * step + nx * elemSize;
            const float deg = degree(pCur, pNb);
            const float cand = std::min(b, deg);

            if (cand > mapData[nIdx])
            {
                mapData[nIdx] = cand;
                heap.push(Entry(cand, nIdx));
            }
        }
    }
}

} // namespace

void lambdaConnected( InputArray _image, OutputArray _mask,
                       Point seed, double lambda,
                       int connectivity, double maxDist,
                       OutputArray _connectivityMap )
{
    CV_INSTRUMENT_REGION();

    Mat img = _image.getMat();
    CV_Assert(!img.empty());

    const int type = img.type();
    CV_CheckType(type, type == CV_8UC1 || type == CV_8UC3,
                 "lambdaConnected supports CV_8UC1 (grayscale) or CV_8UC3 (BGR) images only");

    if (connectivity != 4 && connectivity != 8)
        CV_Error(cv::Error::StsBadFlag, "Connectivity must be 4 or 8");

    if (!(lambda >= 0.0 && lambda <= 1.0))
        CV_Error(cv::Error::StsOutOfRange, "lambda must be in the range [0, 1]");

    if (!(maxDist > 0.0))
        CV_Error(cv::Error::StsOutOfRange, "maxDist must be positive");

    if ((unsigned)seed.x >= (unsigned)img.cols || (unsigned)seed.y >= (unsigned)img.rows)
        CV_Error(cv::Error::StsOutOfRange, "Seed point is outside of image");

    Mat connectivityMap;
    if (type == CV_8UC1)
        bottleneckConnectivity(img, connectivityMap, seed, connectivity, DegreeIntensityDiff(maxDist));
    else
        bottleneckConnectivity(img, connectivityMap, seed, connectivity, DegreeColorEuclid(maxDist));

    _mask.create(img.size(), CV_8UC1);
    Mat mask = _mask.getMat();
    const float lam = (float)lambda;
    for (int y = 0; y < img.rows; y++)
    {
        const float* mp = connectivityMap.ptr<float>(y);
        uchar* mk = mask.ptr<uchar>(y);
        for (int x = 0; x < img.cols; x++)
            mk[x] = (mp[x] >= lam) ? 255 : 0;
    }

    if (_connectivityMap.needed())
        connectivityMap.copyTo(_connectivityMap);
}

} // namespace cv
