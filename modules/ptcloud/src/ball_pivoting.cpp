// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Ball-Pivoting surface reconstruction (Bernardini et al., 1999), multi-radius variant.

#include "precomp.hpp"
#include "ptcloud_utils.hpp"           // toPointVec
#include "opencv2/flann.hpp"           // cv::flann::Index (neighbor indices)

#include <list>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>

namespace cv {

namespace {

// Per-query neighbor cap for the local (radius ~2r) searches; generous but finite.
static const int kMaxNeighbors = 256;

// A directed edge on the advancing front, bounding the triangle whose third vertex is `opp`.
struct FrontEdge
{
    int i, j, opp;
    Point3f center;      // ball center of the triangle this edge belongs to
    bool active;         // false once frozen as a boundary edge (may be retried with a larger ball)
};

// Median nearest-neighbor distance over a bounded sample (shared by both spacing entry points).
static float medianNN(flann::Index& index, const Mat& ptsMat)
{
    const int N = ptsMat.rows;
    const int sample = std::min(N, 1000);
    std::vector<float> nn; nn.reserve(sample);
    for (int s = 0; s < sample; s++)
    {
        int i = (int)((int64_t)s * N / sample);
        float buf[3] = { ptsMat.at<float>(i,0), ptsMat.at<float>(i,1), ptsMat.at<float>(i,2) };
        Mat q(1, 3, CV_32F, buf), qi, qd;
        index.knnSearch(q, qi, qd, 2, flann::SearchParams());   // [0] self, [1] nearest other
        if (qd.cols >= 2) nn.push_back(std::sqrt(qd.at<float>(1)));
    }
    if (nn.empty()) return 0.f;
    std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
    return nn[nn.size() / 2];
}

class BallPivoter
{
public:
    BallPivoter(const std::vector<Point3f>& points, const std::vector<Vec3f>& normals)
        : pts(points), nrm(normals), N((int)points.size()), r(0.f),
          exact(cvflann::FLANN_CHECKS_UNLIMITED),  // BPA predicates need EXACT neighbor sets
          used(N, 0)
    {
        ptsMat = Mat(pts).reshape(1, N);   // N x 3, CV_32F
        index = makePtr<flann::Index>(ptsMat, flann::KDTreeIndexParams(1));   // single tree for exact search
    }

    // Mean point spacing from this pivoter's own index (avoids building a second kd-tree).
    float medianSpacing() { return medianNN(*index, ptsMat); }

    // Roll each radius in turn (ascending): smaller balls first, larger balls fill the leftover gaps.
    void run(std::vector<Vec3i>& tris, const std::vector<float>& radii)
    {
        const int64_t maxIters = (int64_t)100 * N + 1000;   // safety bound against pathological loops
        for (float rad : radii)
        {
            r = rad;
            // Re-arm boundary edges so the larger ball gets a chance to pivot across them.
            for (auto it = front.begin(); it != front.end(); ++it)
                if (!it->active) { it->active = true; work.push_back(ekeyDir(it->i, it->j)); }

            int seedCursor = 0;
            int64_t iters = 0;
            while (iters++ < maxIters)
            {
                std::list<FrontEdge>::iterator e = popActive();
                if (e == front.end())
                {
                    if (!findSeed(seedCursor, tris)) break;     // no more seeds for this radius
                    continue;
                }

                int k; Point3f c;
                if (pivot(*e, k, c) && canAdd(e->j, e->i, k) && !isDuplicate(e->j, e->i, k))
                {
                    int i = e->i, j = e->j;
                    addTriangle(tris, j, i, k);                 // winding consistent with the seed
                    removeEdge(e);
                    insertEdge(i, k, j, c);
                    insertEdge(k, j, i, c);
                }
                else
                {
                    e->active = false;                          // boundary edge (retried at a larger radius)
                }
            }
            if (iters >= maxIters)
                CV_LOG_WARNING(NULL, "BPA: iteration cap hit at radius " << r
                                     << "; the mesh may be incomplete");
        }
    }

private:
    const std::vector<Point3f>& pts;
    const std::vector<Vec3f>& nrm;
    int N;
    float r;
    Mat ptsMat;
    Ptr<flann::Index> index;
    flann::SearchParams exact;                                  // FLANN_CHECKS_UNLIMITED
    std::vector<char> used;                                     // point belongs to >=1 triangle

    std::list<FrontEdge> front;
    std::unordered_map<int64_t, std::list<FrontEdge>::iterator> emap;   // live directed edges
    std::deque<int64_t> work;                                           // directed-edge keys awaiting a pivot
    std::map<std::pair<int,int>, int> ecount;                             // undirected edge use count
    std::set<std::tuple<int,int,int>> triSet;                            // built triangles (sorted)

    int64_t ekeyDir(int a, int b) const { return (int64_t)a * N + b; }
    std::pair<int,int> ekeyUndir(int a, int b) const { return std::make_pair(std::min(a,b), std::max(a,b)); }
    static std::tuple<int,int,int> sorted3(int a, int b, int c)
    {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return std::make_tuple(a, b, c);
    }

    // --- neighbor queries -------------------------------------------------------------------------
    void neighbors(const Point3f& p, float rad, int maxN, std::vector<int>& out) const
    {
        float buf[3] = { p.x, p.y, p.z };                       // stack query (no per-call alloc)
        Mat q(1, 3, CV_32F, buf), idx, dist;
        int found = index->radiusSearch(q, idx, dist, (double)(rad * rad), maxN, exact);
        out.clear();
        int cnt = std::min(found, idx.cols);
        for (int t = 0; t < cnt; t++)
        {
            int id = idx.at<int>(t);
            if (id >= 0) out.push_back(id);
        }
    }

    // Ball of radius r resting on a, b, c; center pushed to the +avgNormal side. False if it can't fit.
    static bool ballCenter(const Point3f& a, const Point3f& b, const Point3f& c,
                           float r, const Vec3f& avgNormal, Point3f& out)
    {
        Vec3f u(b.x-a.x, b.y-a.y, b.z-a.z);
        Vec3f v(c.x-a.x, c.y-a.y, c.z-a.z);
        Vec3f n = u.cross(v);
        float n2 = n.dot(n);
        if (n2 < 1e-16f) return false;                                    // degenerate triangle

        Vec3f o = (u.dot(u) * v - v.dot(v) * u).cross(n) * (1.0f / (2.0f * n2));
        float circ2 = o.dot(o);
        float h2 = r * r - circ2;
        if (h2 < 0.f) return false;                                       // circumradius > r

        Vec3f cc(a.x + o[0], a.y + o[1], a.z + o[2]);
        Vec3f nhat = n * (1.0f / std::sqrt(n2));
        if (nhat.dot(avgNormal) < 0.f) nhat = -nhat;
        float h = std::sqrt(std::max(0.f, h2));
        out = Point3f(cc[0] + nhat[0]*h, cc[1] + nhat[1]*h, cc[2] + nhat[2]*h);
        return true;
    }

    // No point (other than the three it rests on) lies strictly inside the ball.
    bool ballEmpty(const Point3f& center, int e0, int e1, int e2) const
    {
        std::vector<int> nb;
        neighbors(center, r * (1.f - 1e-3f), kMaxNeighbors, nb);
        for (int k : nb)
            if (k != e0 && k != e1 && k != e2) return false;
        return true;
    }

    bool canAdd(int a, int b, int c) const
    {
        auto it = ecount.find(ekeyUndir(a,b)); if (it != ecount.end() && it->second >= 2) return false;
        it = ecount.find(ekeyUndir(b,c));      if (it != ecount.end() && it->second >= 2) return false;
        it = ecount.find(ekeyUndir(c,a));      if (it != ecount.end() && it->second >= 2) return false;
        return true;
    }

    bool isDuplicate(int a, int b, int c) const { return triSet.count(sorted3(a,b,c)) > 0; }

    void addTriangle(std::vector<Vec3i>& tris, int a, int b, int c)
    {
        tris.push_back(Vec3i(a, b, c));
        triSet.insert(sorted3(a, b, c));
        used[a] = used[b] = used[c] = 1;
        ecount[ekeyUndir(a,b)]++;
        ecount[ekeyUndir(b,c)]++;
        ecount[ekeyUndir(c,a)]++;
    }

    // --- front management -------------------------------------------------------------------------
    void insertEdge(int i, int j, int opp, const Point3f& center)
    {
        auto rev = emap.find(ekeyDir(j, i));
        if (rev != emap.end())                        // reverse present -> now an interior edge (glue)
        {
            front.erase(rev->second);
            emap.erase(rev);
            return;
        }
        if (emap.count(ekeyDir(i, j))) return;        // already on the front
        front.push_back(FrontEdge{i, j, opp, center, true});
        auto lit = std::prev(front.end());
        emap[ekeyDir(i, j)] = lit;
        work.push_back(ekeyDir(i, j));
    }

    void removeEdge(std::list<FrontEdge>::iterator lit)
    {
        emap.erase(ekeyDir(lit->i, lit->j));
        front.erase(lit);
    }

    std::list<FrontEdge>::iterator popActive()
    {
        while (!work.empty())
        {
            int64_t key = work.front();
            work.pop_front();
            // look up by key: the edge may have been erased while its key sat in the queue
            auto mit = emap.find(key);
            if (mit != emap.end() && mit->second->active)
                return mit->second;
        }
        return front.end();
    }

    // --- pivoting ---------------------------------------------------------------------------------
    bool pivot(const FrontEdge& e, int& outK, Point3f& outCenter)
    {
        Point3f pi = pts[e.i], pj = pts[e.j];
        Point3f mid((pi.x+pj.x)*0.5f, (pi.y+pj.y)*0.5f, (pi.z+pj.z)*0.5f);
        Vec3f axis = normalize(Vec3f(pj.x-pi.x, pj.y-pi.y, pj.z-pi.z));

        Vec3f a0(e.center.x-mid.x, e.center.y-mid.y, e.center.z-mid.z);
        a0 -= axis * a0.dot(axis);
        if (norm(a0) < 1e-12) return false;
        a0 = normalize(a0);
        Vec3f wdir = axis.cross(a0);

        std::vector<int> nb;
        neighbors(mid, 2.f * r, kMaxNeighbors, nb);

        float best = FLT_MAX; int bestK = -1; Point3f bestC;
        for (int k : nb)
        {
            if (k == e.i || k == e.j || k == e.opp) continue;
            Vec3f an = normalize(nrm[e.i] + nrm[e.j] + nrm[k]);
            Point3f c;
            if (!ballCenter(pi, pj, pts[k], r, an, c)) continue;

            Vec3f a1(c.x-mid.x, c.y-mid.y, c.z-mid.z);
            a1 -= axis * a1.dot(axis);
            if (norm(a1) < 1e-12) continue;
            a1 = normalize(a1);

            float x = a1.dot(a0), y = a1.dot(wdir);
            float ang = std::atan2(y, x);
            if (ang < 0.f) ang += 2.f * (float)CV_PI;      // roll angle in [0, 2pi)
            if (ang < best) { best = ang; bestK = k; bestC = c; }
        }
        if (bestK < 0) return false;
        if (!ballEmpty(bestC, e.i, e.j, bestK)) return false;
        outK = bestK; outCenter = bestC;
        return true;
    }

    // --- seeding ----------------------------------------------------------------------------------
    bool findSeed(int& cursor, std::vector<Vec3i>& tris)
    {
        for (; cursor < N; cursor++)
        {
            int i = cursor;
            if (used[i]) continue;
            std::vector<int> nb;
            neighbors(pts[i], 2.f * r, kMaxNeighbors, nb);
            for (size_t x = 0; x < nb.size(); x++)
            {
                int a = nb[x];
                if (a == i || used[a]) continue;
                for (size_t y = x + 1; y < nb.size(); y++)
                {
                    int b = nb[y];
                    if (b == i || used[b]) continue;

                    Vec3f an = normalize(nrm[i] + nrm[a] + nrm[b]);
                    int aa = a, bb = b;
                    // orient the seed so its geometric normal agrees with the point normals
                    Vec3f gn = Vec3f(pts[aa].x-pts[i].x, pts[aa].y-pts[i].y, pts[aa].z-pts[i].z)
                                   .cross(Vec3f(pts[bb].x-pts[i].x, pts[bb].y-pts[i].y, pts[bb].z-pts[i].z));
                    if (gn.dot(an) < 0.f) std::swap(aa, bb);

                    Point3f c;
                    if (!ballCenter(pts[i], pts[aa], pts[bb], r, an, c)) continue;
                    if (!ballEmpty(c, i, aa, bb)) continue;
                    if (!canAdd(i, aa, bb) || isDuplicate(i, aa, bb)) continue;

                    addTriangle(tris, i, aa, bb);
                    insertEdge(i, aa, bb, c);
                    insertEdge(aa, bb, i, c);
                    insertEdge(bb, i, aa, c);
                    return true;
                }
            }
        }
        return false;
    }
};

} // namespace

float estimateMeanSpacing(InputArray inputCloud)
{
    CV_TRACE_FUNCTION();

    std::vector<Point3f> pts;
    toPointVec(inputCloud, pts);
    const int N = (int)pts.size();
    if (N < 2) return 0.f;

    Mat ptsMat = Mat(pts).reshape(1, N);   // N x 3, CV_32F
    flann::Index index(ptsMat, flann::KDTreeIndexParams(4));
    return medianNN(index, ptsMat);
}

void createMeshBPA(InputArray inputCloud, InputArray normals_, OutputArray vertices,
                   OutputArray triangles, InputArray radii_)
{
    CV_TRACE_FUNCTION();

    std::vector<Point3f> pts;
    toPointVec(inputCloud, pts);
    const int N = (int)pts.size();
    if (N < 3) { vertices.release(); triangles.release(); return; }

    Mat nm = normals_.getMat();
    CV_Assert(!nm.empty() && nm.channels() * (int)nm.total() == 3 * N);
    Mat nmw = nm.isContinuous() ? nm : nm.clone();     // read-only; reshape needs contiguity
    Mat nmf = nmw.reshape(3, N);
    std::vector<Vec3f> nrm(N);
    for (int i = 0; i < N; i++)
        nrm[i] = normalize(nmf.at<Vec3f>(i));

    BallPivoter bp(pts, nrm);   // builds the kd-tree once; reused for spacing and pivoting

    // Radii: caller-supplied, else {1x, 2x, 4x} the mean spacing.
    std::vector<float> radii;
    if (radii_.empty())
    {
        float s = bp.medianSpacing();
        if (s <= 0.f) { vertices.release(); triangles.release(); return; }
        radii = { 1.0f * s, 2.0f * s, 4.0f * s };
    }
    else
    {
        Mat r; radii_.getMat().convertTo(r, CV_64F);
        const double* rp = r.ptr<double>();
        for (int i = 0; i < (int)r.total(); i++)
            if (rp[i] > 0.0) radii.push_back((float)rp[i]);
        std::sort(radii.begin(), radii.end());
    }
    if (radii.empty()) { vertices.release(); triangles.release(); return; }

    std::vector<Vec3i> tris;
    bp.run(tris, radii);

    // BPA is interpolating: the mesh vertices are the input points.
    Mat(pts).copyTo(vertices);                                  // N x 1, CV_32FC3
    if (tris.empty()) { triangles.release(); return; }
    Mat(tris).reshape(1, (int)tris.size()).copyTo(triangles);   // M x 3, CV_32S
}

} // namespace cv
