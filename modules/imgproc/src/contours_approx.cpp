// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/base.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>

using namespace std;
using namespace cv;

namespace {

struct ApproxItem
{
    Point pt;
    int k; // support region
    int s; // 1-curvature
    bool removed;
    ApproxItem() : k(0), s(0), removed(false) {}
    ApproxItem(Point pt_, int s_) : pt(pt_), k(0), s(s_), removed(false) {}
};

static const schar abs_diff[16] = { 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1 };
static const Point chainCodeDeltas[8] =
    { {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1} };

// Pass 0.
// Restores all the digital curve points from the chain code.
// Removes the points (from the resultant polygon)
// that have zero 1-curvature
static vector<ApproxItem> pass_0(const vector<schar> & chain, Point pt, bool isApprox, bool isFull)
{
    vector<ApproxItem> res;
    const size_t len = chain.size();
    for (size_t i = 0; i < len; ++i)
    {
        const schar prev = (i == 0) ? chain[len - 1] : chain[i - 1];
        const schar cur = chain[i];
        const schar s = abs_diff[cur - prev + 7];
        if ((!isApprox && (isFull || s != 0)) || isApprox)
        {
            res.push_back(ApproxItem(pt, s));
            if (s == 0)
                (res.end() - 1)->removed = true;
        }
        pt += chainCodeDeltas[cur];
    }
    return res;
}

static vector<Point> gatherPoints(const vector<ApproxItem> & ares)
{
    vector<Point> res;
    for (const ApproxItem & item : ares)
    {
        if (item.removed)
            continue;
        res.push_back(item.pt);
    }
    return res;
}

static int calc_support(const vector<ApproxItem> & ares, int i)
{
    int k;
    int d_num = 0;
    int l = 0;
    const int len = (int)ares.size();
    /* determine support region */
    for( k = 1;; k++ )
    {
        CV_Assert( k <= len );
        /* calc indices */
        const int i1 = (i >= k) ? (i - k) : (len - k + i);
        const int i2 = (i + k < len) ? (i + k) : (i + k - len);


        int lk, dk_num;
        int dx, dy;
        dx = ares[i2].pt.x - ares[i1].pt.x;
        dy = ares[i2].pt.y - ares[i1].pt.y;

        /* distance between p_(i - k) and p_(i + k) */
        lk = dx * dx + dy * dy;

        /* distance between p_i and the line (p_(i-k), p_(i+k)) */
        dk_num = (ares[i].pt.x - ares[i1].pt.x) * dy - (ares[i].pt.y - ares[i1].pt.y) * dx;

        Cv32suf d;
        d.f = (float) (((double) d_num) * lk - ((double) dk_num) * l);

        if( k > 1 && (l >= lk || ((d_num > 0 && d.i <= 0) || (d_num < 0 && d.i >= 0))))
            break;

        d_num = dk_num;
        l = lk;
    }
    return k - 1;
}

static int calc_cosine(const vector<ApproxItem> & ares, int i)
{
    const int k = ares[i].k;
    int j;
    int s;
    const int len = (int)ares.size();
    /* calc k-cosine curvature */
    for( j = k, s = 0; j > 0; j-- )
    {
        const int i1 = (i >= j) ? (i - j) : (len - j + i);
        const int i2 = (i + j < len) ? (i + j) : (i + j - len);

        const int dx1 = ares[i1].pt.x - ares[i].pt.x;
        const int dy1 = ares[i1].pt.y - ares[i].pt.y;
        const int dx2 = ares[i2].pt.x - ares[i].pt.x;
        const int dy2 = ares[i2].pt.y - ares[i].pt.y;

        if( (dx1 | dy1) == 0 || (dx2 | dy2) == 0 )
            break;

        double temp_num = dx1 * dx2 + dy1 * dy2;
        temp_num =
            (float) (temp_num /
                     sqrt( ((double)dx1 * dx1 + (double)dy1 * dy1) *
                           ((double)dx2 * dx2 + (double)dy2 * dy2) ));
        Cv32suf sk;
        sk.f = (float) (temp_num + 1.1);

        CV_Assert( 0 <= sk.f && sk.f <= 2.2 );
        if( j < k && sk.i <= s )
            break;

        s = sk.i;
    }
    return s;
}

static bool calc_nms_cleanup(const vector<ApproxItem> & ares, int i)
{
    const int k2 = ares[i].k >> 1;
    const int s = ares[i].s;
    const int len = (int)ares.size();
    int j;
    for( j = 1; j <= k2; j++ )
    {
        const int i1 = (i >= j) ? (i - j) : (len - j + i);
        const int i2 = (i + j < len) ? (i + j) : (i + j - len);
        if (ares[i1].s > s || ares[i2].s > s)
            break;
    }
    return j <= k2;
}

static bool calc_dominance(const vector<ApproxItem> & ares, int i)
{
    const int len = (int)ares.size();
    const int i1 = (i >= 1) ? (i - 1) : (len - 1 + i);
    const int i2 = (i + 1 < len) ? (i + 1) : (i + 1 - len);
    return ares[i].s <= ares[i1].s || ares[i].s <= ares[i2].s;
}

inline int get_next_idx(const vector<ApproxItem> & ares, const int start)
{
    const int len = (int)ares.size();
    int res = start + 1;
    for (; res < len; ++res)
    {
        if (!ares[res].removed)
            break;
    }
    return res;
}

inline void clear_until(vector<ApproxItem> & ares, const int start, const int finish)
{
    const int len = (int)ares.size();
    for (int i = start + 1; i < finish && i < len; ++i)
    {
        ares[i].removed = true;
    }
}

static int calc_new_start(vector<ApproxItem> & ares)
{
    const int len = (int)ares.size();
    int i1;
    // remove all previous items from the beginning
    for( i1 = 1; i1 < len && ares[i1].s != 0; i1++ )
    {
        ares[i1 - 1].s = 0;
    }
    if( i1 == len )
    {
        // all points survived - skip to the end
        return -1;
    }
    else
    {
        i1--;

        int i2;
        // remove all following items from the end
        for( i2 = len - 2; i2 > 0 && ares[i2].s != 0; i2-- )
        {
            clear_until(ares, i2, len);
            ares[i2 + 1].s = 0;
        }
        i2++;

        // only two points left
        if( i1 == 0 && i2 == len - 1 )
        {
            // find first non-removed element from the start
            i1 = get_next_idx(ares, 0);
            // append first item to the end
            ares.push_back(ares[0]);
            (ares.end() - 1)->removed = false;
        }
        return i1;
    }
}

static void pass_cleanup(vector<ApproxItem> & ares, int start_idx)
{
    int count = 1;

    const int len = (int)ares.size();
    int first = start_idx;
    for (int i = start_idx, prev = i; i < len; ++i)
    {
        ApproxItem & item = ares[i];
        if (item.removed)
            continue;
        int next_idx = get_next_idx(ares, i);
        if (next_idx == len || next_idx - i != 1)
        {
            if (count >= 2)
            {
                if( count == 2 )
                {
                    const int s1 = ares[prev].s;
                    const int s2 = ares[i].s;

                    if( s1 > s2 || (s1 == s2 && ares[prev].k <= ares[i].k) )
                        /* remove second */
                        clear_until(ares, get_next_idx(ares, prev), get_next_idx(ares, i));
                    else
                        /* remove first */
                        clear_until(ares, first, i);
                }
                else
                {
                    first = get_next_idx(ares, first);
                    clear_until(ares, first, i);
                }
            }
            clear_until(ares, first, i);
            first = i;
            count = 1;
        }
        else
        {
            ++count;
        }
        prev = i;
    }
}

} // <anonymous>::



vector<Point> cv::approximateChainTC89(vector<schar> chain, const Point &origin, const int method)
{
    if (chain.size() == 0)
    {
        return vector<Point>({ origin });
    }

    const bool isApprox = method == CHAIN_APPROX_TC89_L1 || method == CHAIN_APPROX_TC89_KCOS;

    ApproxItem root;
    vector<ApproxItem> ares = pass_0(chain, origin, isApprox, method == CHAIN_APPROX_NONE);

    if (isApprox)
    {
        CV_DbgAssert(ares.size() < (size_t)numeric_limits<int>::max());

        // Pass 1.
        // Determines support region for all the remained points */
        for (int i = 0; i < (int)ares.size(); ++i)
        {
            ApproxItem & item = ares[i];
            if (item.removed)
                continue;
            item.k = calc_support(ares, i);

            if (method == CHAIN_APPROX_TC89_KCOS)
                item.s = calc_cosine(ares, i);
        }

        // Pass 2.
        // Performs non-maxima suppression
        for (int i = 0; i < (int)ares.size(); ++i)
        {
            ApproxItem & item = ares[i];
            if (calc_nms_cleanup(ares, i))
            {
                item.s = 0; // "clear"
                item.removed = true;
            }
        }

        // Pass 3.
        // Removes non-dominant points with 1-length support region */
        for (int i = 0; i < (int)ares.size(); ++i)
        {
            ApproxItem & item = ares[i];
            if (item.removed)
                continue;
            if (item.k == 1 && calc_dominance(ares, i))
            {
                item.s = 0;
                item.removed = true;
            }
        }

        if( method == cv::CHAIN_APPROX_TC89_L1 )
        {
            // Pass 4.
            // Cleans remained couples of points
            bool skip = false;
            int new_start_idx = 0;
            const int len = (int)ares.size();
            if (ares[0].s != 0 && ares[len - 1].s != 0)
            {
                new_start_idx = calc_new_start(ares);
                if (new_start_idx < 0)
                {
                    skip = true;
                }
            }
            if (!skip)
            {
                pass_cleanup(ares, new_start_idx);
            }
        }
    }

    return gatherPoints(ares);
}
