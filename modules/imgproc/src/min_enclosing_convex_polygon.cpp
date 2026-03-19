/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  This file is part of OpenCV project.
 //  It is subject to the license terms in the LICENSE file found in the top-level directory of this
 //  distribution and at http://opencv.org/license.html.
 //
 //  Author: Sara Kuhnert
 //  Created: 20.05.2025
 //  E-mail: <sara.kuhnert[AT]gmx.de>
 //
 //  THE IMPLEMENTATION OF THE MODULES IS BASED ON THE FOLLOWING PAPER:
 //
 //  [1] A. Aggarwal, J. S. Chang, Chee K. Yap: "Minimum area circumscribing Polygons", The Visual
 //  Computer, 1:112-117, 1985
 //
 //  The overall complexity of the algorithm is theta(n^2log(n)log(k)) where "n" represents the number
 //  of vertices in the input convex polygon and "k" the number of vertices in the output polygon.
 //
 //M*/

#include "precomp.hpp"
#include <opencv2/core/utils/logger.hpp>

#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <set>
#include <iostream>
#include <cstdlib>

//////////////////////////// Constants definitions ////////////////////////////

#define EPSILON 1e-6

//////////////////////// Class and struct declarations ////////////////////////

namespace minEnclosingConvexPolygon {

//! Intersection point of two sides with its position relative to the polygon
/*!
 * @param point         Intersection point of the two sides
 * @param position      Is the intersection point a valid solution?
 */
struct IntersectionPoint
{
    cv::Point2f point = {-1.0, -1.0};
    bool position = false;

    IntersectionPoint() = default;
    IntersectionPoint(const cv::Point2f& a, const cv::Point2f& b,
                      const cv::Point2f& c, const cv::Point2f& d);
    IntersectionPoint(int i, int j, const std::vector<cv::Point2f>& ngon);
};

//! Container for the information about the intersection point of two flush sides
/*!
 * @param intersection  Instance of IntersectionPoint
 * @param extra_area    The area that would be added to the kgon compared to the ngon
 * @param done          Set to true once calculated to avoid redundant calculations
 */
struct FlushIntersect
{
    IntersectionPoint intersection = {};
    double extra_area = std::numeric_limits<double>::max();
    bool done = false;
};

//! Container for the information about a balanced side between two points
/*!
 * @param pi            First intersection point
 * @param pj            Second intersection point
 * @param extra_area    The area that would be added to the kgon compared to the ngon
 * @param flush         Flush is a special case of balanced. If the balanced side is also flush, this indicates which of the sides of the ngon it is flush with. The default for not flush sides is -1.
 * @param position      Is the balanced side a valid solution?
 * @param done          Set to true once calculated to avoid redundant calculations
 */
struct BalancedIntersect
{
    cv::Point2f pi = {-1, -1};
    cv::Point2f pj = {-1, -1};
    double extra_area = std::numeric_limits<double>::max();
    int flush = -1;
    bool position = false;
    bool done = false;
};

//! Container for the best segment between two points
/*!
 * @param extra_area    The area that would be added to the kgon compared to the ngon
 * @param side          Side of the ngon to which it is flush (if flush is true) or edge of the ngon which it is touching (if flush is false)
 * @param flush         True if it is flush. If false, it is balanced but not flush
 * @param exists        Does a valid solution exist?
 * @param done          Set to true once calculated to avoid redundant calculations
 */
struct Segment
{
    double extra_area = std::numeric_limits<double>::max();
    int side = -1;
    bool flush = false;
    bool exists = false;
    bool done = false;
};

//! Combination of selected sides and their corresponding chains, which generates the kgon with the minimum area
/*!
 * @param area          Extra area that the minimal kgon is bigger than the input ngon
 * @param i             First side
 * @param j             Second side
 */
struct Minimum
{
    double area = std::numeric_limits<double>::max();
    int i = -1;
    int j = -1;
};

//! Container for a side of the minimal kgon
/*!
 * @param side          Index of the side of the ngon which it is flush with or the vertex which it is touching
 * @param flush         Is this side flush?
 */
struct Side
{
    int side = -1;
    bool flush = false;

    Side() = default;
    Side(int i, bool b);
};

//! Container for the minimal kgon
/*!
 * @param sides         Indices of the corresponding sindes of the ngon
 * @param vertices      Vertices of the kgon
 * @param extra_area    Extra area that the minimal kgon is bigger than the input ngon
 * @param i, @param j   The chains which build the minimal kgon are formed between those two sides
 */
struct Kgon
{
    std::vector<Side> sides;
    std::vector<cv::Point2f> vertices;
    double extra_area = std::numeric_limits<double>::max();
    int i = -1;
    int j = -1;
};

//! Class for all the possible combinations of flush intersections of two sides of the input polygon
/*!
 * @param ngon              Input polygon
 * @param intersections     Matrix of all possible flush intersections of two sides of the ngon
 * @param area_edges        Collection of the points that define the extra area of the kgon
 */
class FlushIntersections
{
private:
    const std::vector<cv::Point2f>& ngon;
    std::vector<std::vector<FlushIntersect>> intersections;
    std::vector<cv::Point2f> area_edges;

public:
    FlushIntersections(const std::vector<cv::Point2f>& _ngon);
    const FlushIntersect& lineIntersect(int i, int j);
};

//! Class for all the possible combinations of balanced intersections with two sides of the input polygon
/*!
 * @param ngon                      Input polygon
 * @param balanced_intersections    atrix of all possible balanced intersections of two sides of the ngon
 * @param area_edges                Collection of the points that define the extra area of the kgon
 */
class BalancedIntersections
{
private:
    const std::vector<cv::Point2f>& ngon;
    std::vector<std::vector<BalancedIntersect>> balanced_intersections;
    std::vector<cv::Point2f> area_edges;

    double extraArea ( int first, int last, const cv::Point2f& extra1, const cv::Point2f& extra2 );

    BalancedIntersect flush(int i, int j, int e);

public:
    BalancedIntersections(const std::vector<cv::Point2f>& _ngon);
    BalancedIntersect balancedIntersect(int i, int j, int e);
    void markAsDone(int i, int j);
    const std::vector<BalancedIntersect>& operator[](size_t i) const;
};

//! Class for building the one-sided and h-sided chains and calculation the minimum enclosing polygon based on those chains
/*!
 * @param ngon              Input polygon
 * @param single_sides      Matrix of the best one-sided chain for each combination of two sides of the ngon
 * @param middle_sides      Matrix of the middle side of each h-sided chain for all relevant h and each combination of two sides of the ngon
 * @param intersections     An instance of FlushIntersections to store already calculated flush intersections
 * @param balanced_inters   An instance of BalancedIntersections to store already calculated balanced intersections
 */
class Chains
{
private:
    const std::vector<cv::Point2f>& ngon;
    std::vector<std::vector<Segment>> single_sides;
    std::vector<std::vector<std::vector<Segment>>> middle_sides;
    FlushIntersections intersections;
    BalancedIntersections balanced_inters;

    void findSingleE(int i, int j, int l, int r);
    void singleSideImpl(int i, int j1, int j2);
    void findMiddleE1(int i, int j, int l, int r);
    void middleSideImpl1(int i, int j1, int j2);
    void findMiddleE(int h, int i, int j, int l, int r);
    void middleSideImpl(int h, int i, int j1, int j2);

public:
    Chains(const std::vector<cv::Point2f>& _ngon, int k);

    std::set<int> relevantChainLengths(int h);
    void calcOneSidedChains();
    void calcMiddleChains(int h);

    Minimum minimumArea(int n, int k);
    std::vector<Side> reconstructHSidedChain(int h, int i, int j);
    std::vector<Side> findKSides(int k, int i, int j);
    std::vector<cv::Point2f> findKVertices(std::vector<Side>& sides);

};

} //namespace


//////////////////////// Class and struct constructors ////////////////////////

namespace minEnclosingConvexPolygon {

//! Constructor for IntersectionPoint. Find the intersection point of two lines given by four points and decide whether it is on the correct side of the polygon
/*!
 * @param a             First point of the first line
 * @param b             Second point of the first line
 * @param c             First point of the second line
 * @param d             Second point of the second line
 */
IntersectionPoint::IntersectionPoint(const cv::Point2f& a, const cv::Point2f& b,
                                     const cv::Point2f& c, const cv::Point2f& d)
{
    const cv::Point2f ab = b - a, cd = d - c, ac = c - a;
    const double det = ab.cross(-cd);
    if(std::abs (det) < EPSILON )
        return;

    const double loc = ac.cross(-cd) / det;
    if(loc <= 0)
        return;

    point = a + loc * ab;
    position = true;
}

//! Constructor for IntersectionPoint. Find the intersection point of two sides of the polygon based on the given side indices.
/*!
 * @param i             Index of the first side
 * @param j             Index of the second side
 * @param ngon          Input polygon with n sides
 */
IntersectionPoint::IntersectionPoint(int i, int j,
                                     const std::vector<cv::Point2f>& ngon) :
    IntersectionPoint(ngon[i],
                      ngon[(i + 1) % ngon.size()],
                      ngon[j],
                      ngon[(j + 1) % ngon.size()])
{}

//! Constructor for FlushIntersections
/*!
 * @param _ngon         Input polygon with n sides
 */
FlushIntersections::FlushIntersections(const std::vector<cv::Point2f>& _ngon) :
    ngon(_ngon),
    intersections(ngon.size(),
                  std::vector<FlushIntersect>(ngon.size()))
{
    area_edges.reserve(ngon.size());
}

//! Constructor for BalancedIntersections
/*!
 * @param _ngon         Input polygon with n sides
 */
BalancedIntersections::BalancedIntersections(const std::vector<cv::Point2f>& _ngon) :
ngon(_ngon),
balanced_intersections(ngon.size(),
                        std::vector<BalancedIntersect>(ngon.size()))
{
    area_edges.reserve(ngon.size());
}

//! Constructor for Side. Assign a side index and weather it is flush or not.
/*!
 * @param i             Side index
 * @param b             Is side i flush?
 */
Side::Side(int i, bool b)
{
    side = i;
    flush = b;
}

//! Constructor for Chains
/*!
 * @param
 */
Chains::Chains(const std::vector<cv::Point2f>& _ngon, int k) :
    ngon(_ngon),
    single_sides(std::vector<std::vector<Segment>>(ngon.size(),
                                                   std::vector<Segment>(ngon.size()))),
    middle_sides(std::vector<std::vector<std::vector<Segment>>>(
        k, std::vector<std::vector<Segment>>(ngon.size(),
                                             std::vector<Segment>(ngon.size())))),
    intersections(ngon),
    balanced_inters(ngon)
{}

} //namespace


////////////////////////// Class and struct functions //////////////////////////

namespace minEnclosingConvexPolygon {

//! Find the intersection point of two sides, decide weather it is a valid point and if so calculate the extra area caused by that intersection.
/*!
 * @param i             Index of the first side
 * @param j             Index of the second side
 */
const FlushIntersect& FlushIntersections::lineIntersect(int i, int j)
{
    FlushIntersect& itr = intersections[i][j];
    if(itr.done)
        return itr;

    const int n = (int)ngon.size();
    if((i + 1) % n == j)
    {
        itr.intersection.point = ngon[j];
        itr.intersection.position = true;
        itr.extra_area = 0.0;
        itr.done = true;
        return itr;
    }
    itr.intersection = IntersectionPoint(i, j, ngon);
    if(itr.intersection.position)
    {
        area_edges.resize(0);
        for(int t = (i + 1) % n; t != (j + 1) % n; t = (t + 1) % n)
        {
            area_edges.push_back(ngon[t]);
        }
        area_edges.push_back(itr.intersection.point);
        itr.extra_area = cv::contourArea(area_edges);
        itr.done = true;
    }
    else
    {
        itr.extra_area = std::numeric_limits<double>::max();
        itr.intersection.position = false;
        itr.done = true;
    }
    return itr;
}

//! Calculate the added area that is enclosed by a sequence of consecutive vertices of the polygon and the intersection point of two sides.
/*!
 * @param first         Index of the first point of the sequence
 * @param last          Index of the last point of the sequence
 * @param extra1        Last point of the sequence
 * @param extra2        Intersection point
 */
double BalancedIntersections::extraArea(int first, int last,
                                        const cv::Point2f& extra1,
                                        const cv::Point2f& extra2)
{
    const size_t n = ngon.size();
    area_edges.resize(0);
    for(int t = first; t != last; t = (t + 1) % n)
        area_edges.push_back(ngon[t]);

    area_edges.push_back(extra1);
    area_edges.push_back(extra2);

    return cv::contourArea(area_edges);
}

//! Determine the intersection points of a flush side e that lies between sides i and j with these two sides. Calculate the extra area and the position of the intersection points relative to the polygon. Update balanced_intersections if the new area is smaller than extraArea.
/*!
 * @param i             Index of first side
 * @param j             Index of second side
 * @param e             Index of a side between i and j
 */
BalancedIntersect BalancedIntersections::flush(int i, int j, int e)
{
    CV_Assert(j != e);

    const int n = (int)ngon.size();
    const int before = (e - 1 + n) % n;
    BalancedIntersect bi = balanced_intersections[i][j];

    const IntersectionPoint left_e(i, e, ngon);
    const IntersectionPoint right_e(e, j, ngon);

    if(left_e.position == true && right_e.position == true)
    {
        double extra_area = extraArea((i + 1) % n, e, ngon[e], left_e.point);
        if(extra_area < bi.extra_area)
        {
            extra_area += extraArea((e + 1) % n, j, ngon[j], right_e.point);
            if(extra_area < bi.extra_area)
            {
                bi.extra_area = extra_area;
                bi.pi = left_e.point;
                bi.pj = right_e.point;
                bi.position = true;
                bi.flush = e;
            }
        }
    }

    if(before != i)
    {
        const IntersectionPoint left_before(i, before, ngon);
        const IntersectionPoint right_before(before, j, ngon);
        if(left_before.position == true && right_before.position == true)
        {
            double extra_area =
            extraArea((i + 1) % n, before, ngon[before], left_before.point);

            if(extra_area < bi.extra_area)
            {
                extra_area += extraArea(e, j, ngon[j], right_before.point);
                if(extra_area < bi.extra_area)
                {
                    bi.extra_area = extra_area;
                    bi.pi = left_before.point;
                    bi.pj = right_before.point;
                    bi.position = true;
                    bi.flush = before;
                }
            }
        }
    }
    return bi;
}

//! Determine the intersection points of a balanced side e that lies between sides i and j with these two sides. If no valid balanced edge is found, ccheck for flush sides. Calculate the extra area and the position of the intersection points relative to the polygon. Update balanced_intersections if the new area is smaller than extraArea.
/*!
 * @param i             Index of first side
 * @param j             Index of second side
 * @param e             Index of a side between i and j
 */
BalancedIntersect BalancedIntersections::balancedIntersect(int i, int j, int e)
{
    if(balanced_intersections[i][j].done)
        return balanced_intersections[i][j];

    const int n = (int)ngon.size();
    if((i + 2) % n == j)
    {
        BalancedIntersect& bi = balanced_intersections[i][j];
        bi.pi = ngon[(i + 1) % n];
        bi.pj = ngon[j];
        bi.flush = (i + 1) % n;
        bi.position = true;
        bi.extra_area = 0.0;
        bi.done = true;
        return bi;
    }

    const cv::Point2f p1 = ngon[i], p2 = ngon[(i + 1) % n], p3 = ngon[e],
    p4 = ngon[j], p5 = ngon[(j + 1) % n];
    const cv::Point2f dir12 = p2 - p1, dir45 = p5 - p4;
    const double det = dir12.cross(dir45);
    if(std::abs (det) < EPSILON )
    {
        flush(i, j, e);
        return balanced_intersections[i][j];
    }

    BalancedIntersect bi;
    cv::Point2f temp = 2 * p3 - p2 - p4;
    const double s = temp.cross(dir45) / det;
    const double t = temp.cross(dir12) / det;
    if(s >= 0 && t >= 0)
    {
        bi.pi = p2 + dir12 * s;
        bi.pj = p4 - dir45 * t;
        bi.position = true;

        const cv::Point2f dir_balanced = bi.pj - bi.pi,
        dir_left = p3 - ngon[(e - 1 + n) % n],
        dir_right = ngon[(e + 1) % n] - p3;

        const double cross_left = dir_balanced.cross(dir_left),
        cross_right = dir_balanced.cross(dir_right);
        if((cross_left < 0 && cross_right < 0) || (cross_left > 0 && cross_right > 0))
        {
            BalancedIntersect reset;
            bi = reset;
            bi = flush(i, j, e);
        }
        else if(std::abs (cross_left) < EPSILON )
        {
            bi.flush = (e - 1 + n) % n;
            bi.extra_area =
            extraArea((i + 1) % n, (e - 1 + n) % n, ngon[(e - 1 + n) % n], bi.pi)
            + extraArea(e, j, ngon[j], bi.pj);
        }
        else if(std::abs (cross_right) < EPSILON )
        {
            bi.flush = e;
            bi.extra_area = extraArea((i + 1) % n, e, ngon[e], bi.pi)
            + extraArea((e + 1) % n, j, ngon[j], bi.pj);
        }
        else
        {
            bi.extra_area = extraArea((i + 1) % n, e, ngon[e], bi.pi)
            + extraArea(e, j, ngon[j], bi.pj);
        }
    }
    else
    {
        flush(i, j, e);
    }

    if(bi.extra_area < balanced_intersections[i][j].extra_area)
    {
        balanced_intersections[i][j] = bi;
    }
    return bi;
}

//! Set function for the done attribute of BalancedIntersections
/*!
 * @param i             Index of first side
 * @param j             Index of second side
 */
void BalancedIntersections::markAsDone(int i, int j)
{
    balanced_intersections[i][j].done = true;
}

//! Operator to get a vector of elements from BalancedIntersections
/*!
 * @param i             index of a side
 */
const std::vector<BalancedIntersect>& BalancedIntersections::operator[](
    size_t i) const
{
    return balanced_intersections[i];
}

//! For a combination of two fixed sides of the ngon test all sides between left and right boundary to find the one sided chain with minimal extra area
/*!
 * @param i     Index of first fixed side
 * @param j     Index of second fixed side
 * @param l     Index of left boundary
 * @param r     Index of right boundary
 */
void Chains::findSingleE(int i, int j, int l, int r)
{
    const size_t n = ngon.size();
    Segment& one = single_sides[i][j];
    if (one.done)
        return;

    double min_area = std::numeric_limits<double>::max();
    for (int e = l; e != r + 1 && e != j; e = (e + 1) %n)
    {
        BalancedIntersect candidate = balanced_inters.balancedIntersect(i, j, e);
        if(candidate.extra_area < min_area)
        {
            min_area = candidate.extra_area;
            one.side = e;
            one.extra_area = candidate.extra_area;
            one.flush = false;
            one.exists = true;
        }
    }
    one.done = true;
    balanced_inters.markAsDone(i, j);
}

//! Recursively repeat the search for the one sided chain with minimal extra area to shrink the boundaries of bigger distances
/*!
 * @param i     Fixed side of the ngon
 * @param j1    Lower boundary of search intervall
 * @param j2    Upper boundary of search intervall
 */
void Chains::singleSideImpl(int i, int j1, int j2)
{
    const int n = (int)ngon.size();
    if((j1 + 1) %n == j2)
    {
        return;
    }

    int mid = (j1 < j2 ? ((j1 + j2) / 2) : ((j1 + n + j2) / 2)) % n;
    int l = single_sides[i][j1].side < 0 ? (j1 + 1) %n : single_sides[i][j1].side;
    int r = single_sides[i][j2].side < 0 ? (j2 - 1 + n) %n : single_sides[i][j2].side;

    findSingleE(i, mid, l, r);
    singleSideImpl(i, j1, mid);
    singleSideImpl(i, mid, j2);

    return;
}

//! For a combination of two fixed sides of the ngon test all sides between left and right boundary to find the middle side of the h-sided chain with minimal extra area. This is the version for h = 1.
/*!
 * @param i     Index of first fixed side
 * @param j     Index of second fixed side
 * @param l     Index of left boundary
 * @param r     Index of right boundary
 */
void Chains::findMiddleE1(int i, int j, int l, int r)
{
    const size_t n = ngon.size();
    Segment& one = middle_sides[1][i][j];
    if (one.done)
        return;

    double min_area = std::numeric_limits<double>::max();
    for (int e = l; e != r + 1 && e != j; e = (e + 1) %n)
    {
        const FlushIntersect& before = intersections.lineIntersect(i, e);
        if(!before.intersection.position)
            continue;
        const FlushIntersect& after = intersections.lineIntersect(e, j);
        if(!after.intersection.position)
            continue;

        double tmp_area = before.extra_area + after.extra_area;
        if(tmp_area < min_area)
        {
            min_area = tmp_area;
            one.side = e;
            one.extra_area = tmp_area;
            one.exists = true;
            one.flush = true;
        }
    }
    one.done = true;
}

//! Recursively repeat the search for the middle side of the h-sided chain with minimal extra area to shrink the boundaries of bigger distances. This is the version for h = 1.
/*!
 * @param i     Fixed side of the ngon
 * @param j1    Lower boundary of search intervall
 * @param j2    Upper boundary of search intervall
 */
void Chains::middleSideImpl1(int i, int j1, int j2)
{
    const int n = (int)ngon.size();
    if((j1 + 1) %n == j2)
    {
        return;
    }

    int mid = (j1 < j2 ? ((j1 + j2) / 2) : ((j1 + n + j2) / 2)) % n;
    int l = middle_sides[1][i][j1].side < 0 ? (j1 + 1) %n : middle_sides[1][i][j1].side;
    int r = middle_sides[1][i][j2].side < 0 ? (j2 - 1 + n) %n : middle_sides[1][i][j2].side;

    findMiddleE1(i, mid, l, r);
    middleSideImpl1(i, j1, mid);
    middleSideImpl1(i, mid, j2);

    return;
}

//! For a combination of two fixed sides of the ngon test all sides between left and right boundary to find the middle side of the h-sided chain with minimal extra area. This is the version for h > 1.
/*!
 * @param h     Length of the h-sided chain
 * @param i     Index of first fixed side
 * @param j     Index of second fixed side
 * @param l     Index of left boundary
 * @param r     Index of right boundary
 */
void Chains::findMiddleE(int h, int i, int j, int l, int r)
{
    const int n = (int)ngon.size();
    Segment& one = middle_sides[h][i][j];
    if (one.done)
        return;

    const int dist = (i <= j ? (j - i) : (j + n - i));
    const int h_floor = (h - 1) / 2;
    const int h_ceil = h - 1 - h_floor;

    CV_Assert(dist != 0);

    if(dist - 1 < h)
    {
        one.done = true;
        return;
    }
    if(dist - 1 == h)
    {
        one.side = (i + h_floor + 1) % n;
        one.extra_area = 0.0;
        one.exists = true;
        one.flush = true;
        one.done = true;
        return;
    }

    double min_area = std::numeric_limits<double>::max();
    for (int e = l; e != r + 1 && e != j; e = (e + 1) %n)
    {
        const Segment& before = middle_sides[h_floor][i][e];
        if (before.extra_area == std::numeric_limits<double>::max())
            continue;
        const Segment& after = middle_sides[h_ceil][e][j];
        if(after.extra_area == std::numeric_limits<double>::max())
            continue;

        double tmp_area = before.extra_area + after.extra_area;
        if(tmp_area < min_area)
        {
            min_area = tmp_area;
            one.side = e;
            one.extra_area = tmp_area;
            one.exists = true;
            one.flush = true;
        }
    }
    one.done = true;
}

//! Recursively repeat the search for the middle side of the h-sided chain with minimal extra area to shrink the boundaries of bigger distances. This is the version for h > 1.
/*!
 * @param h     Length of the h-sided chain
 * @param i     Fixed side of the ngon
 * @param j1    Lower boundary of search intervall
 * @param j2    Upper boundary of search intervall
 */
void Chains::middleSideImpl(int h, int i, int j1, int j2)
{
    const int n = (int)ngon.size();
    if((j1 + 1) %n == j2)
    {
        return;
    }

    int mid = (j1 < j2 ? ((j1 + j2) / 2) : ((j1 + n + j2) / 2)) % n;
    int l = middle_sides[h][i][j1].side < 0 ? (j1 + 1) %n : middle_sides[h][i][j1].side;
    int r = middle_sides[h][i][j2].side < 0 ? (j2 - 1 + n) %n : middle_sides[h][i][j2].side;

    findMiddleE(h, i, mid, l, r);
    middleSideImpl(h, i, j1, mid);
    middleSideImpl(h, i, mid, j2);

    return;
}

//! Calculate the relevant chain lengths. Starting with a maximum chain length of h, down to a chain length of 1, only chains with half the length of the chain before are needed.
/*!
 * @param h     Length of the longest chain (h = k - 3)
 */
std::set<int> Chains::relevantChainLengths(int h)
{
    if(h <= 1)
        return {h};

    std::set<int> hs = {h};
    const int h_floor = (h - 1) / 2;
    const int h_ceil = h - 1 - h_floor;
    std::set<int> h1 = relevantChainLengths(h_floor);
    for(const int& hNew : h1)
        hs.insert(hNew);
    if (h_ceil != h_floor)
    {
        std::set<int> h2 = relevantChainLengths(h_ceil);
        for(const int& hNew : h2)
            hs.insert(hNew);
    }
    return hs;
}

//! Recursively calculate all the onesided chains for each combination of polygon sides.
/*!
 */
void Chains::calcOneSidedChains()
{
    const int n = (int)ngon.size();
    for(int i = 0; i < n; i++)
    {
        int j1 = (i + 2) %n, j2 = (i - 2 + n) %n;

        findSingleE(i, j1, (i + 1) %n, (j1 - 1 + n) %n);
        findSingleE(i, j2, (i + 1) %n, (j2 - 1 + n) %n);
        singleSideImpl(i, j1, j2);
    }
}

//! Recursively calculate the middle sides of the h-sided chains for all combinations of polygon sides.
/*!
 * @param h     Length of the chains
 */
void Chains::calcMiddleChains(int h)
{
    const int n = (int)ngon.size();
    if (h == 0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Segment& one = middle_sides[h][i][j];
                const FlushIntersect itrs = intersections.lineIntersect(i, j);

                one.side = -1;
                one.extra_area = itrs.extra_area;
                one.exists = false;
                one.flush = false;
                one.done = true;
            }
        }
        return;
    }
    if (h == 1)
    {
        for (int i = 0; i < n; i++)
        {
            int j1 = (i + 2) %n, j2 = (i - 2 + n) %n;

            findMiddleE1(i, j1, (i + 1) %n, (j1 - 1 + n) %n);
            findMiddleE1(i, j2, (i + 1) %n, (j2 - 1 + n) %n);
            middleSideImpl1(i, j1, j2);
        }
        return;
    }

    for (int i = 0; i < n; i++)
    {
        int j1 = (i + 2) %n, j2 = (i - 2 + n) %n;

        findMiddleE(h, i, j1, (i + 1) %n, (j1 - 1 + n) %n);
        findMiddleE(h, i, j2, (i + 1) %n, (j2 - 1 + n) %n);
        middleSideImpl(h, i, j1, j2);
    }
}

//! Find the i and j with the smallest extra area.
/*!
 * @param n             Number of sides of the input polygon
 * @param k             Number of sides of the output polygon
 */
Minimum Chains::minimumArea(int n, int k)
{
    Minimum min{};
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(!single_sides[i][j].exists || !middle_sides[k - 3][j][i].exists)
                continue;

            double tmp_area =
            single_sides[i][j].extra_area + middle_sides[k - 3][j][i].extra_area;
            if(tmp_area < min.area)
            {
                min.area = tmp_area;
                min.i = i;
                min.j = j;
            }
        }
    }
    return min;
}

//! Reconstruct the h-sided chain based on the two sides that give the smalles extra area.
/*!
 * @param h             Length of the k - 3 sided chain.
 * @param i             First index from minimumArea
 * @param j             Second index from minimumArea
 */
std::vector<Side> Chains::reconstructHSidedChain(int h, int i, int j)
{
    CV_Assert(h != 0);

    if(h == 1)
    {
        return std::vector<Side>{{middle_sides[h][i][j].side, true}};
    }

    std::vector<Side> before, after;
    const int h_floor = (h - 1) / 2;
    const int h_ceil = h - 1 - h_floor;
    if(h_floor > 0)
        before = reconstructHSidedChain(h_floor, i, middle_sides[h][i][j].side);
    if(h_ceil > 0)
        after = reconstructHSidedChain(h_ceil, middle_sides[h][i][j].side, j);

    std::vector<Side> sides{{middle_sides[h][i][j].side, true}};
    sides.insert(sides.end(), before.begin(), before.end());
    sides.insert(sides.end(), after.begin(), after.end());
    return sides;
}

//! Get the k sides that build the kgon based on the two sides that give the smalles extra area.
/*!
 * @param k             Number of sides of the output polygon
 * @param i             First index from minimumArea
 * @param j             Second index from minimumArea
 */
std::vector<Side> Chains::findKSides(int k, int i, int j)
{
    std::vector<Side> sides;
    sides.push_back({i, true});
    sides.push_back({j, true});

    if(single_sides[i][j].flush)
        sides.push_back({single_sides[i][j].side, true});
    else
        sides.push_back({single_sides[i][j].side, false});

    std::vector<Side> flush_chain = reconstructHSidedChain(k - 3, j, i);
    sides.insert(sides.end(), flush_chain.begin(), flush_chain.end());
    std::sort(sides.begin(), sides.end(),
              [](const Side& lhs, const Side& rhs) { return lhs.side < rhs.side; });

    return sides;
}

//! Calculate the k vertices of the kgon from its k sides.
/*!
 * @param sides         Sides of the output polygon
 */
std::vector<cv::Point2f> Chains::findKVertices(std::vector<Side>& sides)
{
    const int k = (int)sides.size();
    std::vector<cv::Point2f> vertices(k);

    for(int u = 0; u < k; u++)
    {
        const int next = (u + 1) % k;
        if(sides[u].flush && sides[next].flush)
        {
            vertices[u] = intersections.lineIntersect(sides[u].side,
                                                      sides[next].side).intersection.point;
        }
        else if(sides[u].flush && !sides[next].flush)
        {
            vertices[u] = balanced_inters[sides[u].side][sides[(u + 2) % k].side].pi;
        }
        else if(!sides[u].flush && sides[next].flush)
        {
            vertices[u] = balanced_inters[sides[(u - 1 + k) %k].side][sides[next].side].pj;
        }
        else
        {
            CV_Error(cv::Error::StsInternal, "findKVertices logic error!");
        }
    }
    return vertices;
}

} //namespace


///////////////////////// Helper function definitions /////////////////////////

namespace minEnclosingConvexPolygon {

//! Find the minimum enclosing convex polygon and its area by calculating the one-sided and h-sided chains for all possible combinations of sides and then using the one that gives the smallest extra area.
/*!
 * @param ngon           The polygon representing the convex hull of the points
 * @param k              Number of vertices of the output polygon
 * @param kgon           Minimum area convex k-gon enclosing the given polygon
 * @param area           Area of kgon
 */
static double findMinAreaPolygon(const std::vector<cv::Point2f> &ngon,
                                 std::vector<cv::Point2f> &minPolygon, int k)
{
    const int n = (int)ngon.size();

    Chains chains(ngon, k);
    chains.calcOneSidedChains();
    std::set<int> hs = chains.relevantChainLengths(k - 3);
    for(auto& h : hs)
    {
        chains.calcMiddleChains(h);
    }

    Kgon kgon{};

    const Minimum min = chains.minimumArea(n, k);

    kgon.i = min.i;
    kgon.j = min.j;
    kgon.extra_area = min.area;
    kgon.sides = chains.findKSides(k, min.i, min.j);
    kgon.vertices = chains.findKVertices(kgon.sides);

    minPolygon = kgon.vertices;
    return cv::contourArea(minPolygon);
}

} //namespace


//////////////////////////////// Main function ////////////////////////////////

//! Find the minimum enclosing convex polygon for the given set of points and return its area
/*!
 * @param points        Set of points
 * @param polygon       Minimum area convex polygon enclosing the given set of points
 * @param k             Number of vertices of the output polygon
 */

double cv::minEnclosingConvexPolygon(cv::InputArray points, cv::OutputArray polygon, int k)
{
    int n = (int)points.getMat().checkVector(2);

    CV_Assert(!points.empty() && n >= k);
    CV_CheckGE(n, 3, "ngon must have 3 or more different points enclosing an area");
    CV_CheckGE(k, 3, "k must be 3 or higher");

    std::vector<cv::Point2f> ngon, kgon;
    cv::convexHull(points, ngon, true);

    n = (int)ngon.size();

    if (n < k)
    {
        CV_LOG_WARNING(NULL, "convex hull of size " << n << " must have equal or more different points than k = " << k);
        polygon.release();
        return 0.;
    }

    if (n == k)
    {
        cv::Mat(ngon).copyTo(polygon);
        return cv::contourArea(polygon);
    }

    if (cv::contourArea(ngon) < EPSILON)
    {
        CV_LOG_WARNING(NULL, "Singular input poligon");
        polygon.release();
        return 0.;
    }

    double area = minEnclosingConvexPolygon::findMinAreaPolygon(ngon, kgon, k);
    cv::Mat(kgon).copyTo(polygon);

    return area;
}
