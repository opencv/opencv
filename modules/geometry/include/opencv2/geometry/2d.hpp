// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_2D_HPP
#define OPENCV_2D_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace cv {

//! @addtogroup geometry_shape
//! @{

//! types of intersection between rectangles
enum RectanglesIntersectTypes {
    INTERSECT_NONE = 0, //!< No intersection
    INTERSECT_PARTIAL  = 1, //!< There is a partial intersection
    INTERSECT_FULL  = 2 //!< One of the rectangle is fully enclosed in the other
};

//! Distance types for Distance Transform and M-estimators
//! @see distanceTransform, fitLine
enum DistanceTypes {
    DIST_USER    = -1,  //!< User defined distance
    DIST_L1      = 1,   //!< distance = |x1-x2| + |y1-y2|
    DIST_L2      = 2,   //!< the simple euclidean distance
    DIST_C       = 3,   //!< distance = max(|x1-x2|,|y1-y2|)
    DIST_L12     = 4,   //!< L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
    DIST_FAIR    = 5,   //!< distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
    DIST_WELSCH  = 6,   //!< distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
    DIST_HUBER   = 7    //!< distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
};

//! @addtogroup geometry_subdiv2d
//! @{

class CV_EXPORTS_W Subdiv2D
{
public:
    /** Subdiv2D point location cases */
    enum { PTLOC_ERROR        = -2, //!< Point location error
        PTLOC_OUTSIDE_RECT = -1, //!< Point outside the subdivision bounding rect
        PTLOC_INSIDE       = 0, //!< Point inside some facet
        PTLOC_VERTEX       = 1, //!< Point coincides with one of the subdivision vertices
        PTLOC_ON_EDGE      = 2  //!< Point on some edge
    };

    /** Subdiv2D edge type navigation (see: getEdge()) */
    enum { NEXT_AROUND_ORG   = 0x00,
        NEXT_AROUND_DST   = 0x22,
        PREV_AROUND_ORG   = 0x11,
        PREV_AROUND_DST   = 0x33,
        NEXT_AROUND_LEFT  = 0x13,
        NEXT_AROUND_RIGHT = 0x31,
        PREV_AROUND_LEFT  = 0x20,
        PREV_AROUND_RIGHT = 0x02
    };

    /** creates an empty Subdiv2D object.
     *    To create a new empty Delaunay subdivision you need to use the #initDelaunay function.
     */
    CV_WRAP Subdiv2D();

    /** @overload
     *
     *    @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
     *
     *    The function creates an empty Delaunay subdivision where 2D points can be added using the function
     *    insert() . All of the points to be added must be within the specified rectangle, otherwise a runtime
     *    error is raised.
     */
    CV_WRAP Subdiv2D(Rect rect);

    /** @overload */
    CV_WRAP Subdiv2D(Rect2f rect2f);

    /** @overload
     *
     *    @brief Creates a new empty Delaunay subdivision
     *
     *    @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
     *
     */
    CV_WRAP void initDelaunay(Rect rect);

    /** @overload
     *
     *    @brief Creates a new empty Delaunay subdivision
     *
     *    @param rect Rectangle that includes all of the 2d points that are to be added to the subdivision.
     *
     */
    CV_WRAP_AS(initDelaunay2f) CV_WRAP void initDelaunay(Rect2f rect);

    /** @brief Insert a single point into a Delaunay triangulation.
     *
     *    @param pt Point to insert.
     *
     *    The function inserts a single point into a subdivision and modifies the subdivision topology
     *    appropriately. If a point with the same coordinates exists already, no new point is added.
     *    @returns the ID of the point.
     *
     *    @note If the point is outside of the triangulation specified rect a runtime error is raised.
     */
    CV_WRAP int insert(Point2f pt);

    /** @brief Insert multiple points into a Delaunay triangulation.
     *
     *    @param ptvec Points to insert.
     *
     *    The function inserts a vector of points into a subdivision and modifies the subdivision topology
     *    appropriately.
     */
    CV_WRAP void insert(const std::vector<Point2f>& ptvec);

    /** @brief Returns the location of a point within a Delaunay triangulation.
     *
     *    @param pt Point to locate.
     *    @param edge Output edge that the point belongs to or is located to the right of it.
     *    @param vertex Optional output vertex the input point coincides with.
     *
     *    The function locates the input point within the subdivision and gives one of the triangle edges
     *    or vertices.
     *
     *    @returns an integer which specify one of the following five cases for point location:
     *    -  The point falls into some facet. The function returns #PTLOC_INSIDE and edge will contain one of
     *       edges of the facet.
     *    -  The point falls onto the edge. The function returns #PTLOC_ON_EDGE and edge will contain this edge.
     *    -  The point coincides with one of the subdivision vertices. The function returns #PTLOC_VERTEX and
     *       vertex will contain a pointer to the vertex.
     *    -  The point is outside the subdivision reference rectangle. The function returns #PTLOC_OUTSIDE_RECT
     *       and no pointers are filled.
     *    -  One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error
     *       processing mode is selected, #PTLOC_ERROR is returned.
     */
    CV_WRAP int locate(Point2f pt, CV_OUT int& edge, CV_OUT int& vertex);

    /** @brief Finds the subdivision vertex closest to the given point.
     *
     *    @param pt Input point.
     *    @param nearestPt Output subdivision vertex point.
     *
     *    The function is another function that locates the input point within the subdivision. It finds the
     *    subdivision vertex that is the closest to the input point. It is not necessarily one of vertices
     *    of the facet containing the input point, though the facet (located using locate() ) is used as a
     *    starting point.
     *
     *    @returns vertex ID.
     */
    CV_WRAP int findNearest(Point2f pt, CV_OUT Point2f* nearestPt = 0);

    /** @brief Returns a list of all edges.
     *
     *    @param edgeList Output vector.
     *
     *    The function gives each edge as a 4 numbers vector, where each two are one of the edge
     *    vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3].
     */
    CV_WRAP void getEdgeList(CV_OUT std::vector<Vec4f>& edgeList) const;

    /** @brief Returns a list of the leading edge ID connected to each triangle.
     *
     *    @param leadingEdgeList Output vector.
     *
     *    The function gives one edge ID for each triangle.
     */
    CV_WRAP void getLeadingEdgeList(CV_OUT std::vector<int>& leadingEdgeList) const;

    /** @brief Returns a list of all triangles.
     *
     *    @param triangleList Output vector.
     *
     *    The function gives each triangle as a 6 numbers vector, where each two are one of the triangle
     *    vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
     */
    CV_WRAP void getTriangleList(CV_OUT std::vector<Vec6f>& triangleList) const;

    /** @brief Returns a list of all Voronoi facets.
     *
     *    @param idx Vector of vertices IDs to consider. For all vertices you can pass empty vector.
     *    @param facetList Output vector of the Voronoi facets.
     *    @param facetCenters Output vector of the Voronoi facets center points.
     *
     */
    CV_WRAP void getVoronoiFacetList(const std::vector<int>& idx, CV_OUT std::vector<std::vector<Point2f> >& facetList,
                                     CV_OUT std::vector<Point2f>& facetCenters);

    /** @brief Returns vertex location from vertex ID.
     *
     *    @param vertex vertex ID.
     *    @param firstEdge Optional. The first edge ID which is connected to the vertex.
     *    @returns vertex (x,y)
     *
     */
    CV_WRAP Point2f getVertex(int vertex, CV_OUT int* firstEdge = 0) const;

    /** @brief Returns one of the edges related to the given edge.
     *
     *    @param edge Subdivision edge ID.
     *    @param nextEdgeType Parameter specifying which of the related edges to return.
     *    The following values are possible:
     *    -   NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge)
     *    -   NEXT_AROUND_DST next around the edge vertex ( eDnext )
     *    -   PREV_AROUND_ORG previous around the edge origin (reversed eRnext )
     *    -   PREV_AROUND_DST previous around the edge destination (reversed eLnext )
     *    -   NEXT_AROUND_LEFT next around the left facet ( eLnext )
     *    -   NEXT_AROUND_RIGHT next around the right facet ( eRnext )
     *    -   PREV_AROUND_LEFT previous around the left facet (reversed eOnext )
     *    -   PREV_AROUND_RIGHT previous around the right facet (reversed eDnext )
     *
     *    ![sample output](pics/quadedge.png)
     *
     *    @returns edge ID related to the input edge.
     */
    CV_WRAP int getEdge( int edge, int nextEdgeType ) const;

    /** @brief Returns next edge around the edge origin.
     *
     *    @param edge Subdivision edge ID.
     *
     *    @returns an integer which is next edge ID around the edge origin: eOnext on the
     *    picture above if e is the input edge).
     */
    CV_WRAP int nextEdge(int edge) const;

    /** @brief Returns another edge of the same quad-edge.
     *
     *    @param edge Subdivision edge ID.
     *    @param rotate Parameter specifying which of the edges of the same quad-edge as the input
     *    one to return. The following values are possible:
     *    -   0 - the input edge ( e on the picture below if e is the input edge)
     *    -   1 - the rotated edge ( eRot )
     *    -   2 - the reversed edge (reversed e (in green))
     *    -   3 - the reversed rotated edge (reversed eRot (in green))
     *
     *    @returns one of the edges ID of the same quad-edge as the input edge.
     */
    CV_WRAP int rotateEdge(int edge, int rotate) const;
    CV_WRAP int symEdge(int edge) const;

    /** @brief Returns the edge origin.
     *
     *    @param edge Subdivision edge ID.
     *    @param orgpt Output vertex location.
     *
     *    @returns vertex ID.
     */
    CV_WRAP int edgeOrg(int edge, CV_OUT Point2f* orgpt = 0) const;

    /** @brief Returns the edge destination.
     *
     *    @param edge Subdivision edge ID.
     *    @param dstpt Output vertex location.
     *
     *    @returns vertex ID.
     */
    CV_WRAP int edgeDst(int edge, CV_OUT Point2f* dstpt = 0) const;

protected:
    int newEdge();
    void deleteEdge(int edge);
    int newPoint(Point2f pt, bool isvirtual, int firstEdge = 0);
    void deletePoint(int vtx);
    void setEdgePoints( int edge, int orgPt, int dstPt );
    void splice( int edgeA, int edgeB );
    int connectEdges( int edgeA, int edgeB );
    void swapEdges( int edge );
    int isRightOf(Point2f pt, int edge) const;
    void calcVoronoi();
    void clearVoronoi();
    void checkSubdiv() const;

    struct CV_EXPORTS Vertex
    {
        Vertex();
        Vertex(Point2f pt, bool isvirtual, int firstEdge=0);
        bool isvirtual() const;
        bool isfree() const;

        int firstEdge;
        int type;
        Point2f pt;
    };

    struct CV_EXPORTS QuadEdge
    {
        QuadEdge();
        QuadEdge(int edgeidx);
        bool isfree() const;

        int next[4];
        int pt[4];
    };

    //! All of the vertices
    std::vector<Vertex> vtx;
    //! All of the edges
    std::vector<QuadEdge> qedges;
    int freeQEdge;
    int freePoint;
    bool validGeometry;

    int recentEdge;
    //! Top left corner of the bounding rect
    Point2f topLeft;
    //! Bottom right corner of the bounding rect
    Point2f bottomRight;
};

//! @} geometry_subdiv2d

/** @example samples/python/snippets/squares.py
 A n example using approxPolyDP function in python.        *
 */

/** @brief Approximates a polygonal curve(s) with the specified precision.
 *
 T he function cv::approxPolyDP approximates a curve or a p*olygon with another curve/polygon with less
 vertices so that the distance between them is less or equal to the specified precision. It uses the
 Douglas-Peucker algorithm <https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm>

 @param curve Input vector of a 2D point stored in std::vector or Mat
 @param approxCurve Result of the approximation. The type should match the type of the input curve.
 @param epsilon Parameter specifying the approximation accuracy. This is the maximum distance
 between the original curve and its approximation.
 @param closed If true, the approximated curve is closed (its first and last vertices are
 connected). Otherwise, it is not closed.
 */
CV_EXPORTS_W void approxPolyDP( InputArray curve,
                                OutputArray approxCurve,
                                double epsilon, bool closed );

/** @brief Approximates a polygon with a convex hull with a specified accuracy and number of sides.
 *
 T he cv::approxPolyN function approximates a polygon with *a convex hull
 so that the difference between the contour area of the original contour and the new polygon is minimal.
 It uses a greedy algorithm for contracting two vertices into one in such a way that the additional area is minimal.
 Straight lines formed by each edge of the convex contour are drawn and the areas of the resulting triangles are considered.
 Each vertex will lie either on the original contour or outside it.

 The algorithm based on the paper @cite LowIlie2003 .

 @param curve Input vector of a 2D points stored in std::vector or Mat, points must be float or integer.
 @param approxCurve Result of the approximation. The type is vector of a 2D point (Point2f or Point) in std::vector or Mat.
 @param nsides The parameter defines the number of sides of the result polygon.
 @param epsilon_percentage defines the percentage of the maximum of additional area.
 If it equals -1, it is not used. Otherwise algorithm stops if additional area is greater than contourArea(_curve) * percentage.
 If additional area exceeds the limit, algorithm returns as many vertices as there were at the moment the limit was exceeded.
 @param ensure_convex If it is true, algorithm creates a convex hull of input contour. Otherwise input vector should be convex.
 */
CV_EXPORTS_W void approxPolyN(InputArray curve, OutputArray approxCurve,
                              int nsides, float epsilon_percentage = -1.0,
                              bool ensure_convex = true);

/** @brief Finds a rotated rectangle of the minimum area enclosing the input 2D point set.
 *
 * The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a
 * specified point set. The angle of rotation represents the angle between the line connecting the starting
 * and ending points (based on the clockwise order with greatest index for the corner with greatest \f$y\f$)
 * and the horizontal axis. This angle always falls between \f$[-90, 0)\f$ because, if the object
 * rotates more than a rect angle, the next edge is used to measure the angle. The starting and ending points change
 * as the object rotates.Developer should keep in mind that the returned RotatedRect can contain negative
 * indices when data is close to the containing Mat element boundary.
 *
 * @param points Input vector of 2D points, stored in std::vector\<\> or Mat
 */
CV_EXPORTS_W RotatedRect minAreaRect( InputArray points );

/** @brief Finds the four vertices of a rotated rect. Useful to draw the rotated rectangle.
 *
 * The function finds the four vertices of a rotated rectangle. The four vertices are returned
 * in clockwise order starting from the point with greatest \f$y\f$. If two points have the
 * same \f$y\f$ coordinate the rightmost is the starting point. This function is useful to draw the
 * rectangle. In C++, instead of using this function, you can directly use RotatedRect::points method. Please
 * visit the @ref tutorial_bounding_rotated_ellipses "tutorial on Creating Bounding rotated boxes and ellipses
 * for contours" for more information.
 *
 * @param box The input rotated rectangle. It may be the output of @ref minAreaRect.
 * @param points The output array of four vertices of rectangles.
 */
CV_EXPORTS_W void boxPoints(RotatedRect box, OutputArray points);

/** @brief Finds a circle of the minimum area enclosing a 2D point set.
 *
 * The function finds the minimal enclosing circle of a 2D point set using an iterative algorithm.
 *
 * @param points Input vector of 2D points, stored in std::vector\<\> or Mat
 * @param center Output center of the circle.
 * @param radius Output radius of the circle.
 */
CV_EXPORTS_W void minEnclosingCircle( InputArray points,
                                      CV_OUT Point2f& center, CV_OUT float& radius );


/** @brief Finds a triangle of minimum area enclosing a 2D point set and returns its area.
 *
 * The function finds a triangle of minimum area enclosing the given set of 2D points and returns its
 * area. The output for a given 2D point set is shown in the image below. 2D points are depicted in
 *red* and the enclosing triangle in *yellow*.
 *
 * ![Sample output of the minimum enclosing triangle function](pics/minenclosingtriangle.png)
 *
 * The implementation of the algorithm is based on O'Rourke's @cite ORourke86 and Klee and Laskowski's
 * @cite KleeLaskowski85 papers. O'Rourke provides a \f$\theta(n)\f$ algorithm for finding the minimal
 * enclosing triangle of a 2D convex polygon with n vertices. Since the #minEnclosingTriangle function
 * takes a 2D point set as input an additional preprocessing step of computing the convex hull of the
 * 2D point set is required. The complexity of the #convexHull function is \f$O(n log(n))\f$ which is higher
 * than \f$\theta(n)\f$. Thus the overall complexity of the function is \f$O(n log(n))\f$.
 *
 * @param points Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector\<\> or Mat
 * @param triangle Output vector of three 2D points defining the vertices of the triangle. The depth
 * of the OutputArray must be CV_32F.
 */
CV_EXPORTS_W double minEnclosingTriangle( InputArray points, CV_OUT OutputArray triangle );


/**
 * @brief Finds a convex polygon of minimum area enclosing a 2D point set and returns its area.
 *
 * This function takes a given set of 2D points and finds the enclosing polygon with k vertices and minimal
 * area. It takes the set of points and the parameter k as input and returns the area of the minimal
 * enclosing polygon.
 *
 * The Implementation is based on a paper by Aggarwal, Chang and Yap @cite Aggarwal1985. They
 * provide a \f$\theta(n²log(n)log(k))\f$ algorithm for finding the minimal convex polygon with k
 * vertices enclosing a 2D convex polygon with n vertices (k < n). Since the #minEnclosingConvexPolygon
 * function takes a 2D point set as input, an additional preprocessing step of computing the convex hull
 * of the 2D point set is required. The complexity of the #convexHull function is \f$O(n log(n))\f$ which
 * is lower than \f$\theta(n²log(n)log(k))\f$. Thus the overall complexity of the function is
 * \f$O(n²log(n)log(k))\f$.
 *
 * @param points   Input vector of 2D points, stored in std::vector\<\> or Mat
 * @param polygon  Output vector of 2D points defining the vertices of the enclosing polygon
 * @param k        Number of vertices of the output polygon
 */

CV_EXPORTS_W double minEnclosingConvexPolygon ( InputArray points, OutputArray polygon, int k );

/** @brief Calculates all of the moments up to the third order of a polygon or rasterized shape.
 *
 * The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The
 * results are returned in the structure cv::Moments.
 *
 * @param array Single channel raster image (CV_8U, CV_16U, CV_16S, CV_32F, CV_64F) or an array (
 * \f$1 \times N\f$ or \f$N \times 1\f$ ) of 2D points (Point or Point2f).
 * @param binaryImage If it is true, all non-zero image pixels are treated as 1's. The parameter is
 * used for images only.
 * @returns moments.
 *
 * @note Only applicable to contour moments calculations from Python bindings: Note that the numpy
 * type for the input array should be either np.int32 or np.float32.
 *
 * @note For contour-based moments, the zeroth-order moment \c m00 represents
 * the contour area.
 *
 * If the input contour is degenerate (for example, a single point or all points
 * are collinear), the area is zero and therefore \c m00 == 0.
 *
 * In this case, the centroid coordinates (\c m10/m00, \c m01/m00) are undefined
 * and must be handled explicitly by the caller.
 *
 * A common workaround is to compute the center using cv::boundingRect() or by
 * averaging the input points.
 *
 * @sa  contourArea, arcLength
 */
CV_EXPORTS_W Moments moments( InputArray array, bool binaryImage = false );

/** @brief Calculates seven Hu invariants.
 *
 * The function calculates seven Hu invariants (introduced in @cite Hu62; see also
 * <https://en.wikipedia.org/wiki/Image_moment>) defined as:
 *
 * \f[\begin{array}{l} hu[0]= \eta _{20}+ \eta _{02} \\ hu[1]=( \eta _{20}- \eta _{02})^{2}+4 \eta _{11}^{2} \\ hu[2]=( \eta _{30}-3 \eta _{12})^{2}+ (3 \eta _{21}- \eta _{03})^{2} \\ hu[3]=( \eta _{30}+ \eta _{12})^{2}+ ( \eta _{21}+ \eta _{03})^{2} \\ hu[4]=( \eta _{30}-3 \eta _{12})( \eta _{30}+ \eta _{12})[( \eta _{30}+ \eta _{12})^{2}-3( \eta _{21}+ \eta _{03})^{2}]+(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ hu[5]=( \eta _{20}- \eta _{02})[( \eta _{30}+ \eta _{12})^{2}- ( \eta _{21}+ \eta _{03})^{2}]+4 \eta _{11}( \eta _{30}+ \eta _{12})( \eta _{21}+ \eta _{03}) \\ hu[6]=(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}]-( \eta _{30}-3 \eta _{12})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ \end{array}\f]
 *
 * where \f$\eta_{ji}\f$ stands for \f$\texttt{Moments::nu}_{ji}\f$ .
 *
 * These values are proved to be invariants to the image scale, rotation, and reflection except the
 * seventh one, whose sign is changed by reflection. This invariance is proved with the assumption of
 * infinite image resolution. In case of raster images, the computed Hu invariants for the original and
 * transformed images are a bit different.
 *
 * @param moments Input moments computed with moments .
 * @param hu Output Hu invariants.
 *
 * @sa matchShapes
 */
CV_EXPORTS void HuMoments( const Moments& moments, double hu[7] );

/** @overload */
CV_EXPORTS_W void HuMoments( const Moments& m, OutputArray hu );

/** @brief Compares two shapes.
 *
 * The function compares two shapes. All three implemented methods use the Hu invariants (see #HuMoments)
 *
 * @param contour1 First contour or grayscale image.
 * @param contour2 Second contour or grayscale image.
 * @param method Comparison method, see #ShapeMatchModes
 * @param parameter Method-specific parameter (not supported now).
 */
CV_EXPORTS_W double matchShapes( InputArray contour1, InputArray contour2,
                                 int method, double parameter );

/** @example samples/cpp/geometry.cpp
 * An example program illustrates the use of cv::convexHull, cv::fitEllipse, cv::minEnclosingTriangle, cv::minEnclosingCircle and cv::minAreaRect.
 */

/** @brief Finds the convex hull of a point set.
 *
 * The function cv::convexHull finds the convex hull of a 2D point set using the Sklansky's algorithm @cite Sklansky82
 * that has *O(N logN)* complexity in the current implementation.
 *
 * @param points Input 2D point set, stored in std::vector or Mat.
 * @param hull Output convex hull. It is either an integer vector of indices or vector of points. In
 * the first case, the hull elements are 0-based indices of the convex hull points in the original
 * array (since the set of convex hull points is a subset of the original point set). In the second
 * case, hull elements are the convex hull points themselves.
 * @param clockwise Orientation flag. If it is true, the output convex hull is oriented clockwise.
 * Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing
 * to the right, and its Y axis pointing upwards.
 * @param returnPoints Operation flag. In case of a matrix, when the flag is true, the function
 * returns convex hull points. Otherwise, it returns indices of the convex hull points. When the
 * output array is std::vector, the flag is ignored, and the output depends on the type of the
 * vector: std::vector\<int\> implies returnPoints=false, std::vector\<Point\> implies
 * returnPoints=true.
 *
 * @note `points` and `hull` should be different arrays, inplace processing isn't supported.
 *
 * Check @ref tutorial_hull "the corresponding tutorial" for more details.
 *
 * useful links:
 *
 * https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
 */
CV_EXPORTS_W void convexHull( InputArray points, OutputArray hull,
                              bool clockwise = false, bool returnPoints = true );

/** @brief Finds the convexity defects of a contour.
 *
 * The figure below displays convexity defects of a hand contour:
 *
 * ![image](pics/defects.png)
 *
 * @param contour Input contour.
 * @param convexhull Convex hull obtained using convexHull that should contain indices of the contour
 * points that make the hull.
 * @param convexityDefects The output vector of convexity defects. In C++ and the new Python/Java
 * interface each convexity defect is represented as 4-element integer vector (a.k.a. #Vec4i):
 * (start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices
 * in the original contour of the convexity defect beginning, end and the farthest point, and
 * fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the
 * farthest contour point and the hull. That is, to get the floating-point value of the depth will be
 * fixpt_depth/256.0.
 */
CV_EXPORTS_W void convexityDefects( InputArray contour, InputArray convexhull, OutputArray convexityDefects );

/** @brief Tests a contour convexity.
 *
 * The function tests whether the input contour is convex or not. The contour must be simple, that is,
 * without self-intersections. Otherwise, the function output is undefined.
 *
 * @param contour Input vector of 2D points, stored in std::vector\<\> or Mat
 */
CV_EXPORTS_W bool isContourConvex( InputArray contour );

/** @example samples/cpp/snippets/intersectExample.cpp
 * Examples of how intersectConvexConvex works
 */

/** @brief Finds intersection of two convex polygons
 *
 * @param p1 First polygon
 * @param p2 Second polygon
 * @param p12 Output polygon describing the intersecting area
 * @param handleNested When true, an intersection is found if one of the polygons is fully enclosed in the other.
 * When false, no intersection is found. If the polygons share a side or the vertex of one polygon lies on an edge
 * of the other, they are not considered nested and an intersection will be found regardless of the value of handleNested.
 *
 * @returns Area of intersecting polygon. May be negative, if algorithm has not converged, e.g. non-convex input.
 *
 * @note intersectConvexConvex doesn't confirm that both polygons are convex and will return invalid results if they aren't.
 */
CV_EXPORTS_W float intersectConvexConvex( InputArray p1, InputArray p2,
                                          OutputArray p12, bool handleNested = true );


/** @brief Fits an ellipse around a set of 2D points.
 *
 * The function calculates the ellipse that fits (in a least-squares sense) a set of 2D points best of
 * all. It returns the rotated rectangle in which the ellipse is inscribed. The first algorithm described by @cite Fitzgibbon95
 * is used. Developer should keep in mind that it is possible that the returned
 * ellipse/rotatedRect data contains negative indices, due to the data points being close to the
 * border of the containing Mat element.
 *
 * @param points Input 2D point set, stored in std::vector\<\> or Mat
 *
 * @note Input point types are @ref Point2i or @ref Point2f and at least 5 points are required.
 * @note @ref getClosestEllipsePoints function can be used to compute the ellipse fitting error.
 */
CV_EXPORTS_W RotatedRect fitEllipse( InputArray points );

/** @brief Fits an ellipse around a set of 2D points.
 *
 * The function calculates the ellipse that fits a set of 2D points.
 * It returns the rotated rectangle in which the ellipse is inscribed.
 * The Approximate Mean Square (AMS) proposed by @cite Taubin1991 is used.
 *
 * For an ellipse, this basis set is \f$ \chi= \left(x^2, x y, y^2, x, y, 1\right) \f$,
 * which is a set of six free coefficients \f$ A^T=\left\{A_{\text{xx}},A_{\text{xy}},A_{\text{yy}},A_x,A_y,A_0\right\} \f$.
 * However, to specify an ellipse, all that is needed is five numbers; the major and minor axes lengths \f$ (a,b) \f$,
 * the position \f$ (x_0,y_0) \f$, and the orientation \f$ \theta \f$. This is because the basis set includes lines,
 * quadratics, parabolic and hyperbolic functions as well as elliptical functions as possible fits.
 * If the fit is found to be a parabolic or hyperbolic function then the standard #fitEllipse method is used.
 * The AMS method restricts the fit to parabolic, hyperbolic and elliptical curves
 * by imposing the condition that \f$ A^T ( D_x^T D_x  +   D_y^T D_y) A = 1 \f$ where
 * the matrices \f$ Dx \f$ and \f$ Dy \f$ are the partial derivatives of the design matrix \f$ D \f$ with
 * respect to x and y. The matrices are formed row by row applying the following to
 * each of the points in the set:
 * \f{align*}{
 * D(i,:)&=\left\{x_i^2, x_i y_i, y_i^2, x_i, y_i, 1\right\} &
 * D_x(i,:)&=\left\{2 x_i,y_i,0,1,0,0\right\} &
 * D_y(i,:)&=\left\{0,x_i,2 y_i,0,1,0\right\}
 * \f}
 * The AMS method minimizes the cost function
 * \f{equation*}{
 * \epsilon ^2=\frac{ A^T D^T D A }{ A^T (D_x^T D_x +  D_y^T D_y) A^T }
 * \f}
 *
 * The minimum cost is found by solving the generalized eigenvalue problem.
 *
 * \f{equation*}{
 * D^T D A = \lambda  \left( D_x^T D_x +  D_y^T D_y\right) A
 * \f}
 *
 * @param points Input 2D point set, stored in std::vector\<\> or Mat
 *
 * @note Input point types are @ref Point2i or @ref Point2f and at least 5 points are required.
 * @note @ref getClosestEllipsePoints function can be used to compute the ellipse fitting error.
 */
CV_EXPORTS_W RotatedRect fitEllipseAMS( InputArray points );


/** @brief Fits an ellipse around a set of 2D points.
 *
 * The function calculates the ellipse that fits a set of 2D points.
 * It returns the rotated rectangle in which the ellipse is inscribed.
 * The Direct least square (Direct) method by @cite oy1998NumericallySD is used.
 *
 * For an ellipse, this basis set is \f$ \chi= \left(x^2, x y, y^2, x, y, 1\right) \f$,
 * which is a set of six free coefficients \f$ A^T=\left\{A_{\text{xx}},A_{\text{xy}},A_{\text{yy}},A_x,A_y,A_0\right\} \f$.
 * However, to specify an ellipse, all that is needed is five numbers; the major and minor axes lengths \f$ (a,b) \f$,
 * the position \f$ (x_0,y_0) \f$, and the orientation \f$ \theta \f$. This is because the basis set includes lines,
 * quadratics, parabolic and hyperbolic functions as well as elliptical functions as possible fits.
 * The Direct method confines the fit to ellipses by ensuring that \f$ 4 A_{xx} A_{yy}- A_{xy}^2 > 0 \f$.
 * The condition imposed is that \f$ 4 A_{xx} A_{yy}- A_{xy}^2=1 \f$ which satisfies the inequality
 * and as the coefficients can be arbitrarily scaled is not overly restrictive.
 *
 * \f{equation*}{
 * \epsilon ^2= A^T D^T D A \quad \text{with} \quad A^T C A =1 \quad \text{and} \quad C=\left(\begin{matrix}
 * 0 & 0  & 2  & 0  & 0  &  0  \\
 * 0 & -1  & 0  & 0  & 0  &  0 \\
 * 2 & 0  & 0  & 0  & 0  &  0 \\
 * 0 & 0  & 0  & 0  & 0  &  0 \\
 * 0 & 0  & 0  & 0  & 0  &  0 \\
 * 0 & 0  & 0  & 0  & 0  &  0
 * \end{matrix} \right)
 * \f}
 *
 * The minimum cost is found by solving the generalized eigenvalue problem.
 *
 * \f{equation*}{
 * D^T D A = \lambda  \left( C\right) A
 * \f}
 *
 * The system produces only one positive eigenvalue \f$ \lambda\f$ which is chosen as the solution
 * with its eigenvector \f$\mathbf{u}\f$. These are used to find the coefficients
 *
 * \f{equation*}{
 * A = \sqrt{\frac{1}{\mathbf{u}^T C \mathbf{u}}}  \mathbf{u}
 * \f}
 * The scaling factor guarantees that  \f$A^T C A =1\f$.
 *
 * @param points Input 2D point set, stored in std::vector\<\> or Mat
 *
 * @note Input point types are @ref Point2i or @ref Point2f and at least 5 points are required.
 * @note @ref getClosestEllipsePoints function can be used to compute the ellipse fitting error.
 */
CV_EXPORTS_W RotatedRect fitEllipseDirect( InputArray points );

/** @example samples/python/snippets/fitline.py
 * An example for fitting line in python
 */

/** @brief Compute for each 2d point the nearest 2d point located on a given ellipse.
 *
 * The function computes the nearest 2d location on a given ellipse for a vector of 2d points and is based on @cite Chatfield2017 code.
 * This function can be used to compute for instance the ellipse fitting error.
 *
 * @param ellipse_params Ellipse parameters
 * @param points Input 2d points
 * @param closest_pts For each 2d point, their corresponding closest 2d point located on a given ellipse
 *
 * @note Input point types are @ref Point2i or @ref Point2f
 * @see fitEllipse, fitEllipseAMS, fitEllipseDirect
 */
CV_EXPORTS_W void getClosestEllipsePoints( const RotatedRect& ellipse_params, InputArray points, OutputArray closest_pts );

/** @brief Fits a line to a 2D or 3D point set.
 *
 * The function fitLine fits a line to a 2D or 3D point set by minimizing \f$\sum_i \rho(r_i)\f$ where
 * \f$r_i\f$ is a distance between the \f$i^{th}\f$ point, the line and \f$\rho(r)\f$ is a distance function, one
 * of the following:
 * -  DIST_L2
 * \f[\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}\f]
 * - DIST_L1
 * \f[\rho (r) = r\f]
 * - DIST_L12
 * \f[\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)\f]
 * - DIST_FAIR
 * \f[\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998\f]
 * - DIST_WELSCH
 * \f[\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846\f]
 * - DIST_HUBER
 * \f[\rho (r) =  \fork{r^2/2}{if \(r < C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345\f]
 *
 * The algorithm is based on the M-estimator ( <https://en.wikipedia.org/wiki/M-estimator> ) technique
 * that iteratively fits the line using the weighted least-squares algorithm. After each iteration the
 * weights \f$w_i\f$ are adjusted to be inversely proportional to \f$\rho(r_i)\f$ .
 *
 * @param points Input vector of 2D or 3D points, stored in std::vector\<\> or Mat.
 * @param line Output line parameters. In case of 2D fitting, it should be a vector of 4 elements
 * (like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and
 * (x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like
 * Vec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line
 * and (x0, y0, z0) is a point on the line.
 * @param distType Distance used by the M-estimator, see #DistanceTypes
 * @param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value
 * is chosen.
 * @param reps Sufficient accuracy for the radius (distance between the coordinate origin and the line).
 * @param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.
 */
CV_EXPORTS_W void fitLine( InputArray points, OutputArray line, int distType,
                           double param, double reps, double aeps );

/** @brief Performs a point-in-contour test.
 *
 * The function determines whether the point is inside a contour, outside, or lies on an edge (or
 * coincides with a vertex). It returns positive (inside), negative (outside), or zero (on an edge)
 * value, correspondingly. When measureDist=false , the return value is +1, -1, and 0, respectively.
 * Otherwise, the return value is a signed distance between the point and the nearest contour edge.
 *
 * See below a sample output of the function where each image pixel is tested against the contour:
 *
 * ![sample output](pics/pointpolygon.png)
 *
 * @param contour Input contour.
 * @param pt Point tested against the contour.
 * @param measureDist If true, the function estimates the signed distance from the point to the
 * nearest contour edge. Otherwise, the function only checks if the point is inside a contour or not.
 */
CV_EXPORTS_W double pointPolygonTest( InputArray contour, Point2f pt, bool measureDist );

/** @brief Finds out if there is any intersection between two rotated rectangles.
 *
 * If there is then the vertices of the intersecting region are returned as well.
 *
 * Below are some examples of intersection configurations. The hatched pattern indicates the
 * intersecting region and the red vertices are returned by the function.
 *
 * ![intersection examples](pics/intersection.png)
 *
 * @param rect1 First rectangle
 * @param rect2 Second rectangle
 * @param intersectingRegion The output array of the vertices of the intersecting region. It returns
 * at most 8 vertices. Stored as std::vector\<cv::Point2f\> or cv::Mat as Mx1 of type CV_32FC2.
 * @returns One of #RectanglesIntersectTypes
 */
CV_EXPORTS_W int rotatedRectangleIntersection( const RotatedRect& rect1, const RotatedRect& rect2, OutputArray intersectingRegion  );

/** @brief Calculates a contour perimeter or a curve length.
 *
 * The function computes a curve length or a closed contour perimeter.
 *
 * @param curve Input vector of 2D points, stored in std::vector or Mat.
 * @param closed Flag indicating whether the curve is closed or not.
 */
CV_EXPORTS_W double arcLength( InputArray curve, bool closed );

/** @brief Calculates a contour area.
 *
 * The function computes a contour area. Similarly to moments , the area is computed using the Green
 * formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
 * #drawContours or #fillPoly , can be different. Also, the function will most certainly give a wrong
 * results for contours with self-intersections.
 *
 * Example:
 * @code
 *    vector<Point> contour;
 *    contour.push_back(Point2f(0, 0));
 *    contour.push_back(Point2f(10, 0));
 *    contour.push_back(Point2f(10, 10));
 *    contour.push_back(Point2f(5, 4));
 *
 *    double area0 = contourArea(contour);
 *    vector<Point> approx;
 *    approxPolyDP(contour, approx, 5, true);
 *    double area1 = contourArea(approx);
 *
 *    cout << "area0 =" << area0 << endl <<
 *            "area1 =" << area1 << endl <<
 *            "approx poly vertices" << approx.size() << endl;
 * @endcode
 * @param contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
 * @param oriented Oriented area flag. If it is true, the function returns a signed area value,
 * depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can
 * determine orientation of a contour by taking the sign of an area. By default, the parameter is
 * false, which means that the absolute value is returned.
 */
CV_EXPORTS_W double contourArea( InputArray contour, bool oriented = false );

/** @brief Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
 *
 * The function calculates and returns the minimal up-right bounding rectangle for the specified point set or
 * non-zero pixels of gray-scale image.
 *
 * @param array Input gray-scale image or 2D point set, stored in std::vector or Mat.
 */
CV_EXPORTS_W Rect boundingRect( InputArray array );

/** @brief Calculates an affine matrix of 2D rotation.

The function calculates the following matrix:

\f[\begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} + (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}\f]

where

\f[\begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}\f]

The transformation maps the rotation center to itself. If this is not the target, adjust the shift.

@param center Center of the rotation in the source image.
@param angle Rotation angle in degrees. Positive values mean counter-clockwise rotation (the
coordinate origin is assumed to be the top-left corner).
@param scale Isotropic scale factor.

@sa  getAffineTransform, warpAffine, transform
 */
CV_EXPORTS_W Mat getRotationMatrix2D(Point2f center, double angle, double scale);

/** @sa getRotationMatrix2D */
CV_EXPORTS Matx23d getRotationMatrix2D_(Point2f center, double angle, double scale);

inline
Mat getRotationMatrix2D(Point2f center, double angle, double scale)
{
    return Mat(getRotationMatrix2D_(center, angle, scale), true);
}

/** @brief Calculates an affine transform from three pairs of the corresponding points.
 *
 * The function calculates the \f$2 \times 3\f$ matrix of an affine transform so that:
 *
 * \f[\begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}\f]
 *
 * where
 *
 * \f[dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2\f]
 *
 * @param src Coordinates of triangle vertices in the source image.
 * @param dst Coordinates of the corresponding triangle vertices in the destination image.
 *
 * @sa  warpAffine, transform
 */
CV_EXPORTS Mat getAffineTransform( const Point2f src[], const Point2f dst[] );

/** @brief Inverts an affine transformation.
 *
 * The function computes an inverse affine transformation represented by \f$2 \times 3\f$ matrix M:
 *
 * \f[\begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}\f]
 *
 * The result is also a \f$2 \times 3\f$ matrix of the same type as M.
 *
 * @param M Original affine transformation.
 * @param iM Output reverse affine transformation.
 */
CV_EXPORTS_W void invertAffineTransform( InputArray M, OutputArray iM );

/** @brief Calculates a perspective transform from four pairs of the corresponding points.
 *
 * The function calculates the \f$3 \times 3\f$ matrix of a perspective transform so that:
 *
 * \f[\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}\f]
 *
 * where
 *
 * \f[dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3\f]
 *
 * @param src Coordinates of quadrangle vertices in the source image.
 * @param dst Coordinates of the corresponding quadrangle vertices in the destination image.
 * @param solveMethod method passed to cv::solve (#DecompTypes)
 *
 * @sa  findHomography, warpPerspective, perspectiveTransform
 */
CV_EXPORTS_W Mat getPerspectiveTransform(InputArray src, InputArray dst, int solveMethod = DECOMP_LU);

/** @overload */
CV_EXPORTS Mat getPerspectiveTransform(const Point2f src[], const Point2f dst[], int solveMethod = DECOMP_LU);


CV_EXPORTS_W Mat getAffineTransform( InputArray src, InputArray dst );

} // namespace cv

#endif // OPENCV_2D_HPP
