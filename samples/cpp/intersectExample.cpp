/*
 * Author: Steve Nicholson
 *
 * A program that illustrates intersectConvexConvex in various scenarios
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

// Create a vector of points describing a rectangle with the given corners
static vector<Point> makeRectangle(Point topLeft, Point bottomRight)
{
    vector<Point> rectangle;
    rectangle.push_back(topLeft);
    rectangle.push_back(Point(bottomRight.x, topLeft.y));
    rectangle.push_back(bottomRight);
    rectangle.push_back(Point(topLeft.x, bottomRight.y));
    return rectangle;
}

static vector<Point> makeTriangle(Point point1, Point point2, Point point3)
{
    vector<Point> triangle;
    triangle.push_back(point1);
    triangle.push_back(point2);
    triangle.push_back(point3);
    return triangle;
}

// Run intersectConvexConvex on two polygons then draw the polygons and their intersection (if there is one)
// Return the area of the intersection
static float drawIntersection(Mat &image, vector<Point> polygon1, vector<Point> polygon2, bool handleNested = true)
{
    vector<Point> intersectionPolygon;

    vector<vector<Point> > polygons;
    polygons.push_back(polygon1);
    polygons.push_back(polygon2);

    float intersectArea = intersectConvexConvex(polygon1, polygon2, intersectionPolygon, handleNested);

    if (intersectArea > 0)
    {
        Scalar fillColor(200, 200, 200);
        // If the input is invalid, draw the intersection in red
        if (!isContourConvex(polygon1) || !isContourConvex(polygon2))
        {
            fillColor = Scalar(0, 0, 255);
        }
        vector<vector<Point> > pp;
        pp.push_back(intersectionPolygon);
        fillPoly(image, pp, fillColor);
    }
    polylines(image, polygons, true, Scalar(0, 0, 0));

    return intersectArea;
}

static void drawDescription(Mat &image, int intersectionArea, string description, Point origin)
{
    const size_t bufSize=1024;
    char caption[bufSize];
    snprintf(caption, bufSize, "Intersection area: %d%s", intersectionArea, description.c_str());
    putText(image, caption, origin, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0));
}

static void intersectConvexExample()
{
    Mat image(610, 550, CV_8UC3, Scalar(255, 255, 255));
    float intersectionArea;

    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 10), Point(50, 50)),
        makeRectangle(Point(20, 20), Point(60, 60)));

    drawDescription(image, (int)intersectionArea, "", Point(70, 40));

    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 70), Point(35, 95)),
        makeRectangle(Point(35, 95), Point(60, 120)));

    drawDescription(image, (int)intersectionArea, "", Point(70, 100));

    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 130), Point(60, 180)),
        makeRectangle(Point(20, 140), Point(50, 170)),
        true);

    drawDescription(image, (int)intersectionArea, " (handleNested true)", Point(70, 160));

    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 190), Point(60, 240)),
        makeRectangle(Point(20, 200), Point(50, 230)),
        false);

    drawDescription(image, (int)intersectionArea, " (handleNested false)", Point(70, 220));

    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 250), Point(60, 300)),
        makeRectangle(Point(20, 250), Point(50, 290)),
        true);

    drawDescription(image, (int)intersectionArea, " (handleNested true)", Point(70, 280));

    // These rectangles share an edge so handleNested can be false and an intersection is still found
    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 310), Point(60, 360)),
        makeRectangle(Point(20, 310), Point(50, 350)),
        false);

    drawDescription(image, (int)intersectionArea, " (handleNested false)", Point(70, 340));

    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 370), Point(60, 420)),
        makeRectangle(Point(20, 371), Point(50, 410)),
        false);

    drawDescription(image, (int)intersectionArea, " (handleNested false)", Point(70, 400));

    // A vertex of the triangle lies on an edge of the rectangle so handleNested can be false and an intersection is still found
    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 430), Point(60, 480)),
        makeTriangle(Point(35, 430), Point(20, 470), Point(50, 470)),
        false);

    drawDescription(image, (int)intersectionArea, " (handleNested false)", Point(70, 460));

    // Show intersection of overlapping rectangle and triangle
    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 490), Point(40, 540)),
        makeTriangle(Point(25, 500), Point(25, 530), Point(60, 515)),
        false);

    drawDescription(image, (int)intersectionArea, "", Point(70, 520));

    // This concave polygon is invalid input to intersectConvexConvex so it returns an invalid intersection
    vector<Point> notConvex;
    notConvex.push_back(Point(25, 560));
    notConvex.push_back(Point(25, 590));
    notConvex.push_back(Point(45, 580));
    notConvex.push_back(Point(60, 600));
    notConvex.push_back(Point(60, 550));
    notConvex.push_back(Point(45, 570));
    intersectionArea = drawIntersection(image,
        makeRectangle(Point(10, 550), Point(50, 600)),
        notConvex,
        false);

    drawDescription(image, (int)intersectionArea, " (invalid input: not convex)", Point(70, 580));

    imshow("Intersections", image);
    waitKey(0);
}

int main()
{
    intersectConvexExample();
}
