/*
 * =====================================================================================
 *
 *       Filename:  point.hpp
 *
 *    Description:  Wrapper header for the OpenCV Point class(es)
 *
 *        Version:  1.0
 *        Created:  10/02/2013 11:54:37 AM
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar 
 *
 * =====================================================================================
 */

#include <opencv2/c/opencv_generated.hpp>

extern "C" {
    Point* cv_create_Point(int x, int y);
    int cv_Point_getX(Point* self);
    int cv_Point_getY(Point* self);
    int cv_Point_dot(Point* self, Point* other);
    Point2f* cv_create_Point2f(float x, float y);
    float cv_Point2f_getX(Point2f* self);
    float cv_Point2f_getY(Point2f* self);
    float cv_Point2f_dot(Point2f* self, Point2f* other);
    Point2d* cv_create_Point2d(double x, double y);
    double cv_Point2d_getX(Point2d* self);
    double cv_Point2d_getY(Point2d* self);
    double cv_Point2d_dot(Point2d* self, Point2d* other);
    Point3i* cv_create_Point3i(int x, int y, int z);
    int cv_Point3i_getX(Point3i* self);
    int cv_Point3i_getY(Point3i* self);
    int cv_Point3i_getZ(Point3i* self);
    int cv_Point3i_dot(Point3i* self, Point3i* other);
    Point3i* cv_Point3i_cross(Point3i* self, Point3i* other);
    Point3f* cv_create_Point3f(float x, float y, float z);
    float cv_Point3f_getX(Point3f* self);
    float cv_Point3f_getY(Point3f* self);
    float cv_Point3f_getZ(Point3f* self);
    float cv_Point3f_dot(Point3f* self, Point3f* other);
    Point3f* cv_Point3f_cross(Point3f* self, Point3f* other);
    Point3d* cv_create_Point3d(double x, double y, double z);
    double cv_Point3d_getX(Point3d* self);
    double cv_Point3d_getY(Point3d* self);
    double cv_Point3d_getZ(Point3d* self);
    double cv_Point3d_dot(Point3d* self, Point3d* other);
    Point3d* cv_Point3d_cross(Point3d* self, Point3d* other);
}

