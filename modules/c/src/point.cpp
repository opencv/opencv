#include <opencv2/c/point.hpp>

extern "C" {
    Point* cv_create_Point(int x, int y) {
        return new Point(x, y);
    }
    int cv_Point_getX(Point* self) {
        return self->x;
    }
    int cv_Point_getY(Point* self) {
        return self->y;
    }
    int cv_Point_dot(Point* self, Point* other) {
        return self->dot(*other);
    }
    Point2f* cv_create_Point2f(float x, float y) {
        return new Point2f(x, y);
    }
    float cv_Point2f_getX(Point2f* self) {
        return self->x;
    }
    float cv_Point2f_getY(Point2f* self) {
        return self->y;
    }
    float cv_Point2f_dot(Point2f* self, Point2f* other) {
        return (self->dot(*other));
    }
    Point2d* cv_create_Point2d(double x, double y) {
        return new Point2d(x, y);
    }
    double cv_Point2d_getX(Point2d* self) {
        return self->x;
    }
    double cv_Point2d_getY(Point2d* self) {
        return self->y;
    }
    double cv_Point2d_dot(Point2d* self, Point2d* other) {
        return (self->dot(*other));
    }
    Point3i* cv_create_Point3i(int x, int y, int z) {
        return new Point3i(x, y, z);
    }
    int cv_Point3i_getX(Point3i* self) {
        return self->x;
    }
    int cv_Point3i_getY(Point3i* self) {
        return self->y;
    }
    int cv_Point3i_getZ(Point3i* self) {
        return self->z;
    }
    int cv_Point3i_dot(Point3i* self, Point3i* other) {
        return self->dot(*other);
    }
    Point3i* cv_Point3i_cross(Point3i* self, Point3i* other) {
        return new Point3i(self->cross(*other));
    }
    Point3f* cv_create_Point3f(float x, float y, float z) {
        return new Point3f(x, y, z);
    }
    float cv_Point3f_getX(Point3f* self) {
        return self->x;
    }
    float cv_Point3f_getY(Point3f* self) {
        return self->y;
    }
    float cv_Point3f_getZ(Point3f* self) {
        return self->z;
    }
    float cv_Point3f_dot(Point3f* self, Point3f* other) {
        return (self->dot(*other));
    }
    Point3f* cv_Point3f_cross(Point3f* self, Point3f* other) {
        return new Point3f(self->cross(*other));
    }
    Point3d* cv_create_Point3d(double x, double y, double z) {
        return new Point3d(x, y, z);
    }
    double cv_Point3d_getX(Point3d* self) {
        return self->x;
    }
    double cv_Point3d_getY(Point3d* self) {
        return self->y;
    }
    double cv_Point3d_getZ(Point3d* self) {
        return self->z;
    }
    double cv_Point3d_dot(Point3d* self, Point3d* other) {
        return (self->dot(*other));
    }
    Point3d* cv_Point3d_cross(Point3d* self, Point3d* other) {
        return new Point3d(self->cross(*other));
    }
}
