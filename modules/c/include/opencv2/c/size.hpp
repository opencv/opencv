#ifndef __CV_C_SIZE_HPP__
#define __CV_C_SIZE_HPP__
#include <opencv2/c/opencv_generated.hpp>

extern "C" {
    Size* cv_create_Size();
    Size* cv_create_Size2(int width, int height);
    Size* cv_Size_assignTo(Size* self, Size* other);
    Size* cv_Size_fromPoint(Point* p);
    int cv_Size_area(Size* self);
    int cv_Size_width(Size* self);
    int cv_Size_height(Size* self);
}
#endif
