#ifndef __CV_C_RECT_HPP__
#define __CV_C_RECT_HPP__
#include <opencv2/c/opencv_generated.hpp>

extern "C" {
Rect* cv_create_Rect();
Rect* cv_create_Rect4(int x, int y, int width, int height);
Rect* cv_Rect_assignTo(Rect* self, Rect* r);
Point* cv_Rect_tl(Rect* self);
Point* cv_Rect_br(Rect* self);
Size* size(Rect* self);
int cv_Rect_contains(Rect* self, Point* pt);
}
#endif
