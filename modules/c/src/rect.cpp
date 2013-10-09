#include <opencv2/c/rect.hpp>

extern "C" {
Rect* cv_create_Rect() {
    return new Rect;
}
Rect* cv_create_Rect4(int x, int y, int width, int height) {
    return new Rect(x, y, width, height);
}
Rect* cv_Rect_assignTo(Rect* self, Rect* r) {
    *self = *r;
    return self;
}
Point* cv_Rect_tl(Rect* self) {
    return new Point(self->tl());
}
Point* cv_Rect_br(Rect* self) {
    return new Point(self->br());
}
Size* size(Rect* self) {
    return new Size(self->size());
}
int cv_Rect_contains(Rect* self, Point* pt) {
    return self->contains(*pt);
}
}
