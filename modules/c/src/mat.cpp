/*
 * =====================================================================================
 *
 *       Filename:  mat.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09/24/13 20:12:17
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <opencv2/c/mat.hpp>

extern "C" {
Mat* cv_create_Mat() {
    return new Mat();
}

Mat* cv_Mat_assign(Mat* self, Mat* m) {
    *self = *m;
    return self;
}

Mat* cv_Mat_assignVal(Mat* self, Scalar* s) {
    *self = *s;
    return self;
}

Mat* cv_Mat_getRow(Mat* self, int y) {
    return new Mat(self->row(y));
}

Mat* cv_Mat_getCol(Mat* self, int x) {
    return new Mat(self->col(x));
}

Mat* cv_Mat_getRowRange(Mat* self, int startrow, int endrow) {
    return new Mat(self->rowRange(startrow, endrow));
}

Mat* cv_Mat_getColRange(Mat* self, int startcol, int endrow) {
    return new Mat(self->colRange(startcol, endrow));
}

Mat* cv_Mat_diag(Mat* self) {
    return new Mat(self->diag());
}

Mat* cv_Mat_diag_d(Mat* self, int d) {
    return new Mat(self->diag(d));
}

Mat* cv_create_diagMat(Mat* d) {
    return new Mat(Mat::diag(*d));
}

Mat* cv_Mat_clone(Mat* self) {
    return new Mat(self->clone());
}

void cv_Mat_copyTo(Mat* self, Mat* m) {
    self->copyTo(*m);
}

void cv_Mat_copyTo_masked(Mat* self, Mat* m, Mat* mask) {
    self->copyTo(*m, *mask);
}

void cv_Mat_assignTo(Mat* self, Mat* m) {
    self->assignTo(*m);
}

void cv_Mat_assignTo_t(Mat*self, Mat* m, int t) {
    self->assignTo(*m, t);
}

Mat* cv_Mat_setTo(Mat* self, Scalar* value) {
    Mat* m = new Mat;
    *m = *value;
    return new Mat(self->setTo(*m));
}

Mat* cv_Mat_setTo_masked(Mat* self, Scalar* value, Mat* mask) {
    Mat* m = new Mat;
    *m = *value;
    return new Mat(self->setTo(*m, *mask));
}

Mat* cv_Mat_reshape(Mat* self, int cn) {
   return new Mat(self->reshape(cn)); 
}

Mat* cv_Mat_reshape_rows(Mat* self, int cn, int rows) {
   return new Mat(self->reshape(cn, rows)); 
}

size_t cv_Mat_elemSize(Mat* self) {
    return self->elemSize();
}

size_t cv_Mat_elemSize1(Mat* self) {
    return self->elemSize1();
}

int cv_Mat_type(Mat* self) {
    return self->type();
}

int cv_Mat_depth(Mat* self) {
    return self->depth();
}

size_t cv_Mat_total(Mat* self) {
    return self->total();
}

bool cv_Mat_isContinuous(Mat* self) {
    return self->isContinuous();
}

int cv_Mat_channels(Mat* self) {
    return self->channels();
}

int cv_Mat_rows(Mat* self) {
    return self->rows;
}

int cv_Mat_cols(Mat* self) {
    return self->cols;
}

int cv_Mat_empty(Mat* self) {
    return self->empty();
}

Size* cv_Mat_size(Mat* self) {
    return new Size(self->size());
}

size_t cv_Mat_step1(Mat* self) {
    return self->step1();
}

uchar* cv_Mat_ptr(Mat* self) {
    return self->ptr();
}

uchar* cv_Mat_ptr_index(Mat* self, int i) {
    return self->ptr(i);
}

Mat* cv_create_identity(int rows, int cols, int type) {
    return new Mat(Mat::eye(rows, cols, type));
}

Mat* cv_create_ones(int rows, int cols, int type) {
    return new Mat(Mat::ones(rows, cols, type));
}

Mat* cv_create_zeros(int rows, int cols, int type) {
    return new Mat(Mat::zeros(rows, cols, type));
}

MatExpr* cv_Mat_transpose_mat(Mat* self) {
    return new MatExpr(self->t());
}

MatExpr* cv_Mat_inv_mat(Mat* self, int method) {
    return new MatExpr(self->inv(method));
}

MatExpr* cv_Mat_add(Mat* m1, Mat* m2) {
    return new MatExpr(*m1 + *m2);
}

MatExpr* cv_Mat_mult(Mat* m1, Mat* m2) {
    return new MatExpr(*m1 * *m2);
}

Mat* force(MatExpr* expr) {
    Mat* p = new Mat;
    *p = *expr;
    return p;
}

MatExpr* promote(Mat* m) {
    return new MatExpr(*m);
}

MatExpr* cv_Mat_scale(MatExpr* m, double alpha) {
    return new MatExpr(*m * alpha);
}

double cv_Mat_dot(Mat* self, Mat* m) {
    return self->dot(*m);
}

Mat* cv_Mat_cross(Mat* self, Mat* m) {
    return new Mat(self->cross(*m));
}

void cv_Mat_locateROI(Mat* self, Size* s, Point* p) {
    self->locateROI(*s, *p);
}

Mat* cv_Mat_adjustROI(Mat* self, int dtop, int dbottom, int dleft, int dright) {
    return new Mat(self->adjustROI(dtop, dbottom, dleft, dright));
}

}
