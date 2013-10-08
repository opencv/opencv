/*
 * =====================================================================================
 *
 *       Filename:  mat.hpp
 *
 *    Description:  Wrappers for the OpenCV Matrix class
 *
 *        Version:  1.0
 *        Created:  09/24/13 20:01:07
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Arjun Comar
 *
 * =====================================================================================
 */

#include <opencv2/c/opencv_generated.hpp>

extern "C" {
Mat* cv_create_Mat();
Mat* cv_Mat_assign(Mat* self, Mat* m);
Mat* cv_Mat_assignVal(Mat* self, Scalar* s);
Mat* cv_Mat_getRow(Mat* self, int y);
Mat* cv_Mat_getCol(Mat* self, int x);
Mat* cv_Mat_getRowRange(Mat* self, int startrow, int endrow);
Mat* cv_Mat_getColRange(Mat* self, int startrow, int endrow);
Mat* cv_Mat_diag(Mat* self);
Mat* cv_Mat_diag_d(Mat* self, int d);
Mat* cv_create_diagMat(Mat* d);
Mat* cv_Mat_clone(Mat* self);
void cv_Mat_copyTo(Mat* self, Mat* m);
void cv_Mat_copyTo_masked(Mat* self, Mat* m, Mat* mask);
void cv_Mat_assignTo(Mat* self, Mat* m);
void cv_Mat_assignTo_t(Mat*self, Mat* m, int t);
Mat* cv_Mat_setTo(Mat*self, Scalar* value);
Mat* cv_Mat_setTo_masked(Mat* self, Scalar* value, Mat* mask);
Mat* cv_Mat_reshape(Mat* self, int cn);
Mat* cv_Mat_reshape_rows(Mat* self, int cn, int rows);
size_t cv_Mat_elemSize(Mat* self);
size_t cv_Mat_elemSize1(Mat* self);
int cv_Mat_type(Mat* self);
int cv_Mat_depth(Mat* self);
size_t cv_Mat_total(Mat* self);
bool cv_Mat_isContinuous(Mat* self);
int cv_Mat_channels(Mat* self);
int cv_Mat_rows(Mat* self);
int cv_Mat_cols(Mat* self);
int cv_Mat_empty(Mat* self);
Size* cv_Mat_size(Mat* self);
size_t cv_Mat_step1(Mat* self);
uchar* cv_Mat_ptr(Mat* self);
uchar* cv_Mat_ptr_index(Mat* self, int i);
Mat* cv_create_identity(int rows, int cols, int type);
Mat* cv_create_ones(int rows, int cols, int type);
Mat* cv_create_zeros(int rows, int cols, int type);
MatExpr* cv_Mat_transpose_mat(Mat* self);
MatExpr* cv_Mat_inv_mat(Mat* self, int method);
MatExpr* cv_Mat_add(Mat* m1, Mat* m2);
MatExpr* cv_Mat_mult(Mat* m1, Mat* m2);
Mat* force(MatExpr* expr);
MatExpr* promote(Mat* m);
MatExpr* cv_Mat_scale(MatExpr* m, double alpha);
double cv_Mat_dot(Mat* self, Mat* m);
Mat* cv_Mat_cross(Mat* self, Mat* m);
void cv_Mat_locateROI(Mat* self, Size* s, Point* p);
Mat* cv_Mat_adjustROI(Mat* self, int dtop, int dbottom, int dleft, int dright);
}

