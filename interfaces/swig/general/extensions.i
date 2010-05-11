/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


// 2004-03-23, Mark Asbach       <asbach@ient.rwth-aachen.de>
//             Institute of Communications Engineering, RWTH Aachen University
// 2008-05-15, Xavier Delacour   <xavier.delacour@gmail.com>


struct CvLSH {};

%extend IplImage       { ~IplImage       () { IplImage       * dummy = self; cvReleaseImage              (& dummy); } }
%extend CvMat          { ~CvMat          () { CvMat          * dummy = self; cvReleaseMat                (& dummy); } }
%extend CvMatND        { ~CvMatND        () { CvMatND        * dummy = self; cvReleaseMatND              (& dummy); } }
%extend CvSparseMat    { ~CvSparseMat    () { CvSparseMat    * dummy = self; cvReleaseSparseMat          (& dummy); } }
%extend CvMemStorage   { ~CvMemStorage   () { CvMemStorage   * dummy = self; cvReleaseMemStorage         (& dummy); } }
%extend CvGraphScanner { ~CvGraphScanner () { CvGraphScanner * dummy = self; cvReleaseGraphScanner       (& dummy); } }
%extend CvFileStorage  { ~CvFileStorage  () { CvFileStorage  * dummy = self; cvReleaseFileStorage        (& dummy); } }
%extend IplConvKernel  { ~IplConvKernel  () { IplConvKernel  * dummy = self; cvReleaseStructuringElement (& dummy); } }
%extend CvKalman       { ~CvKalman       () { CvKalman       * dummy = self; cvReleaseKalman             (& dummy); } }
%extend CvHistogram    { ~CvHistogram    () { CvHistogram    * dummy = self; cvReleaseHist               (& dummy); } }
%extend CvHaarClassifierCascade { ~CvHaarClassifierCascade () { CvHaarClassifierCascade * dummy = self; cvReleaseHaarClassifierCascade  (& dummy); } }
%extend CvPOSITObject  { ~CvPOSITObject  () { CvPOSITObject  * dummy = self; cvReleasePOSITObject        (& dummy); } }
%extend CvFeatureTree  { ~CvFeatureTree  () { CvFeatureTree  * dummy = self; cvReleaseFeatureTree        (& dummy); } }
%extend CvLSH          { ~CvLSH          () { CvLSH          * dummy = self; cvReleaseLSH                (& dummy); } }

// string operators for some OpenCV types

%extend CvScalar
{
	const char * __str__(){
		static char str[256];
		snprintf(str, 256, "[%f, %f, %f, %f]", self->val[0], self->val[1], self->val[2], self->val[3]);
		return str;
	}
	const char * __repr__(){
		static char str[256];
		snprintf(str, 256, "cvScalar(%f, %f, %f, %f)", self->val[0], self->val[1], self->val[2], self->val[3]);
		return str;
	}
    const double __getitem__ (int index) {
        if (index >= 4) {
#ifdef defined(SWIGPYTHON)
            PyErr_SetString (PyExc_IndexError, "indice must be lower than 4");
#elif defined(SWIGOCTAVE)
            error("indice must be lower than 4");
#endif
            return 0;
        }
        if (index < -4) {
#ifdef defined(SWIGPYTHON)
            PyErr_SetString (PyExc_IndexError, "indice must be bigger or egal to -4");
#elif defined(SWIGOCTAVE)
	    error("indice must be bigger or egal to -4");
#endif
            return 0;
        }
        if (index < 0) {
            /* negative index means from the end in python */
            index = 4 - index;
        }
        return self->val [index];
    }
    void __setitem__ (int index, double value) {
        if (index >= 4) {
#ifdef defined(SWIGPYTHON)
            PyErr_SetString (PyExc_IndexError, "indice must be lower than 4");
#elif defined(SWIGOCTAVE)
	    error("indice must be lower than 4");
#endif
            return;
        }
        if (index < -4) {
#ifdef defined(SWIGPYTHON)
            PyErr_SetString (PyExc_IndexError, "indice must be bigger or egal to -4");
#elif defined(SWIGOCTAVE)
	    error("indice must be bigger or egal to -4");
#endif
            return;
        }
        if (index < 0) {
            /* negative index means from the end in python */
            index = 4 - index;
        }
        self->val [index] = value;
    }
};

%extend CvPoint2D32f
{
	const char * __str__(){
		static char str[64];
		snprintf(str, 64, "[%f %f]", self->x, self->y);
		return str;
	}
	const char * __repr__(){
		static char str[64];
		snprintf(str, 64, "cvPoint2D32f(%f,%f)", self->x, self->y);
		return str;
	}
};

%extend CvPoint
{
	const char * __str__(){
		static char str[64];
		snprintf(str, 64, "[%d %d]", self->x, self->y);
		return str;
	}
	const char * __repr__(){
		static char str[64];
		snprintf(str, 64, "cvPoint(%d,%d)", self->x, self->y);
		return str;
	}
};

// Set up CvMat to emulate IplImage fields
%{
int CvMat_cols_get(CvMat * m){
	return m->cols;
}
void CvMat_cols_set(CvMat * m, int cols){
    m->cols = cols;
}
int CvMat_rows_get(CvMat *m){
	return m->rows;
}
void CvMat_rows_set(CvMat *m, int rows){
    m->rows = rows;
}
int CvMat_width_get(CvMat * m){
	return m->cols;
}
void CvMat_width_set(CvMat * m, int width){
    m->cols = width;
}
int CvMat_height_get(CvMat *m){
	return m->rows;
}
void CvMat_height_set(CvMat * m, int height){
    m->rows = height;
}
int CvMat_depth_get(CvMat * m){
	return cvIplDepth(m->type);
}
void CvMat_depth_set(CvMat *m, int depth){
    cvError(CV_StsNotImplemented, "CvMat_depth_set", "Not Implemented", __FILE__, __LINE__);
}
int CvMat_nChannels_get(CvMat * m){
	return CV_MAT_CN(m->type);
}
void CvMat_nChannels_set(CvMat *m, int nChannels){
    int depth = CV_MAT_DEPTH(m->type);
    m->type = CV_MAKETYPE(depth, nChannels);
}
int CvMat_origin_get(CvMat * m){
    /* Always 0 - top-left origin */
    return 0;
}
void CvMat_origin_set(CvMat * m, int origin){
    cvError(CV_StsNotImplemented, "CvMat_origin_get", "IplImage is replaced by CvMat in Python, so its fields are read-only", __FILE__, __LINE__);
}
int CvMat_dataOrder_get(CvMat * m){
    cvError(CV_StsNotImplemented, "CvMat_dataOrder_get", "Not Implemented", __FILE__, __LINE__);
    return 0;
}
void CvMat_dataOrder_set(CvMat * m, int dataOrder){
    cvError(CV_StsNotImplemented, "CvMat_dataOrder_get", "IplImage is replaced by CvMat in Python, so its fields are read-only", __FILE__, __LINE__);
}
int CvMat_imageSize_get(CvMat * m){
	int step = m->step ? m->step : CV_ELEM_SIZE(m->type) * m->cols;
	return step*m->rows;
}
void CvMat_imageSize_set(CvMat * m, int imageSize){
    cvError(CV_StsNotImplemented, "CvMat_imageSize_set", "IplImage is not implemented in Python, so origin is read-only", __FILE__, __LINE__);
}
int CvMat_widthStep_get(CvMat * m){
	return m->step;
}
void CvMat_widthStep_set(CvMat *m, int widthStep){
    m->step = widthStep;
}
%}

%extend CvMat
{
	int depth;
	int nChannels;
	int dataOrder;
	int origin;
	int width;
	int height;
	int imageSize;
	int widthStep;
	// swig doesn't like the embedded union in CvMat, so re-add these
	int rows;
	int cols;
};
