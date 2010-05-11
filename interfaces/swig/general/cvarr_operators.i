/** This file was automatically generated using util/cvarr_operators.py script */
%extend CvMat {
	%newobject operator &;
	CvMat * operator & (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvAnd(self, src, res);
		return res;
	}
	CvMat * operator &= (CvArr * src){
		cvAnd(self, src, self);
		return self;
	}
	%newobject operator +;
	CvMat * operator + (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvAdd(self, src, res);
		return res;
	}
	CvMat * operator += (CvArr * src){
		cvAdd(self, src, self);
		return self;
	}
	%newobject operator *;
	CvMat * operator * (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvMul(self, src, res);
		return res;
	}
	CvMat * operator *= (CvArr * src){
		cvMul(self, src, self);
		return self;
	}
	%newobject operator -;
	CvMat * operator - (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvSub(self, src, res);
		return res;
	}
	CvMat * operator -= (CvArr * src){
		cvSub(self, src, self);
		return self;
	}
	%newobject operator /;
	CvMat * operator / (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvDiv(self, src, res);
		return res;
	}
	CvMat * operator /= (CvArr * src){
		cvDiv(self, src, self);
		return self;
	}
	%newobject operator |;
	CvMat * operator | (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvOr(self, src, res);
		return res;
	}
	CvMat * operator |= (CvArr * src){
		cvOr(self, src, self);
		return self;
	}
	%newobject operator ^;
	CvMat * operator ^ (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvXor(self, src, res);
		return res;
	}
	CvMat * operator ^= (CvArr * src){
		cvXor(self, src, self);
		return self;
	}
	%newobject operator +;
	CvMat * operator + (CvScalar val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvAddS(self, val, res);
		return res;
	}
	CvMat * operator += (CvScalar val){
		cvAddS(self, val, self);
		return self;
	}
	%newobject operator ^;
	CvMat * operator ^ (CvScalar val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvXorS(self, val, res);
		return res;
	}
	CvMat * operator ^= (CvScalar val){
		cvXorS(self, val, self);
		return self;
	}
	%newobject operator -;
	CvMat * operator - (CvScalar val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvSubS(self, val, res);
		return res;
	}
	CvMat * operator -= (CvScalar val){
		cvSubS(self, val, self);
		return self;
	}
	%newobject operator |;
	CvMat * operator | (CvScalar val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvOrS(self, val, res);
		return res;
	}
	CvMat * operator |= (CvScalar val){
		cvOrS(self, val, self);
		return self;
	}
	%newobject operator &;
	CvMat * operator & (CvScalar val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvAndS(self, val, res);
		return res;
	}
	CvMat * operator &= (CvScalar val){
		cvAndS(self, val, self);
		return self;
	}
	%newobject operator >=;
	CvMat * operator >= (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmp(self, src, res, CV_CMP_GE);
		return res;
	}
	CvMat * operator >= (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmpS(self, val, res, CV_CMP_GE);
		return res;
	}
	%newobject operator ==;
	CvMat * operator == (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmp(self, src, res, CV_CMP_EQ);
		return res;
	}
	CvMat * operator == (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmpS(self, val, res, CV_CMP_EQ);
		return res;
	}
	%newobject operator <=;
	CvMat * operator <= (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmp(self, src, res, CV_CMP_LE);
		return res;
	}
	CvMat * operator <= (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmpS(self, val, res, CV_CMP_LE);
		return res;
	}
	%newobject operator !=;
	CvMat * operator != (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmp(self, src, res, CV_CMP_NE);
		return res;
	}
	CvMat * operator != (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmpS(self, val, res, CV_CMP_NE);
		return res;
	}
	%newobject operator <;
	CvMat * operator < (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmp(self, src, res, CV_CMP_LT);
		return res;
	}
	CvMat * operator < (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmpS(self, val, res, CV_CMP_LT);
		return res;
	}
	%newobject operator >;
	CvMat * operator > (CvArr * src){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmp(self, src, res, CV_CMP_GT);
		return res;
	}
	CvMat * operator > (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, CV_8U);
		cvCmpS(self, val, res, CV_CMP_GT);
		return res;
	}
	%newobject operator *;
	CvMat * operator * (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvScale(self, res, val);
		return res;
	}
	CvMat * operator *= (double val){
		cvScale(self, self, val);
		return self;
	}
	%newobject operator /;
	CvMat * operator / (double val){
		CvMat * res = cvCreateMat(self->rows, self->cols, self->type);
		cvScale(self, res, 1.0/val);
		return res;
	}
	CvMat * operator /= (double val){
		cvScale(self, self, 1.0/val);
		return self;
	}
} /* extend CvMat */

%extend IplImage {
	%newobject operator &;
	IplImage * operator & (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvAnd(self, src, res);
		return res;
	}
	IplImage * operator &= (CvArr * src){
		cvAnd(self, src, self);
		return self;
	}
	%newobject operator +;
	IplImage * operator + (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvAdd(self, src, res);
		return res;
	}
	IplImage * operator += (CvArr * src){
		cvAdd(self, src, self);
		return self;
	}
	%newobject operator *;
	IplImage * operator * (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvMul(self, src, res);
		return res;
	}
	IplImage * operator *= (CvArr * src){
		cvMul(self, src, self);
		return self;
	}
	%newobject operator -;
	IplImage * operator - (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvSub(self, src, res);
		return res;
	}
	IplImage * operator -= (CvArr * src){
		cvSub(self, src, self);
		return self;
	}
	%newobject operator /;
	IplImage * operator / (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvDiv(self, src, res);
		return res;
	}
	IplImage * operator /= (CvArr * src){
		cvDiv(self, src, self);
		return self;
	}
	%newobject operator |;
	IplImage * operator | (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvOr(self, src, res);
		return res;
	}
	IplImage * operator |= (CvArr * src){
		cvOr(self, src, self);
		return self;
	}
	%newobject operator ^;
	IplImage * operator ^ (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvXor(self, src, res);
		return res;
	}
	IplImage * operator ^= (CvArr * src){
		cvXor(self, src, self);
		return self;
	}
	%newobject operator +;
	IplImage * operator + (CvScalar val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvAddS(self, val, res);
		return res;
	}
	IplImage * operator += (CvScalar val){
		cvAddS(self, val, self);
		return self;
	}
	%newobject operator ^;
	IplImage * operator ^ (CvScalar val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvXorS(self, val, res);
		return res;
	}
	IplImage * operator ^= (CvScalar val){
		cvXorS(self, val, self);
		return self;
	}
	%newobject operator -;
	IplImage * operator - (CvScalar val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvSubS(self, val, res);
		return res;
	}
	IplImage * operator -= (CvScalar val){
		cvSubS(self, val, self);
		return self;
	}
	%newobject operator |;
	IplImage * operator | (CvScalar val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvOrS(self, val, res);
		return res;
	}
	IplImage * operator |= (CvScalar val){
		cvOrS(self, val, self);
		return self;
	}
	%newobject operator &;
	IplImage * operator & (CvScalar val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvAndS(self, val, res);
		return res;
	}
	IplImage * operator &= (CvScalar val){
		cvAndS(self, val, self);
		return self;
	}
	%newobject operator >=;
	IplImage * operator >= (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmp(self, src, res, CV_CMP_GE);
		return res;
	}
	IplImage * operator >= (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmpS(self, val, res, CV_CMP_GE);
		return res;
	}
	%newobject operator ==;
	IplImage * operator == (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmp(self, src, res, CV_CMP_EQ);
		return res;
	}
	IplImage * operator == (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmpS(self, val, res, CV_CMP_EQ);
		return res;
	}
	%newobject operator <=;
	IplImage * operator <= (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmp(self, src, res, CV_CMP_LE);
		return res;
	}
	IplImage * operator <= (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmpS(self, val, res, CV_CMP_LE);
		return res;
	}
	%newobject operator !=;
	IplImage * operator != (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmp(self, src, res, CV_CMP_NE);
		return res;
	}
	IplImage * operator != (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmpS(self, val, res, CV_CMP_NE);
		return res;
	}
	%newobject operator <;
	IplImage * operator < (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmp(self, src, res, CV_CMP_LT);
		return res;
	}
	IplImage * operator < (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmpS(self, val, res, CV_CMP_LT);
		return res;
	}
	%newobject operator >;
	IplImage * operator > (CvArr * src){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmp(self, src, res, CV_CMP_GT);
		return res;
	}
	IplImage * operator > (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1);
		cvCmpS(self, val, res, CV_CMP_GT);
		return res;
	}
	%newobject operator *;
	IplImage * operator * (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvScale(self, res, val);
		return res;
	}
	IplImage * operator *= (double val){
		cvScale(self, self, val);
		return self;
	}
	%newobject operator /;
	IplImage * operator / (double val){
		IplImage * res = cvCreateImage(cvGetSize(self), self->depth, self->nChannels);
		cvScale(self, res, 1.0/val);
		return res;
	}
	IplImage * operator /= (double val){
		cvScale(self, self, 1.0/val);
		return self;
	}
} /* extend IplImage */

