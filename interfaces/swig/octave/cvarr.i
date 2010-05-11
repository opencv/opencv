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

// 2006-02-17  Roman Stanchak <rstancha@cse.wustl.edu>
// 2006-07-19  Moved most operators to general/cvarr_operators.i for use with other languages

/*M//////////////////////////////////////////////////////////////////////////////////////////
// Macros for extending CvMat and IplImage -- primarily for operator overloading 
//////////////////////////////////////////////////////////////////////////////////////////M*/

// Macro to define octave function of form B = A.f(c)
// where A is a CvArr type, c and B are arbitrary types
%define %wrap_cvGeneric_CvArr(cname, rettype, octfunc, argtype, cvfunc, newobjcall)
     %newobject cname::octfunc(argtype arg);
     %extend cname { 
  rettype octfunc(argtype arg){
    rettype retarg = newobjcall;
    cvfunc;
    return retarg;
  }
     }
%enddef

// Macro to define octave function of the form B = A.f(c) 
// where A and B are both CvArr of same size and type
%define %wrap_cvArr_binaryop(octfunc, argtype, cvfunc)
     %wrap_cvGeneric_CvArr(CvMat, CvMat *, octfunc, argtype, cvfunc, 
			   cvCreateMat(self->rows, self->cols, self->type));
     %wrap_cvGeneric_CvArr(IplImage, IplImage *, octfunc, argtype, cvfunc,
			   cvCreateImage(cvGetSize(self), self->depth, self->nChannels));
     %enddef

     // Macro to define octave function of the form A = A.f(c) 
     // where f modifies A inplace
     // use for +=, etc
     %define %wrap_cvGeneric_InPlace(cname, rettype, octfunc, argtype, cvfunc)
     %wrap_cvGeneric_CvArr(cname, rettype, octfunc, argtype, cvfunc, self);
     %enddef

     /*M//////////////////////////////////////////////////////////////////////////////////////////
     // Macros to map operators to specific OpenCV functions
     //////////////////////////////////////////////////////////////////////////////////////////M*/

     // map any OpenCV function of form cvFunc(src1, src2, dst)
     %define %wrap_cvArith(octfunc, cvfunc)
     %wrap_cvArr_binaryop(octfunc, CvArr *, cvfunc(self, arg, retarg));
     %enddef

     // map any OpenCV function of form cvFunc(src1, value, dst)
     %define %wrap_cvArithS(octfunc, cvfuncS)
     %wrap_cvArr_binaryop(octfunc, CvScalar, cvfuncS(self, arg, retarg));
     %wrap_cvArr_binaryop(octfunc, double, cvfuncS(self, cvScalar(arg), retarg));
     %enddef

     // same as wrap_cvArith
     %define %wrap_cvLogic(octfunc, cvfunc)
     %wrap_cvArr_binaryop(octfunc, CvArr *, cvfunc(self, arg, retarg))
     %enddef

     // same as wrap_cvArithS
     %define %wrap_cvLogicS(octfunc, cvfuncS)
     %wrap_cvArr_binaryop(octfunc, CvScalar, cvfuncS(self, arg, retarg));
     %wrap_cvArr_binaryop(octfunc, double, cvfuncS(self, cvScalar(arg), retarg));
     %enddef

     // Macro to map logical operations to cvCmp
     %define %wrap_cvCmp(octfunc, cmp_op)
     %wrap_cvGeneric_CvArr(CvMat, CvArr *, octfunc, CvMat *, 
			   cvCmp(self, arg, retarg, cmp_op), 
			   cvCreateMat(self->rows, self->cols, CV_8U));
     %wrap_cvGeneric_CvArr(IplImage, CvArr *, octfunc, IplImage *, 
			   cvCmp(self, arg, retarg, cmp_op), 
			   cvCreateImage(cvGetSize(self), 8, 1));
     %enddef

     %define %wrap_cvCmpS(octfunc, cmp_op)
     %wrap_cvGeneric_CvArr(CvMat, CvArr *, octfunc, double, 
			   cvCmpS(self, arg, retarg, cmp_op), 
			   cvCreateMat(self->rows, self->cols, CV_8U));
     %wrap_cvGeneric_CvArr(IplImage, CvArr *, octfunc, double, 
			   cvCmpS(self, arg, retarg, cmp_op), 
			   cvCreateImage(cvGetSize(self), 8, 1));
     %enddef

     // special case for cvScale, /, * 
     %define %wrap_cvScale(octfunc, scale)
     %wrap_cvGeneric_CvArr(CvMat, CvArr *, octfunc, double,
			   cvScale(self, retarg, scale),
			   cvCreateMat(self->rows, self->cols, self->type));
     %wrap_cvGeneric_CvArr(IplImage, CvArr *, octfunc, double,
			   cvScale(self, retarg, scale),
			   cvCreateImage(cvGetSize(self), self->depth, self->nChannels));
     %enddef

     /*M//////////////////////////////////////////////////////////////////////////////////////////
     // Actual Operator Declarations
     //////////////////////////////////////////////////////////////////////////////////////////M*/

     // Arithmetic operators 
     %wrap_cvArith(__radd__, cvAdd);

// special case for reverse operations
%wrap_cvArr_binaryop(__rsub__, CvArr *, cvSub(arg, self, retarg));
%wrap_cvArr_binaryop(__rdiv__, CvArr *, cvDiv(arg, self, retarg));
%wrap_cvArr_binaryop(__rmul__, CvArr *, cvMatMul(arg, self, retarg));

%wrap_cvArithS(__radd__, cvAddS);
%wrap_cvArithS(__rsub__, cvSubRS);


%wrap_cvScale(__rmul__, arg);
%wrap_cvScale(__rdiv__, 1.0/arg);

%wrap_cvLogicS(__ror__, cvOrS)
     %wrap_cvLogicS(__rand__, cvAndS)
     %wrap_cvLogicS(__rxor__, cvXorS)

     %wrap_cvCmpS(__req__, CV_CMP_EQ);
     %wrap_cvCmpS(__rgt__, CV_CMP_GT);
     %wrap_cvCmpS(__rge__, CV_CMP_GE);
     %wrap_cvCmpS(__rlt__, CV_CMP_LT);
     %wrap_cvCmpS(__rle__, CV_CMP_LE);
     %wrap_cvCmpS(__rne__, CV_CMP_NE);


     // misc operators for octave
     %wrap_cvArr_binaryop(__pow__, double, cvPow(self, retarg, arg))

     // TODO -- other Octave operators listed in SWIG/Octave docs (www.swig.org)

     // __abs__ -- cvAbs
     // __nonzero__
     // __hash__ ??
     // __repr__  -- full string representation
     // __str__  -- compact representation
     // __call__ -- ??
     // __len__ -- number of rows? or elements?
     // __iter__ -- ??
     // __contains__ -- cvCmpS, cvMax ?
     // __floordiv__ ??
     // __mul__ -- cvGEMM
     // __lshift__ -- ??
     // __rshift__ -- ??
     // __pow__ -- cvPow

     // Called to implement the unary arithmetic operations (-, +, abs() and ~). 
     //__neg__(  self)
     //__pos__(  self)
     //__abs__(  self)
     //__invert__(  self)

     // Called to implement the built-in functions complex(), int(), long(), and float(). Should return a value of the appropriate type.  Can I abuse this to return an array of the correct type??? scipy only allows return of length 1 arrays.
     // __complex__( self )
     // __int__( self )
     // __long__( self )
     // __float__( self )

     /*M//////////////////////////////////////////////////////////////////////////////////////////
     // Slice access and assignment for CvArr types
     //////////////////////////////////////////////////////////////////////////////////////////M*/

     // TODO: CvMatND

     %newobject CvMat::__paren(octave_value object);
     %newobject _IplImage::__paren(octave_value object);

     // Macro to check bounds of slice and throw error if outside
     %define CHECK_SLICE_BOUNDS(rect,w,h,retval)
     //printf("__setitem__ slice(%d:%d, %d:%d) array(%d,%d)", rect.x, rect.y, rect.x+rect.width, rect.y+rect.height, w, h);
     if(rect.width<=0 || rect.height<=0 ||
	rect.width>w || rect.height>h ||
	rect.x<0 || rect.y<0 ||
	rect.x>= w || rect.y >=h){
       char errstr[256];
       // previous function already set error string
       if(rect.width==0 && rect.height==0 && rect.x==0 && rect.y==0) return retval;
       sprintf(errstr, "Requested slice [ %d:%d %d:%d ] oversteps array sized [ %d %d ]", 
	       rect.x, rect.y, rect.x+rect.width, rect.y+rect.height, w, h);
       error(errstr);
       return retval;
     }
else{}
%enddef

// slice access and assignment for CvMat
%extend CvMat
{
  char * __str(){
    static char str[8];
    cvArrPrint( self );
    str[0]=0;
    return str;
    return "";
  }

  // scalar assignment
  void __paren_asgn(octave_value object, double val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    CHECK_SLICE_BOUNDS( subrect, self->cols, self->rows, );
    cvGetSubRect(self, &tmp, subrect);
    cvSet(&tmp, cvScalarAll(val));
  }
  void __paren_asgn(octave_value object, CvPoint val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    CHECK_SLICE_BOUNDS( subrect, self->cols, self->rows, );
    cvGetSubRect(self, &tmp, subrect);
    cvSet(&tmp, cvScalar(val.x, val.y));
  }
  void __paren_asgn(octave_value object, CvPoint2D32f val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    CHECK_SLICE_BOUNDS( subrect, self->cols, self->rows, );
    cvSet(&tmp, cvScalar(val.x, val.y));
  }
  void __paren_asgn(octave_value object, CvScalar val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    CHECK_SLICE_BOUNDS( subrect, self->cols, self->rows, );
    cvSet(&tmp, val);
  }

  // array slice assignment
  void __paren_asgn(octave_value object, CvArr * arr){
    CvMat tmp, src_stub, *src;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    CHECK_SLICE_BOUNDS( subrect, self->cols, self->rows, );
    cvGetSubRect(self, &tmp, subrect);

    // Reshape source array to fit destination
    // This will be used a lot for small arrays b/c
    // OctObject_to_CvArr tries to compress a 2-D octave
    // array with 1-4 columns into a multichannel vector
    src=cvReshape(arr, &src_stub, CV_MAT_CN(tmp.type), tmp.rows);

    cvConvert(src, &tmp);
  }

  // slice access
  octave_value __paren(octave_value object){
    CvMat * mat;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    CHECK_SLICE_BOUNDS( subrect, self->cols, self->rows, octave_value() );
    if(subrect.width==1 && subrect.height==1){
      CvScalar * s; 
      int type = cvGetElemType( self );
      if(CV_MAT_CN(type) > 1){
	s = new CvScalar; 
	*s = cvGet2D( self, subrect.y, subrect.x );
	return SWIG_NewPointerObj( s, $descriptor(CvScalar *), 1 );
      }
      switch(CV_MAT_DEPTH(type)){
      case CV_8U:
	return OctLong_FromUnsignedLong( CV_MAT_ELEM(*self, uchar, subrect.y, subrect.x ) );
      case CV_8S:
	return OctLong_FromLong( CV_MAT_ELEM(*self, char, subrect.y, subrect.x ) );
      case CV_16U:
	return OctLong_FromUnsignedLong( CV_MAT_ELEM(*self, ushort, subrect.y, subrect.x ) );
      case CV_16S:
	return OctLong_FromLong( CV_MAT_ELEM(*self, short, subrect.y, subrect.x ) );
      case CV_32S:
	return OctLong_FromLong( CV_MAT_ELEM(*self, int, subrect.y, subrect.x ) );
      case CV_32F:
	return OctFloat_FromDouble( CV_MAT_ELEM(*self, float, subrect.y, subrect.x) );
      case CV_64F:
	return OctFloat_FromDouble( CV_MAT_ELEM(*self, double, subrect.y, subrect.x) );
      }
    }
    mat = (CvMat *) cvAlloc(sizeof(CvMat));
    cvGetSubRect(self, mat, subrect);
		
    // cvGetSubRect doesn't do this since it assumes mat lives on the stack
    mat->hdr_refcount = self->hdr_refcount;
    mat->refcount = self->refcount;
    cvIncRefData(mat);

    return SWIG_NewPointerObj( mat, $descriptor(CvMat *), 1 );
  }

  /*
    %octavecode %{
    def __iter__(self):
    """
    generator function iterating through rows in matrix or elements in vector
    """
    if self.rows==1:
    return self.colrange()
    return self.rowrange()

    def rowrange(self):
    """
    generator function iterating along rows in matrix
    """
    for i in range(self.rows):
    yield self[i]

    def colrange(self):
    """
    generator function iterating along columns in matrix
    """
    for i in range(self.cols):
    yield self[:,i]

    # if arg is None, octave still calls our operator overloads
    # but we want
    # if mat != None
    # if mat == None
    # to do the right thing -- so redefine __ne__ and __eq__

    def __eq__(self, arg):
    """
    __eq__(self, None)
    __eq__(self, CvArr src)
    __eq__(self, double val)
    """

    if not arg:
    return False 
    return _cv.CvMat___eq__(self, arg)
    def __ne__(self, arg):
    """
    __ne__(self, None)
    __ne__(self, CvArr src)
    __ne__(self, double val)
    """

    if not arg:
    return True
    return _cv.CvMat___ne__(self, arg)
    %}
  */

}

// slice access and assignment for IplImage 
%extend _IplImage
{
  char * __str(){
    static char str[8];
    cvArrPrint( self );
    str[0]=0;
    return str;
    return "";
  }

  // scalar assignment
  void __paren_asgn(octave_value object, double val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    cvSet(&tmp, cvScalarAll(val));
  }
  void __paren_asgn(octave_value object, CvPoint val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    cvSet(&tmp, cvScalar(val.x, val.y));
  }
  void __paren_asgn(octave_value object, CvPoint2D32f val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    cvSet(&tmp, cvScalar(val.x, val.y));
  }
  void __paren_asgn(octave_value object, CvScalar val){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    cvSet(&tmp, val);
  }

  // array slice assignment
  void __paren_asgn(octave_value object, CvArr * arr){
    CvMat tmp;
    CvRect subrect = OctSlice_to_CvRect( self, object );
    cvGetSubRect(self, &tmp, subrect);
    cvConvert(arr, &tmp);
  }

  // slice access
  octave_value __paren(octave_value object){
    CvMat mat;
    IplImage * im;
    CvRect subrect = OctSlice_to_CvRect( self, object );

    // return scalar if single element
    if(subrect.width==1 && subrect.height==1){
      CvScalar * s;
      int type = cvGetElemType( self );
      if(CV_MAT_CN(type) > 1){
	s = new CvScalar;
	*s = cvGet2D( self, subrect.y, subrect.x );
	return SWIG_NewPointerObj( s, $descriptor(CvScalar *), 1 );
      }
      switch(CV_MAT_DEPTH(type)){
      case CV_8U:
	return OctLong_FromUnsignedLong( CV_IMAGE_ELEM(self, uchar, subrect.y, subrect.x ) );
      case CV_8S:
	return OctLong_FromLong( CV_IMAGE_ELEM(self, char, subrect.y, subrect.x ) );
      case CV_16U:
	return OctLong_FromUnsignedLong( CV_IMAGE_ELEM(self, ushort, subrect.y, subrect.x ) );
      case CV_16S:
	return OctLong_FromLong( CV_IMAGE_ELEM(self, short, subrect.y, subrect.x ) );
      case CV_32S:
	return OctLong_FromLong( CV_IMAGE_ELEM(self, int, subrect.y, subrect.x ) );
      case CV_32F:
	return OctFloat_FromDouble( CV_IMAGE_ELEM(self, float, subrect.y, subrect.x) );
      case CV_64F:
	return OctFloat_FromDouble( CV_IMAGE_ELEM(self, double, subrect.y, subrect.x) );
      }
    }

    // otherwise return array
    im = (IplImage *) cvAlloc(sizeof(IplImage));
    cvGetSubRect(self, &mat, subrect);
    im = cvGetImage(&mat, im);
    return SWIG_NewPointerObj( im, $descriptor(_IplImage *), 1 );
  }
}

