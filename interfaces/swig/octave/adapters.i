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

// 2008-04-07, Xavier Delacour <xavier.delacour@gmail.com>


// The trouble here is that Octave arrays are in Fortran order, while OpenCV 
// arrays are in C order. Neither Octave nor OpenCV seem to provide n-dim
// transpose, so we do that here.

// For images, we also scale the result to lie within [0-1].


// * add support for sparse matrices

// * add support for complex matrices

// * add support for roi and coi, or complain if either is set

// * test case for channel==1
// * test case for channel=={2,3,4}
// * test case for 2 dim, 1 dim, n dim cases

%{

class ndim_iterator {
  int nd;
  int dims[CV_MAX_DIM];
  int step[CV_MAX_DIM];
  int curr[CV_MAX_DIM];
  uchar* _data;
  int _type;
  bool done;
 public:
  ndim_iterator() {}
  ndim_iterator(CvMat* m) {
    int c = CV_MAT_CN(m->type);
    int elem_size = CV_ELEM_SIZE1(m->type);
    nd = c == 1 ? 2 : 3;
    dims[0] = m->rows;
    dims[1] = m->cols;
    dims[2] = c;
    step[0] = m->step;
    step[1] = c * elem_size;
    step[2] = elem_size;
    curr[0] = curr[1] = curr[2] = 0;
    _data = m->data.ptr;
    _type = m->type;
    done = false;
  }
  ndim_iterator(CvMatND* m) {
    int c = CV_MAT_CN(m->type);
    int elem_size = CV_ELEM_SIZE1(m->type);
    nd = m->dims + (c == 1 ? 0 : 1);
    for (int j = 0; j < m->dims; ++j) {
      dims[j] = m->dim[j].size;
      step[j] = m->dim[j].step;
      curr[j] = 0;
    }
    if (c > 1) {
      dims[m->dims] = c;
      step[m->dims] = elem_size;
      curr[m->dims] = 0;
    }
    _data = m->data.ptr;
    _type = m->type;
    done = false;
  }
  ndim_iterator(IplImage* img) {
    nd = img->nChannels == 1 ? 2 : 3;
    dims[0] = img->height;
    dims[1] = img->width;
    dims[2] = img->nChannels;

    switch (img->depth) {
    case IPL_DEPTH_8U: _type = CV_8U; break;
    case IPL_DEPTH_8S: _type = CV_8S; break;
    case IPL_DEPTH_16U: _type = CV_16U; break;
    case IPL_DEPTH_16S: _type = CV_16S; break;
    case IPL_DEPTH_32S: _type = CV_32S; break;
    case IPL_DEPTH_32F: _type = CV_32F; break;
    case IPL_DEPTH_1U: _type = CV_64F; break;
    default:
      error("unsupported image depth");
      return;
    }

    int elem_size = CV_ELEM_SIZE1(_type);
    step[0] = img->widthStep;
    step[1] = img->nChannels * elem_size;
    step[2] = elem_size;
    curr[0] = curr[1] = curr[2] = 0;
    _data = (uchar*)img->imageData;
    done = false;
  }
  ndim_iterator(NDArray& nda) {
    dim_vector d(nda.dims());
    nd = d.length();
    int last_step = sizeof(double);
    for (int j = 0; j < d.length(); ++j) {
      dims[j] = d(j);
      step[j] = last_step;
      last_step *= dims[j];
      curr[j] = 0;
    }
    _data = (uchar*)const_cast<double*>(nda.data());
    _type = CV_64F;
    done = false;
  }

  operator bool () const {
    return !done;
  }
  uchar* data() {
    return _data;
  }
  int type() const {
    return _type;
  }
  ndim_iterator& operator++ () {
    int curr_dim = 0;
    for (;;) {
      _data += step[curr_dim];
      if (++curr[curr_dim] < dims[curr_dim])
	break;
      curr[curr_dim] = 0;
      _data -= step[curr_dim] * dims[curr_dim];
      ++curr_dim;
      if (curr_dim == nd) {
	done = true;
	break;
      }
    }
    return *this;
  }
};

template <class T1, class T2>
  void transpose_copy_typed(ndim_iterator src_it, ndim_iterator dst_it, 
			   double scale) {
  assert(sizeof(T1) == CV_ELEM_SIZE1(src_it.type()));
  assert(sizeof(T2) == CV_ELEM_SIZE1(dst_it.type()));
  if (scale == 1) {
    while (src_it) {
      *(T2*)dst_it.data() = (T2)*(T1*)src_it.data();
      ++src_it;
      ++dst_it;
    }
  } else {
    while (src_it) {
      *(T2*)dst_it.data() = (T2)(scale * (*(T1*)src_it.data()));
      ++src_it;
      ++dst_it;
    }
  }
}

template <class T1>
void transpose_copy2(ndim_iterator src_it, ndim_iterator dst_it, 
		     double scale) {
  switch (CV_MAT_DEPTH(dst_it.type())) {
  case CV_8U: transpose_copy_typed<T1,unsigned char>(src_it,dst_it,scale); break;
  case CV_8S: transpose_copy_typed<T1,signed char>(src_it,dst_it,scale); break;
  case CV_16U: transpose_copy_typed<T1,unsigned short>(src_it,dst_it,scale); break;
  case CV_16S: transpose_copy_typed<T1,signed short>(src_it,dst_it,scale); break;
  case CV_32S: transpose_copy_typed<T1,signed int>(src_it,dst_it,scale); break;
  case CV_32F: transpose_copy_typed<T1,float>(src_it,dst_it,scale); break;
  case CV_64F: transpose_copy_typed<T1,double>(src_it,dst_it,scale); break;
  default:
    error("unsupported dest array type (supported types are CV_8U, CV_8S, "
	  "CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)");
  }
}

void transpose_copy(ndim_iterator src_it, ndim_iterator dst_it, 
		    double scale = 1) {
  switch (CV_MAT_DEPTH(src_it.type())) {
  case CV_8U: transpose_copy2<unsigned char>(src_it,dst_it,scale); break;
  case CV_8S: transpose_copy2<signed char>(src_it,dst_it,scale); break;
  case CV_16U: transpose_copy2<unsigned short>(src_it,dst_it,scale); break;
  case CV_16S: transpose_copy2<signed short>(src_it,dst_it,scale); break;
  case CV_32S: transpose_copy2<signed int>(src_it,dst_it,scale); break;
  case CV_32F: transpose_copy2<float>(src_it,dst_it,scale); break;
  case CV_64F: transpose_copy2<double>(src_it,dst_it,scale); break;
  default:
    error("unsupported source array type (supported types are CV_8U, CV_8S, "
	  "CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)");
  }
}

octave_value cv2mat(CvArr* arr) {
  dim_vector d;
  NDArray nda;

  if (CV_IS_MAT(arr)) {
    // m x n x c
    CvMat* m = (CvMat*)arr;

    int c = CV_MAT_CN(m->type);
    if (c == 1) {
      d.resize(2);
      d(0) = m->rows;
      d(1) = m->cols;
    } else {
      d.resize(3);
      d(0) = m->rows;
      d(1) = m->cols;
      d(2) = c;
    }

    nda = NDArray(d);
    transpose_copy(m, nda);
  }
  else if (CV_IS_MATND(arr)) {
    // m1 x m2 x ... x mn x c
    CvMatND* m = (CvMatND*)arr;

    int c = CV_MAT_CN(m->type);
    if (c == 1) {
      d.resize(m->dims);
      for (int j = 0; j < m->dims; ++j)
	d(j) = m->dim[j].size;
    } else {
      d.resize(m->dims + 1);
      for (int j = 0; j < m->dims; ++j)
	d(j) = m->dim[j].size;
      d(m->dims) = c;
    }

    nda = NDArray(d);
    transpose_copy(m, nda);
  }
  else if (CV_IS_IMAGE(arr)) {
    // m x n x c
    IplImage* img = (IplImage*)arr;

    if (img->nChannels == 1) {
      d.resize(2);
      d(0) = img->height;
      d(1) = img->width;
    } else {
      d.resize(3);
      d(0) = img->height;
      d(1) = img->width;
      d(2) = img->nChannels;
    }

    nda = NDArray(d);
    transpose_copy(img, nda);
  }
  else {
    error("unsupported array type (supported types are CvMat, CvMatND, IplImage)");
    return octave_value();
  }

  return nda;
}

octave_value mat2cv(const octave_value& ov, int type) {
  NDArray nda(ov.array_value());
  if (error_state)
    return 0;

  dim_vector d = ov.dims();
  assert(d.length() > 0);

  int nd = d.length();
  int last_dim = d(d.length() - 1);
  int c = CV_MAT_CN(type);
  if (c != 1 && c != last_dim) {
    error("last dimension and channel must agree, or channel must equal one");
    return 0;
  }
  if (c > 1)
    --nd;

  if (nd == 2) {
    CvMat *m = cvCreateMat(d(0), d(1), type);
    transpose_copy(nda, m);
    return SWIG_NewPointerObj(m, SWIGTYPE_p_CvMat, SWIG_POINTER_OWN);
  }
  else {
    int tmp[CV_MAX_DIM];
    for (int j = 0; j < nd; ++j)
      tmp[j] = d(j);
    CvMatND *m = cvCreateMatND(nd, tmp, type);
    transpose_copy(nda, m);
    return SWIG_NewPointerObj(m, SWIGTYPE_p_CvMatND, SWIG_POINTER_OWN);
  }
}

octave_value cv2im(CvArr* arr) {
  if (!CV_IS_IMAGE(arr) && !CV_IS_MAT(arr)) {
    error("input is not an OpenCV image or 2D matrix");
    return octave_value();
  }

  dim_vector d;
  NDArray nda;

  if (CV_IS_MAT(arr)) {
    // m x n x c
    CvMat* m = (CvMat*)arr;

    int c = CV_MAT_CN(m->type);
    if (c == 1) {
      d.resize(2);
      d(0) = m->rows;
      d(1) = m->cols;
    } else {
      d.resize(3);
      d(0) = m->rows;
      d(1) = m->cols;
      d(2) = c;
    }

    nda = NDArray(d);
    transpose_copy(m, nda, 1/256.0);
  }
  else if (CV_IS_IMAGE(arr)) {
    // m x n x c
    IplImage* img = (IplImage*)arr;

    if (img->nChannels == 1) {
      d.resize(2);
      d(0) = img->height;
      d(1) = img->width;
    } else {
      d.resize(3);
      d(0) = img->height;
      d(1) = img->width;
      d(2) = img->nChannels;
    }

    nda = NDArray(d);
    transpose_copy(img, nda, 1/256.0);
  }

  return nda;
}

CvMat* im2cv(const octave_value& ov, int depth) {
  NDArray nda(ov.array_value());
  if (error_state)
    return 0;

  dim_vector d = ov.dims();
  assert(d.length() > 0);

  if (d.length() != 2 && d.length() != 3 && 
      !(d.length() == 3 && d(2) <= 4)) {
    error("input must be m x n or m x n x c matrix, where 1<=c<=4");
    return 0;
  }

  int channels = d.length() == 2 ? 1 : d(2);
  int type = CV_MAKETYPE(depth, channels);

  CvMat *m = cvCreateMat(d(0), d(1), type);
  transpose_copy(nda, m, 256);

  return m;
}

%}

%newobject im2cv;

%feature("autodoc", 0) cv2mat;
%feature("autodoc", 0) mat2cv;
%feature("autodoc", 0) cv2im;
%feature("autodoc", 0) im2cv;

%docstring cv2mat {
@deftypefn {Loadable Function} @var{m1} = cv2mat (@var{m2})
Convert the CvMat, CvMatND, or IplImage @var{m2} into an Octave real matrix @var{m1}.
The dimensions @var{m1} are those of @var{m2}, plus an addition dimension 
if @var{m2} has more than one channel.
@end deftypefn
}

%docstring mat2cv {
@deftypefn {Loadable Function} @var{m1} = mat2cv (@var{m2}, @var{type})
Convert the Octave array @var{m2} into either a CvMat or a CvMatND of type
@var{type}.
@var{type} is one of CV_8UC(n), CV_8SC(n), CV_16UC(n), CV_16SC(n), CV_32SC(n), 
CV_32FC(n), CV_64FC(n), where n indicates channel and is between 1 and 4.
If the dimension of @var{m2} is equal to 2 (not counting channels),
a CvMat is returned. Otherwise, a CvMatND is returned.
@end deftypefn
}

%docstring cv2im {
@deftypefn {Loadable Function} @var{im} = cv2im (@var{I})
Convert the OpenCV image or 2D matrix @var{I} into an Octave image @var{im}.
@var{im} is a real matrix of dimension height x width or 
height x width x channels, with values within the interval [0,1]).
@end deftypefn
}

%docstring im2cv {
@deftypefn {Loadable Function} @var{I} = im2cv (@var{im}, @var{depth})
Convert the Octave image @var{im} into the OpenCV image @var{I} of depth
@var{depth}.
@var{im} is a real matrix of dimension height x width or 
height x width x channels, with values within the interval [0,1].
@var{depth} must be one of CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.
@end deftypefn
}

octave_value cv2mat(CvArr* arr);
octave_value mat2cv(const octave_value& ov, int type);
octave_value cv2im(CvArr* arr);
CvMat* im2cv(const octave_value& ov, int depth);
