/* These functions need the SWIG_* functions defined in the wrapper */
%{

#include "pyhelpers.h"

static CvArr * PyObject_to_CvArr(PyObject * obj, bool * freearg);

// convert a python sequence/array/list object into a c-array
#define PyObject_AsArrayImpl(func, ctype, ptype)                              \
  int func(PyObject * obj, ctype * array, int len){                         \
  void * mat_vptr=NULL;                                                     \
  void * im_vptr=NULL;                                                      \
  if(PyNumber_Check(obj)){                                                  \
    memset( array, 0, sizeof(ctype)*len );                                \
    array[0] = PyObject_As##ptype( obj );                                 \
  }                                                                         \
  else if(PyList_Check(obj) || PyTuple_Check(obj)){                         \
    int seqsize = PySequence_Size(obj);                                   \
    for(int i=0; i<len && i<seqsize; i++){                                \
      if(i<seqsize){                                                    \
              array[i] =  PyObject_As##ptype( PySequence_GetItem(obj, i) ); \
      }                                                                 \
      else{                                                             \
        array[i] = 0;                                                 \
      }                                                                 \
    }                                                                     \
  }                                                                         \
  else if( SWIG_ConvertPtr(obj, &mat_vptr, SWIGTYPE_p_CvMat, 0)!=-1 ||      \
           SWIG_ConvertPtr(obj, &im_vptr, SWIGTYPE_p__IplImage, 0)!=-1)     \
  {                                                                         \
    CvMat * mat = (CvMat *) mat_vptr;                                     \
    CvMat stub;                                                           \
    if(im_vptr) mat = cvGetMat(im_vptr, &stub);                           \
    if( mat->rows!=1 && mat->cols!=1 ){                                   \
      PyErr_SetString( PyExc_TypeError,                                 \
           "PyObject_As*Array: CvArr must be row or column vector" );   \
      return -1;                                                        \
    }                                                                     \
    if( mat->rows==1 && mat->cols==1 ){                                   \
      CvScalar val;                                                     \
      if( len!=CV_MAT_CN(mat->type) ){                                  \
        PyErr_SetString( PyExc_TypeError,                             \
        "PyObject_As*Array: CvArr channels != length" );              \
        return -1;                                                    \
      }                                                                 \
      val = cvGet1D(mat, 0);                                            \
      for(int i=0; i<len; i++){                                         \
        array[i] = (ctype) val.val[i];                                \
      }                                                                 \
    }                                                                     \
    else{                                                                 \
      mat = cvReshape(mat, &stub, -1, mat->rows*mat->cols);             \
      if( mat->rows != len ){                                           \
        PyErr_SetString( PyExc_TypeError,                             \
         "PyObject_As*Array: CvArr rows or cols must equal length" ); \
         return -1;                                                   \
      }                                                                 \
      for(int i=0; i<len; i++){                                         \
        CvScalar val = cvGet1D(mat, i);                               \
        array[i] = (ctype) val.val[0];                                \
      }                                                                 \
    }                                                                     \
  }                                                                         \
  else{                                                                     \
    PyErr_SetString( PyExc_TypeError,                                     \
        "PyObject_As*Array: Expected a number, sequence or CvArr" );  \
    return -1;                                                            \
  }                                                                         \
  return 0;                                                                 \
}

PyObject_AsArrayImpl( PyObject_AsFloatArray, float, Double );
PyObject_AsArrayImpl( PyObject_AsDoubleArray, double, Double );
PyObject_AsArrayImpl( PyObject_AsLongArray, int, Long );

static CvPoint PyObject_to_CvPoint(PyObject * obj){
  CvPoint val;
  CvPoint *ptr;
  CvPoint2D32f * ptr2D32f;
  CvScalar * scalar;

  if( SWIG_ConvertPtr(obj, (void**)&ptr, SWIGTYPE_p_CvPoint, 0) != -1) {
    return *ptr;
  }
  if( SWIG_ConvertPtr(obj, (void**)&ptr2D32f, SWIGTYPE_p_CvPoint2D32f, 0) != -1) {
    return cvPointFrom32f( *ptr2D32f );
  }
  if( SWIG_ConvertPtr(obj, (void**)&scalar, SWIGTYPE_p_CvScalar, 0) != -1) {
    return cvPointFrom32f(cvPoint2D32f( scalar->val[0], scalar->val[1] ));
  }
  if(PyObject_AsLongArray(obj, (int *) &val, 2) != -1){
    return val;
  }

  PyErr_SetString( PyExc_TypeError, "could not convert to CvPoint");
  return cvPoint(0,0);
}

static CvPoint2D32f PyObject_to_CvPoint2D32f(PyObject * obj){
    CvPoint2D32f val;
    CvPoint2D32f *ptr2D32f;
  CvPoint *ptr;
  CvScalar * scalar;
    if( SWIG_ConvertPtr(obj, (void**)&ptr2D32f, SWIGTYPE_p_CvPoint2D32f, 0) != -1) {
    return *ptr2D32f;
  }
  if( SWIG_ConvertPtr(obj, (void**)&ptr, SWIGTYPE_p_CvPoint, 0) != -1) {
    return cvPointTo32f(*ptr);
  }
  if( SWIG_ConvertPtr(obj, (void**)&scalar, SWIGTYPE_p_CvScalar, 0) != -1) {
    return cvPoint2D32f( scalar->val[0], scalar->val[1] );
  }
  if(PyObject_AsFloatArray(obj, (float *) &val, 2) != -1){
    return val;
  }
  PyErr_SetString(PyExc_TypeError, "could not convert to CvPoint2D32f");
  return cvPoint2D32f(0,0);
}

/* Check if this object can be interpreted as a CvScalar */
static bool CvScalar_Check(PyObject * obj){
  void * vptr;
    CvScalar val;
  return SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvScalar,     0 ) != -1 ||
         SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint2D32f, 0 ) != -1 ||
           SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint,      0 ) != -1 ||
         PyObject_AsDoubleArray(obj, val.val, 4) !=-1;
}

static CvScalar PyObject_to_CvScalar(PyObject * obj){
  CvScalar val;
  CvScalar * ptr;
    CvPoint2D32f *ptr2D32f;
  CvPoint *pt_ptr;
  void * vptr;
  if( SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvScalar, 0 ) != -1)
  {
    ptr = (CvScalar *) vptr;
    return *ptr;
  }
  if( SWIG_ConvertPtr(obj, (void**)&ptr2D32f, SWIGTYPE_p_CvPoint2D32f, 0) != -1) {
        return cvScalar(ptr2D32f->x, ptr2D32f->y);
    }
    if( SWIG_ConvertPtr(obj, (void**)&pt_ptr, SWIGTYPE_p_CvPoint, 0) != -1) {
        return cvScalar(pt_ptr->x, pt_ptr->y);
    }
  if(PyObject_AsDoubleArray(obj, val.val, 4)!=-1){
    return val;
  }
  return cvScalar(-1,-1,-1,-1); 
}

static int CvArr_Check( PyObject * obj )
{
	void *ptr;
	if( obj == Py_None ||
	    SWIG_IsOK( SWIG_ConvertPtr(obj, &ptr, SWIGTYPE_p_void,       0) ) ||
	    SWIG_IsOK( SWIG_ConvertPtr(obj, &ptr, SWIGTYPE_p_CvMat,       0) ) ||
        SWIG_IsOK( SWIG_ConvertPtr(obj, &ptr, SWIGTYPE_p_CvSeq,       0) ) ||
        SWIG_IsOK( SWIG_ConvertPtr(obj, &ptr, SWIGTYPE_p_CvContour,   0) ) ||
        SWIG_IsOK( SWIG_ConvertPtr(obj, &ptr, SWIGTYPE_p_CvSparseMat, 0) ) ||
        SWIG_IsOK( SWIG_ConvertPtr(obj, &ptr, SWIGTYPE_p_CvMatND,     0) ) ||
        PyObject_HasAttrString(obj, "__array_interface__") ||
        PySequence_Check(obj) ) 
    { 
        return 1;
	}
    PyErr_Clear();
    return 0;
}

/* if python sequence type, convert to CvMat or CvMatND */
static CvArr * PyObject_to_CvArr (PyObject * obj, bool * freearg)
{
  CvArr * cvarr = NULL;
  *freearg = false;

  if ( obj == Py_None )
  {
    // Interpret None as NULL pointer 
    return NULL;
  }
  else if( SWIG_IsOK( SWIG_ConvertPtr(obj, (void **)& cvarr, SWIGTYPE_p_void,       0) ) ||
      SWIG_IsOK( SWIG_ConvertPtr (obj, (void** )& cvarr, SWIGTYPE_p_CvMat, 0) ) ||
      SWIG_IsOK( SWIG_ConvertPtr (obj, (void **)& cvarr, SWIGTYPE_p_CvSeq, 0) ) ||
      SWIG_IsOK( SWIG_ConvertPtr (obj, (void **)& cvarr, SWIGTYPE_p_CvContour, 0) ) ||
      SWIG_IsOK( SWIG_ConvertPtr (obj, (void **)& cvarr, SWIGTYPE_p_CvSparseMat, 0) ) ||
      SWIG_IsOK( SWIG_ConvertPtr (obj, (void **)& cvarr, SWIGTYPE_p_CvMatND, 0) ))
  {
    // we got a directly wrapped void * pointer, OpenCV array or sequence type
    return cvarr;
  }
  else if (PyObject_HasAttrString (obj, "__array_interface__"))
  {
    // if we didn't get our own datatype, let's see if it supports the array protocol
    // array protocol is great because we just have to create another header but can
    // use the original data without copying
    cvarr = PyArray_to_CvArr (obj);
    *freearg = (cvarr != NULL);
  }
  else if (PySequence_Check (obj))
  {
    // our next bet is a tuple or list of tuples or lists this has to be copied over, however
    cvarr = PySequence_to_CvArr (obj);
    *freearg = (cvarr != NULL);
  }
  else if (PyLong_Check (obj) && PyLong_AsLong (obj) == 0)
  {
    // Interpret a '0' integer as a NULL pointer
    * freearg = false;
    return NULL;
  }
  else 
  {
    // TODO, throw an error here
    return NULL;
  }
  
  return cvarr;
}


static int PyObject_GetElemType(PyObject * obj){
  void *vptr;
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint, 0) != -1) return CV_32SC2; 
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvSize, 0) != -1) return CV_32SC2;  
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvRect, 0) != -1) return CV_32SC4;  
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvSize2D32f, 0) != -1) return CV_32FC2; 
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint2D32f, 0) != -1) return CV_32FC2;  
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint3D32f, 0) != -1) return CV_32FC3;  
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint2D64f, 0) != -1) return CV_64FC2;  
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint3D64f, 0) != -1) return CV_64FC3;  
  if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvScalar, 0) != -1) return CV_64FC4;  
  if(PyTuple_Check(obj) || PyList_Check(obj)) return CV_MAKE_TYPE(CV_32F, PySequence_Size( obj ));
  if(PyLong_Check(obj)) return CV_32S;
  return CV_32F;
}

%}
