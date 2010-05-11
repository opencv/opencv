#include "pyhelpers.h"
#include <iostream>
#include <sstream>

int PySwigObject_Check(PyObject *op);

/* Py_ssize_t for old Pythons */
#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#endif

PyObject * PyTuple_FromIntArray(int * arr, int len){
  PyObject * obj = PyTuple_New(len);
  for(int i=0; i<len; i++){
    PyTuple_SetItem(obj, i, PyLong_FromLong( arr[i] ) );
  }
  return obj;
}

PyObject * SWIG_SetResult(PyObject * result, PyObject * obj){
  if(result){
    Py_DECREF(result);
  }
  result = PyTuple_New(1);
  PyTuple_SetItem(result, 0, obj);
  return result;
}

PyObject * SWIG_AppendResult(PyObject * result, PyObject ** to_add, int num){
  if ((!result) || (result == Py_None)) {
    /* no other results, so just add our values */

    /* if only one object, return that */
    if(num==1){
      return to_add[0];
    }
    
    /* create a new tuple to put in our new pointer python objects */
    result = PyTuple_New (num);

    /* put in our new pointer python objects */
    for(int i=0; i<num; i++){
      PyTuple_SetItem (result, i, to_add[i]);
    } 
  }
  else {
    /* we have other results, so add it to the end */

    if (!PyTuple_Check (result)) {
      /* previous result is not a tuple, so create one and put
         previous result and current pointer in it */

      /* first, save previous result */
      PyObject *obj_save = result;

      /* then, create the tuple */
      result = PyTuple_New (1);

      /* finaly, put the saved value in the tuple */
      PyTuple_SetItem (result, 0, obj_save);
    }

    /* create a new tuple to put in our new pointer python object */
    PyObject *my_obj = PyTuple_New (num);

    /* put in our new pointer python object */
    for( int i=0; i<num ; i++ ){
      PyTuple_SetItem (my_obj, i, to_add[i]);
    }

    /* save the previous result */
    PyObject *obj_save = result;

    /* concat previous and our new result */
    result = PySequence_Concat (obj_save, my_obj);

    /* decrement the usage of no more used objects */
    Py_DECREF (obj_save);
    Py_DECREF (my_obj);
  }
  return result;
}

template <typename T>
void cv_arr_write(FILE * f, const char * fmt, T * data, size_t rows, size_t nch, size_t step){
    size_t i,j,k;
    char * cdata = (char *) data;
    const char * chdelim1="", * chdelim2="";

    // only output channel parens if > 1
    if(nch>1){
        chdelim1="(";
        chdelim2=")";
    }

    fputs("[",f);
    for(i=0; i<rows; i++){
    fputs("[",f);

        // first element
        // out<<chdelim1;
    fputs(chdelim1, f);
        fprintf(f, fmt, ((T*)(cdata+i*step))[0]);
        for(k=1; k<nch; k++){
      fputs(", ", f);
      fprintf(f, fmt, ((T*)(cdata+i*step))[k]);
        }
    fputs(chdelim2,f);

        // remaining elements
        for(j=nch*sizeof(T); j<step; j+=(nch*sizeof(T))){
      fprintf(f, ",%s", chdelim1);
          fprintf(f, fmt, ((T*)(cdata+i*step+j))[0]);
            for(k=1; k<nch; k++){
        fputs(", ", f);
        fprintf(f, fmt, ((T*)(cdata+i*step+j))[k]);
            }
      fputs(chdelim2, f);
        }
    fputs( "]\n", f );
    }
  fputs( "]", f );
}

void cvArrPrint(CvArr * arr){
  CvMat * mat;
  CvMat stub;

  mat = cvGetMat(arr, &stub);
  
  int cn = CV_MAT_CN(mat->type);
  int depth = CV_MAT_DEPTH(mat->type);
  int step = MAX(mat->step, cn*mat->cols*CV_ELEM_SIZE(depth));


  switch(depth){
    case CV_8U:
      cv_arr_write(stdout, "%u", (uchar *)mat->data.ptr, mat->rows, cn, step);
      break;
    case CV_8S:
      cv_arr_write(stdout, "%d", (char *)mat->data.ptr, mat->rows, cn, step);
      break;
    case CV_16U:
      cv_arr_write(stdout, "%u", (ushort *)mat->data.ptr, mat->rows, cn, step);
      break;
    case CV_16S:
      cv_arr_write(stdout, "%d", (short *)mat->data.ptr, mat->rows, cn, step);
      break;
    case CV_32S:
      cv_arr_write(stdout, "%d", (int *)mat->data.ptr, mat->rows, cn, step);
      break;
    case CV_32F:
      cv_arr_write(stdout, "%f", (float *)mat->data.ptr, mat->rows, cn, step);
      break;
    case CV_64F:
      cv_arr_write(stdout, "%g", (double *)mat->data.ptr, mat->rows, cn, step);
      break;
    default:
      CV_Error( CV_StsError, "Unknown element type");
      break;
  }
}

// deal with negative array indices
int PyLong_AsIndex( PyObject * idx_object, int len ){
  int idx = PyLong_AsLong( idx_object );
  if(idx<0) return len+idx;
  return idx;
}

CvRect PySlice_to_CvRect(CvArr * src, PyObject * idx_object){
  CvSize sz = cvGetSize(src);
  //printf("Size %dx%d\n", sz.height, sz.width);
  int lower[2], upper[2];
  Py_ssize_t len, start, stop, step, slicelength;

  if(PyInt_Check(idx_object) || PyLong_Check(idx_object)){
    // if array is a row vector, assume index into columns
    if(sz.height>1){
      lower[0] = PyLong_AsIndex( idx_object, sz.height );
      upper[0] = lower[0] + 1;
      lower[1] = 0;
      upper[1] = sz.width;
    }
    else{
      lower[0] = 0;
      upper[0] = sz.height;
      lower[1] = PyLong_AsIndex( idx_object, sz.width );
      upper[1] = lower[1]+1;
    }
  }

  // 1. Slice
  else if(PySlice_Check(idx_object)){
    len = sz.height;
    if(PySlice_GetIndicesEx( (PySliceObject*)idx_object, len, &start, &stop, &step, &slicelength )!=0){
      printf("Error in PySlice_GetIndicesEx: returning NULL");
      PyErr_SetString(PyExc_Exception, "Error");
      return cvRect(0,0,0,0);
    }
    // if array is a row vector, assume index bounds are into columns
    if(sz.height>1){
      lower[0] = (int) start; // use c convention of start index = 0
      upper[0] = (int) stop;    // use c convention
      lower[1] = 0;
      upper[1] = sz.width;
    }
    else{
      lower[1] = (int) start; // use c convention of start index = 0
      upper[1] = (int) stop;    // use c convention
      lower[0] = 0;
      upper[0] = sz.height;
    }
  }

  // 2. Tuple
  else if(PyTuple_Check(idx_object)){
    //printf("PyTuple{\n");
    if(PyObject_Length(idx_object)!=2){
      //printf("Expected a sequence of length 2: returning NULL");
      PyErr_SetString(PyExc_ValueError, "Expected a sequence with 2 elements");
      return cvRect(0,0,0,0);
    }
    for(int i=0; i<2; i++){
      PyObject *o = PyTuple_GetItem(idx_object, i);

      // 2a. Slice -- same as above
      if(PySlice_Check(o)){
        //printf("PySlice\n");
        len = (i==0 ? sz.height : sz.width);
        if(PySlice_GetIndicesEx( (PySliceObject*)o, len, &start, &stop, &step, &slicelength )!=0){
          PyErr_SetString(PyExc_Exception, "Error");
          printf("Error in PySlice_GetIndicesEx: returning NULL");
          return cvRect(0,0,0,0);
        }
        //printf("PySlice_GetIndecesEx(%d, %d, %d, %d, %d)\n", len, start, stop, step, slicelength);
        lower[i] = start;
        upper[i] = stop;

      }

      // 2b. Integer
      else if(PyInt_Check(o) || PyLong_Check(o)){
        //printf("PyInt\n");
        lower[i] = PyLong_AsIndex(o, i==0 ? sz.height : sz.width);
        upper[i] = lower[i]+1;
      }

      else {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of slices or integers");
        printf("Expected a slice or int as sequence item: returning NULL");
        return cvRect(0,0,0,0);
      }
    }
  }

  else {
    PyErr_SetString( PyExc_TypeError, "Expected a slice or sequence");
    printf("Expected a slice or sequence: returning NULL");
    return cvRect(0,0,0,0);
  }

  //lower[0] = MAX(0, lower[0]);
  //lower[1] = MAX(0, lower[1]);
  //upper[0] = MIN(sz.height, upper[0]);
  //upper[1] = MIN(sz.width, upper[1]);
  //printf("Slice=%d %d %d %d\n", lower[0], upper[0], lower[1], upper[1]);
  return cvRect(lower[1],lower[0], upper[1]-lower[1], upper[0]-lower[0]);
}

int CheckSliceBounds(CvRect * rect, int w, int h){
	//printf("__setitem__ slice(%d:%d, %d:%d) array(%d,%d)", rect.x, rect.y, rect.x+rect.width, rect.y+rect.height, w, h);
	if(rect->width<=0 || rect->height<=0 ||
	   	rect->width>w || rect->height>h ||
	   	rect->x<0 || rect->y<0 ||
	   	rect->x>= w || rect->y >=h){
	   	char errstr[256];

		// previous function already set error string
		if(rect->width==0 && rect->height==0 && rect->x==0 && rect->y==0) return -1;

	   	sprintf(errstr, "Requested slice [ %d:%d %d:%d ] oversteps array sized [ %d %d ]", 
	   		rect->x, rect->y, rect->x+rect->width, rect->y+rect->height, w, h);
		PyErr_SetString(PyExc_IndexError, errstr);
		//PyErr_SetString(PyExc_ValueError, errstr);
		return 0;
	}
    return 1;
}

double PyObject_AsDouble(PyObject * obj){
  if(PyNumber_Check(obj)){
    if(PyFloat_Check(obj)){
      return PyFloat_AsDouble(obj);
    }
    else if(PyInt_Check(obj) || PyLong_Check(obj)){
      return (double) PyLong_AsLong(obj);
    }
  }
  PyErr_SetString( PyExc_TypeError, "Could not convert python object to Double");
  return -1;
}

long PyObject_AsLong(PyObject * obj){
    if(PyNumber_Check(obj)){
        if(PyFloat_Check(obj)){
            return (long) PyFloat_AsDouble(obj);
        }
        else if(PyInt_Check(obj) || PyLong_Check(obj)){
            return PyLong_AsLong(obj);
        }
    }
  PyErr_SetString( PyExc_TypeError, "Could not convert python object to Long");
  return -1;
}

CvArr * PyArray_to_CvArr (PyObject * obj)
{
  // let's try to create a temporary CvMat header that points to the
  // data owned by obj and reflects its memory layout
  
  CvArr * cvarr  = NULL;
  
  void * raw_data = 0;
  long   rows;
  long   cols;
  long   channels;
  long   step;
  long   mat_type     = 7;
  long   element_size = 1;
  
  // infer layout from array interface
  PyObject * interface = PyObject_GetAttrString (obj, "__array_interface__");
  
  
  // the array interface should be a dict
  if (PyMapping_Check (interface))
  {
    if (PyMapping_HasKeyString (interface, (char*)"version") &&
        PyMapping_HasKeyString (interface, (char*)"shape")   &&
        PyMapping_HasKeyString (interface, (char*)"typestr") &&
        PyMapping_HasKeyString (interface, (char*)"data"))
    {
      PyObject * version = PyMapping_GetItemString (interface, (char*)"version");
      PyObject * shape   = PyMapping_GetItemString (interface, (char*)"shape");
      PyObject * typestr = PyMapping_GetItemString (interface, (char*)"typestr");
      PyObject * data    = PyMapping_GetItemString (interface, (char*)"data");
      
      if (!PyInt_Check (version)  ||  PyInt_AsLong (version) != 3)
        PyErr_SetString(PyExc_TypeError, "OpenCV understands version 3 of the __array_interface__ only");
      else
      {
        if (!PyTuple_Check (shape)  ||  PyTuple_Size (shape) < 2  ||  PyTuple_Size (shape) > 3)
          PyErr_SetString(PyExc_TypeError, "arrays must have a shape with 2 or 3 dimensions");
        else
        {
          rows     = PyInt_AsLong (PyTuple_GetItem (shape, 0));
          cols     = PyInt_AsLong (PyTuple_GetItem (shape, 1));
          channels = PyTuple_Size (shape) < 3 ? 1 : PyInt_AsLong (PyTuple_GetItem (shape, 2));
          
          if (rows < 1  ||  cols < 1  ||  channels < 1  ||  channels > 4)
            PyErr_SetString(PyExc_TypeError, "rows and columns must be positive, channels from 1 to 4");
          else
          {
//              fprintf (stderr, "rows: %ld, cols: %ld, channels %ld\n", rows, cols, channels); fflush (stderr);
            
            if (! PyTuple_Check (data)  ||  PyTuple_Size (data) != 2  ||  
                !(PyInt_Check (PyTuple_GetItem (data,0)) || PyLong_Check (PyTuple_GetItem (data,0))) ||
                !(PyBool_Check (PyTuple_GetItem (data,1)) && !PyInt_AsLong (PyTuple_GetItem (data,1))))
              PyErr_SetString (PyExc_TypeError, "arrays must have a pointer to writeable data");
            else
            {
              raw_data = PyLong_AsVoidPtr (PyTuple_GetItem (data,0));
//                fprintf(stderr, "raw_data: %p\n", raw_data); fflush (stderr);
              
              char *      format_str = NULL;
              Py_ssize_t  len        = 0;
              
              if (!PyString_Check (typestr)  ||  PyString_AsStringAndSize (typestr, & format_str, &len) == -1  ||  len !=3)
                PyErr_SetString(PyExc_TypeError, "there is something wrong with the format string");
              else
              {
//                fprintf(stderr, "format: %c %c\n", format_str[1], format_str[2]); fflush (stderr);
              
                if      (format_str[1] == 'u'  &&  format_str[2] == '1')
                {
                  element_size = 1;
                  mat_type     = CV_MAKETYPE(CV_8U, channels);
                }
                else if (format_str[1] == 'i'  &&  format_str[2] == '1')
                {
                  element_size = 1;
                  mat_type     = CV_MAKETYPE(CV_8S, channels);
                }
                else if (format_str[1] == 'u'  &&  format_str[2] == '2')
                {
                  element_size = 2;
                  mat_type     = CV_MAKETYPE(CV_16U, channels);
                }
                else if (format_str[1] == 'i'  &&  format_str[2] == '2')
                {
                  element_size = 2;
                  mat_type     = CV_MAKETYPE(CV_16S, channels);
                }
                else if (format_str[1] == 'i'  &&  format_str[2] == '4')
                {
                  element_size = 4;
                  mat_type     = CV_MAKETYPE(CV_32S, channels);
                }
                else if (format_str[1] == 'f'  &&  format_str[2] == '4')
                {
                  element_size = 4;
                  mat_type     = CV_MAKETYPE(CV_32F, channels);
                }
                else if (format_str[1] == 'f'  &&  format_str[2] == '8')
                {
                  element_size = 8;
                  mat_type     = CV_MAKETYPE(CV_64F, channels);
                }
                else
                {
                  PyErr_SetString(PyExc_TypeError, "unknown or unhandled element format");
                  mat_type     = CV_USRTYPE1;
                }
                
                // handle strides if given
                // TODO: implement stride handling
                step = cols * channels * element_size;
                if (PyMapping_HasKeyString (interface, (char*)"strides"))
                {
                  PyObject * strides = PyMapping_GetItemString (interface, (char*)"strides");
                  
                  if (strides != Py_None)
                  {
                    fprintf(stderr, "we have strides ... not handled!\n"); fflush (stderr);
                    PyErr_SetString(PyExc_TypeError, "arrays with strides not handled yet");
                    mat_type = CV_USRTYPE1; // use this to denote, we've got an error
                  }
                  
                  Py_DECREF (strides);
                }
                
                // create matrix header if everything is okay
                if (mat_type != CV_USRTYPE1)
                {
                  CvMat * temp_matrix = cvCreateMatHeader (rows, cols, mat_type);
                  cvSetData (temp_matrix, raw_data, step);
                  cvarr = temp_matrix;
                  
//                    fprintf(stderr, "step_size: %ld, type: %ld\n", step, mat_type); fflush (stderr);
                }
              }
            }
          }
        }
      }
      
      Py_DECREF (data);
      Py_DECREF (typestr);
      Py_DECREF (shape);
      Py_DECREF (version);
    }
  
  }
  
  Py_DECREF (interface);
  
  return cvarr;
}


// Convert Python lists to CvMat *
CvArr * PySequence_to_CvArr (PyObject * obj)
{
  int        dims     [CV_MAX_DIM]   = {   1,    1,    1};
  PyObject * container[CV_MAX_DIM+1] = {NULL, NULL, NULL, NULL};
  int        ndim                    = 0;
  PyObject * item                    = Py_None;
  
  // TODO: implement type detection - currently we create CV_64F only
  // scan full array to
  // - figure out dimensions
  // - check consistency of dimensions
  // - find appropriate data-type and signedness
  //  enum NEEDED_DATATYPE { NEEDS_CHAR, NEEDS_INTEGER, NEEDS_FLOAT, NEEDS_DOUBLE };
  //  NEEDED_DATATYPE needed_datatype = NEEDS_CHAR;
  //  bool            needs_sign      = false;
  
  // scan first entries to find out dimensions
  for (item = obj, ndim = 0; PySequence_Check (item) && ndim <= CV_MAX_DIM; ndim++)
  {
    dims [ndim]      = PySequence_Size    (item);
    container [ndim] = PySequence_GetItem (item, 0); 
    item             = container[ndim];
  }
  
  // in contrast to PyTuple_GetItem, PySequence_GetItame returns a NEW reference
  if (container[0])
  {
    Py_DECREF (container[0]);
  }
  if (container[1])
  {
    Py_DECREF (container[1]);
  }
  if (container[2])
  {
    Py_DECREF (container[2]);
  }
  if (container[3])
  {
    Py_DECREF (container[3]);
  }
  
  // it only makes sense to support 2 and 3 dimensional data at this time
  if (ndim < 2 || ndim > 3)
  {
    PyErr_SetString (PyExc_TypeError, "Nested sequences should have 2 or 3 dimensions");
    return NULL;
  }
  
  // also, the number of channels should match what's typical for OpenCV
  if (ndim == 3  &&  (dims[2] < 1  ||  dims[2] > 4))
  {
    PyErr_SetString (PyExc_TypeError, "Currently, the third dimension of CvMat only supports 1 to 4 channels");
    return NULL;
  }
  
  // CvMat
  CvMat * matrix = cvCreateMat (dims[0], dims[1], CV_MAKETYPE (CV_64F, dims[2]));
  
  for (int y = 0; y < dims[0]; y++)
  {
    PyObject * rowobj = PySequence_GetItem (obj, y);
    
    // double check size
    if (PySequence_Check (rowobj)  &&  PySequence_Size (rowobj) == dims[1])
    {
      for (int x = 0; x < dims[1]; x++)
      {
        PyObject * colobj = PySequence_GetItem (rowobj, x);
        
        if (dims [2] > 1)
        {
          if (PySequence_Check (colobj)  &&  PySequence_Size (colobj) == dims[2])
          {
            PyObject * tuple = PySequence_Tuple (colobj);
            
            double  a, b, c, d;
            if (PyArg_ParseTuple (colobj, "d|d|d|d", &a, &b, &c, &d))
            {
              cvSet2D (matrix, y, x, cvScalar (a, b, c, d));
            }
            else 
            {
              PyErr_SetString (PyExc_TypeError, "OpenCV only accepts numbers that can be converted to float");
              cvReleaseMat (& matrix);
              Py_DECREF (tuple);
              Py_DECREF (colobj);
              Py_DECREF (rowobj);
              return NULL;
            }

            Py_DECREF (tuple);
          }
          else
          {
            PyErr_SetString (PyExc_TypeError, "All sub-sequences must have the same number of entries");
            cvReleaseMat (& matrix);
            Py_DECREF (colobj);
            Py_DECREF (rowobj);
            return NULL;
          }
        }
        else
        {
          if (PyFloat_Check (colobj) || PyInt_Check (colobj))
          {
            cvmSet (matrix, y, x, PyFloat_AsDouble (colobj));
          }
          else
          {
            PyErr_SetString (PyExc_TypeError, "OpenCV only accepts numbers that can be converted to float");
            cvReleaseMat (& matrix);
            Py_DECREF (colobj);
            Py_DECREF (rowobj);
            return NULL;
          }
        }
        
        Py_DECREF (colobj);
      }
    }
    else
    {
      PyErr_SetString (PyExc_TypeError, "All sub-sequences must have the same number of entries");
      cvReleaseMat (& matrix);
      Py_DECREF (rowobj);
      return NULL;
    }
    
    Py_DECREF (rowobj);
  }
  
  return matrix;
}
