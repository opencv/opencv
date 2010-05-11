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
%include "exception.i"
%include "./pyhelpers.i"

%typemap(in) (CvArr *) (bool freearg=false) 
{
  $1 = PyObject_to_CvArr($input, &freearg);
}

%typemap(freearg) (CvArr *) 
{
  if($1!=NULL && freearg$argnum)
  {
    cvReleaseData( $1 );
    cvFree(&($1));
  }
}

%typemap(in) CvMat* (bool freearg=false), const CvMat* (bool freearg=false)
{
  $1 = (CvMat*)PyObject_to_CvArr($input, &freearg);
}

%typemap(freearg) CvMat*,const CvMat* {
  if($1!=NULL && freearg$argnum){
    cvReleaseData( $1 );
    cvFree(&($1));
  }
}

/* typecheck typemaps */
%typecheck(SWIG_TYPECHECK_POINTER) CvArr * {
    $1 = CvArr_Check( $input );
}

%typecheck(SWIG_TYPECHECK_POINTER) CvScalar {
    $1 = CvScalar_Check( $input );
}

/* copy built-in swig typemaps for these types */
%typemap(typecheck) CvPoint = SWIGTYPE;
%typemap(typecheck) CvPoint2D32f = SWIGTYPE;
%typemap(typecheck) CvPoint3D32f = SWIGTYPE;
%typemap(typecheck) CvPoint2D64f = SWIGTYPE;
%typemap(typecheck) CvPoint3D64f = SWIGTYPE;
%typemap(typecheck) CvRect = SWIGTYPE;
%typemap(typecheck) CvSize = SWIGTYPE;
%typemap(typecheck) CvSize2D32f = SWIGTYPE;
%typemap(typecheck) CvSlice = SWIGTYPE;
%typemap(typecheck) CvBox2D = SWIGTYPE;
%typemap(typecheck) CvTermCriteria = SWIGTYPE;


// for cvReshape, cvGetRow, where header is passed, then filled in
%typemap(in, numinputs=0) CvMat * OUTPUT (CvMat * header, bool freearg=false) {
	header = (CvMat *)cvAlloc(sizeof(CvMat));
   	$1 = header;
}

%apply CvMat * OUTPUT {CvMat * header};
%apply CvMat * OUTPUT {CvMat * submat};

%newobject cvReshape;
%newobject cvGetRow;
%newobject cvGetRows;
%newobject cvGetCol;
%newobject cvGetCols;
%newobject cvGetSubRect;
%newobject cvGetDiag;

/**
 * In C, these functions assume input will always be around at least as long as header,
 * presumably because the most common usage is to pass in a reference to a stack object.  
 * i.e
 * CvMat row;
 * cvGetRow(A, &row, 0);
 *
 * As a result, the header is not refcounted (see the C source for cvGetRow, Reshape, in cxarray.cpp)
 * However, in python, the header parameter is implicitly created so it is easier to create
 * situations where the sub-array outlives the original header.  A simple example is:
 * A = cvReshape(A, -1, A.rows*A.cols)
 *
 * since python doesn't have an assignment operator, the new header simply replaces the original,
 * the refcount of the original goes to zero, and cvReleaseMat is called on the original, freeing both
 * the header and data.  The new header is left pointing to invalid data.  To avoid this, need to add
 * refcount field to the returned header.
*/
%typemap(argout) (const CvArr* arr, CvMat* header) 
{
	$2->hdr_refcount = ((CvMat *)$1)->hdr_refcount;
	$2->refcount = ((CvMat *)$1)->refcount;
	cvIncRefData($2);
}

%typemap(argout) (const CvArr* arr, CvMat* submat) 
{
	$2->hdr_refcount = ((CvMat *)$1)->hdr_refcount;
	$2->refcount = ((CvMat *)$1)->refcount;
	cvIncRefData($2);
}

/* map scalar or sequence to CvScalar, CvPoint2D32f, CvPoint */
%typemap(in) (CvScalar) 
{
	$1 = PyObject_to_CvScalar( $input );
}

//%typemap(in) (CvPoint) {
//	$1 = PyObject_to_CvPoint($input);
//}
//%typemap(in) (CvPoint2D32f) {
//	$1 = PyObject_to_CvPoint2D32f($input);
//}


// ============================================================================================

%define TUPLE_OR_TYPE (item, destination, typename, number, description, ...)
{
  if (PySequence_Check(item)  &&  PySequence_Length(item) == number) 
  {
    PyObject * as_tuple = PySequence_Tuple (item);
    if (!PyArg_ParseTuple (as_tuple, __VA_ARGS__)) 
    {
      PyErr_SetString(PyExc_TypeError, "each entry must consist of " # number " values " # description);
      Py_DECREF (as_tuple);
      return NULL;
    }
    Py_DECREF (as_tuple);
  } 
  else
  {
    typename * ptr;
    if (SWIG_ConvertPtr (item, (void **) & ptr, $descriptor(typename *), SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError, "expected a sequence of " # number " values " # description " or a " # typename);
      Py_DECREF (item);
      return NULL;
    }
    destination = *ptr;
  }
}
%enddef

%define INPUT_ARRAY_OF_TUPLES_OR_TYPES (typename, number, description, ...) 
{
	if(! PySequence_Check ($input))
  {
		PyErr_SetString(PyExc_TypeError, "Expected a list or tuple");
		return NULL;
	}
  
  // TODO: will this ever be freed?
  int count = PySequence_Size ($input);
  int array = (typename *) malloc (count * sizeof (typename));
  
  // extract all the points values from the list */
  typename * element = array;
  for (int i = 0; i < count; i++, element++) 
  {
    PyObject * item = PySequence_GetItem ($input, i);
    
    // use the macro we have to expand a single entry
    TUPLE_OR_TYPE (item, *element, typename, number, description, __VA_ARGS__)
    // *corner, "ff", & corner->x, & corner->y
  }
  
  // these are the arguments passed to the OpenCV function
  $1 = array;
  $2 = count;
}
%enddef


// ============================================================================================
// Tiny typemaps for tiny types ...

%typemap(in) CvRect (CvRect temp) 
//TUPLE_OR_TYPE ($input, $1, CvRect, 4, "(x,y,w,h)", "iiii", & temp.x, & temp.y, & temp.width, & temp.height)
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"iiii", & temp.x, & temp.y, & temp.width, & temp.height)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 4 integers (x, y, width, height)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvRect * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvRect, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvRect");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvTermCriteria (CvTermCriteria temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"iid", & temp.type, & temp.max_iter, & temp.epsilon)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 integers and a float (type, max_iter, epsilon)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvTermCriteria * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvTermCriteria, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvTermCriteria");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvPoint (CvPoint temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"ii", & temp.x, & temp.y)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 integers (x, y)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvPoint * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvPoint, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvPoint");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvPoint2D32f (CvPoint2D32f temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"ff", & temp.x, & temp.y)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 floats (x, y)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvPoint2D32f * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvPoint2D32f, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvPoint2D32f");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvPoint3D32f (CvPoint3D32f temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"fff", & temp.x, & temp.y, &temp.z)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 3 floats (x, y, z)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvPoint3D32f * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvPoint3D32f, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvPoint3D32f");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvPoint2D64f (CvPoint2D64f temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"dd", & temp.x, & temp.y)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 floats (x, y)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvPoint2D64f * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvPoint2D64f, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvPoint2D64f");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvPoint3D64f (CvPoint3D64f temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"ddd", & temp.x, & temp.y, &temp.z)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 3 floats (x, y, z)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvPoint3D64f * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvPoint3D64f, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvPoint3D64f");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvSize (CvSize temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"ii", & temp.width, & temp.height)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 integers (width, height)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvSize * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvSize, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvSize");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvSize2D32f (CvSize2D32f temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"ff", & temp.width, & temp.height)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 floats (width, height)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvSize2D32f * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvSize2D32f, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvSize2D32f");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvBox2D (CvBox2D temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"fffff",  & temp.center.x, & temp.center.y, & temp.size.width, & temp.size.height, & temp.angle)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 5 floats (center_x, center_y, width, height, angle)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvBox2D * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvBox2D, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvBox2D");
      return NULL;
    }
    $1 = *ptr;
  }
}


%typemap(in) CvSlice (CvSlice temp) 
{
  if (PyTuple_Check($input)) 
  {
    if (!PyArg_ParseTuple($input,"ii", & temp.start_index, & temp.end_index)) 
    {
      PyErr_SetString(PyExc_TypeError,"tuple must consist of 2 integers (start_index, end_index)");
      return NULL;
    }
    $1 = temp;
  } 
  else
  {
    CvSlice * ptr;
    if (SWIG_ConvertPtr ($input, (void **) & ptr, SWIGTYPE_p_CvSlice, SWIG_POINTER_EXCEPTION) == -1)
    {
      PyErr_SetString (PyExc_TypeError,"expected a tuple or a CvSlice");
      return NULL;
    }
    $1 = *ptr;
  }
}


/* typemap for cvGetDims */
%typemap(in) (const CvArr * arr, int * sizes = NULL) (void * myarr, int mysizes[CV_MAX_DIM]){
	SWIG_Python_ConvertPtr($input, &myarr, 0, SWIG_POINTER_EXCEPTION);
	$1=(CvArr *)myarr;
	$2=mysizes;
}

%typemap(argout) (const CvArr * arr, int * sizes = NULL) {
	int len = PyInt_AsLong( $result );
	PyObject * obj = PyTuple_FromIntArray( $2, len );
	Py_DECREF( $result );
	$result = obj;
}
				
/* map one list of integer to the two parameters dimension/sizes */
%typemap(in) (int dims, int* sizes) {
    int i;

    /* get the size of the dimention array */
    $1 = PyList_Size ($input);

    /* allocate the needed memory */
    $2 = (int *)malloc ($1 * sizeof (int));

    /* extract all the integer values from the list */
    for (i = 0; i < $1; i++) {
	PyObject *item = PyList_GetItem ($input, i);
	$2 [i] = (int)PyInt_AsLong (item);
    }
}

/* map one list of integer to the parameter idx of
   cvGetND, cvSetND, cvClearND, cvGetRealND, cvSetRealND and cvClearRealND */
%typemap(in) (int* idx) {
    int i;
    int size;

    /* get the size of the dimention array */
    size = PyList_Size ($input);

    /* allocate the needed memory */
    $1 = (int *)malloc (size * sizeof (int));

    /* extract all the integer values from the list */
    for (i = 0; i < size; i++) {
	PyObject *item = PyList_GetItem ($input, i);
	$1 [i] = (int)PyInt_AsLong (item);
    }
}

/* map a list of list of float to an matrix of floats*/
%typemap(in) float** ranges {
    int i1;
    int i2;
    int size1;
    int size2 = 0;

    /* get the number of lines of the matrix */
    size1 = PyList_Size ($input);

    /* allocate the correct number of lines for the destination matrix */
    $1 = (float **)malloc (size1 * sizeof (float *));

    for (i1 = 0; i1 < size1; i1++) {

	/* extract all the lines of the matrix */
	PyObject *list = PyList_GetItem ($input, i1);

	if (size2 == 0) {
	    /* size 2 wasn't computed before */
	    size2 = PyList_Size (list);
	} else if (size2 != PyList_Size (list)) {
	    /* the current line as a different size than the previous one */
	    /* so, generate an exception */
	    SWIG_exception (SWIG_ValueError, "Lines must be the same size");
	}

	/* allocate the correct number of rows for the current line */
	$1 [i1] = (float *)malloc (size2 * sizeof (float));

	/* extract all the float values of this row */
	for (i2 = 0; i2 < size2; i2++) {
	    PyObject *item = PyList_GetItem (list, i2);
	    $1 [i1][i2] = (float)PyFloat_AsDouble (item);
	}
    }
}

/**
 * map the output parameter of the cvGetMinMaxHistValue()
 * so, we can call cvGetMinMaxHistValue() in Python like:
 * min_value, max_value = cvGetMinMaxHistValue (hist, None, None)
 */
%apply int *OUTPUT {int *min_idx};
%apply int *OUTPUT {int *max_idx}; 
%apply float *OUTPUT {float *min_value};
%apply float *OUTPUT {float *max_value};

/**
 * map output parameters of cvMinMaxLoc
 */
%apply double *OUTPUT {double* min_val};
%apply double *OUTPUT {double* max_val};

%typemap(in, numinputs=0) CvPoint * OUTPUT {
    $1 = (CvPoint *) malloc(sizeof(CvPoint));
}

%typemap(argout) CvPoint * OUTPUT { 
	PyObject * to_add = SWIG_NewPointerObj ($1, $descriptor(CvPoint *), SWIG_POINTER_OWN);
	$result = SWIG_AppendOutput( $result, to_add );
}

%apply CvPoint *OUTPUT {CvPoint *min_loc};
%apply CvPoint *OUTPUT {CvPoint *max_loc}; 

/**
 * the input argument of cvPolyLine "CvPoint** pts" is converted from 
 * a "list of list" (aka. an array) of CvPoint().
 * The next parameters "int* npts" and "int contours" are computed from
 * the givne list.
 */
%typemap(in) (CvPoint** pts, int* npts, int contours){
    int i;
    int j;
    int size2 = -1;
    CvPoint **points = NULL;
    int *nb_vertex = NULL;

	if(!PySequence_Check($input)){
		SWIG_exception(SWIG_TypeError, "Expected a list for argument $argnum\n");
		return NULL;
	}

    /* get the number of polylines input array */
    int size1 = PySequence_Size ($input);
    $3 = size1;

	if(size1>0){
	    /* create the points array */
	    points = (CvPoint **)malloc (size1 * sizeof (CvPoint *));

	    /* point to the created array for passing info to the C function */
	    $1 = points;

	    /* create the array for memorizing the vertex */
	    nb_vertex = (int *)malloc (size1 * sizeof (int));
	    $2 = nb_vertex;
	}
    for (i = 0; i < size1; i++) {

		/* get the current item */
		PyObject *line = PySequence_GetItem ($input, i);

		if(!PySequence_Check(line)){
			SWIG_exception(SWIG_TypeError, "Expected a sequence of sequences of integers for argument $argnum\n");
			// TODO: cleanup here
		}

		/* get the size of the current polyline */
		size2 = PySequence_Size (line);


		if(size2>0){
			/* allocate the necessary memory to store the points */
			points [i] = (CvPoint *)malloc (size2 * sizeof (CvPoint));
	 	}

		/* memorize the size of the polyline in the vertex list */
		nb_vertex [i] = size2;

		for (j = 0; j < size2; j++) {
		    /* get the current item */
		    PyObject *item = PySequence_GetItem (line, j);
			points[i][j] = PyObject_to_CvPoint( item );
    	}
	}
}
/** Free arguments allocated before the function call */
%typemap(freearg) (CvPoint **pts, int* npts, int contours){
	int i;
	for(i=0;i<$3;i++){
		free($1[i]);
	}
	free($1);
	free($2);
}


/** this typemap is meant to help cvCalcOpticalFlowPyrLK */
%typemap(in, numinputs = 0) (int count, char* status, float* track_error) {
   	$1 [count] = (char *)  malloc (count * sizeof (char));
   	$2 [count] = (float *) malloc (count * sizeof (float));
}

%typemap(argout) float *track_error { 
	PyObject * to_add = SWIG_NewPointerObj ($1, $descriptor(float *), SWIG_POINTER_OWN);
	$result = SWIG_AppendOutput( $result, to_add );
}

/** Macro to define typemaps to convert a python list of CvPoints to a C array of CvPoints */
%define %typemap_CvPoint_CArr(points_arg, numpoints_arg)

%typemap(in, numinputs=1) (CvPoint * points_arg, int numpoints_arg){
	int i;

	if(!PySequence_Check($input)){
		SWIG_exception(SWIG_TypeError, "Expected a list for argument $argnum\n");
		return NULL;
	}
	int size = PySequence_Size($input);
	CvPoint * points = (CvPoint *)malloc(size*sizeof(CvPoint));
	for(i=0; i<size; i++){
		PyObject *item = PySequence_GetItem($input, i);
		points[i] = PyObject_to_CvPoint( item );
	}
	$1 = points;
	$2 = size;
}
%typemap(freearg) (CvPoint *points_arg, int numpoints_arg){
	free((char *)$1);
}
%enddef

/* apply to cvFillConvexPoly */
%typemap_CvPoint_CArr(pts, npts)

/**
 * this is mainly an "output parameter"
 * So, just allocate the memory as input
 */
%typemap (in, numinputs=0) (CvSeq ** OUTPUT) (CvSeq * seq) {
    $1 = &seq;
}

/**
 * return the contours with all the others parametres
 */
%typemap(argout) (CvSeq ** OUTPUT) {
    PyObject *to_add;

    /* extract the pointer we want to add to the returned tuple */
	/* sequence is allocated in CvMemStorage, so python_ownership=0 */
    to_add = SWIG_NewPointerObj (*$1, $descriptor(CvSeq*), 0); 

	$result = SWIG_AppendResult($result, &to_add, 1);
}
%apply CvSeq **OUTPUT {CvSeq **first_contour};
%apply CvSeq **OUTPUT {CvSeq **comp};

/**
 * CvArr ** image can be either one CvArr or one array of CvArr
 * (for example like in cvCalcHist() )
 * From Python, the array of CvArr can be a tuple.
 */
%typemap(in) (CvArr ** INPUT) (
    CvArr * one_image=NULL, 
    bool free_one_arg=false, 
    CvArr ** many_images=NULL, 
    bool *free_many_args=NULL, 
    int nimages=0 ) {

    /* first, check if this is a tuple */
    if PyTuple_Check ($input) {
        /* This is a tuple, so we need to test each element and pass
            them to the called function */

        int i;

        /* get the size of the tuple */
        nimages = PyTuple_Size ($input);

        /* allocate the necessary place */
        many_images = (CvArr **)malloc (nimages * sizeof (CvArr *));
        free_many_args = (bool *)malloc(nimages * sizeof(bool));

        for (i = 0; i < nimages; i++) {

            /* convert the current tuple element to a CvArr *, and
               store to many_images [i] */
            many_images[i] = PyObject_to_CvArr (PyTuple_GetItem ($input, i),
                                                free_many_args+i);

            /* check that the current item is a correct type */
            if(!many_images[i]) {
                /* incorrect ! */
                SWIG_fail;
            }
        }

        /* what to give to the called function */
        $1 = many_images;

    } else if((one_image = PyObject_to_CvArr( $input, &free_one_arg ))){

        /* this is just one CvArr *, so one_image will receive it */
        $1 = &one_image;

    } else {
        /* not a CvArr *, not a tuple, this is wrong */
        SWIG_fail;
    }
}
%apply CvArr ** INPUT {CvArr ** img};
%apply CvArr ** INPUT {CvArr ** image};
%apply CvArr ** INPUT {CvArr ** arr};
%apply CvArr ** INPUT {CvArr ** vects};

%typemap(freearg) (CvArr ** FREEARG) {
	if(free_one_arg$argnum){
		cvFree(&(one_image$argnum));
	}
	else if(free_many_args$argnum){
		int i;
		for (i=0; i<nimages$argnum; i++){
			if(free_many_args$argnum[i]){
				cvReleaseData(many_images$argnum[i]);
				cvFree(many_images$argnum+i);
			}
		}
		free(many_images$argnum);
		free(free_many_args$argnum);
	}

}
%apply CvArr ** FREEARG {CvArr ** img};
%apply CvArr ** FREEARG {CvArr ** image};
%apply CvArr ** FREEARG {CvArr ** arr};
%apply CvArr ** FREEARG {CvArr ** vects};

/**
 * Map the CvFont * parameter from the cvInitFont() as an output parameter
 */
%typemap (in, numinputs=1) (CvFont* font, int font_face) {
    $1 = (CvFont *)malloc (sizeof (CvFont));
    $2 = (int)PyInt_AsLong ($input); 
    if (SWIG_arg_fail($argnum)) SWIG_fail;
}
%typemap(argout) (CvFont* font, int font_face) {
    PyObject *to_add;

    /* extract the pointer we want to add to the returned tuple */
    to_add = SWIG_NewPointerObj ($1, $descriptor(CvFont *), 0);

	$result = SWIG_AppendResult($result, &to_add, 1);
}

/**
 * these are output parameters for cvGetTextSize
 */
%typemap (in, numinputs=0) (CvSize* text_size, int* baseline) {
    CvSize *size = (CvSize *)malloc (sizeof (CvSize));
    int *baseline = (int *)malloc (sizeof (int));
    $1 = size;
    $2 = baseline;
}

/**
 * return the finded parameters for cvGetTextSize
 */
%typemap(argout) (CvSize* text_size, int* baseline) {
    PyObject * to_add[2];

    /* extract the pointers we want to add to the returned tuple */
    to_add [0] = SWIG_NewPointerObj ($1, $descriptor(CvSize *), 0);
    to_add [1] = PyInt_FromLong (*$2);

    $result = SWIG_AppendResult($result, to_add, 2);
}


/**
 * curr_features is output parameter for cvCalcOpticalFlowPyrLK
 */
%typemap (in, numinputs=1) (CvPoint2D32f* curr_features, int count)
     (int tmpCount) {
    /* as input, we only need the size of the wanted features */

    /* memorize the size of the wanted features */
    tmpCount = (int)PyInt_AsLong ($input);

    /* create the array for the C call */
    $1 = (CvPoint2D32f *) malloc(tmpCount * sizeof (CvPoint2D32f));

    /* the size of the array for the C call */
    $2 = tmpCount;
}

/**
 * the features returned by cvCalcOpticalFlowPyrLK
 */
%typemap(argout) (CvPoint2D32f* curr_features, int count) {
    int i;
    PyObject *to_add;
    
    /* create the list to return */
    to_add = PyList_New (tmpCount$argnum);

    /* extract all the points values of the result, and add it to the
       final resulting list */
    for (i = 0; i < tmpCount$argnum; i++) {
	PyList_SetItem (to_add, i,
			SWIG_NewPointerObj (&($1 [i]),
					    $descriptor(CvPoint2D32f *), 0));
    }

	$result = SWIG_AppendResult($result, &to_add, 1);
}

/**
 * status is an output parameters for cvCalcOpticalFlowPyrLK
 */
%typemap (in, numinputs=1) (char *status) (int tmpCountStatus){
    /* as input, we still need the size of the status array */

    /* memorize the size of the status array */
    tmpCountStatus = (int)PyInt_AsLong ($input);

    /* create the status array for the C call */
    $1 = (char *)malloc (tmpCountStatus * sizeof (char));
}

/**
 * the status returned by cvCalcOpticalFlowPyrLK
 */
%typemap(argout) (char *status) {
    int i;
    PyObject *to_add;

    /* create the list to return */
    to_add = PyList_New (tmpCountStatus$argnum);

    /* extract all the integer values of the result, and add it to the
       final resulting list */
    for (i = 0; i < tmpCountStatus$argnum; i++) {
		PyList_SetItem (to_add, i, PyBool_FromLong ($1 [i]));
    }

	$result = SWIG_AppendResult($result, &to_add, 1); 
}

/* map one list of points to the two parameters dimenssion/sizes
 for cvCalcOpticalFlowPyrLK */
%typemap(in) (CvPoint2D32f* prev_features) 
{
  int i;
  int size;
  
  /* get the size of the input array */
  size = PyList_Size ($input);
  
  /* allocate the needed memory */
  CvPoint2D32f * features = (CvPoint2D32f *) malloc (size * sizeof (CvPoint2D32f));
  
  /* extract all the points values from the list */
  for (i = 0; i < size; i++)
  {
    PyObject *item = PyList_GetItem ($input, i);
    
    void * vptr;
    SWIG_Python_ConvertPtr (item, &vptr,
                            $descriptor(CvPoint2D32f*),
                            SWIG_POINTER_EXCEPTION);
    CvPoint2D32f *p = (CvPoint2D32f *)vptr;
    features[i].x = p->x;
    features[i].y = p->y;
  }
  
  // these are the arguments passed to the OpenCV function
  $1 = features;
}

/**
 * the corners returned by cvGoodFeaturesToTrack
 */
%typemap (in, numinputs=1) (CvPoint2D32f* corners, int* corner_count) (int tmpCount) 
{
  /* as input, we still need the size of the corners array */
  
  /* memorize the size of the status corners */
  tmpCount = (int) PyInt_AsLong ($input);
  
  // these are the arguments passed to the OpenCV function
  $1 = (CvPoint2D32f *) malloc (tmpCount * sizeof (CvPoint2D32f));
  $2 = &tmpCount;
}

/**
 * the corners returned by cvGoodFeaturesToTrack
 */
%typemap(argout) (CvPoint2D32f* corners, int* corner_count) 
{
  int i;
  PyObject *to_add;
  
  /* create the list to return */
  to_add = PyList_New (tmpCount$argnum);
  
  /* extract all the integer values of the result, and add it to the final resulting list */
  for (i = 0; i < tmpCount$argnum; i++)
    PyList_SetItem (to_add, i, SWIG_NewPointerObj (&($1 [i]), $descriptor(CvPoint2D32f *), 0));
  
  $result = SWIG_AppendResult($result, &to_add, 1);
}

/* map one list of points to the two parameters dimension/sizes for cvFindCornerSubPix */
%typemap(in, numinputs=1) (CvPoint2D32f* corners, int count) (int cornersCount, CvPoint2D32f* corners)
{
	if(! PySequence_Check ($input))
  {
		PyErr_SetString(PyExc_TypeError, "Expected a list or tuple");
		return NULL;
	}
  
  // TODO: will this ever be freed?
  cornersCount = PySequence_Size ($input);
  corners = (CvPoint2D32f *) malloc (cornersCount * sizeof (CvPoint2D32f));
  
  // extract all the points values from the list */
  CvPoint2D32f * corner = corners;
  for (int i = 0; i < cornersCount; i++, corner++) 
  {
    PyObject * item = PySequence_GetItem ($input, i);
        
    if (PySequence_Check(item)  &&  PySequence_Length(item) == 2) 
    {
      PyObject * tuple = PySequence_Tuple (item);
      if (!PyArg_ParseTuple (tuple, "ff", & corner->x, & corner->y)) 
      {
        PyErr_SetString(PyExc_TypeError,"each entry must consist of 2 floats (x, y)");
        Py_DECREF (tuple);
        Py_DECREF (item);
        return NULL;
      }
      Py_DECREF (tuple);
    } 
    else
    {
      CvPoint2D32f * ptr;
      if (SWIG_ConvertPtr (item, (void **) & ptr, $descriptor(CvPoint2D32f *), SWIG_POINTER_EXCEPTION) == -1)
      {
        PyErr_SetString (PyExc_TypeError,"expected a sequence of 2 floats (x, y) or a CvPoint2D32f");
        Py_DECREF (item);
        return NULL;
      }
      *corner = *ptr;
    }
    
    Py_DECREF (item);
  }
  
  // these are the arguments passed to the OpenCV function
  $1 = corners;
  $2 = cornersCount;
}

/**
 * the corners returned by cvFindCornerSubPix
 */
%typemap(argout) (CvPoint2D32f* corners, int count) 
{
  int i;
  PyObject *to_add;
  
  /* create the list to return */
  to_add = PyList_New (cornersCount$argnum);
  
  /* extract all the corner values of the result, and add it to the
   final resulting list */
  for (i = 0; i < cornersCount$argnum; i++)
    PyList_SetItem (to_add, i, SWIG_NewPointerObj (&(corners$argnum [i]), $descriptor(CvPoint2D32f *), 0));
  
	$result = SWIG_AppendResult( $result, &to_add, 1);
}

/**
 * return the corners for cvFindChessboardCorners
 */
%typemap(in, numinputs=1) (CvSize pattern_size, CvPoint2D32f * corners, int * corner_count ) 
     (CvSize * pattern_size, CvPoint2D32f * tmp_corners, int tmp_ncorners) {
	 void * vptr;
	if( SWIG_ConvertPtr($input, &vptr, $descriptor( CvSize * ), SWIG_POINTER_EXCEPTION ) == -1){
		return NULL;
	}
	pattern_size=(CvSize *)vptr;
	tmp_ncorners = pattern_size->width*pattern_size->height;

	tmp_corners = (CvPoint2D32f *) malloc(sizeof(CvPoint2D32f)*tmp_ncorners);
	$1 = *pattern_size;
	$2 = tmp_corners;
	$3 = &tmp_ncorners;
}

%typemap(argout) (CvSize pattern_size, CvPoint2D32f * corners, int * corner_count){
    int i;
    PyObject *to_add;

    /* create the list to return */
    to_add = PyList_New ( tmp_ncorners$argnum );

    /* extract all the corner values of the result, and add it to the
       final resulting list */
    for (i = 0; i < tmp_ncorners$argnum; i++) {
		CvPoint2D32f * pt = new CvPoint2D32f;
		pt->x = tmp_corners$argnum[i].x;
		pt->y = tmp_corners$argnum[i].y;
		
    	PyList_SetItem (to_add, i,
            SWIG_NewPointerObj( pt, $descriptor(CvPoint2D32f *), 0));
    }

	$result = SWIG_AppendResult( $result, &to_add, 1);
    free(tmp_corners$argnum);
}

/**
 * return the matrices for cvCameraCalibrate
 */
%typemap(in, numinputs=0) (CvMat * intrinsic_matrix, CvMat * distortion_coeffs)
{
	$1 = cvCreateMat(3,3,CV_32F);
	$2 = cvCreateMat(4,1,CV_32F);
}

%typemap(argout) (CvMat * intrinsic_matrix, CvMat * distortion_coeffs)
{
	PyObject * to_add[2] = {NULL, NULL};
	to_add[0] = SWIG_NewPointerObj($1, $descriptor(CvMat *), 1);
	to_add[1] = SWIG_NewPointerObj($2, $descriptor(CvMat *), 1);
	$result = SWIG_AppendResult( $result, to_add, 2 );
}

/**
 * Fix OpenCV inheritance for CvSeq, CvSet, CvGraph
 * Otherwise, can't call CvSeq functions on CvSet or CvGraph
*/
%typemap(in, numinputs=1) (CvSeq *) (void * ptr)
{
	
	if( SWIG_ConvertPtr($input, &ptr, $descriptor(CvSeq *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvSet *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvGraph *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvSubdiv2D *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvChain *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvContour *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvContourTree *), 0) == -1 )
	{
		SWIG_exception (SWIG_TypeError, "could not convert to CvSeq");
		return NULL;
	}
	$1 = (CvSeq *) ptr;
}

%typemap(in, numinputs=1) (CvSet *) (void * ptr)
{
	if( SWIG_ConvertPtr($input, &ptr, $descriptor(CvSet *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvGraph *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvSubdiv2D *), 0) == -1) 
	{
		SWIG_exception (SWIG_TypeError, "could not convert to CvSet");
		return NULL;
	}
	$1 = (CvSet *)ptr;
}

%typemap(in, numinputs=1) (CvGraph *) (void * ptr)
{
	if( SWIG_ConvertPtr($input, &ptr, $descriptor(CvGraph *), 0) == -1 &&
	    SWIG_ConvertPtr($input, &ptr, $descriptor(CvSubdiv2D *), 0) == -1) 
	{
		SWIG_exception (SWIG_TypeError, "could not convert to CvGraph");
		return NULL;
	}
	$1 = (CvGraph *)ptr;
}

/**
 * Remap output arguments to multiple return values for cvMinEnclosingCircle
 */
%typemap(in, numinputs=0) (CvPoint2D32f * center, float * radius) (CvPoint2D32f * tmp_center, float tmp_radius) 
{
	tmp_center = (CvPoint2D32f *) malloc(sizeof(CvPoint2D32f));
	$1 = tmp_center;
	$2 = &tmp_radius;
}
%typemap(argout) (CvPoint2D32f * center, float * radius)
{
    PyObject * to_add[2] = {NULL, NULL};
	to_add[0] = SWIG_NewPointerObj( tmp_center$argnum, $descriptor(CvPoint2D32f *), 1); 
	to_add[1] = PyFloat_FromDouble( tmp_radius$argnum );

    $result = SWIG_AppendResult($result, to_add, 2);
}

/** BoxPoints */
%typemap(in, numinputs=0) (CvPoint2D32f pt[4]) (CvPoint2D32f tmp_pts[4])
{
	$1 = tmp_pts;
}
%typemap(argout) (CvPoint2D32f pt[4])
{
	PyObject * to_add = PyList_New(4);
	int i;
	for(i=0; i<4; i++){
		CvPoint2D32f * p = new CvPoint2D32f;
		*p = tmp_pts$argnum[i];
		PyList_SetItem(to_add, i, SWIG_NewPointerObj( p, $descriptor(CvPoint2D32f *), 1 ) );
	}
	$result = SWIG_AppendResult($result, &to_add, 1);
}

/** Macro to wrap a built-in type that is used as an object like CvRNG and CvSubdiv2DEdge */
%define %wrap_builtin(type)
%inline %{
// Wrapper class
class type##_Wrapper {
private:
	type m_val;
public:
	type##_Wrapper( const type & val ) :
		m_val(val)
	{
	}
	type * ptr() { return &m_val; }
	type & ref() { return m_val; }
	bool operator==(const type##_Wrapper & x){
		return m_val==x.m_val;
	}
	bool operator!=(const type##_Wrapper & x){
		return m_val!=x.m_val;
	}
};
%}
%typemap(out) type
{
	type##_Wrapper * wrapper = new type##_Wrapper( $1 );
	$result = SWIG_NewPointerObj( wrapper, $descriptor( type##_Wrapper * ), 1 );
}
%typemap(out) type *
{
	type##_Wrapper * wrapper = new type##_Wrapper( *($1) );
	$result = SWIG_NewPointerObj( wrapper, $descriptor( type##_Wrapper * ), 1 );
}

%typemap(in) (type *) (void * vptr, type##_Wrapper * wrapper){
	if(SWIG_ConvertPtr($input, &vptr, $descriptor(type##_Wrapper *), 0)==-1){
		SWIG_exception( SWIG_TypeError, "could not convert Python object to C value");
		return NULL;
	}
	wrapper = (type##_Wrapper *) vptr;
	$1 = wrapper->ptr();
}
%typemap(in) (type) (void * vptr, type##_Wrapper * wrapper){
	if(SWIG_ConvertPtr($input, &vptr, $descriptor(type##_Wrapper *), 0)==-1){
		SWIG_exception( SWIG_TypeError, "could not convert Python object to C value");
		return NULL;
	}
	wrapper = (type##_Wrapper *) vptr;
	$1 = wrapper->ref();
}
%enddef 

/** Application of wrapper class to built-in types */
%wrap_builtin(CvRNG);
%wrap_builtin(CvSubdiv2DEdge);

/**
 * Allow CvQuadEdge2D to be interpreted as CvSubdiv2DEdge
 */
%typemap(in, numinputs=1) (CvSubdiv2DEdge) (CvSubdiv2DEdge_Wrapper * wrapper, CvQuadEdge2D * qedge, void *vptr)
{
	if( SWIG_ConvertPtr($input, &vptr, $descriptor(CvSubdiv2DEdge_Wrapper *), 0) != -1 ){
		wrapper = (CvSubdiv2DEdge_Wrapper *) vptr;
		$1 = wrapper->ref();
	}
	else if( SWIG_ConvertPtr($input, &vptr, $descriptor(CvQuadEdge2D *), 0) != -1 ){
		qedge = (CvQuadEdge2D *) vptr;
		$1 = (CvSubdiv2DEdge)qedge;
	}
	else{
		 SWIG_exception( SWIG_TypeError, "could not convert to CvSubdiv2DEdge");
		 return NULL;
	}
}

/**
 * return the vertex and edge for cvSubdiv2DLocate
 */
%typemap(in, numinputs=0) (CvSubdiv2DEdge * edge, CvSubdiv2DPoint ** vertex) 
	(CvSubdiv2DEdge tmpEdge, CvSubdiv2DPoint * tmpVertex)
{
	$1 = &tmpEdge;
	$2 = &tmpVertex;
}
%typemap(argout) (CvSubdiv2DEdge * edge, CvSubdiv2DPoint ** vertex)
{
	PyObject * to_add[2] = {NULL, NULL};
	if(result==CV_PTLOC_INSIDE || result==CV_PTLOC_ON_EDGE){
		CvSubdiv2DEdge_Wrapper * wrapper = new CvSubdiv2DEdge_Wrapper( tmpEdge$argnum );
		to_add[0] = SWIG_NewPointerObj( wrapper, $descriptor(CvSubdiv2DEdge_Wrapper *), 0);
		to_add[1] = Py_None;
	}
	if(result==CV_PTLOC_VERTEX){
		to_add[0] = Py_None;
		to_add[1] = SWIG_NewPointerObj( tmpVertex$argnum, $descriptor(CvSubdiv2DPoint *), 0);
	}
	
	$result = SWIG_AppendResult($result, to_add, 2);
}

/**
 * int *value  in cvCreateTrackbar() is only used for input in the Python wrapper.
 * for output, use the pos in the callback
 * TODO: remove the memory leak introducted by the malloc () (if needed).
 */
%typemap(in, numinputs=1) (int *value)
{
    $1 = (int *)malloc (sizeof (int));
    *$1 = PyInt_AsLong ($input);
}


/**
 * take (query_points,k) and return (indices,dist)
 * for cvLSHQuery, cvFindFeatures
 */
%typemap(in, noblock=1) (const CvMat* query_points) (bool freearg=false, int num_query_points)
{
  $1 = (CvMat*)PyObject_to_CvArr($input, &freearg);
  num_query_points = $1->rows;
}
%typemap(freearg) (const CvMat* query_points) {
  if($1!=NULL && freearg$argnum){
    cvReleaseData( $1 );
    cvFree(&($1));
  }
}
%typemap(in) (CvMat* indices, CvMat* dist, int k)
{
  $3 = (int)PyInt_AsLong($input);
  $1 = cvCreateMat(num_query_points2, $3, CV_32SC1);
  $2 = cvCreateMat(num_query_points2, $3, CV_64FC1);
}
%typemap(argout) (CvMat* indices, CvMat* dist, int k)
{
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($1, $descriptor(CvMat *), 1) );
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($2, $descriptor(CvMat *), 1) );
}

/**
 * take (data) and return (indices)
 * for cvLSHAdd
 */
%typemap(in) (const CvMat* data, CvMat* indices) (bool freearg=false)
{
  $1 = (CvMat*)PyObject_to_CvArr($input, &freearg);
  CvMat* m = (CvMat*)$1;
  $2 = cvCreateMat(m->rows, 1, CV_32SC1 );
}
%typemap(argout) (const CvMat* data, CvMat* indices)
{
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($2, $descriptor(CvMat *), 1) );
}
%typemap(freearg) (const CvMat* data, CvMat* indices) {
  if($1!=NULL && freearg$argnum){
    cvReleaseData( $1 );
    cvFree(&($1));
  }
}

/**
 * take (max_out_indices) and return (indices)
 * for cvFindFeaturesBoxed
 */
%typemap(in) (CvMat* out_indices) (bool freearg=false)
{
  int max_out_indices = (int)PyInt_AsLong($input);
  $1 = cvCreateMat(max_out_indices, 1, CV_32SC1 );
}
%typemap(argout) (CvMat* out_indices)
{
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($1, $descriptor(CvMat *), 1) );
}

/**
 * suppress (CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage) and return (keypoints, descriptors) 
 * for cvExtractSURF
 */
%typemap(in,numinputs=0) (CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage)
     (CvSeq* keypoints = 0, CvSeq* descriptors = 0,CvMemStorage* storage)
{
  storage = cvCreateMemStorage();
  $1 = &keypoints;
  $2 = &descriptors;
  $3 = storage;
}
%typemap(argout) (CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage)
{
  const int n1 = 6;
  int n2 = (*$2)->elem_size / sizeof(float);
  assert((*$2)->elem_size == 64 * sizeof(float) ||
	 (*$2)->elem_size == 128 * sizeof(float));
  assert((*$2)->total == (*$1)->total);
  CvMat* m1 = cvCreateMat((*$2)->total,n1,CV_32FC1);
  CvMat* m2 = cvCreateMat((*$2)->total,n2,CV_32FC1);
  CvSeqReader r1;
  cvStartReadSeq(*$1, &r1);
  float* m1p = m1->data.fl;
  for (int j=0;j<(*$2)->total;++j) {
    CvSURFPoint* sp = (CvSURFPoint*)r1.ptr;
    m1p[0] = sp->pt.x;
    m1p[1] = sp->pt.y;
    m1p[2] = sp->laplacian;
    m1p[3] = sp->size;
    m1p[4] = sp->dir;
    m1p[5] = sp->hessian;
    m1p += n1;
    CV_NEXT_SEQ_ELEM((*$1)->elem_size, r1);
  }
  CvSeqReader r2;
  cvStartReadSeq(*$2, &r2);
  float* m2p = m2->data.fl;
  for (int j=0;j<(*$2)->total;++j) {
    memcpy(m2p,r2.ptr,n2*sizeof(float));
    m2p += n2;
    CV_NEXT_SEQ_ELEM((*$2)->elem_size, r2);
  }
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj(m1, $descriptor(CvMat *), 1) );
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj(m2, $descriptor(CvMat *), 1) );
  cvReleaseMemStorage(&$3);
}

/**
 * suppress (CvMat* homography) and return (homography)
 * for cvFindHomography
 */
%typemap(in,numinputs=0) (CvMat* homography) (bool freearg=false)
{
  $1 = cvCreateMat(3,3,CV_64FC1);
}
%typemap(argout) (CvMat* homography)
{
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($1, $descriptor(CvMat *), 1) );
}

/**
 * take (coeffs) for (const CvMat* coeffs, CvMat *roots) and return (roots)
 * for cvSolveCubic
 */
%typemap(in) (const CvMat* coeffs, CvMat *roots) (bool freearg=false)
{
  $1 = (CvMat*)PyObject_to_CvArr($input, &freearg);
  int m = $1->rows * $1->cols;
  if (m<2) {
    PyErr_SetString (PyExc_TypeError,"must give at least 2 coefficients");
    return NULL;
  }
  $2 = cvCreateMat(m-1, 1, CV_MAKETYPE(CV_MAT_DEPTH($1->type),1));
}
%typemap(argout) (const CvMat* coeffs, CvMat *roots)
{
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($2, $descriptor(CvMat *), 1) );
}
%typemap(freearg) (const CvMat* coeffs, CvMat *roots)
{
  if($1!=NULL && freearg$argnum){
    cvReleaseData( $1 );
    cvFree(&($1));
  }
}

/**
 * take (coeffs) for (const CvMat* coeffs, CvMat *roots2) and return (roots2)
 * for cvSolvePoly
 */
%typemap(in) (const CvMat* coeffs, CvMat *roots2) (bool freearg=false)
{
  $1 = (CvMat*)PyObject_to_CvArr($input, &freearg);
  int m = $1->rows * $1->cols;
  if (m<2) {
    PyErr_SetString (PyExc_TypeError,"must give at least 2 coefficients");
    return NULL;
  }
  $2 = cvCreateMat(m-1, 1, CV_MAKETYPE(CV_MAT_DEPTH($1->type),2));
}
%typemap(argout) (const CvMat* coeffs, CvMat *roots2)
{
  $result = SWIG_AppendOutput( $result, SWIG_NewPointerObj($2, $descriptor(CvMat *), 1) );
}
%typemap(freearg) (const CvMat* coeffs, CvMat *roots2)
{
  if($1!=NULL && freearg$argnum){
    cvReleaseData( $1 );
    cvFree(&($1));
  }
}

