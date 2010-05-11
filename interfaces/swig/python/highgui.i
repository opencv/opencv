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


// 2004-03-16, Gabriel Schreiber <schreiber@ient.rwth-aachen.de>
//             Mark Asbach       <asbach@ient.rwth-aachen.de>
//             Institute of Communications Engineering, RWTH Aachen University

%{
	#include <cxtypes.h>
	#include <cv.h>
	#include <highgui.h>
	#include "pyhelpers.h"
	#include "pycvseq.hpp"
%}
// include python-specific files
%include "./nointpb.i"
%include "./pytypemaps.i"
%include "exception.i"
%include "cvswigmacros.i"

// handle camera and video writer destruction
%myrelease(highgui, cvReleaseCapture, CvCapture);
%myrelease(highgui, cvReleaseVideoWriter, CvVideoWriter);

/* the wrapping code to enable the use of Python-based mouse callbacks */
%header %{
	/* This encapsulates the python callback and user_data for mouse callback */
	struct PyCvMouseCBData {
		PyObject * py_func;
		PyObject * user_data;
	};
	/* This encapsulates the python callback and user_data for mouse callback */
    /* C helper function which is responsible for calling
       the Python real trackbar callback function */
    static void icvPyOnMouse (int event, int x, int y,
					 int flags, PyCvMouseCBData * param) {

		/* Must ensure this thread has a lock on the interpreter */
		PyGILState_STATE state = PyGILState_Ensure();

		PyObject *result;

		/* the argument of the callback ready to be passed to Python code */
		PyObject *arg1 = PyInt_FromLong (event);
		PyObject *arg2 = PyInt_FromLong (x);
		PyObject *arg3 = PyInt_FromLong (y);
		PyObject *arg4 = PyInt_FromLong (flags);
		PyObject *arg5 = param->user_data;  // assume this is already a PyObject

		/* build the tuple for calling the Python callback */
		PyObject *arglist = Py_BuildValue ("(OOOOO)",
				arg1, arg2, arg3, arg4, arg5);

		/* call the Python callback */
		result = PyEval_CallObject (param->py_func, arglist);

		/* Errors in Python callback get swallowed, so report them here */
		if(!result){
			PyErr_Print();
			cvError( CV_StsInternal, "icvPyOnMouse", "", __FILE__, __LINE__);
		}

		/* cleanup */
		Py_XDECREF (result);

		/* Release Interpreter lock */
		PyGILState_Release(state);
	}
%}
/**
 * adapt cvSetMouseCallback to use python callback
 */
%rename (cvSetMouseCallbackOld) cvSetMouseCallback;
%rename (cvSetMouseCallback) cvSetMouseCallbackPy;
%inline %{
	void cvSetMouseCallbackPy( const char* window_name, PyObject * on_mouse, PyObject * param=NULL ){
		// TODO potential memory leak if mouse callback is redefined
		PyCvMouseCBData * py_callback = new PyCvMouseCBData;
		py_callback->py_func = on_mouse;
		py_callback->user_data = param ? param : Py_None;

        Py_XINCREF(py_callback->py_func);
        Py_XINCREF(py_callback->user_data);
            
		cvSetMouseCallback( window_name, (CvMouseCallback) icvPyOnMouse, (void *) py_callback );
	}
%}



/**
 * The following code enables trackbar callbacks from python.  Unfortunately, there is no 
 * way to distinguish which trackbar the event originated from, so must hard code a 
 * fixed number of unique c callback functions using the macros below
 */
%wrapper %{
    /* C helper function which is responsible for calling
       the Python real trackbar callback function */
    static void icvPyOnTrackbar( PyObject * py_cb_func, int pos) {
	
		/* Must ensure this thread has a lock on the interpreter */
		PyGILState_STATE state = PyGILState_Ensure();

		PyObject *result;

		/* the argument of the callback ready to be passed to Python code */
		PyObject *arg1 = PyInt_FromLong (pos);

		/* build the tuple for calling the Python callback */
		PyObject *arglist = Py_BuildValue ("(O)", arg1);

		/* call the Python callback */
		result = PyEval_CallObject (py_cb_func, arglist);

		/* Errors in Python callback get swallowed, so report them here */
		if(!result){
			PyErr_Print();
			cvError( CV_StsInternal, "icvPyOnTrackbar", "", __FILE__, __LINE__);
		}


		/* cleanup */
		Py_XDECREF (result);

		/* Release Interpreter lock */
		PyGILState_Release(state);
	}

#define ICV_PY_MAX_CB 10

	struct PyCvTrackbar {
		CvTrackbarCallback cv_func;
		PyObject * py_func;
		PyObject * py_pos;
	};

	static int my_trackbar_cb_size=0;
	extern PyCvTrackbar my_trackbar_cb_funcs[ICV_PY_MAX_CB];
%}

/* Callback table entry */
%define %ICV_PY_CB_TAB_ENTRY(idx)
	{(CvTrackbarCallback) icvPyTrackbarCB##idx, NULL, NULL }
%enddef

/* Table of callbacks */
%define %ICV_PY_CB_TAB
%wrapper %{
	PyCvTrackbar my_trackbar_cb_funcs[ICV_PY_MAX_CB] = {
		%ICV_PY_CB_TAB_ENTRY(0),
		%ICV_PY_CB_TAB_ENTRY(1),
		%ICV_PY_CB_TAB_ENTRY(2),
		%ICV_PY_CB_TAB_ENTRY(3),
		%ICV_PY_CB_TAB_ENTRY(4),
		%ICV_PY_CB_TAB_ENTRY(5),
		%ICV_PY_CB_TAB_ENTRY(6),
		%ICV_PY_CB_TAB_ENTRY(7),
		%ICV_PY_CB_TAB_ENTRY(8),
		%ICV_PY_CB_TAB_ENTRY(9)
	};
%}	 
%enddef

/* Callback definition */
%define %ICV_PY_CB_IMPL(idx) 
%wrapper %{
static void icvPyTrackbarCB##idx(int pos){                                      
	if(!my_trackbar_cb_funcs[idx].py_func) return;                              
	icvPyOnTrackbar( my_trackbar_cb_funcs[idx].py_func, pos );                    
}                                                                               
%}
%enddef


%ICV_PY_CB_IMPL(0);
%ICV_PY_CB_IMPL(1);
%ICV_PY_CB_IMPL(2);
%ICV_PY_CB_IMPL(3);
%ICV_PY_CB_IMPL(4);
%ICV_PY_CB_IMPL(5);
%ICV_PY_CB_IMPL(6);
%ICV_PY_CB_IMPL(7);
%ICV_PY_CB_IMPL(8);
%ICV_PY_CB_IMPL(9);

%ICV_PY_CB_TAB;


/**
 * typemap to memorize the Python callback when doing cvCreateTrackbar ()
 */
%typemap(in) CvTrackbarCallback {

	if(my_trackbar_cb_size == ICV_PY_MAX_CB){
		SWIG_exception(SWIG_IndexError, "Exceeded maximum number of trackbars");
	}

	my_trackbar_cb_size++;

    if (!PyCallable_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return 0;
    }
    Py_XINCREF((PyObject*) $input);         /* Add a reference to new callback */
    Py_XDECREF(my_trackbar_cb_funcs[my_trackbar_cb_size-1].py_func);  /* Dispose of previous callback */
	my_trackbar_cb_funcs[my_trackbar_cb_size-1].py_func = (PyObject *) $input;

	/* prepare to call the C function who will register the callback */
	$1 = my_trackbar_cb_funcs[ my_trackbar_cb_size-1 ].cv_func;
}

/**
 * typemap so that cvWaitKey returns a character in all cases except -1
 */
%rename (cvWaitKeyC) cvWaitKey;
%rename (cvWaitKey) cvWaitKeyPy;
%inline %{
	PyObject * cvWaitKeyPy(int delay=0){
		// In order for the event processing thread to run a python callback
		// it must acquire the global interpreter lock, but  cvWaitKey blocks, so
		// this thread can never release the lock. So release it here.
		PyThreadState * thread_state = PyEval_SaveThread();
		int res = cvWaitKey(delay);
		PyEval_RestoreThread( thread_state );
		
		char str[2]={(char)res,0};
		if(res==-1){
			return PyLong_FromLong(-1);
		}
		return PyString_FromString(str);
	}
%}
/* HighGUI Python module initialization
 * needed for callbacks to work in a threaded environment 
 */
%init %{
	PyEval_InitThreads();
%}


%include "../general/highgui.i"

%pythoncode 
%{

__doc__ = """HighGUI provides minimalistic user interface parts and video input/output.

Dependent on the platform it was compiled on, this library provides methods
to draw a window for image display, capture video from a camera or framegrabber
or read/write video streams from/to the file system.

This wrapper was semi-automatically created from the C/C++ headers and therefore
contains no Python documentation. Because all identifiers are identical to their
C/C++ counterparts, you can consult the standard manuals that come with OpenCV.
"""

%}
