/* File : android-cv.i

import this file, and make sure to add the System.loadlibrary("android-opencv")
before loading any lib that depends on this.
 */

%module opencv
%{
#include "image_pool.h"
#include "glcamera.h"
using namespace cv;
%}
#ifndef SWIGIMPORTED
%include "various.i"
%include "typemaps.i"
%include "arrays_java.i"
#endif

/**
 * Make all the swig pointers public, so that
 * external libraries can refer to these, otherwise they default to 
 * protected...
 */
%typemap(javabody) SWIGTYPE %{
  private long swigCPtr;
  protected boolean swigCMemOwn;
  public $javaclassname(long cPtr, boolean cMemoryOwn) {
	swigCMemOwn = cMemoryOwn;
	swigCPtr = cPtr;
  }
  public static long getCPtr($javaclassname obj) {
	return (obj == null) ? 0 : obj.swigCPtr;
  }
%}


%pragma(java) jniclasscode=%{
  static {
    try {
    	//load the library, make sure that libandroid-opencv.so is in your <project>/libs/armeabi directory
    	//so that android sdk automatically installs it along with the app.
        System.loadLibrary("android-opencv");
    } catch (UnsatisfiedLinkError e) {
    	//badness
    	throw e;
     
    }
  }
%}


%include "cv.i"

%include "glcamera.i"

%include "image_pool.i"

%include "Calibration.i"
