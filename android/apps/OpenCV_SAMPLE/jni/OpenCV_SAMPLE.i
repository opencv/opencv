/* File : foobar.i */
%module OpenCV_SAMPLE

/*
 * the java import code muse be included for the opencv jni wrappers
 * this means that the android project must reference opencv/android as a project
 * see the default.properties for how this is done
 */
%pragma(java) jniclassimports=%{
import com.opencv.jni.*; //import the android-opencv jni wrappers
%}

%pragma(java) jniclasscode=%{
  static {
    try {
        //load up our shared libraries
        System.loadLibrary("android-opencv");
        System.loadLibrary("OpenCV_SAMPLE");
      } catch (UnsatisfiedLinkError e) {
        //badness
        throw e;
    }
  }

%}

//import the android-cv.i file so that swig is aware of all that has been previous defined
//notice that it is not an include....
%import "android-cv.i"

%{
#include "cvsample.h"
using cv::Mat;
%}

//make sure to import the image_pool as it is 
//referenced by the Processor java generated
//class
%typemap(javaimports) CVSample "
import com.opencv.jni.*;// import the opencv java bindings
"
class CVSample
{
public:
  void canny(const Mat& input, Mat& output, int edgeThresh);
  void invert(Mat& inout);
  void blur(Mat& inout, int half_kernel_size);
};
