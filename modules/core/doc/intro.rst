************
Introduction
************

.. highlight:: cpp

OpenCV (Open Source Computer Vision Library: http://opencv.willowgarage.com/wiki/) is open-source BSD-licensed library that includes several hundreds computer vision algorithms. It is very popular in the Computer Vision community. Some people call it “de-facto standard” API. The document aims to specify the stable parts of the library, as well as some abstract interfaces for high-level interfaces, with the final goal to make it an official standard.

API specifications in the document use the standard C++ (http://www.open-std.org/jtc1/sc22/wg21/) and the standard C++ library.

The current OpenCV implementation has a modular structure (i.e. the binary package includes several shared or static libraries), where we have:

 * **core** - the compact module defining basic data structures, including the dense multi-dimensional array ``Mat``, and basic functions, used by all other modules.
 * **imgproc** - image processing module that includes linear and non-linear image filtering, geometrical image transformations (resize, affine and perspective warping, generic table-based remap), color space conversion, histograms etc.
 * **video** - video analysis module that includes motion estimation, background subtraction and object tracking algorithms.
 * **calib3d** - basic multiple-view geometry algorithms, single and stereo camera calibration, object pose estimation, stereo correspondence algorithms, elements of 3d reconstruction.
 * **features2d** - salient feature detectors, descriptors and the descriptor matchers.
 * **objdetect** - detection of objects, instances of the predefined classes (e.g faces, eyes, mugs, people, cars etc.)
 * **highgui** - easy-to-use interface to video capturing, image and video codecs APIs, as well as simple UI capabilities.
 * **gpu** - GPU-accelerated algorithms from different OpenCV modules.
 * ... some other helper modules, such as FLANN and Google test wrappers, Python bindings etc.

Although the alternative implementations of the proposed standard may be structured differently, the proposed standard draft is organized by the functionality groups that reflect the decomposition of the library by modules.

Below are the other main concepts of the OpenCV API, implied everywhere in the document.

The API Concepts
================

*"cv"* namespace
----------------

All the OpenCV classes and functions are placed into *"cv"* namespace. Therefore, to access this functionality from your code, use 
``cv::`` specifier or ``using namespace cv;`` directive:

.. code-block:: c
    
    #include "opencv2/core/core.hpp"
    ...
    cv::Mat H = cv::findHomography(points1, points2, CV_RANSAC, 5);
    ...

or

::
    
    #include "opencv2/core/core.hpp"
    using namespace cv;
    ...
    Mat H = findHomography(points1, points2, CV_RANSAC, 5 );
    ...

It is probable that some of the current or future OpenCV external names conflict with STL
or other libraries, in this case use explicit namespace specifiers to resolve the name conflicts:

::
    
    Mat a(100, 100, CV_32F);
    randu(a, Scalar::all(1), Scalar::all(std::rand()));
    cv::log(a, a);
    a /= std::log(2.);


Automatic Memory Management
---------------------------

OpenCV handles all the memory automatically.

First of all, ``std::vector``, ``Mat`` and other data structures used by the functions and methods have destructors that deallocate the underlying memory buffers when needed.

Secondly, in the case of ``Mat`` this *when needed* means that the destructors do not always deallocate the buffers, they take into account possible data sharing. That is, destructor decrements the reference counter, associated with the matrix data buffer, and the buffer is deallocated if and only if the reference counter reaches zero, that is, when no other structures refer to the same buffer. Similarly, when ``Mat`` instance is copied, not actual data is really copied; instead, the associated with it reference counter is incremented to memorize that there is another owner of the same data. There is also ``Mat::clone`` method that creates a full copy of the matrix data. Here is the example

::
    
    // create a big 8Mb matrix
    Mat A(1000, 1000, CV_64F);
    
    // create another header for the same matrix;
    // this is instant operation, regardless of the matrix size.
    Mat B = A;
    // create another header for the 3-rd row of A; no data is copied either
    Mat C = B.row(3);
    // now create a separate copy of the matrix
    Mat D = B.clone();
    // copy the 5-th row of B to C, that is, copy the 5-th row of A 
    // to the 3-rd row of A.
    B.row(5).copyTo(C);
    // now let A and D share the data; after that the modified version
    // of A is still referenced by B and C.
    A = D;
    // now make B an empty matrix (which references no memory buffers),
    // but the modified version of A will still be referenced by C,
    // despite that C is just a single row of the original A
    B.release(); 
                 
    // finally, make a full copy of C. In result, the big modified
    // matrix will be deallocated, since it's not referenced by anyone
    C = C.clone();

Therefore, ``Mat`` and other basic structures use is simple. But what about high-level classes or even user data types that have been created without automatic memory management in mind? For them OpenCV offers ``Ptr<>`` template class, which is similar to the ``std::shared_ptr`` from C++ TR1. That is, instead of using plain pointers::

   T* ptr = new T(...);

one can use::

   Ptr<T> ptr = new T(...);

That is, ``Ptr<T> ptr`` incapsulates a pointer to ``T`` instance and a reference counter associated with the pointer. See ``Ptr`` description for details.


.. todo::

  Should we replace Ptr<> with the semi-standard shared_ptr<>?

Automatic Allocation of the Output Data
---------------------------------------

OpenCV does not only deallocate the memory automatically, it can also allocate memory for the output function parameters automatically most of the time. That is, if a function has one or more input arrays (``cv::Mat`` instances) and some output arrays, the output arrays automatically allocated or reallocated. The size and type of the output arrays are determined from the input arrays' size and type. If needed, the functions take extra parameters that help to figure out the output array properties.

Here is the example: ::
    
    #include "cv.h"
    #include "highgui.h"
    
    using namespace cv;
    
    int main(int, char**)
    {
        VideoCapture cap(0);
        if(!cap.isOpened()) return -1;
    
        Mat frame, edges;
        namedWindow("edges",1);
        for(;;)
        {
            cap >> frame;
            cvtColor(frame, edges, CV_BGR2GRAY);
            GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
            Canny(edges, edges, 0, 30, 3);
            imshow("edges", edges);
            if(waitKey(30) >= 0) break;
        }
        return 0;
    }
..

The array ``frame`` is automatically allocated by ``>>`` operator, since the video frame resolution and bit-depth is known to the video capturing module. The array ``edges`` is automatically allocated by ``cvtColor`` function. It will have the same size and the bit-depth as the input array, and the number of channels will be 1, because we passed the color conversion code ``CV_BGR2GRAY`` (that means color to grayscale conversion). Note that ``frame`` and ``edges`` will be allocated only once during the first execution of the loop body, since all the next video frames will have the same resolution (unless user somehow changes the video resolution, in this case the arrays will be automatically reallocated).

The key component of this technology is the method ``Mat::create``. It takes the desired array size and type. If the array already has the specified size and type, the method does nothing. Otherwise, it releases the previously allocated data, if any (this part involves decrementing the reference counter and comparing it with zero), and then allocates a new buffer of the required size. Most functions call this ``Mat::create`` method for each output array and so the automatic output data allocation is implemented.

Some notable exceptions from this scheme are ``cv::mixChannels``, ``cv::RNG::fill`` and a few others functions and methods. They are not able to allocate the output array, so the user has to do that in advance.


Saturation Arithmetics
----------------------

As computer vision library, OpenCV deals a lot with image pixels that are often encoded in a compact 8- or 16-bit per channel form and thus have a limited value range. Furthermore, certain operations on images, like color space conversions, brightness/contrast adjustments, sharpening, complex interpolation (bi-cubic, Lanczos) can produce values out of the available range. If we just store the lowest 8 (16) bit of the result, that will result in some visual artifacts and may affect the further image analysis. To solve this problem, we use so-called *saturation* arithmetics, e.g. to store ``r``, a result of some operation, to 8-bit image, we find the nearest value within 0..255 range: 

.. math::

    I(x,y)= \min ( \max (\textrm{round}(r), 0), 255) 

The similar rules are applied to 8-bit signed and 16-bit signed and unsigned types. This semantics is used everywhere in the library. In C++ code it is done using ``saturate_cast<>`` functions that resembler the standard C++ cast operations. Here is the implementation of the above formula::

    I.at<uchar>(y, x) = saturate_cast<uchar>(r);

where ``cv::uchar`` is OpenCV's 8-bit unsigned integer type. In optimized SIMD code we use specialized instructions, like SSE2' ``paddusb``, ``packuswb`` etc. to achieve exactly the same behavior as in C++ code.


Fixed Pixel Types. Limited Use of Templates
-------------------------------------------

Templates is a great feature of C++ that enables implementation of very powerful, efficient and yet safe data structures and algorithms. However, the extensive use of templates may dramatically increase compile time and code size. Besides, it is difficult to separate interface and implementation when templates are used exclusively, which is fine for basic algorithms, but not good for computer vision libraries, where a single algorithm may span a thousands lines of code. Because of this, and also to simplify development of bindings for other languages, like Python, Java, Matlab, that do not have templates at all or have limited template capabilities, we prefer polymorphism and runtime dispatching over templates. In the places where runtime dispatching would be too slow (like pixel access operators), impossible (generic Ptr<> implementation) or just very inconvenient (saturate_cast<>()) we introduce small template classes, methods and functions. Everywhere else we prefer not to use templates.

Because of this, there is a limited fixed set of primitive data types that the library can operate on. That is, an array elements should have one of the following types:

  * 8-bit unsigned integer (uchar)
  * 8-bit signed integer (schar)
  * 16-bit unsigned integer (ushort)
  * 16-bit signed integer (short)
  * 32-bit signed integer (int)
  * 32-bit floating-point number (float)
  * 64-bit floating-point number (double)
  * a tuple of several elements, where all elements have the same type (one of the above). Array, which elements are such tuples, are called multi-channel arrays, as opposite to the single-channel arrays, which elements are scalar values. The maximum possible number of channels is defined by ``CV_CN_MAX`` constant (which is not smaller than 32).
  
.. todo::
  Need we extend the above list? Shouldn't we throw away 8-bit signed (schar)?
  
For these basic types there is enumeration::

  enum { CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6 };
  
Multi-channel (``n``-channel) types can be specified using ``CV_8UC1`` ... ``CV_64FC4`` constants (for number of channels from 1 to 4), or using ``CV_8UC(n)`` ... ``CV_64FC(n)`` or ``CV_MAKETYPE(CV_8U, n)`` ... ``CV_MAKETYPE(CV_64F, n)`` macros when the number of channels is more than 4 or unknown at compile time.

.. note::
  ``CV_32FC1 == CV_32F``, ``CV_32FC2 == CV_32FC(2) == CV_MAKETYPE(CV_32F, 2)`` and ``CV_MAKETYPE(depth, n) == ((x&7)<<3) + (n-1)``, that is, the type constant is formed from the ``depth``, taking the lowest 3 bits, and the number of channels minus 1, taking the next ``log2(CV_CN_MAX)`` bits.

Here are some examples::

   Mat mtx(3, 3, CV_32F); // make 3x3 floating-point matrix
   Mat cmtx(10, 1, CV_64FC2); // make 10x1 2-channel floating-point
                              // matrix (i.e. 10-element complex vector)
   Mat img(Size(1920, 1080), CV_8UC3); // make 3-channel (color) image
                                       // of 1920 columns and 1080 rows.
   Mat grayscale(image.size(), CV_MAKETYPE(image.depth(), 1)); // make 1-channel image of
                                                               // the same size and same
                                                               // channel type as img

Arrays, which elements are more complex, can not be constructed or processed using OpenCV. Furthermore, each function or method can handle only a subset of all possible array types. Usually, the more complex is the algorithm, the smaller is the supported subset of formats. Here are some typical examples of such limitations:

  * The face detection algorithm only works with 8-bit grayscale or color images.
  * Linear algebra functions and most of the machine learning algorithms work with floating-point arrays only.
  * Basic functions, such as ``cv::add``, support all types, except for ``CV_8SC(n)``.
  * Color space conversion functions support 8-bit unsigned, 16-bit unsigned and 32-bit floating-point types.

The subset of supported types for each functions has been defined from practical needs. All this information about supported types can be put together into a special table. In different implementations of the standard the tables may look differently, for example, on embedded platforms double-precision floating-point type (``CV_64F``) may be unavailable.

.. todo::
  Should we include such a table into the standard?
  Should we specify minimum "must-have" set of supported formats for each functions?


Error handling
--------------

OpenCV uses exceptions to signal about the critical errors. When the input data has correct format and within the specified value range, but the algorithm can not succeed for some reason (e.g. the optimization algorithm did not converge), it returns a special error code (typically, just a boolean variable).

The exceptions can be instances of ``cv::Exception`` class or its derivatives. In its turn, ``cv::Exception`` is a derivative of std::exception, so it can be gracefully handled in the code using other standard C++ library components.

The exception is typically thrown using ``CV_Error(errcode, description)`` macro, or its printf-like ``CV_Error_(errcode, printf-spec, (printf-args))`` variant, or using ``CV_Assert(condition)`` macro that checks the condition and throws exception when it is not satisfied. For performance-critical code there is ``CV_DbgAssert(condition)`` that is only retained in Debug configuration. Thanks to the automatic memory management, all the intermediate buffers are automatically deallocated in the case of sudden error; user only needs to put a try statement to catch the exceptions, if needed:

::
  
    try
    {
        ... // call OpenCV
    }
    catch( cv::Exception& e )
    {
        const char* err_msg = e.what();
        std::cout << "exception caught: " << err_msg << std::endl;
    }


Multi-threading and reenterability
----------------------------------

The current OpenCV implementation is fully reenterable, and so should be any alternative implementation targeted for multi-threaded environments. That is, the same function, the same *constant* method of a class instance, or the same *non-constant* method of different class instances can be called from different threads. Also, the same ``cv::Mat`` can be used in different threads, because the reference-counting operations use the architecture-specific atomic instructions.
