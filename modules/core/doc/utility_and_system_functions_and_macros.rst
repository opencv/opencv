Utility and System Functions and Macros
=======================================

.. highlight:: cpp



.. index:: alignPtr


cv::alignPtr
------------

`id=0.732441674276 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/alignPtr>`__




.. cfunction:: template<typename _Tp> _Tp* alignPtr(_Tp* ptr, int n=sizeof(_Tp))

    Aligns pointer to the specified number of bytes





    
    :param ptr: The aligned pointer 
    
    
    :param n: The alignment size; must be a power of two 
    
    
    
The function returns the aligned pointer of the same type as the input pointer:


.. math::

    \texttt{(\_Tp*)(((size\_t)ptr + n-1) \& -n)} 



.. index:: alignSize


cv::alignSize
-------------

`id=0.0293178300141 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/alignSize>`__




.. cfunction:: size_t alignSize(size_t sz, int n)

    Aligns a buffer size to the specified number of bytes





    
    :param sz: The buffer size to align 
    
    
    :param n: The alignment size; must be a power of two 
    
    
    
The function returns the minimum number that is greater or equal to 
``sz``
and is divisble by 
``n``
:


.. math::

    \texttt{(sz + n-1) \& -n} 



.. index:: allocate


cv::allocate
------------

`id=0.672857293821 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/allocate>`__




.. cfunction:: template<typename _Tp> _Tp* allocate(size_t n)

    Allocates an array of elements





    
    :param n: The number of elements to allocate 
    
    
    
The generic function 
``allocate``
allocates buffer for the specified number of elements. For each element the default constructor is called.



.. index:: deallocate


cv::deallocate
--------------

`id=0.907199792708 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/deallocate>`__




.. cfunction:: template<typename _Tp> void deallocate(_Tp* ptr, size_t n)

    Allocates an array of elements





    
    :param ptr: Pointer to the deallocated buffer 
    
    
    :param n: The number of elements in the buffer 
    
    
    
The generic function 
``deallocate``
deallocates the buffer allocated with 
:func:`allocate`
. The number of elements must match the number passed to 
:func:`allocate`
.


.. index:: CV_Assert

.. _CV_Assert:

CV_Assert
---------

`id=0.132247699783 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/CV_Assert>`__




.. cfunction:: CV_Assert(expr)

    Checks a condition at runtime.






::


    
    #define CV_Assert( expr ) ...
    #define CV_DbgAssert(expr) ...
    

..



    
    :param expr: The checked expression 
    
    
    
The macros 
``CV_Assert``
and 
``CV_DbgAssert``
evaluate the specified expression and if it is 0, the macros raise an error (see 
:func:`error`
). The macro 
``CV_Assert``
checks the condition in both Debug and Release configurations, while 
``CV_DbgAssert``
is only retained in the Debug configuration.


.. index:: error


cv::error
---------

`id=0.274198769781 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/error>`__




.. cfunction:: void error( const Exception\& exc )



.. cfunction:: \#define CV_Error( code, msg ) <...>



.. cfunction:: \#define CV_Error_( code, args ) <...>

    Signals an error and raises the exception





    
    :param exc: The exception to throw 
    
    
    :param code: The error code, normally, a negative value. The list of pre-defined error codes can be found in  ``cxerror.h`` 
    
    
    :param msg: Text of the error message 
    
    
    :param args: printf-like formatted error message in parantheses 
    
    
    
The function and the helper macros 
``CV_Error``
and 
``CV_Error_``
call the error handler. Currently, the error handler prints the error code (
``exc.code``
), the context (
``exc.file``
, 
``exc.line``
and the error message 
``exc.err``
to the standard error stream 
``stderr``
. In Debug configuration it then provokes memory access violation, so that the execution stack and all the parameters can be analyzed in debugger. In Release configuration the exception 
``exc``
is thrown.

The macro 
``CV_Error_``
can be used to construct the error message on-fly to include some dynamic information, for example:




::


    
    // note the extra parentheses around the formatted text message
    CV_Error_(CV_StsOutOfRange,
        ("the matrix element (
        i, j, mtx.at<float>(i,j)))
    

..


.. index:: Exception

.. _Exception:

Exception
---------

`id=0.792198322059 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/Exception>`__

.. ctype:: Exception



The exception class passed to error




::


    
    class  Exception
    {
    public:
        // various constructors and the copy operation
        Exception() { code = 0; line = 0; }
        Exception(int _code, const string& _err,
                  const string& _func, const string& _file, int _line);
        Exception(const Exception& exc);
        Exception& operator = (const Exception& exc);
    
        // the error code
        int code;
        // the error text message
        string err;
        // function name where the error happened
        string func;
        // the source file name where the error happened
        string file;
        // the source file line where the error happened
        int line;
    };
    

..

The class 
``Exception``
encapsulates all or almost all the necessary information about the error happened in the program. The exception is usually constructed and thrown implicitly, via 
``CV_Error``
and 
``CV_Error_``
macros, see 
:func:`error`
.



.. index:: fastMalloc


cv::fastMalloc
--------------

`id=0.913748026438 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/fastMalloc>`__




.. cfunction:: void* fastMalloc(size_t size)

    Allocates aligned memory buffer





    
    :param size: The allocated buffer size 
    
    
    
The function allocates buffer of the specified size and returns it. When the buffer size is 16 bytes or more, the returned buffer is aligned on 16 bytes.


.. index:: fastFree


cv::fastFree
------------

`id=0.486348253472 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/fastFree>`__




.. cfunction:: void fastFree(void* ptr)

    Deallocates memory buffer





    
    :param ptr: Pointer to the allocated buffer 
    
    
    
The function deallocates the buffer, allocated with 
:func:`fastMalloc`
.
If NULL pointer is passed, the function does nothing.


.. index:: format


cv::format
----------

`id=0.359045522761 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/format>`__




.. cfunction:: string format( const char* fmt, ... )

    Returns a text string formatted using printf-like expression





    
    :param fmt: The printf-compatible formatting specifiers 
    
    
    
The function acts like 
``sprintf``
, but forms and returns STL string. It can be used for form the error message in 
:func:`Exception`
constructor.


.. index:: getNumThreads


cv::getNumThreads
-----------------

`id=0.665594834701 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/getNumThreads>`__




.. cfunction:: int getNumThreads()

    Returns the number of threads used by OpenCV



The function returns the number of threads that is used by OpenCV.

See also: 
:func:`setNumThreads`
, 
:func:`getThreadNum`
.



.. index:: getThreadNum


cv::getThreadNum
----------------

`id=0.835208450402 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/getThreadNum>`__




.. cfunction:: int getThreadNum()

    Returns index of the currently executed thread



The function returns 0-based index of the currently executed thread. The function is only valid inside a parallel OpenMP region. When OpenCV is built without OpenMP support, the function always returns 0.

See also: 
:func:`setNumThreads`
, 
:func:`getNumThreads`
.


.. index:: getTickCount


cv::getTickCount
----------------

`id=0.682548115061 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/getTickCount>`__




.. cfunction:: int64 getTickCount()

    Returns the number of ticks



The function returns the number of ticks since the certain event (e.g. when the machine was turned on).
It can be used to initialize 
:func:`RNG`
or to measure a function execution time by reading the tick count before and after the function call. See also the tick frequency.


.. index:: getTickFrequency


cv::getTickFrequency
--------------------

`id=0.85013360741 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/getTickFrequency>`__




.. cfunction:: double getTickFrequency()

    Returns the number of ticks per second



The function returns the number of ticks per second.
That is, the following code computes the execution time in seconds.



::


    
    double t = (double)getTickCount();
    // do something ...
    t = ((double)getTickCount() - t)/getTickFrequency();
    

..


.. index:: setNumThreads


cv::setNumThreads
-----------------

`id=0.215563071229 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/setNumThreads>`__




.. cfunction:: void setNumThreads(int nthreads)

    Sets the number of threads used by OpenCV





    
    :param nthreads: The number of threads used by OpenCV 
    
    
    
The function sets the number of threads used by OpenCV in parallel OpenMP regions. If 
``nthreads=0``
, the function will use the default number of threads, which is usually equal to the number of the processing cores.

See also: 
:func:`getNumThreads`
, 
:func:`getThreadNum`
