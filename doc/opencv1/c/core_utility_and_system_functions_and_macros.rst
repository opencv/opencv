Utility and System Functions and Macros
=======================================

.. highlight:: c



Error Handling
--------------


Error handling in OpenCV is similar to IPL (Image Processing
Library). In the case of an error, functions do not return the error
code. Instead, they raise an error using 
``CV_ERROR``
macro that calls 
:ref:`Error`
that, in its turn, sets the error
status with 
:ref:`SetErrStatus`
and calls a standard or user-defined
error handler (that can display a message box, write to log, etc., see
:ref:`RedirectError`
).  There is a global variable, one per each program
thread, that contains current error status (an integer value). The status
can be retrieved with the 
:ref:`GetErrStatus`
function.

There are three modes of error handling (see 
:ref:`SetErrMode`
and
:ref:`GetErrMode`
):



    

*
    **Leaf**
    . The program is terminated after the error handler is
    called. This is the default value. It is useful for debugging, as the
    error is signalled immediately after it occurs. However, for production
    systems, other two methods may be preferable as they provide more
    control.
    

*
    **Parent**
    . The program is not terminated, but the error handler
    is called. The stack is unwound (it is done w/o using a C++ exception
    mechanism). The user may check error code after calling the 
    ``CxCore``
    function with
    :ref:`GetErrStatus`
    and react.
    

*
    **Silent**
    . Similar to 
    ``Parent``
    mode, but no error handler
    is called.
    
    
Actually, the semantics of the 
``Leaf``
and 
``Parent``
modes are implemented by error handlers and the above description is true for them. 
:ref:`GuiBoxReport`
behaves slightly differently, and some custom error handlers may implement quite different semantics.  

Macros for raising an error, checking for errors, etc.



::


    
    
    /* special macros for enclosing processing statements within a function and separating
       them from prologue (resource initialization) and epilogue (guaranteed resource release) */
    #define __BEGIN__       {
    #define __END__         goto exit; exit: ; }
    /* proceeds to "resource release" stage */
    #define EXIT            goto exit
    
    /* Declares locally the function name for CV_ERROR() use */
    #define CV_FUNCNAME( Name )  \
        static char cvFuncName[] = Name
    
    /* Raises an error within the current context */
    #define CV_ERROR( Code, Msg )                                       \
    
    
    /* Checks status after calling CXCORE function */
    #define CV_CHECK()                                                  \
    
    
    /* Provies shorthand for CXCORE function call and CV_CHECK() */
    #define CV_CALL( Statement )                                        \
    
    
    /* Checks some condition in both debug and release configurations */
    #define CV_ASSERT( Condition )                                          \
    
    
    /* these macros are similar to their CV_... counterparts, but they
       do not need exit label nor cvFuncName to be defined */
    #define OPENCV_ERROR(status,func_name,err_msg) ...
    #define OPENCV_ERRCHK(func_name,err_msg) ...
    #define OPENCV_ASSERT(condition,func_name,err_msg) ...
    #define OPENCV_CALL(statement) ...
    
    

..

Instead of a discussion, below is a documented example of a typical CXCORE function and an example of the function use.


Example: Use of Error Handling Macros
-------------------------------------





::


    
    
    #include "cxcore.h"
    #include <stdio.h>
    
    void cvResizeDCT( CvMat* input_array, CvMat* output_array )
    {
        CvMat* temp_array = 0; // declare pointer that should be released anyway.
    
        CV_FUNCNAME( "cvResizeDCT" ); // declare cvFuncName
    
        __BEGIN__; // start processing. There may be some declarations just after 
                  // this macro, but they could not be accessed from the epilogue.
    
        if( !CV_IS_MAT(input_array) || !CV_IS_MAT(output_array) )
            // use CV_ERROR() to raise an error
            CV_ERROR( CV_StsBadArg, 
            "input_array or output_array are not valid matrices" );
    
        // some restrictions that are going to be removed later, may be checked 
        // with CV_ASSERT()
        CV_ASSERT( input_array->rows == 1 && output_array->rows == 1 );
    
        // use CV_CALL for safe function call
        CV_CALL( temp_array = cvCreateMat( input_array->rows,
                                           MAX(input_array->cols,
                                           output_array->cols),
                                           input_array->type ));
    
        if( output_array->cols > input_array->cols )
            CV_CALL( cvZero( temp_array ));
    
        temp_array->cols = input_array->cols;
        CV_CALL( cvDCT( input_array, temp_array, CV_DXT_FORWARD ));
        temp_array->cols = output_array->cols;
        CV_CALL( cvDCT( temp_array, output_array, CV_DXT_INVERSE ));
        CV_CALL( cvScale( output_array,
                          output_array,
                          1./sqrt((double)input_array->cols*output_array->cols), 0 ));
    
        __END__; // finish processing. Epilogue follows after the macro.
    
        // release temp_array. If temp_array has not been allocated
        // before an error occured, cvReleaseMat
        // takes care of it and does nothing in this case.
        cvReleaseMat( &temp_array );
    }
    
    int main( int argc, char** argv )
    {
        CvMat* src = cvCreateMat( 1, 512, CV_32F );
    #if 1 /* no errors */
        CvMat* dst = cvCreateMat( 1, 256, CV_32F );
    #else
        CvMat* dst = 0; /* test error processing mechanism */
    #endif
        cvSet( src, cvRealScalar(1.), 0 );
    #if 0 /* change 0 to 1 to suppress error handler invocation */
        cvSetErrMode( CV_ErrModeSilent );
    #endif
        cvResizeDCT( src, dst ); // if some error occurs, the message
                                 // box will popup, or a message will be
                                 // written to log, or some user-defined
                                 // processing will be done
        if( cvGetErrStatus() < 0 )
            printf("Some error occured" );
        else
            printf("Everything is OK" );
        return 0;
    }
    

..


.. index:: GetErrStatus

.. _GetErrStatus:

GetErrStatus
------------






.. cfunction:: int cvGetErrStatus( void )

    Returns the current error status.



The function returns the current error status -
the value set with the last 
:ref:`SetErrStatus`
call. Note that in
``Leaf``
mode, the program terminates immediately after an
error occurs, so to always gain control after the function call,
one should call 
:ref:`SetErrMode`
and set the 
``Parent``
or 
``Silent``
error mode.


.. index:: SetErrStatus

.. _SetErrStatus:

SetErrStatus
------------






.. cfunction:: void cvSetErrStatus( int status )

    Sets the error status.





    
    :param status: The error status 
    
    
    
The function sets the error status to the specified value. Mostly, the function is used to reset the error status (set to it 
``CV_StsOk``
) to recover after an error. In other cases it is more natural to call 
:ref:`Error`
or 
``CV_ERROR``
.


.. index:: GetErrMode

.. _GetErrMode:

GetErrMode
----------






.. cfunction:: int cvGetErrMode(void)

    Returns the current error mode.



The function returns the current error mode - the value set with the last 
:ref:`SetErrMode`
call.


.. index:: SetErrMode

.. _SetErrMode:

SetErrMode
----------







::


    
    

..



.. cfunction:: int cvSetErrMode( int mode )

    Sets the error mode.

#define CV_ErrModeLeaf    0
#define CV_ErrModeParent  1
#define CV_ErrModeSilent  2




    
    :param mode: The error mode 
    
    
    
The function sets the specified error mode. For descriptions of different error modes, see the beginning of the error section.


.. index:: Error

.. _Error:

Error
-----






.. cfunction:: int cvError(  int status, const char* func_name, const char* err_msg, const char* filename, int line )

    Raises an error.





    
    :param status: The error status 
    
    
    :param func_name: Name of the function where the error occured 
    
    
    :param err_msg: Additional information/diagnostics about the error 
    
    
    :param filename: Name of the file where the error occured 
    
    
    :param line: Line number, where the error occured 
    
    
    
The function sets the error status to the specified value (via 
:ref:`SetErrStatus`
) and, if the error mode is not 
``Silent``
, calls the error handler.


.. index:: ErrorStr

.. _ErrorStr:

ErrorStr
--------






.. cfunction:: const char* cvErrorStr( int status )

    Returns textual description of an error status code.





    
    :param status: The error status 
    
    
    
The function returns the textual description for
the specified error status code. In the case of unknown status, the function
returns a NULL pointer.


.. index:: RedirectError

.. _RedirectError:

RedirectError
-------------






.. cfunction:: CvErrorCallback cvRedirectError(  CvErrorCallback error_handler, void* userdata=NULL, void** prevUserdata=NULL )

    Sets a new error handler.






    
    :param error_handler: The new error _ handler 
    
    
    :param userdata: Arbitrary pointer that is transparently passed to the error handler 
    
    
    :param prevUserdata: Pointer to the previously assigned user data pointer 
    
    
    



::


    
    typedef int (CV_CDECL *CvErrorCallback)( int status, const char* func_name,
                        const char* err_msg, const char* file_name, int line );
    

..

The function sets a new error handler that
can be one of the standard handlers or a custom handler
that has a specific interface. The handler takes the same parameters
as the 
:ref:`Error`
function. If the handler returns a non-zero value, the
program is terminated; otherwise, it continues. The error handler may
check the current error mode with 
:ref:`GetErrMode`
to make a decision.



.. index:: cvNulDevReport cvStdErrReport cvGuiBoxReport

.. _cvNulDevReport cvStdErrReport cvGuiBoxReport:

cvNulDevReport cvStdErrReport cvGuiBoxReport
--------------------------------------------






.. cfunction:: int cvNulDevReport( int status, const char* func_name,                     const char* err_msg, const char* file_name,                     int line, void* userdata )



.. cfunction:: int cvStdErrReport( int status, const char* func_name,                     const char* err_msg, const char* file_name,                     int line, void* userdata )



.. cfunction:: int cvGuiBoxReport( int status, const char* func_name,                     const char* err_msg, const char* file_name,                     int line, void* userdata )

    Provide standard error handling.





    
    :param status: The error status 
    
    
    :param func_name: Name of the function where the error occured 
    
    
    :param err_msg: Additional information/diagnostics about the error 
    
    
    :param filename: Name of the file where the error occured 
    
    
    :param line: Line number, where the error occured 
    
    
    :param userdata: Pointer to the user data. Ignored by the standard handlers 
    
    
    
The functions 
``cvNullDevReport``
, 
``cvStdErrReport``
,
and 
``cvGuiBoxReport``
provide standard error
handling. 
``cvGuiBoxReport``
is the default error
handler on Win32 systems, 
``cvStdErrReport``
is the default on other
systems. 
``cvGuiBoxReport``
pops up a message box with the error
description and suggest a few options. Below is an example message box
that may be recieved with the sample code above, if one introduces an
error as described in the sample.

**Error Message Box**


.. image:: ../pics/errmsg.png



If the error handler is set to 
``cvStdErrReport``
, the above message will be printed to standard error output and the program will be terminated or continued, depending on the current error mode.

**Error Message printed to Standard Error Output (in ``Leaf`` mode)**



::


    
    OpenCV ERROR: Bad argument (input_array or output_array are not valid matrices)
            in function cvResizeDCT, D:UserVPProjectsavl_probaa.cpp(75)
    Terminating the application...
    

..


.. index:: Alloc

.. _Alloc:

Alloc
-----






.. cfunction:: void* cvAlloc( size_t size )

    Allocates a memory buffer.





    
    :param size: Buffer size in bytes 
    
    
    
The function allocates 
``size``
bytes and returns
a pointer to the allocated buffer. In the case of an error the function reports an
error and returns a NULL pointer. By default, 
``cvAlloc``
calls
``icvAlloc``
which
itself calls 
``malloc``
. However it is possible to assign user-defined memory
allocation/deallocation functions using the 
:ref:`SetMemoryManager`
function.


.. index:: Free

.. _Free:

Free
----






.. cfunction:: void cvFree( void** ptr )

    Deallocates a memory buffer.





    
    :param ptr: Double pointer to released buffer 
    
    
    
The function deallocates a memory buffer allocated by
:ref:`Alloc`
. It clears the pointer to buffer upon exit, which is why
the double pointer is used. If the 
``*buffer``
is already NULL, the function
does nothing.


.. index:: GetTickCount

.. _GetTickCount:

GetTickCount
------------






.. cfunction:: int64 cvGetTickCount( void )

    Returns the number of ticks.



The function returns number of the ticks starting from some platform-dependent event (number of CPU ticks from the startup, number of milliseconds from 1970th year, etc.). The function is useful for accurate measurement of a function/user-code execution time. To convert the number of ticks to time units, use 
:ref:`GetTickFrequency`
.


.. index:: GetTickFrequency

.. _GetTickFrequency:

GetTickFrequency
----------------






.. cfunction:: double cvGetTickFrequency( void )

    Returns the number of ticks per microsecond.



The function returns the number of ticks per microsecond. Thus, the quotient of 
:ref:`GetTickCount`
and 
:ref:`GetTickFrequency`
will give the number of microseconds starting from the platform-dependent event.


.. index:: RegisterModule

.. _RegisterModule:

RegisterModule
--------------







::


    
    

..



.. cfunction:: int cvRegisterModule( const CvModuleInfo* moduleInfo )

    Registers another module.

typedef struct CvPluginFuncInfo
{
    void** func_addr;
    void* default_func_addr;
    const char* func_names;
    int search_modules;
    int loaded_from;
}
CvPluginFuncInfo;

typedef struct CvModuleInfo
{
    struct CvModuleInfo* next;
    const char* name;
    const char* version;
    CvPluginFuncInfo* func_tab;
}
CvModuleInfo;




    
    :param moduleInfo: Information about the module 
    
    
    
The function adds a module to the list of
registered modules. After the module is registered, information about
it can be retrieved using the 
:ref:`GetModuleInfo`
function. Also, the
registered module makes full use of optimized plugins (IPP, MKL, ...),
supported by CXCORE. CXCORE itself, CV (computer vision), CVAUX (auxilary
computer vision), and HIGHGUI (visualization and image/video acquisition) are
examples of modules. Registration is usually done when the shared library
is loaded. See 
``cxcore/src/cxswitcher.cpp``
and
``cv/src/cvswitcher.cpp``
for details about how registration is done
and look at 
``cxcore/src/cxswitcher.cpp``
, 
``cxcore/src/_cxipp.h``
on how IPP and MKL are connected to the modules.


.. index:: GetModuleInfo

.. _GetModuleInfo:

GetModuleInfo
-------------






.. cfunction:: void  cvGetModuleInfo(  const char* moduleName, const char** version, const char** loadedAddonPlugins)

    Retrieves information about registered module(s) and plugins.





    
    :param moduleName: Name of the module of interest, or NULL, which means all the modules 
    
    
    :param version: The output parameter. Information about the module(s), including version 
    
    
    :param loadedAddonPlugins: The list of names and versions of the optimized plugins that CXCORE was able to find and load 
    
    
    
The function returns information about one or
all of the registered modules. The returned information is stored inside
the libraries, so the user should not deallocate or modify the returned
text strings.


.. index:: UseOptimized

.. _UseOptimized:

UseOptimized
------------






.. cfunction:: int cvUseOptimized( int onoff )

    Switches between optimized/non-optimized modes.





    
    :param onoff: Use optimized ( :math:`\ne 0` ) or not ( :math:`=0` ) 
    
    
    
The function switches between the mode, where
only pure C implementations from cxcore, OpenCV, etc. are used, and
the mode, where IPP and MKL functions are used if available. When
``cvUseOptimized(0)``
is called, all the optimized libraries are
unloaded. The function may be useful for debugging, IPP and MKL upgrading on
the fly, online speed comparisons, etc. It returns the number of optimized
functions loaded. Note that by default, the optimized plugins are loaded,
so it is not necessary to call 
``cvUseOptimized(1)``
in the beginning of
the program (actually, it will only increase the startup time).


.. index:: SetMemoryManager

.. _SetMemoryManager:

SetMemoryManager
----------------







::


    
    

..



.. cfunction:: void cvSetMemoryManager(  CvAllocFunc allocFunc=NULL, CvFreeFunc freeFunc=NULL, void* userdata=NULL )

    Accesses custom/default memory managing functions.

typedef void* (CV_CDECL *CvAllocFunc)(size_t size, void* userdata);
typedef int (CV_CDECL *CvFreeFunc)(void* pptr, void* userdata);




    
    :param allocFunc: Allocation function; the interface is similar to  ``malloc`` , except that  ``userdata``  may be used to determine the context 
    
    
    :param freeFunc: Deallocation function; the interface is similar to  ``free`` 
    
    
    :param userdata: User data that is transparently passed to the custom functions 
    
    
    
The function sets user-defined memory
managment functions (substitutes for 
``malloc``
and 
``free``
) that will be called
by 
``cvAlloc, cvFree``
and higher-level functions (e.g., 
``cvCreateImage``
). Note
that the function should be called when there is data allocated using
``cvAlloc``
. Also, to avoid infinite recursive calls, it is not
allowed to call 
``cvAlloc``
and 
:ref:`Free`
from the custom
allocation/deallocation functions.

If the 
``alloc_func``
and 
``free_func``
pointers are
``NULL``
, the default memory managing functions are restored.


.. index:: SetIPLAllocators

.. _SetIPLAllocators:

SetIPLAllocators
----------------







::


    \
    \
    
    

..



.. cfunction:: void cvSetIPLAllocators(                          Cv_iplCreateImageHeader create_header,                          Cv_iplAllocateImageData allocate_data,                          Cv_iplDeallocate deallocate,                          Cv_iplCreateROI create_roi,                          Cv_iplCloneImage clone_image )

    Switches to IPL functions for image allocation/deallocation.

typedef IplImage* (CV_STDCALL* Cv_iplCreateImageHeader)
                            (int,int,int,char*,char*,int,int,int,int,int,
                            IplROI*,IplImage*,void*,IplTileInfo*);
typedef void (CV_STDCALL* Cv_iplAllocateImageData)(IplImage*,int,int);
typedef void (CV_STDCALL* Cv_iplDeallocate)(IplImage*,int);
typedef IplROI* (CV_STDCALL* Cv_iplCreateROI)(int,int,int,int,int);
typedef IplImage* (CV_STDCALL* Cv_iplCloneImage)(const IplImage*);

#define CV_TURN_ON_IPL_COMPATIBILITY()                                      cvSetIPLAllocators( iplCreateImageHeader, iplAllocateImage,                                 iplDeallocate, iplCreateROI, iplCloneImage )




    
    :param create_header: Pointer to iplCreateImageHeader 
    
    
    :param allocate_data: Pointer to iplAllocateImage 
    
    
    :param deallocate: Pointer to iplDeallocate 
    
    
    :param create_roi: Pointer to iplCreateROI 
    
    
    :param clone_image: Pointer to iplCloneImage 
    
    
    
The function causes CXCORE to use IPL functions
for image allocation/deallocation operations. For convenience, there
is the wrapping macro 
``CV_TURN_ON_IPL_COMPATIBILITY``
. The
function is useful for applications where IPL and CXCORE/OpenCV are used
together and still there are calls to 
``iplCreateImageHeader``
,
etc. The function is not necessary if IPL is called only for data
processing and all the allocation/deallocation is done by CXCORE, or
if all the allocation/deallocation is done by IPL and some of OpenCV
functions are used to process the data.

