User Interface
==============

.. highlight:: c



.. index:: ConvertImage

.. _ConvertImage:

ConvertImage
------------






.. cfunction:: void cvConvertImage( const CvArr* src, CvArr* dst, int flags=0 )

    Converts one image to another with an optional vertical flip.





    
    :param src: Source image. 
    
    
    :param dst: Destination image. Must be single-channel or 3-channel 8-bit image. 
    
    
    :param flags: The operation flags: 
         
            * **CV_CVTIMG_FLIP** Flips the image vertically 
            
            * **CV_CVTIMG_SWAP_RB** Swaps the red and blue channels. In OpenCV color images have  ``BGR``  channel order, however on some systems the order needs to be reversed before displaying the image ( :ref:`ShowImage`  does this automatically). 
            
            
    
    
    
The function 
``cvConvertImage``
converts one image to another and flips the result vertically if desired. The function is used by 
:ref:`ShowImage`
.


.. index:: CreateTrackbar

.. _CreateTrackbar:

CreateTrackbar
--------------






.. cfunction:: int cvCreateTrackbar(  const char* trackbarName,  const char* windowName,                        int* value,  int count,  CvTrackbarCallback onChange )

    Creates a trackbar and attaches it to the specified window





    
    :param trackbarName: Name of the created trackbar. 
    
    
    :param windowName: Name of the window which will be used as a parent for created trackbar. 
    
    
    :param value: Pointer to an integer variable, whose value will reflect the position of the slider. Upon creation, the slider position is defined by this variable. 
    
    
    :param count: Maximal position of the slider. Minimal position is always 0. 
    
    
    :param onChange: 
        Pointer to the function to be called every time the slider changes position.
        This function should be prototyped as  ``void Foo(int);``   Can be NULL if callback is not required. 
    
    
    
The function 
``cvCreateTrackbar``
creates a trackbar (a.k.a. slider or range control) with the specified name and range, assigns a variable to be syncronized with trackbar position and specifies a callback function to be called on trackbar position change. The created trackbar is displayed on the top of the given window.
\
\
**[Qt Backend Only]**
qt-specific details:


    
    * **windowName** Name of the window which will be used as a parent for created trackbar. Can be NULL if the trackbar should be attached to the control panel. 
    
    
    
The created trackbar is displayed at the bottom of the given window if 
*windowName*
is correctly provided, or displayed on the control panel if 
*windowName*
is NULL.

By clicking on the label of each trackbar, it is possible to edit the trackbar's value manually for a more accurate control of it.




::


    
    CV_EXTERN_C_FUNCPTR( void (*CvTrackbarCallback)(int pos) );
    

..


.. index:: DestroyAllWindows

.. _DestroyAllWindows:

DestroyAllWindows
-----------------






.. cfunction:: void cvDestroyAllWindows(void)

    Destroys all of the HighGUI windows.



The function 
``cvDestroyAllWindows``
destroys all of the opened HighGUI windows.


.. index:: DestroyWindow

.. _DestroyWindow:

DestroyWindow
-------------






.. cfunction:: void cvDestroyWindow( const char* name )

    Destroys a window.





    
    :param name: Name of the window to be destroyed. 
    
    
    
The function 
``cvDestroyWindow``
destroys the window with the given name.


.. index:: GetTrackbarPos

.. _GetTrackbarPos:

GetTrackbarPos
--------------






.. cfunction:: int cvGetTrackbarPos(  const char* trackbarName,  const char* windowName )

    Returns the trackbar position.





    
    :param trackbarName: Name of the trackbar. 
    
    
    :param windowName: Name of the window which is the parent of the trackbar. 
    
    
    
The function 
``cvGetTrackbarPos``
returns the current position of the specified trackbar.
\
\
**[Qt Backend Only]**
qt-specific details:


    
    * **windowName** Name of the window which is the parent of the trackbar. Can be NULL if the trackbar is attached to the control panel. 
    
    
    

.. index:: GetWindowHandle

.. _GetWindowHandle:

GetWindowHandle
---------------






.. cfunction:: void* cvGetWindowHandle( const char* name )

    Gets the window's handle by its name.





    
    :param name: Name of the window 
    
    .
    
    
The function 
``cvGetWindowHandle``
returns the native window handle (HWND in case of Win32 and GtkWidget in case of GTK+).
\
\
**[Qt Backend Only]**
qt-specific details:
The function 
``cvGetWindowHandle``
returns the native window handle inheriting from the Qt class QWidget.


.. index:: GetWindowName

.. _GetWindowName:

GetWindowName
-------------






.. cfunction:: const char* cvGetWindowName( void* windowHandle )

    Gets the window's name by its handle.





    
    :param windowHandle: Handle of the window. 
    
    
    
The function 
``cvGetWindowName``
returns the name of the window given its native handle (HWND in case of Win32 and GtkWidget in case of GTK+).
\
\
**[Qt Backend Only]**
qt-specific details:
The function 
``cvGetWindowName``
returns the name of the window given its native handle (QWidget).


.. index:: InitSystem

.. _InitSystem:

InitSystem
----------






.. cfunction:: int cvInitSystem( int argc, char** argv )

    Initializes HighGUI.





    
    :param argc: Number of command line arguments 
    
    
    :param argv: Array of command line arguments 
    
    
    
The function 
``cvInitSystem``
initializes HighGUI. If it wasn't
called explicitly by the user before the first window was created, it is
called implicitly then with 
``argc=0``
, 
``argv=NULL``
. Under
Win32 there is no need to call it explicitly. Under X Window the arguments
may be used to customize a look of HighGUI windows and controls.
\
\
**[Qt Backend Only]**
qt-specific details:
The function 
``cvInitSystem``
is automatically called at the first cvNameWindow call. 


.. index:: MoveWindow

.. _MoveWindow:

MoveWindow
----------






.. cfunction:: void cvMoveWindow( const char* name, int x, int y )

    Sets the position of the window.





    
    :param name: Name of the window to be moved. 
    
    
    :param x: New x coordinate of the top-left corner 
    
    
    :param y: New y coordinate of the top-left corner 
    
    
    
The function 
``cvMoveWindow``
changes the position of the window.


.. index:: NamedWindow

.. _NamedWindow:

NamedWindow
-----------






.. cfunction:: int cvNamedWindow( const char* name, int flags )

    Creates a window.





    
    :param name: Name of the window in the window caption that may be used as a window identifier. 
    
    
    :param flags: Flags of the window. Currently the only supported flag is  ``CV_WINDOW_AUTOSIZE`` . If this is set, window size is automatically adjusted to fit the displayed image (see  :ref:`ShowImage` ), and the user can not change the window size manually. 
    
    
    
The function 
``cvNamedWindow``
creates a window which can be used as a placeholder for images and trackbars. Created windows are referred to by their names.

If a window with the same name already exists, the function does nothing.
\
\
**[Qt Backend Only]**
qt-specific details:


    
    * **flags** Flags of the window. Currently the supported flags are: 
        
                              
            * **CV_WINDOW_NORMAL or CV_WINDOW_AUTOSIZE:**   ``CV_WINDOW_NORMAL``  let the user resize the window, whereas   ``CV_WINDOW_AUTOSIZE``  adjusts automatically the window's size to fit the displayed image (see  :ref:`ShowImage` ), and the user can not change the window size manually. 
            
                             
            * **CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO:** ``CV_WINDOW_FREERATIO``  adjust the image without respect the its ration, whereas  ``CV_WINDOW_KEEPRATIO``  keep the image's ratio. 
            
                             
            * **CV_GUI_NORMAL or CV_GUI_EXPANDED:**   ``CV_GUI_NORMAL``  is the old way to draw the window without statusbar and toolbar, whereas  ``CV_GUI_EXPANDED``  is the new enhance GUI. 
            
            
        
        This parameter is optional. The default flags set for a new window are  ``CV_WINDOW_AUTOSIZE`` ,  ``CV_WINDOW_KEEPRATIO`` , and  ``CV_GUI_EXPANDED`` .
        
        However, if you want to modify the flags, you can combine them using OR operator, ie: 
        
        
        ::
        
        
            
            cvNamedWindow( ``myWindow'',  ``CV_WINDOW_NORMAL``   textbar   ``CV_GUI_NORMAL`` ); 
            
            
        
        ..
        
        
        
    
.. index:: ResizeWindow

.. _ResizeWindow:

ResizeWindow
------------






.. cfunction:: void cvResizeWindow( const char* name, int width, int height )

    Sets the window size.





    
    :param name: Name of the window to be resized. 
    
    
    :param width: New width 
    
    
    :param height: New height 
    
    
    
The function 
``cvResizeWindow``
changes the size of the window.


.. index:: SetMouseCallback

.. _SetMouseCallback:

SetMouseCallback
----------------






.. cfunction:: void cvSetMouseCallback( const char* windowName, CvMouseCallback onMouse, void* param=NULL )

    Assigns callback for mouse events.





    
    :param windowName: Name of the window. 
    
    
    :param onMouse: Pointer to the function to be called every time a mouse event occurs in the specified window. This function should be prototyped as ``void Foo(int event, int x, int y, int flags, void* param);`` 
        where  ``event``  is one of  ``CV_EVENT_*`` ,  ``x``  and  ``y``  are the coordinates of the mouse pointer in image coordinates (not window coordinates),  ``flags``  is a combination of  ``CV_EVENT_FLAG_*`` , and  ``param``  is a user-defined parameter passed to the  ``cvSetMouseCallback``  function call. 
    
    
    :param param: User-defined parameter to be passed to the callback function. 
    
    
    
The function 
``cvSetMouseCallback``
sets the callback function for mouse events occuring within the specified window. 

The 
``event``
parameter is one of:



    
    * **CV_EVENT_MOUSEMOVE** Mouse movement 
    
    
    * **CV_EVENT_LBUTTONDOWN** Left button down 
    
    
    * **CV_EVENT_RBUTTONDOWN** Right button down 
    
    
    * **CV_EVENT_MBUTTONDOWN** Middle button down 
    
    
    * **CV_EVENT_LBUTTONUP** Left button up 
    
    
    * **CV_EVENT_RBUTTONUP** Right button up 
    
    
    * **CV_EVENT_MBUTTONUP** Middle button up 
    
    
    * **CV_EVENT_LBUTTONDBLCLK** Left button double click 
    
    
    * **CV_EVENT_RBUTTONDBLCLK** Right button double click 
    
    
    * **CV_EVENT_MBUTTONDBLCLK** Middle button double click 
    
    
    
The 
``flags``
parameter is a combination of :



    
    * **CV_EVENT_FLAG_LBUTTON** Left button pressed 
    
    
    * **CV_EVENT_FLAG_RBUTTON** Right button pressed 
    
    
    * **CV_EVENT_FLAG_MBUTTON** Middle button pressed 
    
    
    * **CV_EVENT_FLAG_CTRLKEY** Control key pressed 
    
    
    * **CV_EVENT_FLAG_SHIFTKEY** Shift key pressed 
    
    
    * **CV_EVENT_FLAG_ALTKEY** Alt key pressed 
    
    
    

.. index:: SetTrackbarPos

.. _SetTrackbarPos:

SetTrackbarPos
--------------






.. cfunction:: void cvSetTrackbarPos(  const char* trackbarName,  const char* windowName,  int pos )

    Sets the trackbar position.





    
    :param trackbarName: Name of the trackbar. 
    
    
    :param windowName: Name of the window which is the parent of trackbar. 
    
    
    :param pos: New position. 
    
    
    
The function 
``cvSetTrackbarPos``
sets the position of the specified trackbar.
\
\
**[Qt Backend Only]**
qt-specific details:


    
    * **windowName** Name of the window which is the parent of trackbar.  Can be NULL if the trackbar is attached to the control panel. 
    
    
    

.. index:: ShowImage

.. _ShowImage:

ShowImage
---------






.. cfunction:: void cvShowImage( const char* name, const CvArr* image )

    Displays the image in the specified window





    
    :param name: Name of the window. 
    
    
    :param image: Image to be shown. 
    
    
    
The function 
``cvShowImage``
displays the image in the specified window. If the window was created with the 
``CV_WINDOW_AUTOSIZE``
flag then the image is shown with its original size, otherwise the image is scaled to fit in the window. The function may scale the image, depending on its depth:


    

*
    If the image is 8-bit unsigned, it is displayed as is.
        
    

*
    If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the value range [0,255*256] is mapped to [0,255].
        
    

*
    If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255].
    
    

.. index:: WaitKey

.. _WaitKey:

WaitKey
-------






.. cfunction:: int cvWaitKey( int delay=0 )

    Waits for a pressed key.





    
    :param delay: Delay in milliseconds. 
    
    
    
The function 
``cvWaitKey``
waits for key event infinitely (
:math:`\texttt{delay} <= 0`
) or for 
``delay``
milliseconds. Returns the code of the pressed key or -1 if no key was pressed before the specified time had elapsed.

**Note:**
This function is the only method in HighGUI that can fetch and handle events, so it needs to be called periodically for normal event processing, unless HighGUI is used within some environment that takes care of event processing.
\
\
**[Qt Backend Only]**
qt-specific details:
With this current Qt implementation, this is the only way to process event such as repaint for the windows, and so on 
ldots
