User Interface
==============

.. highlight:: python



.. index:: CreateTrackbar

.. _CreateTrackbar:

CreateTrackbar
--------------

`id=0.859200002353 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/CreateTrackbar>`__


.. function:: CreateTrackbar(trackbarName, windowName, value, count, onChange) -> None

    Creates a trackbar and attaches it to the specified window





    
    :param trackbarName: Name of the created trackbar. 
    
    :type trackbarName: str
    
    
    :param windowName: Name of the window which will be used as a parent for created trackbar. 
    
    :type windowName: str
    
    
    :param value: Initial value for the slider position, between 0 and  ``count`` . 
    
    :type value: int
    
    
    :param count: Maximal position of the slider. Minimal position is always 0. 
    
    :type count: int
    
    
    :param onChange: 
        OpenCV calls  ``onChange``  every time the slider changes position.
        OpenCV will call it as  ``func(x)``  where  ``x``  is the new position of the slider. 
    
    :type onChange: :class:`PyCallableObject`
    
    
    
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


.. index:: DestroyAllWindows

.. _DestroyAllWindows:

DestroyAllWindows
-----------------

`id=0.386578572057 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/DestroyAllWindows>`__


.. function:: DestroyAllWindows()-> None

    Destroys all of the HighGUI windows.



The function 
``cvDestroyAllWindows``
destroys all of the opened HighGUI windows.


.. index:: DestroyWindow

.. _DestroyWindow:

DestroyWindow
-------------

`id=0.0256606142145 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/DestroyWindow>`__


.. function:: DestroyWindow(name)-> None

    Destroys a window.





    
    :param name: Name of the window to be destroyed. 
    
    :type name: str
    
    
    
The function 
``cvDestroyWindow``
destroys the window with the given name.


.. index:: GetTrackbarPos

.. _GetTrackbarPos:

GetTrackbarPos
--------------

`id=0.0119794922165 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/GetTrackbarPos>`__


.. function:: GetTrackbarPos(trackbarName,windowName)-> None

    Returns the trackbar position.





    
    :param trackbarName: Name of the trackbar. 
    
    :type trackbarName: str
    
    
    :param windowName: Name of the window which is the parent of the trackbar. 
    
    :type windowName: str
    
    
    
The function 
``cvGetTrackbarPos``
returns the current position of the specified trackbar.
\
\
**[Qt Backend Only]**
qt-specific details:


    
    * **windowName** Name of the window which is the parent of the trackbar. Can be NULL if the trackbar is attached to the control panel. 
    
    
    

.. index:: MoveWindow

.. _MoveWindow:

MoveWindow
----------

`id=0.0432662100889 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/MoveWindow>`__


.. function:: MoveWindow(name,x,y)-> None

    Sets the position of the window.





    
    :param name: Name of the window to be moved. 
    
    :type name: str
    
    
    :param x: New x coordinate of the top-left corner 
    
    :type x: int
    
    
    :param y: New y coordinate of the top-left corner 
    
    :type y: int
    
    
    
The function 
``cvMoveWindow``
changes the position of the window.


.. index:: NamedWindow

.. _NamedWindow:

NamedWindow
-----------

`id=0.155885062255 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/NamedWindow>`__


.. function:: NamedWindow(name,flags=CV_WINDOW_AUTOSIZE)-> None

    Creates a window.





    
    :param name: Name of the window in the window caption that may be used as a window identifier. 
    
    :type name: str
    
    
    :param flags: Flags of the window. Currently the only supported flag is  ``CV_WINDOW_AUTOSIZE`` . If this is set, window size is automatically adjusted to fit the displayed image (see  :ref:`ShowImage` ), and the user can not change the window size manually. 
    
    :type flags: int
    
    
    
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

`id=0.266699312987 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/ResizeWindow>`__


.. function:: ResizeWindow(name,width,height)-> None

    Sets the window size.





    
    :param name: Name of the window to be resized. 
    
    :type name: str
    
    
    :param width: New width 
    
    :type width: int
    
    
    :param height: New height 
    
    :type height: int
    
    
    
The function 
``cvResizeWindow``
changes the size of the window.


.. index:: SetMouseCallback

.. _SetMouseCallback:

SetMouseCallback
----------------

`id=0.299310906828 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/SetMouseCallback>`__


.. function:: SetMouseCallback(windowName, onMouse, param) -> None

    Assigns callback for mouse events.





    
    :param windowName: Name of the window. 
    
    :type windowName: str
    
    
    :param onMouse: Callable to be called every time a mouse event occurs in the specified window. This callable should have signature `` Foo(event, x, y, flags, param)-> None `` 
        where  ``event``  is one of  ``CV_EVENT_*`` ,  ``x``  and  ``y``  are the coordinates of the mouse pointer in image coordinates (not window coordinates),  ``flags``  is a combination of  ``CV_EVENT_FLAG_*`` , and  ``param``  is a user-defined parameter passed to the  ``cvSetMouseCallback``  function call. 
    
    :type onMouse: :class:`PyCallableObject`
    
    
    :param param: User-defined parameter to be passed to the callback function. 
    
    :type param: object
    
    
    
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

`id=0.722744232916 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/SetTrackbarPos>`__


.. function:: SetTrackbarPos(trackbarName,windowName,pos)-> None

    Sets the trackbar position.





    
    :param trackbarName: Name of the trackbar. 
    
    :type trackbarName: str
    
    
    :param windowName: Name of the window which is the parent of trackbar. 
    
    :type windowName: str
    
    
    :param pos: New position. 
    
    :type pos: int
    
    
    
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

`id=0.260802502296 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/ShowImage>`__


.. function:: ShowImage(name,image)-> None

    Displays the image in the specified window





    
    :param name: Name of the window. 
    
    :type name: str
    
    
    :param image: Image to be shown. 
    
    :type image: :class:`CvArr`
    
    
    
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

`id=0.742095797983 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/py/highgui/WaitKey>`__


.. function:: WaitKey(delay=0)-> int

    Waits for a pressed key.





    
    :param delay: Delay in milliseconds. 
    
    :type delay: int
    
    
    
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
