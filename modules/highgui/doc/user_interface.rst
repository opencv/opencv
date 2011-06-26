User Interface
==============

.. highlight:: cpp

createTrackbar
------------------
Creates a trackbar and attaches it to the specified window.

.. ocv:function:: int createTrackbar( const string& trackbarname, const string& winname, int* value, int count, TrackbarCallback onChange=0, void* userdata=0)

.. ocv:cfunction:: int cvCreateTrackbar( const char* trackbarName, const char* windowName, int* value, int count, CvTrackbarCallback onChange )
.. ocv:pyoldfunction:: CreateTrackbar(trackbarName, windowName, value, count, onChange) -> None

    :param trackbarname: Name of the created trackbar.

    :param winname: Name of the window that will be used as a parent of the created trackbar.

    :param value: Optional pointer to an integer variable whose value reflects the position of the slider. Upon creation, the slider position is defined by this variable.

    :param count: Maximal position of the slider. The minimal position is always 0.

    :param onChange: Pointer to the function to be called every time the slider changes position. This function should be prototyped as  ``void Foo(int,void*);`` , where the first parameter is the trackbar position and the second parameter is the user data (see the next parameter). If the callback is the NULL pointer, no callbacks are called, but only  ``value``  is updated.

    :param userdata: User data that is passed as is to the callback. It can be used to handle trackbar events without using global variables.

The function ``createTrackbar`` creates a trackbar (a.k.a. slider or range control) with the specified name and range, assigns a variable ``value`` to be syncronized with the trackbar position and specifies the callback function ``onChange`` to be called on the trackbar position change. The created trackbar is displayed on top of the given window.


**[Qt Backend Only]**
Qt-specific details:

    * **winname** Name of the window that will be used as a parent for created trackbar. It can be NULL if the trackbar should be attached to the control panel.

The created trackbar is displayed at the bottom of the given window if
*winname*
is correctly provided, or displayed on the control panel if
*winname*
is NULL.

Clicking the label of each trackbar enables editing the trackbar values manually for a more accurate control of it.

getTrackbarPos
------------------
Returns the trackbar position.

.. ocv:function:: int getTrackbarPos( const string& trackbarname, const string& winname )

.. ocv:cfunction:: int cvGetTrackbarPos( const char* trackbarName, const char* windowName )
.. ocv:pyoldfunction:: GetTrackbarPos(trackbarName, windowName)-> None

    :param trackbarname: Name of the trackbar.

    :param winname: Name of the window that is the parent of the trackbar.

The function returns the current position of the specified trackbar.


**[Qt Backend Only]**
Qt-specific details:

    * **winname** Name of the window that is the parent of the trackbar. It can be NULL if the trackbar is attached to the control panel.

imshow
----------
Displays an image in the specified window.

.. ocv:function:: void imshow( const string& winname, InputArray image )

    :param winname: Name of the window.

    :param image: Image to be shown.

The function ``imshow`` displays an image in the specified window. If the window was created with the ``CV_WINDOW_AUTOSIZE`` flag, the image is shown with its original size. Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:

*
    If the image is 8-bit unsigned, it is displayed as is.

*
    If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the value range [0,255*256] is mapped to [0,255].

*
    If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255].


namedWindow
---------------
Creates a window.

.. ocv:function:: void namedWindow( const string& winname, int flags )

.. ocv:cfunction:: int cvNamedWindow( const char* name, int flags )
.. ocv:pyoldfunction:: NamedWindow(name, flags=CV_WINDOW_AUTOSIZE)-> None

    :param name: Name of the window in the window caption that may be used as a window identifier.

    :param flags: Flags of the window. Currently the only supported flag is  ``CV_WINDOW_AUTOSIZE`` . If this is set, the window size is automatically adjusted to fit the displayed image (see  :ref:`imshow` ), and you cannot change the window size manually.

The function ``namedWindow`` creates a window that can be used as a placeholder for images and trackbars. Created windows are referred to by their names.

If a window with the same name already exists, the function does nothing.

You can call :cpp:func:`destroyWindow` or :cpp:func:`destroyAllWindows` to close the window and de-allocate any associated memory usage. For a simple program, you do not really have to call these functions because all the resources and windows of the application are closed automatically by the operating system upon exit.


**[Qt Backend Only]**
Qt-specific details:

    * **flags** Flags of the window. Currently the supported flags are:

            * **CV_WINDOW_NORMAL or CV_WINDOW_AUTOSIZE:**   ``CV_WINDOW_NORMAL``  enables you to resize the window, whereas   ``CV_WINDOW_AUTOSIZE``  adjusts automatically the window size to fit the displayed image (see  :ref:`imshow` ), and you cannot change the window size manually.

            * **CV_WINDOW_FREERATIO or CV_WINDOW_KEEPRATIO:** ``CV_WINDOW_FREERATIO``  adjusts the image with no respect to its ratio, whereas  ``CV_WINDOW_KEEPRATIO``  keeps the image ratio.

            * **CV_GUI_NORMAL or CV_GUI_EXPANDED:**   ``CV_GUI_NORMAL``  is the old way to draw the window without statusbar and toolbar, whereas  ``CV_GUI_EXPANDED``  is a new enhanced GUI.

        This parameter is optional. The default flags set for a new window are  ``CV_WINDOW_AUTOSIZE`` , ``CV_WINDOW_KEEPRATIO`` , and  ``CV_GUI_EXPANDED`` .

        However, if you want to modify the flags, you can combine them using the OR operator, that is:

        ::

            namedWindow( "myWindow", CV_WINDOW_NORMAL | CV_GUI_NORMAL );

        ..


destroyWindow
-------------
Destroys a window.

.. ocv:function:: void destroyWindow( const string &winname )
            
.. ocv:cfunction:: void cvDestroyWindow( const char* name )
.. ocv:pyoldfunction:: DestroyWindow(name)-> None

    :param winname: Name of the window to be destroyed. 
                                           
The function ``destroyWindow`` destroys the window with the given name.


destroyAllWindows
-----------------
Destroys all of the HighGUI windows.

.. ocv:function:: void destroyAllWindows()

The function ``destroyAllWindows`` destroys all of the opened HighGUI windows.


setTrackbarPos
------------------
Sets the trackbar position.

.. ocv:function:: void setTrackbarPos( const string& trackbarname, const string& winname, int pos )

.. ocv:cfunction:: void cvSetTrackbarPos( const char* trackbarName, const char* windowName, int pos )
.. ocv:pyoldfunction:: SetTrackbarPos(trackbarName, windowName, pos)-> None

    :param trackbarname: Name of the trackbar.

    :param winname: Name of the window that is the parent of trackbar.

    :param pos: New position.

The function sets the position of the specified trackbar in the specified window.


**[Qt Backend Only]**
Qt-specific details:

    * **winname** Name of the window that is the parent of the trackbar. It can be NULL if the trackbar is attached to the control panel.

waitKey
-----------
Waits for a pressed key.

.. ocv:function:: int waitKey(int delay=0)

.. ocv:cfunction:: int cvWaitKey( int delay=0 )
.. ocv:pyoldfunction:: WaitKey(delay=0)-> int

    :param delay: Delay in milliseconds. 0 is the special value that means "forever".

The function ``waitKey`` waits for a key event infinitely (when
:math:`\texttt{delay}\leq 0` ) or for ``delay`` milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the function will not wait exactly ``delay`` ms, it will wait at least ``delay`` ms, depending on what else is running on your computer at that time. It returns the code of the pressed key or -1 if no key was pressed before the specified time had elapsed.

**Notes:**

* This function is the only method in HighGUI that can fetch and handle events, so it needs to be called periodically for normal event processing unless HighGUI is used within an environment that takes care of event processing.

* The function only works if there is at least one HighGUI window created and the window is active. If there are several HighGUI windows, any of them can be active.

