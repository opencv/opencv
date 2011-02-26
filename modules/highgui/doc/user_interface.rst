User Interface
==============

.. highlight:: cpp

.. index:: createTrackbar

cv::createTrackbar
------------------
.. cfunction:: int createTrackbar( const string\& trackbarname,                    const string\& winname,                    int* value, int count,                    TrackbarCallback onChange CV_DEFAULT(0),                    void* userdata CV_DEFAULT(0))

    Creates a trackbar and attaches it to the specified window

    :param trackbarname: Name of the created trackbar.

    :param winname: Name of the window which will be used as a parent of the created trackbar.

    :param value: The optional pointer to an integer variable, whose value will reflect the position of the slider. Upon creation, the slider position is defined by this variable.

    :param count: The maximal position of the slider. The minimal position is always 0.

    :param onChange: Pointer to the function to be called every time the slider changes position. This function should be prototyped as  ``void Foo(int,void*);`` , where the first parameter is the trackbar position and the second parameter is the user data (see the next parameter). If the callback is NULL pointer, then no callbacks is called, but only  ``value``  is updated

    :param userdata: The user data that is passed as-is to the callback; it can be used to handle trackbar events without using global variables

The function ``createTrackbar`` creates a trackbar (a.k.a. slider or range control) with the specified name and range, assigns a variable ``value`` to be syncronized with trackbar position and specifies a callback function ``onChange`` to be called on the trackbar position change. The created trackbar is displayed on the top of the given window.
\
\
**[Qt Backend Only]**
qt-specific details:

    * **winname** Name of the window which will be used as a parent for created trackbar. Can be NULL if the trackbar should be attached to the control panel.

The created trackbar is displayed at the bottom of the given window if
*winname*
is correctly provided, or displayed on the control panel if
*winname*
is NULL.

By clicking on the label of each trackbar, it is possible to edit the trackbar's value manually for a more accurate control of it.

.. index:: getTrackbarPos

cv::getTrackbarPos
------------------
.. cfunction:: int getTrackbarPos( const string\& trackbarname,  const string\& winname )

    Returns the trackbar position.

    :param trackbarname: Name of the trackbar.

    :param winname: Name of the window which is the parent of the trackbar.

The function returns the current position of the specified trackbar.
\
\
**[Qt Backend Only]**
qt-specific details:

    * **winname** Name of the window which is the parent of the trackbar. Can be NULL if the trackbar is attached to the control panel.

.. index:: imshow

cv::imshow
----------
.. cfunction:: void imshow( const string\& winname,  const Mat\& image )

    Displays the image in the specified window

    :param winname: Name of the window.

    :param image: Image to be shown.

The function ``imshow`` displays the image in the specified window. If the window was created with the ``CV_WINDOW_AUTOSIZE`` flag then the image is shown with its original size, otherwise the image is scaled to fit in the window. The function may scale the image, depending on its depth:

*
    If the image is 8-bit unsigned, it is displayed as is.

*
    If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the value range [0,255*256] is mapped to [0,255].

*
    If the image is 32-bit floating-point, the pixel values are multiplied by 255. That is, the value range [0,1] is mapped to [0,255].

.. index:: namedWindow

cv::namedWindow
---------------
.. cfunction:: void namedWindow( const string\& winname,  int flags )

    Creates a window.

    :param name: Name of the window in the window caption that may be used as a window identifier.

    :param flags: Flags of the window. Currently the only supported flag is  ``CV_WINDOW_AUTOSIZE`` . If this is set, the window size is automatically adjusted to fit the displayed image (see  :ref:`imshow` ), and the user can not change the window size manually.

The function ``namedWindow`` creates a window which can be used as a placeholder for images and trackbars. Created windows are referred to by their names.

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

            namedWindow( ``myWindow'',  ``CV_WINDOW_NORMAL``   textbar   ``CV_GUI_NORMAL`` );

        ..

.. index:: setTrackbarPos

cv::setTrackbarPos
------------------
.. cfunction:: void setTrackbarPos( const string\& trackbarname,  const string\& winname, int pos )

    Sets the trackbar position.

    :param trackbarname: Name of the trackbar.

    :param winname: Name of the window which is the parent of trackbar.

    :param pos: The new position.

The function sets the position of the specified trackbar in the specified window.
\
\
**[Qt Backend Only]**
qt-specific details:

    * **winname** Name of the window which is the parent of trackbar. Can be NULL if the trackbar is attached to the control panel.

.. index:: waitKey

cv::waitKey
-----------
.. cfunction:: int waitKey(int delay=0)

    Waits for a pressed key.

    :param delay: Delay in milliseconds. 0 is the special value that means "forever"

The function ``waitKey`` waits for key event infinitely (when
:math:`\texttt{delay}\leq 0` ) or for ``delay`` milliseconds, when it's positive. Returns the code of the pressed key or -1 if no key was pressed before the specified time had elapsed.

**Note:**
This function is the only method in HighGUI that can fetch and handle events, so it needs to be called periodically for normal event processing, unless HighGUI is used within some environment that takes care of event processing.

**Note 2:**
The function only works if there is at least one HighGUI window created and the window is active. If there are several HighGUI windows, any of them can be active.

