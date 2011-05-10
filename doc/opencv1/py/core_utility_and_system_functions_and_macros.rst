Utility and System Functions and Macros
=======================================

.. highlight:: python



Error Handling
--------------


Errors in argument type cause a 
``TypeError``
exception.
OpenCV errors cause an 
``cv.error``
exception.

For example a function argument that is the wrong type produces a 
``TypeError``
:




.. doctest::


    
    >>> import cv
    >>> cv.LoadImage(4)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: argument 1 must be string, not int
    

..

A function with the 




.. doctest::


    
    >>> cv.CreateMat(-1, -1, cv.CV_8UC1)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    error: Non-positive width or height
    

..


.. index:: GetTickCount

.. _GetTickCount:

GetTickCount
------------




.. function:: GetTickCount() -> long

    Returns the number of ticks.



The function returns number of the ticks starting from some platform-dependent event (number of CPU ticks from the startup, number of milliseconds from 1970th year, etc.). The function is useful for accurate measurement of a function/user-code execution time. To convert the number of ticks to time units, use 
:ref:`GetTickFrequency`
.


.. index:: GetTickFrequency

.. _GetTickFrequency:

GetTickFrequency
----------------




.. function:: GetTickFrequency() -> long

    Returns the number of ticks per microsecond.



The function returns the number of ticks per microsecond. Thus, the quotient of 
:ref:`GetTickCount`
and 
:ref:`GetTickFrequency`
will give the number of microseconds starting from the platform-dependent event.

