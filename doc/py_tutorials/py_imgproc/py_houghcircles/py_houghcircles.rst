.. _Hough_Circles:

Hough Circle Transform
**************************

Goal
=====

In this chapter,
    * We will learn to use Hough Transform to find circles in an image.
    * We will see these functions: **cv2.HoughCircles()**

Theory
========

A circle is represented mathematically as :math:`(x-x_{center})^2 + (y - y_{center})^2 = r^2` where :math:`(x_{center},y_{center})` is the center of the circle, and :math:`r` is the radius of the circle. From equation, we can see we have 3 parameters, so we need a 3D accumulator for hough transform, which would be highly ineffective. So OpenCV uses more trickier method, **Hough Gradient Method** which uses the gradient information of edges.

The function we use here is **cv2.HoughCircles()**. It has plenty of arguments which are well explained in the documentation. So we directly go to the code.
::

    import cv2
    import numpy as np

    img = cv2.imread('opencv_logo.png',0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Result is shown below:

    .. image:: images/houghcircles2.jpg
        :alt: Hough Circles
        :align: center

Additional Resources
=====================

Exercises
===========
