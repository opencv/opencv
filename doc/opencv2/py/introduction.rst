************
Introduction
************


Starting with release 2.0, OpenCV has a new Python interface. This replaces the previous 
`SWIG-based Python interface <http://opencv.willowgarage.com/wiki/SwigPythonInterface>`_
.

Some highlights of the new bindings:



    

* single import of all of OpenCV using ``import cv``
    

* OpenCV functions no longer have the "cv" prefix
    

* simple types like CvRect and CvScalar use Python tuples
    

* sharing of Image storage, so image transport between OpenCV and other systems (e.g. numpy and ROS) is very efficient
    

* complete documentation for the Python functions
    
    
This cookbook section contains a few illustrative examples of OpenCV Python code.


.. toctree::
    :maxdepth: 2

    cookbook
