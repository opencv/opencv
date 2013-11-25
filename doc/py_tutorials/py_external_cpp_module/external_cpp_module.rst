.. _PY_ExternalModule:

Use C++ code from Python
************************

Goal
=====

OpenCV provides an bindings to Python which transparently maps cv::Mat C++ objects to NumPy matrices. This tutorial explains how to integrate your own C++ code or third party modules into you Python projects by using these bindings.

This tutorial will explain the following:

.. container:: enumeratevisibleitemswithsquare

    + generating bindings for your own C++ code
    + generating bindings for third party C++ code
    + integrating bindings into the CMake build process


Bindings for your own C++ code
==============================

We generate bindings for the following C++ code, containing a single function and a basic class:

.. code-block:: cpp
   namespace cv2test
   {

     CV_EXPORTS_W int image_height(const cv::Mat& image);

     class

   }
