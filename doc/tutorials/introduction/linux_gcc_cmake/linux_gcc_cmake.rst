.. _Linux_GCC_Usage:

Using OpenCV with gcc and CMake
*********************************

.. note::
   We assume that you have successfully installed OpenCV in your workstation.

.. container:: enumeratevisibleitemswithsquare

   * The easiest way of using OpenCV in your code is to use `CMake <http://www.cmake.org/>`_. A few advantages (taken from the Wiki):

     #. No need to change anything when porting between Linux and Windows
     #. Can easily be combined with other tools by CMake( i.e. Qt, ITK and VTK )

   * If you are not familiar with CMake, checkout the `tutorial <http://www.cmake.org/cmake/help/cmake_tutorial.html>`_ on its website.

Steps
======

Create a program using OpenCV
-------------------------------

Let's use a simple program such as DisplayImage.cpp shown below.

.. code-block:: cpp

   #include <stdio.h>
   #include <opencv2/opencv.hpp>

   using namespace cv;

   int main( int argc, char** argv )
   {
     Mat image;
     image = imread( argv[1], 1 );

     if( argc != 2 || !image.data )
       {
         printf( "No image data \n" );
         return -1;
       }

     namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );
     imshow( "Display Image", image );

     waitKey(0);

     return 0;
   }

Create a CMake file
---------------------
Now you have to create your CMakeLists.txt file. It should look like this:

.. code-block:: cmake

   cmake_minimum_required(VERSION 2.8)
   project( DisplayImage )
   find_package( OpenCV REQUIRED )
   include_directories( ${OpenCV_INCLUDE_DIRS} )
   add_executable( DisplayImage DisplayImage.cpp )
   target_link_libraries( DisplayImage ${OpenCV_LIBS} )

Generate the executable
-------------------------
This part is easy, just proceed as with any other project using CMake:

.. code-block:: bash

   cd <DisplayImage_directory>
   cmake .
   make

Result
--------
By now you should have an executable (called DisplayImage in this case). You just have to run it giving an image location as an argument, i.e.:

.. code-block:: bash

   ./DisplayImage lena.jpg

You should get a nice window as the one shown below:

.. image:: images/GCC_CMake_Example_Tutorial.jpg
   :alt: Display Image - Lena
   :align: center
