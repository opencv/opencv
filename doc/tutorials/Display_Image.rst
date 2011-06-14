.. _Display_Image:

Display an Image
*****************

Goal
=====

In this tutorial you will learn how to:

* Load an image using :imread:`imread <>`
* Create a named window (using :named_window:`namedWindow <>`)
* Display an image in an OpenCV window (using :imshow:`imshow <>`)

Code
=====

Here it is:

.. code-block:: cpp

   #include <cv.h>
   #include <highgui.h>

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


Explanation
============

#. .. code-block:: cpp

      #include <cv.h>
      #include <highgui.h>
   
      using namespace cv;

   These are OpenCV headers:

   * *cv.h* : Main OpenCV functions
   * *highgui.h* : Graphical User Interface (GUI) functions

   Now, let's analyze the *main* function:

#. .. code-block:: cpp 

      Mat image;
   
   We create a Mat object to store the data of the image to load.

#. .. code-block:: cpp
    
      image = imread( argv[1], 1 );

   Here, we called the function :imread:`imread <>` which basically loads the image specified by the first argument (in this case *argv[1]*). The second argument is by default.

#. After checking that the image data was loaded correctly, we want to display our image, so we create a window:

   .. code-block:: cpp

      namedWindow( "Display Image", CV_WINDOW_AUTOSIZE );


   :named_window:`namedWindow <>` receives as arguments the window name ("Display Image") and an additional argument that defines windows properties. In this case **CV_WINDOW_AUTOSIZE** indicates that the window will adopt the size of the image to be displayed.

#. Finally, it is time to show the image, for this we use :imshow:`imshow <>` 

   .. code-block:: cpp
  
      imshow( "Display Image", image )

#. Finally, we want our window to be displayed until the user presses a key (otherwise the program would end far too quickly):

   .. code-block:: cpp
 
      waitKey(0);

   We use the :wait_key:`waitKey <>` function, which allow us to wait for a keystroke during a number of milliseconds (determined by the argument). If the argument is zero, then it will wait indefinitely.

Result
=======

* Compile your code and then run the executable giving a image path as argument:

  .. code-block:: bash
 
     ./DisplayImage HappyFish.jpg

* You should get a nice window as the one shown below:

  .. image:: images/Display_Image_Tutorial_Result.png
     :alt: Display Image Tutorial - Final Result
     :align: center 
