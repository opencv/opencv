.. _Load_Save_Image:

Load, Modify, and Save an Image
*******************************

.. note::

   We assume that by now you know how to load an image using :imread:`imread <>` and to display it in a window (using :imshow:`imshow <>`). Read the :ref:`Display_Image` tutorial otherwise.

Goals
======

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Load an image using :imread:`imread <>`
   * Transform an image from BGR to Grayscale format by using :cvt_color:`cvtColor <>`
   * Save your transformed image in a file on disk (using :imwrite:`imwrite <>`)

Code
======

Here it is:

.. code-block:: cpp
   :linenos:

   #include <cv.h>
   #include <highgui.h>

   using namespace cv;

   int main( int argc, char** argv )
   {
    char* imageName = argv[1];

    Mat image;
    image = imread( imageName, 1 );

    if( argc != 2 || !image.data )
    {
      printf( " No image data \n " );
      return -1;
    }

    Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    imwrite( "../../images/Gray_Image.jpg", gray_image );

    namedWindow( imageName, CV_WINDOW_AUTOSIZE );
    namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );

    imshow( imageName, image );
    imshow( "Gray image", gray_image );

    waitKey(0);

    return 0;
   }

Explanation
============

#. We begin by:

   * Creating a Mat object to store the image information
   * Load an image using :imread:`imread <>`, located in the path given by *imageName*. Fort this example, assume you are loading a RGB image.

#. Now we are going to convert our image from BGR to Grayscale format. OpenCV has a really nice function to do this kind of transformations:

   .. code-block:: cpp

      cvtColor( image, gray_image, CV_BGR2GRAY );

   As you can see, :cvt_color:`cvtColor <>` takes as arguments:

   .. container:: enumeratevisibleitemswithsquare

      * a source image (*image*)
      * a destination image (*gray_image*), in which we will save the converted image.
      * an additional parameter that indicates what kind of transformation will be performed. In this case we use **CV_BGR2GRAY** (because of :imread:`imread <>` has BGR default channel order in case of color images).

#. So now we have our new *gray_image* and want to save it on disk (otherwise it will get lost after the program ends). To save it, we will use a function analagous to :imread:`imread <>`: :imwrite:`imwrite <>`

   .. code-block:: cpp

      imwrite( "../../images/Gray_Image.jpg", gray_image );

   Which will save our *gray_image* as *Gray_Image.jpg* in the folder *images* located two levels up of my current location.

#. Finally, let's check out the images. We create two windows and use them to show the original image as well as the new one:

   .. code-block:: cpp

      namedWindow( imageName, CV_WINDOW_AUTOSIZE );
      namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );

      imshow( imageName, image );
      imshow( "Gray image", gray_image );

#. Add add the *waitKey(0)* function call for the program to wait forever for an user key press.


Result
=======

When you run your program you should get something like this:

 .. image:: images/Load_Save_Image_Result_1.jpg
    :alt: Load Save Image Result 1
    :align: center

And if you check in your folder (in my case *images*), you should have a newly .jpg file named *Gray_Image.jpg*:

 .. image:: images/Load_Save_Image_Result_2.jpg
    :alt: Load Save Image Result 2
    :align: center

Congratulations, you are done with this tutorial!
