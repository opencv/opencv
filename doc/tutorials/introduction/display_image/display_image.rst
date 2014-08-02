.. _Display_Image:

Load and Display an Image
*************************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   * Load an image (using :imread:`imread <>`)
   * Create a named OpenCV window (using :named_window:`namedWindow <>`)
   * Display an image in an OpenCV window (using :imshow:`imshow <>`)

Source Code
===========

Download the source code from `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/introduction/display_image/display_image.cpp>`_.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :tab-width: 4
   :linenos:

Explanation
============

In OpenCV 2 we have multiple modules. Each one takes care of a different area or approach towards image processing. You could already observe this in the structure of the user guide of these tutorials itself. Before you use any of them you first need to include the header files where the content of each individual module is declared.

You'll almost always end up using the:

.. container:: enumeratevisibleitemswithsquare

   + *core* section, as here are defined the basic building blocks of the library
   + *highgui* module, as this contains the functions for input and output operations

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :tab-width: 4
   :lines:  1-4

We also include the *iostream* to facilitate console line output and input. To avoid data structure and function name conflicts with other libraries, OpenCV has its own namespace: *cv*. To avoid the need appending prior each of these the *cv::* keyword you can import the namespace in the whole file by using the lines:

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :tab-width: 4
   :lines:  6-7

This is true for the STL library too (used for console I/O). Now, let's analyze the *main* function. We start up assuring that we acquire a valid image name argument from the command line.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :tab-width: 4
   :lines: 11-15

Then create a *Mat* object that will store the data of the loaded image.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :tab-width: 4
   :lines: 17

Now we call the :imread:`imread <>` function which loads the image name specified by the first argument (*argv[1]*). The second argument specifies the format in what we want the image. This may be:

.. container:: enumeratevisibleitemswithsquare

   + CV_LOAD_IMAGE_UNCHANGED (<0) loads the image as is (including the alpha channel if present)
   + CV_LOAD_IMAGE_GRAYSCALE ( 0) loads the image as an intensity one
   + CV_LOAD_IMAGE_COLOR     (>0) loads the image in the RGB format

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :tab-width: 4
   :lines: 18

.. note::

   OpenCV offers support for the image formats Windows bitmap (bmp), portable image formats (pbm, pgm, ppm) and Sun raster (sr, ras). With help of plugins (you need to specify to use them if you build yourself the library, nevertheless in the packages we ship present by default) you may also load image formats like JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 - codenamed in the CMake as Jasper), TIFF files (tiff, tif) and portable network graphics (png). Furthermore, OpenEXR is also a possibility.

After checking that the image data was loaded correctly, we want to display our image, so we create an OpenCV window using the :named_window:`namedWindow <>` function. These are automatically managed by OpenCV once you create them. For this you need to specify its name and how it should handle the change of the image it contains from a size point of view. It may be:

.. container:: enumeratevisibleitemswithsquare

   + *CV_WINDOW_AUTOSIZE* is the only supported one if you do not use the Qt backend. In this case the window size will take up the size of the image it shows. No resize permitted!
   + *CV_WINDOW_NORMAL* on Qt you may use this to allow window resize. The image will resize itself according to the current window size. By using the | operator you also need to specify if you would like the image to keep its aspect ratio (*CV_WINDOW_KEEPRATIO*) or not (*CV_WINDOW_FREERATIO*).

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :lines: 26
   :tab-width: 4

Finally, to update the content of the OpenCV window with a new image use the :imshow:`imshow <>` function. Specify the OpenCV window name to update and the image to use during this operation:

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :lines: 27
   :tab-width: 4

Because we want our window to be displayed until the user presses a key (otherwise the program would end far too quickly), we use the :wait_key:`waitKey <>` function whose only parameter is just how long should it wait for a user input (measured in milliseconds). Zero means to wait forever.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
   :language: cpp
   :lines: 29
   :tab-width: 4

Result
=======

.. container:: enumeratevisibleitemswithsquare

   * Compile your code and then run the executable giving an image path as argument. If you're on Windows the executable will of course contain an *exe* extension too. Of course assure the image file is near your program file.

     .. code-block:: bash

        ./DisplayImage HappyFish.jpg

   * You should get a nice window as the one shown below:

     .. image:: images/Display_Image_Tutorial_Result.jpg
        :alt: Display Image Tutorial - Final Result
        :align: center

   .. raw:: html

     <div align="center">
     <iframe title="Introduction - Display an Image" width="560" height="349" src="http://www.youtube.com/embed/1OJEqpuaGc4?rel=0&loop=1" frameborder="0" allowfullscreen align="middle"></iframe>
     </div>
