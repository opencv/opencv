.. _back_projection:

Back Projection
****************


Goal
====

In this tutorial you will learn:

.. container:: enumeratevisibleitemswithsquare

   * What is Back Projection and why it is useful

   * How to use the OpenCV function :calc_back_project:`calcBackProject <>` to calculate Back Projection

   * How to mix different channels of an image by using the OpenCV function :mix_channels:`mixChannels <>`


Theory
======

What is Back Projection?
---------------------------

.. container:: enumeratevisibleitemswithsquare

   * Back Projection is a way of recording how well the pixels of a given image fit the distribution of pixels in a histogram model.

   * To make it simpler: For Back Projection, you calculate the histogram model of a feature and then use it to find this feature in an image.

   * Application example: If you have a histogram of flesh color (say, a Hue-Saturation histogram ), then you can use it to find flesh color areas in an image:


How does it work?
------------------

.. container:: enumeratevisibleitemswithsquare

   * We explain this by using the skin example:

   * Let's say you have gotten a skin histogram (Hue-Saturation) based on the image below. The histogram besides is going to be our *model histogram* (which we know represents a sample of skin tonality). You applied some mask to capture only the histogram of the skin area:

     ======  ======
      |T0|    |T1|
     ======  ======

     .. |T0| image:: images/Back_Projection_Theory0.jpg
                   :align: middle

     .. |T1| image:: images/Back_Projection_Theory1.jpg
                   :align: middle


   * Now, let's imagine that you get another hand image (Test Image) like the one below: (with its respective histogram):

     ======  ======
      |T2|    |T3|
     ======  ======

     .. |T2| image:: images/Back_Projection_Theory2.jpg
                   :align: middle

     .. |T3| image:: images/Back_Projection_Theory3.jpg
                   :align: middle


   * What we want to do is to use our *model histogram* (that we know represents a skin tonality) to detect skin areas in our Test Image. Here are the steps

     a. In each pixel of our Test Image (i.e. :math:`p(i,j)` ), collect the data and find the correspondent bin location for that pixel (i.e. :math:`( h_{i,j}, s_{i,j} )` ).

     b. Lookup the *model histogram* in the correspondent bin - :math:`( h_{i,j}, s_{i,j} )` - and read the bin value.

     c. Store this bin value in a new image (*BackProjection*). Also, you may consider to normalize the *model histogram* first, so the output for the Test Image can be visible for you.

     d. Applying the steps above, we get the following BackProjection image for our Test Image:

        .. image:: images/Back_Projection_Theory4.jpg
                 :align: center

     e. In terms of statistics, the values stored in *BackProjection* represent the *probability* that a pixel in *Test Image* belongs to a skin area, based on the *model histogram* that we use. For instance in our Test image, the brighter areas are more probable to be skin area (as they actually are), whereas the darker areas have less probability (notice that these "dark" areas belong to surfaces that have some shadow on it, which in turns affects the detection).


Code
====

.. container:: enumeratevisibleitemswithsquare

   * **What does this program do?**

     .. container:: enumeratevisibleitemswithsquare

        * Loads an image
        * Convert the original to HSV format and separate only *Hue* channel to be used for the Histogram (using the OpenCV function :mix_channels:`mixChannels <>`)
        * Let the user to enter the number of bins to be used in the calculation of the histogram.
        * Calculate the histogram (and update it if the bins change) and the backprojection of the same image.
        * Display the backprojection and the histogram in windows.

   * **Downloadable code**:

      a. Click `here <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo1.cpp>`_ for the basic version (explained in this tutorial).
      b. For stuff slightly fancier (using H-S histograms and floodFill to define a mask for the skin area) you can check the `improved demo <https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/Histograms_Matching/calcBackProject_Demo2.cpp>`_
      c. ...or you can always check out the classical `camshiftdemo <https://github.com/Itseez/opencv/tree/master/samples/cpp/camshiftdemo.cpp>`_ in samples.

   * **Code at glance:**

.. code-block:: cpp

   #include "opencv2/imgproc.hpp"
   #include "opencv2/highgui.hpp"

   #include <iostream>

   using namespace cv;
   using namespace std;

   /// Global Variables
   Mat src; Mat hsv; Mat hue;
   int bins = 25;

   /// Function Headers
   void Hist_and_Backproj(int, void* );

   /** @function main */
   int main( int argc, char** argv )
   {
     /// Read the image
     src = imread( argv[1], 1 );
     /// Transform it to HSV
     cvtColor( src, hsv, CV_BGR2HSV );

     /// Use only the Hue value
     hue.create( hsv.size(), hsv.depth() );
     int ch[] = { 0, 0 };
     mixChannels( &hsv, 1, &hue, 1, ch, 1 );

     /// Create Trackbar to enter the number of bins
     char* window_image = "Source image";
     namedWindow( window_image, CV_WINDOW_AUTOSIZE );
     createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj );
     Hist_and_Backproj(0, 0);

     /// Show the image
     imshow( window_image, src );

     /// Wait until user exits the program
     waitKey(0);
     return 0;
   }


   /**
    * @function Hist_and_Backproj
    * @brief Callback to Trackbar
    */
   void Hist_and_Backproj(int, void* )
   {
     MatND hist;
     int histSize = MAX( bins, 2 );
     float hue_range[] = { 0, 180 };
     const float* ranges = { hue_range };

     /// Get the Histogram and normalize it
     calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
     normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

     /// Get Backprojection
     MatND backproj;
     calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

     /// Draw the backproj
     imshow( "BackProj", backproj );

     /// Draw the histogram
     int w = 400; int h = 400;
     int bin_w = cvRound( (double) w / histSize );
     Mat histImg = Mat::zeros( w, h, CV_8UC3 );

     for( int i = 0; i < bins; i ++ )
        { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

     imshow( "Histogram", histImg );
   }

Explanation
===========

#. Declare the matrices to store our images and initialize the number of bins to be used by our histogram:

   .. code-block:: cpp

      Mat src; Mat hsv; Mat hue;
      int bins = 25;

#. Read the input image and transform it to HSV format:

   .. code-block:: cpp

      src = imread( argv[1], 1 );
      cvtColor( src, hsv, CV_BGR2HSV );

#. For this tutorial, we will use only the Hue value for our 1-D histogram (check out the fancier code in the links above if you want to use the more standard H-S histogram, which yields better results):

   .. code-block:: cpp

      hue.create( hsv.size(), hsv.depth() );
      int ch[] = { 0, 0 };
      mixChannels( &hsv, 1, &hue, 1, ch, 1 );

   as you see, we use the function :mix_channels:`mixChannels` to get only the channel 0 (Hue) from the hsv image. It gets the following parameters:

   .. container:: enumeratevisibleitemswithsquare

      + **&hsv:** The source array from which the channels will be copied
      + **1:** The number of source arrays
      + **&hue:** The destination array of the copied channels
      + **1:** The number of destination arrays
      + **ch[] = {0,0}:** The array of index pairs indicating how the channels are copied. In this case, the Hue(0) channel of &hsv is being copied to the 0 channel of &hue (1-channel)
      + **1:** Number of index pairs

#. Create a Trackbar for the user to enter the bin values. Any change on the Trackbar means a call to the **Hist_and_Backproj** callback function.

   .. code-block:: cpp

      char* window_image = "Source image";
      namedWindow( window_image, CV_WINDOW_AUTOSIZE );
      createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj );
      Hist_and_Backproj(0, 0);

#. Show the image and wait for the user to exit the program:

   .. code-block:: cpp

     imshow( window_image, src );

     waitKey(0);
     return 0;

#. **Hist_and_Backproj function:** Initialize the arguments needed for :calc_hist:`calcHist <>`. The number of bins comes from the Trackbar:


   .. code-block:: cpp

      void Hist_and_Backproj(int, void* )
      {
        MatND hist;
        int histSize = MAX( bins, 2 );
        float hue_range[] = { 0, 180 };
        const float* ranges = { hue_range };

#. Calculate the Histogram and normalize it to the range :math:`[0,255]`

   .. code-block:: cpp

      calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
      normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

#. Get the Backprojection of the same image by calling the function :calc_back_project:`calcBackProject <>`

   .. code-block:: cpp

      MatND backproj;
      calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

   all the arguments are known (the same as used to calculate the histogram), only we add the backproj matrix, which will store the backprojection of the source image (&hue)

#. Display backproj:

   .. code-block:: cpp

      imshow( "BackProj", backproj );

#. Draw the 1-D Hue histogram of the image:

   .. code-block:: cpp

      int w = 400; int h = 400;
      int bin_w = cvRound( (double) w / histSize );
      Mat histImg = Mat::zeros( w, h, CV_8UC3 );

      for( int i = 0; i < bins; i ++ )
         { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

      imshow( "Histogram", histImg );



Results
=======

#. Here are the output by using a sample image ( guess what? Another hand ). You can play with the bin values and you will observe how it affects the results:

   ======  ======  ======
    |R0|    |R1|    |R2|
   ======  ======  ======

   .. |R0| image:: images/Back_Projection1_Source_Image.jpg
                 :align: middle

   .. |R1| image:: images/Back_Projection1_Histogram.jpg
                 :align: middle

   .. |R2| image:: images/Back_Projection1_BackProj.jpg
                 :align: middle
