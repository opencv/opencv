.. _Morphology_2:

More Morphology Transformations
*********************************

Goal
=====

In this tutorial you will learn how to:

* Use the OpenCV function :morphology_ex:`morphologyEx <>` to apply Morphological Transformation such as:
  
  * Opening 
  * Closing
  * Morphological Gradient
  * Top Hat
  * Black Hat

Cool Theory
============

.. note::
   The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

In the previous tutorial we covered two basic Morphology operations: 

* Erosion

* Dilation. 

Based on these two we can effectuate more sophisticated transformations to our images. Here we discuss briefly 05 operations offered by OpenCV:

Opening
---------

* It is obtained by the erosion of an image followed by a dilation.

  .. math::
     
     dst = open( src, element) = dilate( erode( src, element ) )

* Useful for removing small objects (it is assumed that the objects are bright on a dark foreground)

* For instance, check out the example below. The image at the left is the original and the image at the right is the result after applying the opening transformation. We can observe that the small spaces in the corners of the letter tend to dissapear.

  .. image:: images/Morphology_2_Tutorial_Theory_Opening.png
     :height: 150pt
     :alt: Opening
     :align: center  

Closing
---------

* It is obtained by the dilation of an image followed by an erosion.

  .. math::
    
     dst = close( src, element ) = erode( dilate( src, element ) )

* Useful to remove small holes (dark regions). 

  .. image:: images/Morphology_2_Tutorial_Theory_Closing.png
     :height: 150pt
     :alt: Closing example
     :align: center  


Morphological Gradient
------------------------

* It is the difference between the dilation and the erosion of an image.

  .. math::

     dst = morph_{grad}( src, element ) = dilate( src, element ) - erode( src, element )

* It is useful for finding the outline of an object as can be seen below:

  .. image:: images/Morphology_2_Tutorial_Theory_Gradient.png
     :height: 150pt
     :alt: Gradient
     :align: center  


Top Hat
---------

* It is the difference between an input image and its opening.

  .. math::
   
     dst = tophat( src, element ) = src - open( src, element )

  .. image:: images/Morphology_2_Tutorial_Theory_TopHat.png
     :height: 150pt
     :alt: Top Hat
     :align: center  

Black Hat
----------

* It is the difference between the closing and its input image

  .. math::
 
     dst = blackhat( src, element ) = close( src, element ) - src

  .. image:: images/Morphology_2_Tutorial_Theory_BlackHat.png
     :height: 150pt
     :alt: Black Hat
     :align: center    

Code
======

This tutorial code's is shown lines below. You can also download it from `here <https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/tutorial_code/Image_Processing/Morphology_2.cpp>`_

.. code-block:: cpp 

   #include "opencv2/imgproc/imgproc.hpp"
   #include "opencv2/highgui/highgui.hpp"
   #include <stdlib.h>
   #include <stdio.h>

   using namespace cv;

   /// Global variables
   Mat src, dst;

   int morph_elem = 0;
   int morph_size = 0;
   int morph_operator = 0;
   int const max_operator = 4;
   int const max_elem = 2;
   int const max_kernel_size = 21;

   char* window_name = "Morphology Transformations Demo";

   /** Function Headers */
   void Morphology_Operations( int, void* );

   /** @function main */
   int main( int argc, char** argv )
   {
     /// Load an image
     src = imread( argv[1] );

     if( !src.data )
     { return -1; }
    
    /// Create window
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    /// Create Trackbar to select Morphology operation
    createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations );

    /// Create Trackbar to select kernel type
    createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name, 
		    &morph_elem, max_elem, 
		    Morphology_Operations );

    /// Create Trackbar to choose kernel size
    createTrackbar( "Kernel size:\n 2n +1", window_name, 
		    &morph_size, max_kernel_size,
		    Morphology_Operations );

    /// Default start
    Morphology_Operations( 0, 0 );

    waitKey(0);
    return 0;
    }

    /** 
     * @function Morphology_Operations
     */
   void Morphology_Operations( int, void* )
   {
     // Since MORPH_X : 2,3,4,5 and 6
     int operation = morph_operator + 2;

     Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) ); 

     /// Apply the specified morphology operation
     morphologyEx( src, dst, operation, element );
     imshow( window_name, dst );  
     }


Explanation
=============

#. Let's check the general structure of the program:

   * Load an image

   * Create a window to display results of the Morphological operations

   * Create 03 Trackbars for the user to enter parameters:

     * The first trackbar **"Operator"** returns the kind of morphology operation to use (**morph_operator**).

       .. code-block:: cpp

          createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", 
                         window_name, &morph_operator, max_operator, 
                         Morphology_Operations );



     * The second trackbar **"Element"** returns **morph_elem**, which indicates what kind of structure our kernel is: 

       .. code-block:: cpp

          createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name, 
		          &morph_elem, max_elem, 
		          Morphology_Operations );

     * The final trackbar **"Kernel Size"** returns the size of the kernel to be used (**morph_size**)

       .. code-block:: cpp

          createTrackbar( "Kernel size:\n 2n +1", window_name, 
		          &morph_size, max_kernel_size,
		          Morphology_Operations );


   * Every time we move any slider, the user's function **Morphology_Operations** will be called to effectuate a new morphology operation and it will update the output image based on the current trackbar values.
  
     .. code-block:: cpp

        /** 
         * @function Morphology_Operations
         */
       void Morphology_Operations( int, void* )
       {
         // Since MORPH_X : 2,3,4,5 and 6
         int operation = morph_operator + 2;

         Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) ); 

         /// Apply the specified morphology operation
         morphologyEx( src, dst, operation, element );
         imshow( window_name, dst );  
        }


     We can observe that the key function to perform the morphology transformations is :morphology_ex:`morphologyEx <>`. In this example we use four arguments (leaving the rest as defaults):

     * **src** : Source (input) image
     * **dst**: Output image
     * **operation**: The kind of morphology transformation to be performed. Note that we have 5 alternatives:

       * *Opening*: MORPH_OPEN : 2
       * *Closing*: MORPH_CLOSE: 3
       * *Gradient*: MORPH_GRADIENT: 4
       * *Top Hat*: MORPH_TOPHAT: 5
       * *Black Hat*: MORPH_BLACKHAT: 6

       As you can see the values range from <2-6>, that is why we add (+2) to the values entered by the Trackbar:

       ..  code-block:: cpp

           int operation = morph_operator + 2;    

     * **element**: The kernel to be used. We use the function :get_structuring_element:`getStructuringElement <>` to define our own structure.

   

Results
========

* After compiling the code above we can execute it giving an image path as an argument. For this tutorial we use as input the image: **baboon.jpg**:

  .. image:: images/Morphology_2_Tutorial_Original_Image.jpg
     :height: 200pt
     :alt: Morphology 2: Original image
     :align: center 

* And here are two snapshots of the display window. The first picture shows the output after using the operator **Opening** with a cross kernel. The second picture (right side, shows the result of using a **Blackhat** operator with an ellipse kernel.
 
  .. image:: images/Morphology_2_Tutorial_Cover.png
     :height: 300pt
     :alt: Morphology 2: Result sample
     :align: center 

