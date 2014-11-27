Adding borders to your images {#tutorial_copyMakeBorder}
=============================

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::copyMakeBorder to set the borders (extra padding to your
    image).

Theory
------

@note The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

1.  In our previous tutorial we learned to use convolution to operate on images. One problem that
    naturally arises is how to handle the boundaries. How can we convolve them if the evaluated
    points are at the edge of the image?
2.  What most of OpenCV functions do is to copy a given image onto another slightly larger image and
    then automatically pads the boundary (by any of the methods explained in the sample code just
    below). This way, the convolution can be performed over the needed pixels without problems (the
    extra padding is cut after the operation is done).
3.  In this tutorial, we will briefly explore two ways of defining the extra padding (border) for an
    image:

    -#  **BORDER_CONSTANT**: Pad the image with a constant value (i.e. black or \f$0\f$
    -#  **BORDER_REPLICATE**: The row or column at the very edge of the original is replicated to
        the extra border.

    This will be seen more clearly in the Code section.

Code
----

1.  **What does this program do?**
    -   Load an image
    -   Let the user choose what kind of padding use in the input image. There are two options:

        1.  *Constant value border*: Applies a padding of a constant value for the whole border.
            This value will be updated randomly each 0.5 seconds.
        2.  *Replicated border*: The border will be replicated from the pixel values at the edges of
            the original image.

        The user chooses either option by pressing 'c' (constant) or 'r' (replicate)
    -   The program finishes when the user presses 'ESC'

2.  The tutorial code's is shown lines below. You can also download it from
    [here](https://github.com/Itseez/opencv/tree/master/samples/cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp)
@code{.cpp}
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global Variables
Mat src, dst;
int top, bottom, left, right;
int borderType;
Scalar value;
char* window_name = "copyMakeBorder Demo";
RNG rng(12345);

/* @function main  */
int main( int argc, char** argv )
{

  int c;

  /// Load an image
  src = imread( argv[1] );

  if( !src.data )
  { return -1;
    printf(" No data entered, please enter the path to an image file \n");
  }

  /// Brief how-to for this program
  printf( "\n \t copyMakeBorder Demo: \n" );
  printf( "\t -------------------- \n" );
  printf( " ** Press 'c' to set the border to a random constant value \n");
  printf( " ** Press 'r' to set the border to be replicated \n");
  printf( " ** Press 'ESC' to exit the program \n");

  /// Create window
  namedWindow( window_name, WINDOW_AUTOSIZE );

  /// Initialize arguments for the filter
  top = (int) (0.05*src.rows); bottom = (int) (0.05*src.rows);
  left = (int) (0.05*src.cols); right = (int) (0.05*src.cols);
  dst = src;

  imshow( window_name, dst );

  while( true )
    {
      c = waitKey(500);

      if( (char)c == 27 )
        { break; }
      else if( (char)c == 'c' )
        { borderType = BORDER_CONSTANT; }
      else if( (char)c == 'r' )
        { borderType = BORDER_REPLICATE; }

      value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
      copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );

      imshow( window_name, dst );
    }

  return 0;
}
@endcode
Explanation
-----------

1.  First we declare the variables we are going to use:
    @code{.cpp}
    Mat src, dst;
    int top, bottom, left, right;
    int borderType;
    Scalar value;
    char* window_name = "copyMakeBorder Demo";
    RNG rng(12345);
    @endcode
    Especial attention deserves the variable *rng* which is a random number generator. We use it to
    generate the random border color, as we will see soon.

2.  As usual we load our source image *src*:
    @code{.cpp}
    src = imread( argv[1] );

    if( !src.data )
    { return -1;
      printf(" No data entered, please enter the path to an image file \n");
    }
    @endcode
3.  After giving a short intro of how to use the program, we create a window:
    @code{.cpp}
    namedWindow( window_name, WINDOW_AUTOSIZE );
    @endcode
4.  Now we initialize the argument that defines the size of the borders (*top*, *bottom*, *left* and
    *right*). We give them a value of 5% the size of *src*.
    @code{.cpp}
    top = (int) (0.05*src.rows); bottom = (int) (0.05*src.rows);
    left = (int) (0.05*src.cols); right = (int) (0.05*src.cols);
    @endcode
5.  The program begins a *while* loop. If the user presses 'c' or 'r', the *borderType* variable
    takes the value of *BORDER_CONSTANT* or *BORDER_REPLICATE* respectively:
    @code{.cpp}
    while( true )
     {
       c = waitKey(500);

       if( (char)c == 27 )
         { break; }
       else if( (char)c == 'c' )
         { borderType = BORDER_CONSTANT; }
       else if( (char)c == 'r' )
         { borderType = BORDER_REPLICATE; }
    @endcode
6.  In each iteration (after 0.5 seconds), the variable *value* is updated...
    @code{.cpp}
    value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
    @endcode
    with a random value generated by the **RNG** variable *rng*. This value is a number picked
    randomly in the range \f$[0,255]\f$

7.  Finally, we call the function @ref cv::copyMakeBorder to apply the respective padding:
    @code{.cpp}
    copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
    @endcode
    The arguments are:

    -#  *src*: Source image
    -#  *dst*: Destination image
    -#  *top*, *bottom*, *left*, *right*: Length in pixels of the borders at each side of the image.
        We define them as being 5% of the original size of the image.
    -#  *borderType*: Define what type of border is applied. It can be constant or replicate for
        this example.
    -#  *value*: If *borderType* is *BORDER_CONSTANT*, this is the value used to fill the border
        pixels.

8.  We display our output image in the image created previously
    @code{.cpp}
    imshow( window_name, dst );
    @endcode
Results
-------

1.  After compiling the code above, you can execute it giving as argument the path of an image. The
    result should be:

    -   By default, it begins with the border set to BORDER_CONSTANT. Hence, a succession of random
        colored borders will be shown.
    -   If you press 'r', the border will become a replica of the edge pixels.
    -   If you press 'c', the random colored borders will appear again
    -   If you press 'ESC' the program will exit.

    Below some screenshot showing how the border changes color and how the *BORDER_REPLICATE*
    option looks:

    ![image](images/CopyMakeBorder_Tutorial_Results.jpg)
