Adding (blending) two images using OpenCV {#tutorial_adding_images}
=========================================

Goal
----

In this tutorial you will learn:

-   what is *linear blending* and why it is useful;
-   how to add two images using @ref cv::addWeighted

Theory
------

@note
   The explanation below belongs to the book [Computer Vision: Algorithms and
    Applications](http://szeliski.org/Book/) by Richard Szeliski

From our previous tutorial, we know already a bit of *Pixel operators*. An interesting dyadic
(two-input) operator is the *linear blend operator*:

\f[g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)\f]

By varying \f$\alpha\f$ from \f$0 \rightarrow 1\f$ this operator can be used to perform a temporal
*cross-dissolve* between two images or videos, as seen in slide shows and film productions (cool,
eh?)

Code
----

As usual, after the not-so-lengthy explanation, let's go to the code:
@code{.cpp}
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main( int argc, char** argv )
{
 double alpha = 0.5; double beta; double input;

 Mat src1, src2, dst;

 /// Ask the user enter alpha
 std::cout<<" Simple Linear Blender "<<std::endl;
 std::cout<<"-----------------------"<<std::endl;
 std::cout<<"* Enter alpha [0-1]: ";
 std::cin>>input;

 /// We use the alpha provided by the user if it is between 0 and 1
 if( input >= 0.0 && input <= 1.0 )
   { alpha = input; }

 /// Read image ( same size, same type )
 src1 = imread("../../images/LinuxLogo.jpg");
 src2 = imread("../../images/WindowsLogo.jpg");

 if( !src1.data ) { printf("Error loading src1 \n"); return -1; }
 if( !src2.data ) { printf("Error loading src2 \n"); return -1; }

 /// Create Windows
 namedWindow("Linear Blend", 1);

 beta = ( 1.0 - alpha );
 addWeighted( src1, alpha, src2, beta, 0.0, dst);

 imshow( "Linear Blend", dst );

 waitKey(0);
 return 0;
}
@endcode
Explanation
-----------

-#  Since we are going to perform:

    \f[g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)\f]

    We need two source images (\f$f_{0}(x)\f$ and \f$f_{1}(x)\f$). So, we load them in the usual way:
    @code{.cpp}
    src1 = imread("../../images/LinuxLogo.jpg");
    src2 = imread("../../images/WindowsLogo.jpg");
    @endcode
    **warning**

    Since we are *adding* *src1* and *src2*, they both have to be of the same size (width and
    height) and type.

-#  Now we need to generate the `g(x)` image. For this, the function add_weighted:addWeighted  comes quite handy:
    @code{.cpp}
    beta = ( 1.0 - alpha );
    addWeighted( src1, alpha, src2, beta, 0.0, dst);
    @endcode
    since @ref cv::addWeighted  produces:
    \f[dst = \alpha \cdot src1 + \beta \cdot src2 + \gamma\f]
    In this case, `gamma` is the argument \f$0.0\f$ in the code above.

-#  Create windows, show the images and wait for the user to end the program.

Result
------

![](images/Adding_Images_Tutorial_Result_Big.jpg)
