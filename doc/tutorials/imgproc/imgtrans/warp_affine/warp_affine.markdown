Affine Transformations {#tutorial_warp_affine}
======================

Goal
----

In this tutorial you will learn how to:

-  Use the OpenCV function @ref cv::warpAffine to implement simple remapping routines.
-  Use the OpenCV function @ref cv::getRotationMatrix2D to obtain a \f$2 \times 3\f$ rotation matrix

Theory
------

### What is an Affine Transformation?

-#  It is any transformation that can be expressed in the form of a *matrix multiplication* (linear
    transformation) followed by a *vector addition* (translation).
-#  From the above, We can use an Affine Transformation to express:

    -#  Rotations (linear transformation)
    -#  Translations (vector addition)
    -#  Scale operations (linear transformation)

    you can see that, in essence, an Affine Transformation represents a **relation** between two
    images.

-#  The usual way to represent an Affine Transform is by using a \f$2 \times 3\f$ matrix.

    \f[
    A = \begin{bmatrix}
        a_{00} & a_{01} \\
        a_{10} & a_{11}
        \end{bmatrix}_{2 \times 2}
    B = \begin{bmatrix}
        b_{00} \\
        b_{10}
        \end{bmatrix}_{2 \times 1}
    \f]
    \f[
    M = \begin{bmatrix}
        A & B
        \end{bmatrix}
    =
   \begin{bmatrix}
        a_{00} & a_{01} & b_{00} \\
        a_{10} & a_{11} & b_{10}
   \end{bmatrix}_{2 \times 3}
   \f]

    Considering that we want to transform a 2D vector \f$X = \begin{bmatrix}x \\ y\end{bmatrix}\f$ by
    using \f$A\f$ and \f$B\f$, we can do it equivalently with:

    \f$T = A \cdot \begin{bmatrix}x \\ y\end{bmatrix} + B\f$ or \f$T = M \cdot  [x, y, 1]^{T}\f$

    \f[T =  \begin{bmatrix}
        a_{00}x + a_{01}y + b_{00} \\
        a_{10}x + a_{11}y + b_{10}
        \end{bmatrix}\f]

### How do we get an Affine Transformation?

-#  Excellent question. We mentioned that an Affine Transformation is basically a **relation**
    between two images. The information about this relation can come, roughly, in two ways:
    -#  We know both \f$X\f$ and T and we also know that they are related. Then our job is to find \f$M\f$
    -#  We know \f$M\f$ and \f$X\f$. To obtain \f$T\f$ we only need to apply \f$T = M \cdot X\f$. Our information
        for \f$M\f$ may be explicit (i.e. have the 2-by-3 matrix) or it can come as a geometric relation
        between points.

-#  Let's explain a little bit better (b). Since \f$M\f$ relates 02 images, we can analyze the simplest
    case in which it relates three points in both images. Look at the figure below:

    ![](images/Warp_Affine_Tutorial_Theory_0.jpg)

    the points 1, 2 and 3 (forming a triangle in image 1) are mapped into image 2, still forming a
    triangle, but now they have changed notoriously. If we find the Affine Transformation with these
    3 points (you can choose them as you like), then we can apply this found relation to the whole
    pixels in the image.

Code
----

-#  **What does this program do?**
    -   Loads an image
    -   Applies an Affine Transform to the image. This Transform is obtained from the relation
        between three points. We use the function @ref cv::warpAffine for that purpose.
    -   Applies a Rotation to the image after being transformed. This rotation is with respect to
        the image center
    -   Waits until the user exits the program

-#  The tutorial code's is shown lines below. You can also download it from
    [here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp)
    @include samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp

Explanation
-----------

-#  Declare some variables we will use, such as the matrices to store our results and 2 arrays of
    points to store the 2D points that define our Affine Transform.
    @code{.cpp}
    Point2f srcTri[3];
    Point2f dstTri[3];

    Mat rot_mat( 2, 3, CV_32FC1 );
    Mat warp_mat( 2, 3, CV_32FC1 );
    Mat src, warp_dst, warp_rotate_dst;
    @endcode
-#  Load an image:
    @code{.cpp}
    src = imread( argv[1], 1 );
    @endcode
-#  Initialize the destination image as having the same size and type as the source:
    @code{.cpp}
    warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
    @endcode
-#  **Affine Transform:** As we explained lines above, we need two sets of 3 points to derive the
    affine transform relation. Take a look:
    @code{.cpp}
    srcTri[0] = Point2f( 0,0 );
    srcTri[1] = Point2f( src.cols - 1, 0 );
    srcTri[2] = Point2f( 0, src.rows - 1 );

    dstTri[0] = Point2f( src.cols*0.0, src.rows*0.33 );
    dstTri[1] = Point2f( src.cols*0.85, src.rows*0.25 );
    dstTri[2] = Point2f( src.cols*0.15, src.rows*0.7 );
    @endcode
    You may want to draw the points to make a better idea of how they change. Their locations are
    approximately the same as the ones depicted in the example figure (in the Theory section). You
    may note that the size and orientation of the triangle defined by the 3 points change.

-#  Armed with both sets of points, we calculate the Affine Transform by using OpenCV function @ref
    cv::getAffineTransform :
    @code{.cpp}
    warp_mat = getAffineTransform( srcTri, dstTri );
    @endcode
    We get as an output a \f$2 \times 3\f$ matrix (in this case **warp_mat**)

-#  We apply the Affine Transform just found to the src image
    @code{.cpp}
    warpAffine( src, warp_dst, warp_mat, warp_dst.size() );
    @endcode
    with the following arguments:

    -   **src**: Input image
    -   **warp_dst**: Output image
    -   **warp_mat**: Affine transform
    -   **warp_dst.size()**: The desired size of the output image

    We just got our first transformed image! We will display it in one bit. Before that, we also
    want to rotate it...

-#  **Rotate:** To rotate an image, we need to know two things:

    -#  The center with respect to which the image will rotate
    -#  The angle to be rotated. In OpenCV a positive angle is counter-clockwise
    -#  *Optional:* A scale factor

    We define these parameters with the following snippet:
    @code{.cpp}
    Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
    double angle = -50.0;
    double scale = 0.6;
    @endcode
-#  We generate the rotation matrix with the OpenCV function @ref cv::getRotationMatrix2D , which
    returns a \f$2 \times 3\f$ matrix (in this case *rot_mat*)
    @code{.cpp}
    rot_mat = getRotationMatrix2D( center, angle, scale );
    @endcode
-#  We now apply the found rotation to the output of our previous Transformation.
    @code{.cpp}
    warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );
    @endcode
-#  Finally, we display our results in two windows plus the original image for good measure:
    @code{.cpp}
    namedWindow( source_window, WINDOW_AUTOSIZE );
    imshow( source_window, src );

    namedWindow( warp_window, WINDOW_AUTOSIZE );
    imshow( warp_window, warp_dst );

    namedWindow( warp_rotate_window, WINDOW_AUTOSIZE );
    imshow( warp_rotate_window, warp_rotate_dst );
    @endcode
-#  We just have to wait until the user exits the program
    @code{.cpp}
    waitKey(0);
    @endcode

Result
------

-#  After compiling the code above, we can give it the path of an image as argument. For instance,
    for a picture like:

    ![](images/Warp_Affine_Tutorial_Original_Image.jpg)

    after applying the first Affine Transform we obtain:

    ![](images/Warp_Affine_Tutorial_Result_Warp.jpg)

    and finally, after applying a negative rotation (remember negative means clockwise) and a scale
    factor, we get:

    ![](images/Warp_Affine_Tutorial_Result_Warp_Rotate.jpg)
