Affine Transformations {#tutorial_warp_affine}
======================

@prev_tutorial{tutorial_remap}
@next_tutorial{tutorial_histogram_equalization}

Goal
----

In this tutorial you will learn how to:

-  Use the OpenCV function @ref cv::warpAffine to implement simple remapping routines.
-  Use the OpenCV function @ref cv::getRotationMatrix2D to obtain a \f$2 \times 3\f$ rotation matrix

Theory
------

### What is an Affine Transformation?

-#  A transformation that can be expressed in the form of a *matrix multiplication* (linear
    transformation) followed by a *vector addition* (translation).
-#  From the above, we can use an Affine Transformation to express:

    -#  Rotations (linear transformation)
    -#  Translations (vector addition)
    -#  Scale operations (linear transformation)

    you can see that, in essence, an Affine Transformation represents a **relation** between two
    images.

-#  The usual way to represent an Affine Transformation is by using a \f$2 \times 3\f$ matrix.

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
    using \f$A\f$ and \f$B\f$, we can do the same with:

    \f$T = A \cdot \begin{bmatrix}x \\ y\end{bmatrix} + B\f$ or \f$T = M \cdot  [x, y, 1]^{T}\f$

    \f[T =  \begin{bmatrix}
        a_{00}x + a_{01}y + b_{00} \\
        a_{10}x + a_{11}y + b_{10}
        \end{bmatrix}\f]

### How do we get an Affine Transformation?

-#  We mentioned that an Affine Transformation is basically a **relation**
    between two images. The information about this relation can come, roughly, in two ways:
    -#  We know both \f$X\f$ and T and we also know that they are related. Then our task is to find \f$M\f$
    -#  We know \f$M\f$ and \f$X\f$. To obtain \f$T\f$ we only need to apply \f$T = M \cdot X\f$. Our information
        for \f$M\f$ may be explicit (i.e. have the 2-by-3 matrix) or it can come as a geometric relation
        between points.

-#  Let's explain this in a better way (b). Since \f$M\f$ relates 2 images, we can analyze the simplest
    case in which it relates three points in both images. Look at the figure below:

    ![](images/Warp_Affine_Tutorial_Theory_0.jpg)

    the points 1, 2 and 3 (forming a triangle in image 1) are mapped into image 2, still forming a
    triangle, but now they have changed notoriously. If we find the Affine Transformation with these
    3 points (you can choose them as you like), then we can apply this found relation to all the
    pixels in an image.

Code
----

-   **What does this program do?**
    -   Loads an image
    -   Applies an Affine Transform to the image. This transform is obtained from the relation
        between three points. We use the function @ref cv::warpAffine for that purpose.
    -   Applies a Rotation to the image after being transformed. This rotation is with respect to
        the image center
    -   Waits until the user exits the program

@add_toggle_cpp
-   The tutorial's code is shown below. You can also download it
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/ImgProc/Smoothing/Smoothing.cpp)
    @include samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp
@end_toggle

@add_toggle_java
-   The tutorial's code is shown below. You can also download it
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/ImgProc/Smoothing/Smoothing.cpp)
    @include samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java
@end_toggle

@add_toggle_python
-   The tutorial's code is shown below. You can also download it
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py)
    @include samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py
@end_toggle

Explanation
-----------

-   Load an image:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Load the image
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Load the image
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Load the image
    @end_toggle

-   **Affine Transform:** As we explained in lines above, we need two sets of 3 points to derive the
    affine transform relation. Have a look:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Set your 3 points to calculate the  Affine Transform
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Set your 3 points to calculate the  Affine Transform
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Set your 3 points to calculate the  Affine Transform
    @end_toggle
    You may want to draw these points to get a better idea on how they change. Their locations are
    approximately the same as the ones depicted in the example figure (in the Theory section). You
    may note that the size and orientation of the triangle defined by the 3 points change.

-   Armed with both sets of points, we calculate the Affine Transform by using OpenCV function @ref
    cv::getAffineTransform :

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Get the Affine Transform
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Get the Affine Transform
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Get the Affine Transform
    @end_toggle
    We get a \f$2 \times 3\f$ matrix as an output (in this case **warp_mat**)

-   We then apply the Affine Transform just found to the src image

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Apply the Affine Transform just found to the src image
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Apply the Affine Transform just found to the src image
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Apply the Affine Transform just found to the src image
    @end_toggle
    with the following arguments:

    -   **src**: Input image
    -   **warp_dst**: Output image
    -   **warp_mat**: Affine transform
    -   **warp_dst.size()**: The desired size of the output image

    We just got our first transformed image! We will display it in one bit. Before that, we also
    want to rotate it...

-   **Rotate:** To rotate an image, we need to know two things:

    -#  The center with respect to which the image will rotate
    -#  The angle to be rotated. In OpenCV a positive angle is counter-clockwise
    -#  *Optional:* A scale factor

    We define these parameters with the following snippet:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Compute a rotation matrix with respect to the center of the image
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Compute a rotation matrix with respect to the center of the image
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Compute a rotation matrix with respect to the center of the image
    @end_toggle

-   We generate the rotation matrix with the OpenCV function @ref cv::getRotationMatrix2D , which
    returns a \f$2 \times 3\f$ matrix (in this case *rot_mat*)

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Get the rotation matrix with the specifications above
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Get the rotation matrix with the specifications above
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Get the rotation matrix with the specifications above
    @end_toggle

-   We now apply the found rotation to the output of our previous Transformation:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Rotate the warped image
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Rotate the warped image
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Rotate the warped image
    @end_toggle

-   Finally, we display our results in two windows plus the original image for good measure:

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Show what you got
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Show what you got
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Show what you got
    @end_toggle

-   We just have to wait until the user exits the program

    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgTrans/Geometric_Transforms_Demo.cpp Wait until user exits the program
    @end_toggle

    @add_toggle_java
    @snippet samples/java/tutorial_code/ImgTrans/warp_affine/GeometricTransformsDemo.java Wait until user exits the program
    @end_toggle

    @add_toggle_python
    @snippet samples/python/tutorial_code/ImgTrans/warp_affine/Geometric_Transforms_Demo.py Wait until user exits the program
    @end_toggle

Result
------

-   After compiling the code above, we can give it the path of an image as argument. For instance,
    for a picture like:

    ![](images/Warp_Affine_Tutorial_Original_Image.jpg)

    after applying the first Affine Transform we obtain:

    ![](images/Warp_Affine_Tutorial_Result_Warp.jpg)

    and finally, after applying a negative rotation (remember negative means clockwise) and a scale
    factor, we get:

    ![](images/Warp_Affine_Tutorial_Result_Warp_Rotate.jpg)
