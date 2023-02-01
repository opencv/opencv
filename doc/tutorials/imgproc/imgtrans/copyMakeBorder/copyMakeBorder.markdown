Adding borders to your images {#tutorial_copyMakeBorder}
=============================

@prev_tutorial{tutorial_filter_2d}
@next_tutorial{tutorial_sobel_derivatives}

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function **copyMakeBorder()** to set the borders (extra padding to your
    image).

Theory
------

@note The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

-#  In our previous tutorial we learned to use convolution to operate on images. One problem that
    naturally arises is how to handle the boundaries. How can we convolve them if the evaluated
    points are at the edge of the image?
-#  What most of OpenCV functions do is to copy a given image onto another slightly larger image and
    then automatically pads the boundary (by any of the methods explained in the sample code just
    below). This way, the convolution can be performed over the needed pixels without problems (the
    extra padding is cut after the operation is done).
-#  In this tutorial, we will briefly explore two ways of defining the extra padding (border) for an
    image:

    -#  **BORDER_CONSTANT**: Pad the image with a constant value (i.e. black or \f$0\f$
    -#  **BORDER_REPLICATE**: The row or column at the very edge of the original is replicated to
        the extra border.

    This will be seen more clearly in the Code section.

-   **What does this program do?**
    -   Load an image
    -   Let the user choose what kind of padding use in the input image. There are two options:

        -#  *Constant value border*: Applies a padding of a constant value for the whole border.
            This value will be updated randomly each 0.5 seconds.
        -#  *Replicated border*: The border will be replicated from the pixel values at the edges of
            the original image.

        The user chooses either option by pressing 'c' (constant) or 'r' (replicate)
    -   The program finishes when the user presses 'ESC'

Code
----

The tutorial code's is shown lines below.

@add_toggle_cpp
You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/3.4/samples/cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp)
@include samples/cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp
@end_toggle

@add_toggle_java
You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/3.4/samples/java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java)
@include samples/java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java
@end_toggle

@add_toggle_python
You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/3.4/samples/python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py)
@include samples/python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py
@end_toggle

Explanation
-----------

#### Declare the variables

First we declare the variables we are going to use:

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp variables
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java variables
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py variables
@end_toggle

Especial attention deserves the variable *rng* which is a random number generator. We use it to
generate the random border color, as we will see soon.

#### Load an image

As usual we load our source image *src*:

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp load
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java load
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py load
@end_toggle

#### Create a window

After giving a short intro of how to use the program, we create a window:

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp create_window
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java create_window
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py create_window
@end_toggle

#### Initialize arguments

Now we initialize the argument that defines the size of the borders (*top*, *bottom*, *left* and
*right*). We give them a value of 5% the size of *src*.

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp init_arguments
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java init_arguments
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py init_arguments
@end_toggle

#### Loop

The program runs in an infinite loop while the key **ESC** isn't pressed.
If the user presses '**c**' or '**r**', the *borderType* variable
takes the value of *BORDER_CONSTANT* or *BORDER_REPLICATE* respectively:

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp check_keypress
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java check_keypress
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py check_keypress
@end_toggle

#### Random color

In each iteration (after 0.5 seconds), the random border color (*value*) is updated...

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp update_value
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java update_value
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py update_value
@end_toggle

This value is a set of three numbers picked randomly in the range \f$[0,255]\f$.

#### Form a border around the image

Finally, we call the function **copyMakeBorder()** to apply the respective padding:

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp copymakeborder
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java copymakeborder
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py copymakeborder
@end_toggle

-   The arguments are:

    -#  *src*: Source image
    -#  *dst*: Destination image
    -#  *top*, *bottom*, *left*, *right*: Length in pixels of the borders at each side of the image.
        We define them as being 5% of the original size of the image.
    -#  *borderType*: Define what type of border is applied. It can be constant or replicate for
        this example.
    -#  *value*: If *borderType* is *BORDER_CONSTANT*, this is the value used to fill the border
        pixels.

#### Display the results

We display our output image in the image created previously

@add_toggle_cpp
@snippet cpp/tutorial_code/ImgTrans/copyMakeBorder_demo.cpp display
@end_toggle

@add_toggle_java
@snippet java/tutorial_code/ImgTrans/MakeBorder/CopyMakeBorder.java display
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/ImgTrans/MakeBorder/copy_make_border.py display
@end_toggle

Results
-------

-#  After compiling the code above, you can execute it giving as argument the path of an image. The
    result should be:

    -   By default, it begins with the border set to BORDER_CONSTANT. Hence, a succession of random
        colored borders will be shown.
    -   If you press 'r', the border will become a replica of the edge pixels.
    -   If you press 'c', the random colored borders will appear again
    -   If you press 'ESC' the program will exit.

    Below some screenshot showing how the border changes color and how the *BORDER_REPLICATE*
    option looks:

    ![](images/CopyMakeBorder_Tutorial_Results.jpg)
