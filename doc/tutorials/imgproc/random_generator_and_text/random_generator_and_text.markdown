Random generator and text with OpenCV {#tutorial_random_generator_and_text}
=====================================

@prev_tutorial{tutorial_basic_geometric_drawing}
@next_tutorial{tutorial_gausian_median_blur_bilateral_filter}

Goals
-----

In this tutorial you will learn how to:

-   Use the *Random Number generator class* (@ref cv::RNG ) and how to get a random number from a uniform distribution.
-   Display text on an OpenCV window by using the function @ref cv::putText

Code
----

-   In the previous tutorial (@ref tutorial_basic_geometric_drawing) we drew diverse geometric figures, giving as input parameters such as coordinates (in the form of @ref cv::Point), color, thickness, etc. You might have noticed that we gave specific values for these arguments.
-   In this tutorial, we intend to use *random* values for the drawing parameters. Also, we intend to populate our image with a big number of geometric figures. Since we will be initializing them in a random fashion, this process will be automatic and made by using *loops* .
@add_toggle_cpp
-   This code is in your OpenCV sample folder. Otherwise you can grab it from
    [here](https://raw.githubusercontent.com/opencv/opencv/3.4/samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp).
    @include samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp
@end_toggle
@add_toggle_python
-   This code is in your OpenCV sample folder. Otherwise you can grab it from
    [here](https://raw.githubusercontent.com/opencv/opencv/3.4/samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py).
    @include samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py
@end_toggle

Explanation
-----------

-#  Let's start by checking out the *main* function.
@add_toggle_cpp
We observe that first thing we do is creating a *Random Number Generator* object (RNG):
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp create_RNG
    RNG implements a random number generator. In this example, *rng* is a RNG element initialized with the value *0xFFFFFFFF*
@end_toggle
@add_toggle_python
    In Python we can perform random number generation with NumPy's `randint()` function. There are no bindings for the OpenCV RNG class in Python.
@end_toggle
-#  Then we create a matrix initialized to *zeros* (which means that it will appear as black), specifying its height, width and its type:
@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp create_zeros
@end_toggle
@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py create_zeros
@end_toggle
-#  Then we proceed to draw crazy stuff. After taking a look at the code, you can see that it is mainly divided in 8 sections, defined as functions:
@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp main_drawing
@end_toggle
@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py main_drawing
@end_toggle
    All of these functions follow the same pattern, so we will analyze only a couple of them, since the same explanation applies for all.

-#  Checking out the function **Drawing_Random_Lines**:
@add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp random_lines
    We can observe the following:
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp line_points
    -   The *for* loop will repeat **NUMBER** times. Since the function @ref cv::line is inside this loop, that means that **NUMBER** lines will be generated.
    -   The line extremes are given by *pt1* and *pt2*. For *pt1* we can see that:
        -   We know that **rng** is a *Random number generator* object. In the code above we are calling **rng.uniform(a,b)**. This generates a randomly uniformed distribution between the values **a** and **b** (inclusive in **a**, exclusive in **b**).
        -   From the explanation above, we deduce that the extremes *pt1* and *pt2* will be random values, so the lines positions will be quite impredictable, giving a nice visual effect (check out the Result section below).
        -   As another observation, we notice that in the @ref cv::line arguments, for the *color* input we enter:
            @code{.cpp}
            randomColor(rng)
            @endcode
            Let's check the function implementation:
            @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp random_color
            As we can see, the return value is an *Scalar* with 3 randomly initialized values, which are used as the *R*, *G* and *B* parameters for the line color. Hence, the color of the lines will be random too!
@end_toggle
@add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py random_lines
    We can observe the following:
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py line_points
    -   The *for* loop will repeat **NUMBER** times. Since the function @ref cv::line is inside this loop, that means that **NUMBER** lines will be generated.
    -   The line extremes are given by *pt1* and *pt2*. For *pt1* we can see that:
        -   In the code above we are calling **np.random.randint(a,b)**. This generates a randomly uniformed distribution between the values **a** and **b** (inclusive in **a**, exclusive in **b**).
        -   From the explanation above, we deduce that the extremes *pt1* and *pt2* will be random values, so the lines positions will be quite impredictable, giving a nice visual effect (check out the Result section below).
        -   As another observation, we notice that in the @ref cv::line arguments, for the *color* input we generate a random color by using a tuple of three random values generated like so:
        @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py random_color
@end_toggle
-#  The explanation above applies for the other functions generating circles, ellipses, polygons, etc. The parameters such as *center* and *vertices* are also generated randomly.
-#  Before finishing, we also should take a look at the functions *Display_Random_Text* and *Displaying_Big_End*, since they both have a few interesting features:
-#  **Display_Random_Text:**
    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp random_text
    Everything looks familiar but the expression:
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp put_text
    So, what does the function @ref cv::putText do? In our example:

    -   Draws the text **"Testing text rendering"** in **image**
    -   The bottom-left corner of the text will be located in the Point **org**
    -   The font type is a random integer value in the range: \f$[0, 8>\f$.
    -   The scale of the font is denoted by the expression **rng.uniform(0, 100)x0.05 + 0.1** (meaning its range is: \f$[0.1, 5.1>\f$)
    -   The text color is random (denoted by **randomColor(rng)**)
    -   The text thickness ranges between 1 and 10, as specified by **rng.uniform(1,10)**
    @end_toggle
    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py random_text
    Everything looks familiar but the expression:
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py put_text
    So, what does the function @ref cv::putText do? In our example:

    -   Draws the text **"Testing text rendering"** in **image**
    -   The bottom-left corner of the text will be located at the point defined by the tuple **org**
    -   The font type is a random integer value in the range: \f$[0, 8>\f$.
    -   The scale of the font is denoted by the expression **rng.uniform(0, 100)x0.05 + 0.1** (meaning its range is: \f$[0.1, 5.1>\f$)
    -   The text color is random, generated in the same fashion as above
    -   The text thickness ranges between 1 and 10, as specified by **np.random.randint(1, 10)**
    @end_toggle

    As a result, we will get (analagously to the other drawing functions) **NUMBER** texts over our image, in random locations.

-#  **Displaying_Big_End**
    @add_toggle_cpp
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp big_end
    Besides the function **getTextSize** (which gets the size of the argument text), the new operation we can observe is inside the *for* loop:
    @snippet samples/cpp/tutorial_code/ImgProc/basic_drawing/Drawing_2.cpp subtract
    So, **image2** is the subtraction of **image** and **Scalar::all(i)**. In fact, what happens here is that every pixel of **image2** will be the result of subtracting every pixel of **image** minus the value of **i** (remember that for each pixel we are considering three values such as R, G and B, so each of them will be affected)

    Also remember that the subtraction operation *always* performs internally a **saturate** operation, which means that the result obtained will always be inside the allowed range (no negative and between 0 and 255 for our example).
    @end_toggle
    @add_toggle_python
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py big_end
    Besides the function **getTextSize** (which gets the size of the argument text), the new operation we can observe is inside the *for* loop:
    @snippet samples/python/tutorial_code/imgProc/BasicGeometricDrawing/drawing.py subtract
    So, **image2** is the subtraction of **image** and the value of **i**. In fact, what happens here is that every pixel of **image2** will be the result of subtracting every pixel of **image** minus the value of **i** (remember that for each pixel we are considering three values such as R, G and B, so each of them will be affected)

    Note that NumPy subtraction is equivalent to regular subtraction and is not saturated: meaning that values can become negative. Thus rather than values simply being set to 0 if negative, the values will wrap around beginning from 255 and count down.
    @end_toggle

Result
------

As you just saw in the Code section, the program will sequentially execute diverse drawing functions, which will produce:

-#  First a random set of *NUMBER* lines will appear on screen such as it can be seen in this screenshot:

    ![](images/Drawing_2_Tutorial_Result_0.jpg)

-#  Then, a new set of figures, these time *rectangles* will follow.
-#  Now some ellipses will appear, each of them with random position, size, thickness and arc length:

    ![](images/Drawing_2_Tutorial_Result_2.jpg)

-#  Now, *polylines* with 03 segments will appear on screen, again in random configurations.

    ![](images/Drawing_2_Tutorial_Result_3.jpg)

-#  Filled polygons (in this example triangles) will follow.
-#  The last geometric figure to appear: circles!

    ![](images/Drawing_2_Tutorial_Result_5.jpg)

-#  Near the end, the text *"Testing Text Rendering"* will appear in a variety of fonts, sizes, colors and positions.
-#  And the big end (which by the way expresses a big truth too):

    ![](images/Drawing_2_Tutorial_Result_big.jpg)
