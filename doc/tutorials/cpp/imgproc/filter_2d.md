# Making your own linear filters!

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

:::

## Goal

In this tutorial you will learn how to:

-   Use the OpenCV function **filter2D()** to create your own linear filters.

## Theory

:::{note}
The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.
:::
#### Correlation

In a very general sense, correlation is an operation between every part of an image and an operator
(kernel).

#### What is a kernel?

A kernel is essentially a fixed size array of numerical coefficients along with an *anchor point* in
that array, which is typically located at the center.

![](images/filter_2d_tutorial_kernel_theory.png)

#### How does correlation with a kernel work?

Assume you want to know the resulting value of a particular location in the image. The value of the
correlation is calculated in the following way:

1. Place the kernel anchor on top of a determined pixel, with the rest of the kernel overlaying the
   corresponding local pixels in the image.
1. Multiply the kernel coefficients by the corresponding image pixel values and sum the result.
1. Place the result to the location of the *anchor* in the input image.
1. Repeat the process for all pixels by scanning the kernel over the entire image.

Expressing the procedure above in the form of an equation we would have:

$$

H(x,y) = \sum_{i=0}^{M_{i} - 1} \sum_{j=0}^{M_{j}-1} I(x+i - a_{i}, y + j - a_{j})K(i,j)

$$

Fortunately, OpenCV provides you with the function **filter2D()** so you do not have to code all
these operations.

####  What does this program do?
-   Loads an image
-   Performs a *normalized box filter*. For instance, for a kernel of size $size = 3$, the
    kernel would be:

$$

K = \dfrac{1}{3 \cdot 3} \begin{bmatrix}
1 & 1 & 1  \\
        1 & 1 & 1  \\
        1 & 1 & 1
\end{bmatrix}

$$

The program will perform the filter operation with kernels of sizes 3, 5, 7, 9 and 11.

-   The filter output (with each kernel) will be shown during 500 milliseconds

## Code

The tutorial code's is shown in the lines below.

::::{tab-set}
:::{tab-item} C++
:sync: cpp

You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/cpp/tutorial_code/ImgTrans/filter2D_demo.cpp)

```{doxyinclude} cpp/tutorial_code/ImgTrans/filter2D_demo.cpp
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/java/tutorial_code/ImgTrans/Filter2D/Filter2D_Demo.java)

```{doxyinclude} java/tutorial_code/ImgTrans/Filter2D/Filter2D_Demo.java
:language: java
```

:::
:::{tab-item} Python
:sync: python

You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/python/tutorial_code/ImgTrans/Filter2D/filter2D.py)

```{doxyinclude} python/tutorial_code/ImgTrans/Filter2D/filter2D.py
:language: python
```

:::
::::

## Explanation

####  Load an image

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/ImgTrans/filter2D_demo.cpp
:tag: load
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/ImgTrans/Filter2D/Filter2D_Demo.java
:tag: load
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/ImgTrans/Filter2D/filter2D.py
:tag: load
:language: python
```

:::
::::

####  Initialize the arguments

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/ImgTrans/filter2D_demo.cpp
:tag: init_arguments
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/ImgTrans/Filter2D/Filter2D_Demo.java
:tag: init_arguments
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/ImgTrans/Filter2D/filter2D.py
:tag: init_arguments
:language: python
```

:::
::::

#### Loop

Perform an infinite loop updating the kernel size and applying our linear filter to the input
image. Let's analyze that more in detail:

-  First we define the kernel our filter is going to use. Here it is:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/ImgTrans/filter2D_demo.cpp
:tag: update_kernel
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/ImgTrans/Filter2D/Filter2D_Demo.java
:tag: update_kernel
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/ImgTrans/Filter2D/filter2D.py
:tag: update_kernel
:language: python
```

:::
::::

The first line is to update the *kernel_size* to odd values in the range: $[3,11]$.
The second line actually builds the kernel by setting its value to a matrix filled with
$1's$ and normalizing it by dividing it between the number of elements.

-  After setting the kernel, we can generate the filter by using the function **filter2D()** :

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/ImgTrans/filter2D_demo.cpp
:tag: apply_filter
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/ImgTrans/Filter2D/Filter2D_Demo.java
:tag: apply_filter
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/ImgTrans/Filter2D/filter2D.py
:tag: apply_filter
:language: python
```

:::
::::

-  The arguments denote:
       -  *src*: Source image
       -  *dst*: Destination image
       -  *ddepth*: The depth of *dst*. A negative value (such as $-1$) indicates that the depth is
        the same as the source.
       -  *kernel*: The kernel to be scanned through the image
       -  *anchor*: The position of the anchor relative to its kernel. The location *Point(-1, -1)*
       indicates the center by default.
       -  *delta*: A value to be added to each pixel during the correlation. By default it is $0$
       -  *BORDER_DEFAULT*: We let this value by default (more details in the following tutorial)

-  Our program will effectuate a *while* loop, each 500 ms the kernel size of our filter will be
    updated in the range indicated.

## Results

1. After compiling the code above, you can execute it giving as argument the path of an image. The
   result should be a window that shows an image blurred by a normalized filter. Each 0.5 seconds
   the kernel size should change, as can be seen in the series of snapshots below:

![](images/filter_2d_tutorial_result.jpg)
