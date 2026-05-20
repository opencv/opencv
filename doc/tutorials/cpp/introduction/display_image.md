# Getting Started with Images

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.4.4 |

:::

:::{warning}
This tutorial can contain obsolete information.
:::
## Goal

In this tutorial you will learn how to:

-   Read an image from file (using [cv::imread](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html#gaffb68fce322c6e52841d7d9357b9ad2d))
-   Display an image in an OpenCV window (using [cv::imshow](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563))
-   Write an image to a file (using [cv::imwrite](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce))

## Source Code

::::{tab-set}
:::{tab-item} C++
:sync: cpp

-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/introduction/display_image/display_image.cpp)

-   **Code at glance:**

```{doxyinclude} samples/cpp/tutorial_code/introduction/display_image/display_image.cpp
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/introduction/display_image/display_image.py)

-   **Code at glance:**

```{doxyinclude} samples/python/tutorial_code/introduction/display_image/display_image.py
:language: python
```

:::
::::

## Explanation

::::{tab-set}
:::{tab-item} C++
:sync: cpp

In OpenCV 3 we have multiple modules. Each one takes care of a different area or approach towards
image processing. You could already observe this in the structure of the user guide of these
tutorials itself. Before you use any of them you first need to include the header files where the
content of each individual module is declared.

You'll almost always end up using the:

- [core](https://docs.opencv.org/5.x/d0/de1/group__core.html) section, as here are defined the basic building blocks of the library
- [imgcodecs](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html) module, which provides functions for reading and writing
- [highgui](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html) module, as this contains the functions to show an image in a window

We also include the *iostream* to facilitate console line output and input.

By declaring `using namespace cv;`, in the following, the library functions can be accessed without explicitly stating the namespace.

```{doxysnippet} cpp/tutorial_code/introduction/display_image/display_image.cpp
:tag: includes
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

As a first step, the OpenCV python library is imported.
The proper way to do this is to additionally assign it the name *cv*, which is used in the following to reference the library.

```{doxysnippet} samples/python/tutorial_code/introduction/display_image/display_image.py
:tag: imports
:language: python
```

:::
::::

Now, let's analyze the main code.
As a first step, we read the image "starry_night.jpg" from the OpenCV samples.
In order to do so, a call to the [cv::imread](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html#gaffb68fce322c6e52841d7d9357b9ad2d) function loads the image using the file path specified by the first argument.
The second argument is optional and specifies the format in which we want the image. This may be:

-   IMREAD_COLOR loads the image in the BGR 8-bit format. This is the **default** that is used here.
-   IMREAD_UNCHANGED loads the image as is (including the alpha channel if present)
-   IMREAD_GRAYSCALE loads the image as an intensity one

After reading in the image data will be stored in a [cv::Mat](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html) object.

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/introduction/display_image/display_image.cpp
:tag: imread
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/introduction/display_image/display_image.py
:tag: imread
:language: python
```

:::
::::

:::{note}
OpenCV offers support for the image formats Windows bitmap (bmp), portable image formats (pbm,
 pgm, ppm) and Sun raster (sr, ras). With help of plugins (you need to specify to use them if you
 build yourself the library, nevertheless in the packages we ship present by default) you may
 also load image formats like JPEG (jpeg, jpg, jpe), JPEG 2000 (jp2 - codenamed in the CMake as
 Jasper), TIFF files (tiff, tif) and portable network graphics (png). Furthermore, OpenEXR is
 also a possibility.
:::
Afterwards, a check is executed, if the image was loaded correctly.
::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/introduction/display_image/display_image.cpp
:tag: empty
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/introduction/display_image/display_image.py
:tag: empty
:language: python
```

:::
::::

Then, the image is shown using a call to the [cv::imshow](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563) function.
The first argument is the title of the window and the second argument is the [cv::Mat](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html) object that will be shown.

Because we want our window to be displayed until the user presses a key (otherwise the program would
end far too quickly), we use the [cv::waitKey](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7) function whose only parameter is just how long
should it wait for a user input (measured in milliseconds). Zero means to wait forever.
The return value is the key that was pressed.

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/introduction/display_image/display_image.cpp
:tag: imshow
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/introduction/display_image/display_image.py
:tag: imshow
:language: python
```

:::
::::

In the end, the image is written to a file if the pressed key was the "s"-key.
For this the [cv::imwrite](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) function is called that has the file path and the [cv::Mat](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html) object as an argument.

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/introduction/display_image/display_image.cpp
:tag: imsave
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/introduction/display_image/display_image.py
:tag: imsave
:language: python
```

:::
::::
