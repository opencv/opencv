# Operations with images

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

:::

## Input/Output

#### Images

Load an image from a file:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Load an image from a file
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Load an image from a file
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Load an image from a file
:language: python
```

:::
::::

If you read a jpg file, a 3 channel image is created by default. If you need a grayscale image, use:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Load an image from a file in grayscale
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Load an image from a file in grayscale
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Load an image from a file in grayscale
:language: python
```

:::
::::

:::{note}
Format of the file is determined by its content (first few bytes). To save an image to a file:
:::
::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Save image
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Save image
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Save image
:language: python
```

:::
::::

:::{note}
Format of the file is determined by its extension.
Use [cv::imdecode](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e) and [cv::imencode](https://docs.opencv.org/5.x/d4/da8/group__imgcodecs.html#ga461f9ac09887e47797a54567df3b8b63) to read and write an image from/to memory rather than a file.
:::
## Basic operations with images

#### Accessing pixel intensity values

In order to get pixel intensity value, you have to know the type of an image and the number of
channels. Here is an example for a single channel grey scale image (type 8UC1) and pixel coordinates
x and y:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Pixel access 1
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Pixel access 1
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Pixel access 1
:language: python
```

:::
::::

C++ version only:
intensity.val[0] contains a value from 0 to 255. Note the ordering of x and y. Since in OpenCV
images are represented by the same structure as matrices, we use the same convention for both
cases - the 0-based row index (or y-coordinate) goes first and the 0-based column index (or
x-coordinate) follows it. Alternatively, you can use the following notation (**C++ only**):

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Pixel access 2
:language: cpp
```

Now let us consider a 3 channel image with BGR color ordering (the default format returned by
imread):

**C++ code**

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Pixel access 3
:language: cpp
```

**Python Python**

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Pixel access 3
:language: python
```

You can use the same method for floating-point images (for example, you can get such an image by
running Sobel on a 3 channel image) (**C++ only**):

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Pixel access 4
:language: cpp
```

The same method can be used to change pixel intensities:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Pixel access 5
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Pixel access 5
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Pixel access 5
:language: python
```

:::
::::

There are functions in OpenCV, especially from calib3d module, such as [cv::projectPoints](https://docs.opencv.org/5.x/da/d35/group____3d.html#gaa9baca117e5ab7d588e23b50b2ff4509), that take an
array of 2D or 3D points in the form of Mat. Matrix should contain exactly one column, each row
corresponds to a point, matrix type should be 32FC2 or 32FC3 correspondingly. Such a matrix can be
easily constructed from `std::vector` (**C++ only**):

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Mat from points vector
:language: cpp
```

One can access a point in this matrix using the same method `Mat::at` (**C++ only**):

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Point access
:language: cpp
```

#### Memory management and reference counting

Mat is a structure that keeps matrix/image characteristics (rows and columns number, data type etc)
and a pointer to data. So nothing prevents us from having several instances of Mat corresponding to
the same data. A Mat keeps a reference count that tells if data has to be deallocated when a
particular instance of Mat is destroyed. Here is an example of creating two matrices without copying
data (**C++ only**):

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Reference counting 1
:language: cpp
```

As a result, we get a 32FC1 matrix with 3 columns instead of 32FC3 matrix with 1 column. `pointsMat`
uses data from points and will not deallocate the memory when destroyed. In this particular
instance, however, developer has to make sure that lifetime of `points` is longer than of `pointsMat`
If we need to copy the data, this is done using, for example, [cv::Mat::copyTo](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77) or [cv::Mat::clone](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html#a03d2a2570d06dcae378f788725789aa4):

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Reference counting 2
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Reference counting 2
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Reference counting 2
:language: python
```

:::
::::

An empty output Mat can be supplied to each function.
Each implementation calls Mat::create for a destination matrix.
This method allocates data for a matrix if it is empty.
If it is not empty and has the correct size and type, the method does nothing.
If however, size or type are different from the input arguments, the data is deallocated (and lost) and a new data is allocated.
For example:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Reference counting 3
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Reference counting 3
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Reference counting 3
:language: python
```

:::
::::

#### Primitive operations

There is a number of convenient operators defined on a matrix. For example, here is how we can make
a black image from an existing greyscale image `img`

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Set image to black
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Set image to black
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Set image to black
:language: python
```

:::
::::

Selecting a region of interest:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Select ROI
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Select ROI
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Select ROI
:language: python
```

:::
::::

Conversion from color to greyscale:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: BGR to Gray
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: BGR to Gray
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: BGR to Gray
:language: python
```

:::
::::

Change image type from 8UC1 to 32FC1:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: Convert to CV_32F
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: Convert to CV_32F
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: Convert to CV_32F
:language: python
```

:::
::::

#### Visualizing images

It is very useful to see intermediate results of your algorithm during development process. OpenCV
provides a convenient way of visualizing images. A 8U image can be shown using:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: imshow 1
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: imshow 1
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: imshow 1
:language: python
```

:::
::::

A call to waitKey() starts a message passing cycle that waits for a key stroke in the "image"
window. A 32F image needs to be converted to 8U type. For example:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp
:tag: imshow 2
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} samples/java/tutorial_code/core/mat_operations/MatOperations.java
:tag: imshow 2
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} samples/python/tutorial_code/core/mat_operations/mat_operations.py
:tag: imshow 2
:language: python
```

:::
::::

:::{note}
Here [cv::namedWindow](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b) is not necessary since it is immediately followed by [cv::imshow](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563).
Nevertheless, it can be used to change the window properties or when using [cv::createTrackbar](https://docs.opencv.org/5.x/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b)
:::
