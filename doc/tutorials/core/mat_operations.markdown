Operations with images {#tutorial_mat_operations}
======================

@tableofcontents

@prev_tutorial{tutorial_mat_mask_operations}
@next_tutorial{tutorial_adding_images}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

Input/Output
------------

### Images

Load an image from a file:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Load an image from a file
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Load an image from a file
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Load an image from a file
@end_toggle

If you read a jpg file, a 3 channel image is created by default. If you need a grayscale image, use:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Load an image from a file in grayscale
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Load an image from a file in grayscale
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Load an image from a file in grayscale
@end_toggle

@note Format of the file is determined by its content (first few bytes). To save an image to a file:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Save image
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Save image
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Save image
@end_toggle

@note Format of the file is determined by its extension.

@note Use cv::imdecode and cv::imencode to read and write an image from/to memory rather than a file.

Basic operations with images
----------------------------

### Accessing pixel intensity values

In order to get pixel intensity value, you have to know the type of an image and the number of
channels. Here is an example for a single channel grey scale image (type 8UC1) and pixel coordinates
x and y:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Pixel access 1
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Pixel access 1
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Pixel access 1
@end_toggle

C++ version only:
intensity.val[0] contains a value from 0 to 255. Note the ordering of x and y. Since in OpenCV
images are represented by the same structure as matrices, we use the same convention for both
cases - the 0-based row index (or y-coordinate) goes first and the 0-based column index (or
x-coordinate) follows it. Alternatively, you can use the following notation (**C++ only**):

@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Pixel access 2

Now let us consider a 3 channel image with BGR color ordering (the default format returned by
imread):

**C++ code**
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Pixel access 3

**Python Python**
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Pixel access 3

You can use the same method for floating-point images (for example, you can get such an image by
running Sobel on a 3 channel image) (**C++ only**):

@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Pixel access 4

The same method can be used to change pixel intensities:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Pixel access 5
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Pixel access 5
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Pixel access 5
@end_toggle

There are functions in OpenCV, especially from calib3d module, such as cv::projectPoints, that take an
array of 2D or 3D points in the form of Mat. Matrix should contain exactly one column, each row
corresponds to a point, matrix type should be 32FC2 or 32FC3 correspondingly. Such a matrix can be
easily constructed from `std::vector` (**C++ only**):

@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Mat from points vector

One can access a point in this matrix using the same method `Mat::at` (**C++ only**):

@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Point access

### Memory management and reference counting

Mat is a structure that keeps matrix/image characteristics (rows and columns number, data type etc)
and a pointer to data. So nothing prevents us from having several instances of Mat corresponding to
the same data. A Mat keeps a reference count that tells if data has to be deallocated when a
particular instance of Mat is destroyed. Here is an example of creating two matrices without copying
data (**C++ only**):

@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Reference counting 1

As a result, we get a 32FC1 matrix with 3 columns instead of 32FC3 matrix with 1 column. `pointsMat`
uses data from points and will not deallocate the memory when destroyed. In this particular
instance, however, developer has to make sure that lifetime of `points` is longer than of `pointsMat`
If we need to copy the data, this is done using, for example, cv::Mat::copyTo or cv::Mat::clone:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Reference counting 2
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Reference counting 2
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Reference counting 2
@end_toggle

An empty output Mat can be supplied to each function.
Each implementation calls Mat::create for a destination matrix.
This method allocates data for a matrix if it is empty.
If it is not empty and has the correct size and type, the method does nothing.
If however, size or type are different from the input arguments, the data is deallocated (and lost) and a new data is allocated.
For example:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Reference counting 3
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Reference counting 3
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Reference counting 3
@end_toggle

### Primitive operations

There is a number of convenient operators defined on a matrix. For example, here is how we can make
a black image from an existing greyscale image `img`

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Set image to black
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Set image to black
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Set image to black
@end_toggle

Selecting a region of interest:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Select ROI
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Select ROI
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Select ROI
@end_toggle

Conversion from color to greyscale:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp BGR to Gray
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java BGR to Gray
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py BGR to Gray
@end_toggle

Change image type from 8UC1 to 32FC1:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp Convert to CV_32F
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java Convert to CV_32F
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py Convert to CV_32F
@end_toggle

### Visualizing images

It is very useful to see intermediate results of your algorithm during development process. OpenCV
provides a convenient way of visualizing images. A 8U image can be shown using:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp imshow 1
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java imshow 1
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py imshow 1
@end_toggle

A call to waitKey() starts a message passing cycle that waits for a key stroke in the "image"
window. A 32F image needs to be converted to 8U type. For example:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/core/mat_operations/mat_operations.cpp imshow 2
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/core/mat_operations/MatOperations.java imshow 2
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/core/mat_operations/mat_operations.py imshow 2
@end_toggle

@note Here cv::namedWindow is not necessary since it is immediately followed by cv::imshow.
Nevertheless, it can be used to change the window properties or when using cv::createTrackbar
