Image Segmentation with Distance Transform and Watershed Algorithm {#tutorial_distance_transform}
=============

@prev_tutorial{tutorial_point_polygon_test}
@next_tutorial{tutorial_out_of_focus_deblur_filter}

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV function @ref cv::filter2D in order to perform some laplacian filtering for image sharpening
-   Use the OpenCV function @ref cv::distanceTransform in order to obtain the derived representation of a binary image, where the value of each pixel is replaced by its distance to the nearest background pixel
-   Use the OpenCV function @ref cv::watershed in order to isolate objects in the image from the background

Theory
------

Code
----

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp).
@include samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java)
@include samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py)
@include samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py
@end_toggle

Explanation / Result
--------------------

-   Load the source image and check if it is loaded without any problem, then show it:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp load_image
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java load_image
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py load_image
@end_toggle

![](images/source.jpeg)

-   Then if we have an image with a white background, it is good to transform it to black. This will help us to discriminate the foreground objects easier when we will apply the Distance Transform:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp black_bg
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java black_bg
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py black_bg
@end_toggle

![](images/black_bg.jpeg)

-   Afterwards we will sharpen our image in order to acute the edges of the foreground objects. We will apply a laplacian filter with a quite strong filter (an approximation of second derivative):

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp sharp
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java sharp
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py sharp
@end_toggle

![](images/laplace.jpeg)
![](images/sharp.jpeg)

-   Now we transform our new sharpened source image to a grayscale and a binary one, respectively:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp bin
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java bin
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py bin
@end_toggle

![](images/bin.jpeg)

-   We are ready now to apply the Distance Transform on the binary image. Moreover, we normalize the output image in order to be able visualize and threshold the result:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp dist
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java dist
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py dist
@end_toggle

![](images/dist_transf.jpeg)

-   We threshold the *dist* image and then perform some morphology operation (i.e. dilation) in order to extract the peaks from the above image:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp peaks
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java peaks
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py peaks
@end_toggle

![](images/peaks.jpeg)

-   From each blob then we create a seed/marker for the watershed algorithm with the help of the @ref cv::findContours function:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp seeds
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java seeds
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py seeds
@end_toggle

![](images/markers.jpeg)

-   Finally, we can apply the watershed algorithm, and visualize the result:

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ImgTrans/imageSegmentation.cpp watershed
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ImgTrans/distance_transformation/ImageSegmentationDemo.java watershed
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ImgTrans/distance_transformation/imageSegmentation.py watershed
@end_toggle

![](images/final.jpeg)
