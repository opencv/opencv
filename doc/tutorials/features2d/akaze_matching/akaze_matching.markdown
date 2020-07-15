AKAZE local features matching {#tutorial_akaze_matching}
=============================

@prev_tutorial{tutorial_detection_of_planar_objects}
@next_tutorial{tutorial_akaze_tracking}

Introduction
------------

In this tutorial we will learn how to use AKAZE @cite ANB13 local features to detect and match keypoints on
two images.
We will find keypoints on a pair of images with given homography matrix, match them and count the
number of inliers (i.e. matches that fit in the given homography).

You can find expanded version of this example here:
<https://github.com/pablofdezalc/test_kaze_akaze_opencv>

Data
----

We are going to use images 1 and 3 from *Graffiti* sequence of [Oxford dataset](http://www.robots.ox.ac.uk/~vgg/data/data-aff.html).

![](images/graf.png)

Homography is given by a 3 by 3 matrix:
@code{.none}
7.6285898e-01  -2.9922929e-01   2.2567123e+02
3.3443473e-01   1.0143901e+00  -7.6999973e+01
3.4663091e-04  -1.4364524e-05   1.0000000e+00
@endcode
You can find the images (*graf1.png*, *graf3.png*) and homography (*H1to3p.xml*) in
*opencv/samples/data/*.

### Source Code

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/features2D/AKAZE_match.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/features2D/AKAZE_match.cpp
@end_toggle

@add_toggle_java
-   **Downloadable code**: Click
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java)

-   **Code at glance:**
    @include samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py
@end_toggle

### Explanation

-   **Load images and homography**

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/features2D/AKAZE_match.cpp load
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java load
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py load
@end_toggle

We are loading grayscale images here. Homography is stored in the xml created with FileStorage.

-   **Detect keypoints and compute descriptors using AKAZE**

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/features2D/AKAZE_match.cpp AKAZE
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java AKAZE
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py AKAZE
@end_toggle

We create AKAZE and detect and compute AKAZE keypoints and descriptors. Since we don't need the *mask*
parameter, *noArray()* is used.

-   **Use brute-force matcher to find 2-nn matches**

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/features2D/AKAZE_match.cpp 2-nn matching
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java 2-nn matching
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py 2-nn matching
@end_toggle

We use Hamming distance, because AKAZE uses binary descriptor by default.

-   **Use 2-nn matches and ratio criterion to find correct keypoint matches**
@add_toggle_cpp
@snippet samples/cpp/tutorial_code/features2D/AKAZE_match.cpp ratio test filtering
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java ratio test filtering
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py ratio test filtering
@end_toggle

If the closest match distance is significantly lower than the second closest one, then the match is correct (match is not ambiguous).

-   **Check if our matches fit in the homography model**

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/features2D/AKAZE_match.cpp homography check
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java homography check
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py homography check
@end_toggle

If the distance from first keypoint's projection to the second keypoint is less than threshold,
then it fits the homography model.

We create a new set of matches for the inliers, because it is required by the drawing function.

-   **Output results**

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/features2D/AKAZE_match.cpp draw final matches
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/features2D/akaze_matching/AKAZEMatchDemo.java draw final matches
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/features2D/akaze_matching/AKAZE_match.py draw final matches
@end_toggle

Here we save the resulting image and print some statistics.

Results
-------

### Found matches

![](images/res.png)

Depending on your OpenCV version, you should get results coherent with:

@code{.none}
 Keypoints 1:   2943
 Keypoints 2:   3511
 Matches:       447
 Inliers:       308
 Inlier Ratio: 0.689038
@endcode
