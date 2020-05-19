Optical Flow {#tutorial_optical_flow}
============

@prev_tutorial{tutorial_meanshift}

Goal
----

In this chapter,
    -   We will understand the concepts of optical flow and its estimation using Lucas-Kanade
        method.
    -   We will use functions like **cv.calcOpticalFlowPyrLK()** to track feature points in a
        video.
    -   We will create a dense optical flow field using the **cv.calcOpticalFlowFarneback()** method.

Optical Flow
------------

Optical flow is the pattern of apparent motion of image objects between two consecutive frames
caused by the movement of object or camera. It is 2D vector field where each vector is a
displacement vector showing the movement of points from first frame to second. Consider the image
below (Image Courtesy: [Wikipedia article on Optical Flow](http://en.wikipedia.org/wiki/Optical_flow)).

![image](images/optical_flow_basic1.jpg)

It shows a ball moving in 5 consecutive frames. The arrow shows its displacement vector. Optical
flow has many applications in areas like :

-   Structure from Motion
-   Video Compression
-   Video Stabilization ...

Optical flow works on several assumptions:

-#  The pixel intensities of an object do not change between consecutive frames.
2.  Neighbouring pixels have similar motion.

Consider a pixel \f$I(x,y,t)\f$ in first frame (Check a new dimension, time, is added here. Earlier we
were working with images only, so no need of time). It moves by distance \f$(dx,dy)\f$ in next frame
taken after \f$dt\f$ time. So since those pixels are the same and intensity does not change, we can say,

\f[I(x,y,t) = I(x+dx, y+dy, t+dt)\f]

Then take taylor series approximation of right-hand side, remove common terms and divide by \f$dt\f$ to
get the following equation:

\f[f_x u + f_y v + f_t = 0 \;\f]

where:

\f[f_x = \frac{\partial f}{\partial x} \; ; \; f_y = \frac{\partial f}{\partial y}\f]\f[u = \frac{dx}{dt} \; ; \; v = \frac{dy}{dt}\f]

Above equation is called Optical Flow equation. In it, we can find \f$f_x\f$ and \f$f_y\f$, they are image
gradients. Similarly \f$f_t\f$ is the gradient along time. But \f$(u,v)\f$ is unknown. We cannot solve this
one equation with two unknown variables. So several methods are provided to solve this problem and
one of them is Lucas-Kanade.

### Lucas-Kanade method

We have seen an assumption before, that all the neighbouring pixels will have similar motion.
Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion. We
can find \f$(f_x, f_y, f_t)\f$ for these 9 points. So now our problem becomes solving 9 equations with
two unknown variables which is over-determined. A better solution is obtained with least square fit
method. Below is the final solution which is two equation-two unknown problem and solve to get the
solution.

\f[\begin{bmatrix} u \\ v \end{bmatrix} =
\begin{bmatrix}
    \sum_{i}{f_{x_i}}^2  &  \sum_{i}{f_{x_i} f_{y_i} } \\
    \sum_{i}{f_{x_i} f_{y_i}} & \sum_{i}{f_{y_i}}^2
\end{bmatrix}^{-1}
\begin{bmatrix}
    - \sum_{i}{f_{x_i} f_{t_i}} \\
    - \sum_{i}{f_{y_i} f_{t_i}}
\end{bmatrix}\f]

( Check similarity of inverse matrix with Harris corner detector. It denotes that corners are better
points to be tracked.)

So from the user point of view, the idea is simple, we give some points to track, we receive the optical
flow vectors of those points. But again there are some problems. Until now, we were dealing with
small motions, so it fails when there is a large motion. To deal with this we use pyramids. When we go up in
the pyramid, small motions are removed and large motions become small motions. So by applying
Lucas-Kanade there, we get optical flow along with the scale.

Lucas-Kanade Optical Flow in OpenCV
-----------------------------------

OpenCV provides all these in a single function, **cv.calcOpticalFlowPyrLK()**. Here, we create a
simple application which tracks some points in a video. To decide the points, we use
**cv.goodFeaturesToTrack()**. We take the first frame, detect some Shi-Tomasi corner points in it,
then we iteratively track those points using Lucas-Kanade optical flow. For the function
**cv.calcOpticalFlowPyrLK()** we pass the previous frame, previous points and next frame. It
returns next points along with some status numbers which has a value of 1 if next point is found,
else zero. We iteratively pass these next points as previous points in next step. See the code
below:

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/3.4/samples/cpp/tutorial_code/video/optical_flow/optical_flow.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/video/optical_flow/optical_flow.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/3.4/samples/python/tutorial_code/video/optical_flow/optical_flow.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/video/optical_flow/optical_flow.py
@end_toggle


@add_toggle_java
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/3.4/samples/java/tutorial_code/video/optical_flow/OpticalFlowDemo.java)

-   **Code at glance:**
    @include samples/java/tutorial_code/video/optical_flow/OpticalFlowDemo.java
@end_toggle

(This code doesn't check how correct are the next keypoints. So even if any feature point disappears
in image, there is a chance that optical flow finds the next point which may look close to it. So
actually for a robust tracking, corner points should be detected in particular intervals. OpenCV
samples comes up with such a sample which finds the feature points at every 5 frames. It also run a
backward-check of the optical flow points got to select only good ones. Check
samples/python/lk_track.py).

See the results we got:

![image](images/opticalflow_lk.jpg)

Dense Optical Flow in OpenCV
----------------------------

Lucas-Kanade method computes optical flow for a sparse feature set (in our example, corners detected
using Shi-Tomasi algorithm). OpenCV provides another algorithm to find the dense optical flow. It
computes the optical flow for all the points in the frame. It is based on Gunner Farneback's
algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by
Gunner Farneback in 2003.

Below sample shows how to find the dense optical flow using above algorithm. We get a 2-channel
array with optical flow vectors, \f$(u,v)\f$. We find their magnitude and direction. We color code the
result for better visualization. Direction corresponds to Hue value of the image. Magnitude
corresponds to Value plane. See the code below:

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/3.4/samples/cpp/tutorial_code/video/optical_flow/optical_flow_dense.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/video/optical_flow/optical_flow_dense.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/3.4/samples/python/tutorial_code/video/optical_flow/optical_flow_dense.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/video/optical_flow/optical_flow_dense.py
@end_toggle


@add_toggle_java
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/3.4/samples/java/tutorial_code/video/optical_flow/OpticalFlowDenseDemo.java)

-   **Code at glance:**
    @include samples/java/tutorial_code/video/optical_flow/OpticalFlowDenseDemo.java
@end_toggle


See the result below:

![image](images/opticalfb.jpg)
