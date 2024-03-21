Introduction to Principal Component Analysis (PCA) {#tutorial_introduction_to_pca}
=======================================

@tableofcontents

@prev_tutorial{tutorial_optical_flow}

|    |    |
| -: | :- |
| Original author | Theodore Tsesmelis |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use the OpenCV class @ref cv::PCA to calculate the orientation of an object.

What is PCA?
--------------

Principal Component Analysis (PCA) is a statistical procedure that extracts the most important features of a dataset.

![](images/pca_line.png)

Consider that you have a set of 2D points as it is shown in the figure above. Each dimension corresponds to a feature you are interested in. Here some could argue that the points are set in a random order. However, if you have a better look you will see that there is a linear pattern (indicated by the blue line) which is hard to dismiss. A key point of PCA is the Dimensionality Reduction. Dimensionality Reduction is the process of reducing the number of the dimensions of the given dataset. For example, in the above case it is possible to approximate the set of points to a single line and therefore, reduce the dimensionality of the given points from 2D to 1D.

Moreover, you could also see that the points vary the most along the blue line, more than they vary along the Feature 1 or Feature 2 axes. This means that if you know the position of a point along the blue line you have more information about the point than if you only knew where it was on Feature 1 axis or Feature 2 axis.

Hence, PCA allows us to find the direction along which our data varies the most. In fact, the result of running PCA on the set of points in the diagram consist of 2 vectors called _eigenvectors_ which are the _principal components_ of the data set.

![](images/pca_eigen.png)

The size of each eigenvector is encoded in the corresponding eigenvalue and indicates how much the data vary along the principal component. The beginning of the eigenvectors is the center of all points in the data set. Applying PCA to N-dimensional data set yields N N-dimensional eigenvectors, N eigenvalues and 1 N-dimensional center point. Enough theory, let’s see how we can put these ideas into code.

How are the eigenvectors and eigenvalues computed?
--------------------------------------------------

The goal is to transform a given data set __X__ of dimension _p_ to an alternative data set __Y__ of smaller dimension _L_. Equivalently, we are seeking to find the matrix __Y__, where __Y__ is the _Karhunen–Loève transform_ (KLT) of matrix __X__:

\f[ \mathbf{Y} = \mathbb{K} \mathbb{L} \mathbb{T} \{\mathbf{X}\} \f]

__Organize the data set__

Suppose you have data comprising a set of observations of _p_ variables, and you want to reduce the data so that each observation can be described with only _L_ variables, _L_ < _p_. Suppose further, that the data are arranged as a set of _n_ data vectors \f$ x_1...x_n \f$ with each \f$ x_i \f$  representing a single grouped observation of the _p_ variables.

- Write \f$ x_1...x_n \f$ as row vectors, each of which has _p_ columns.
- Place the row vectors into a single matrix __X__ of dimensions \f$ n\times p \f$.

__Calculate the empirical mean__

- Find the empirical mean along each dimension \f$ j = 1, ..., p \f$.

- Place the calculated mean values into an empirical mean vector __u__ of dimensions \f$ p\times 1 \f$.

  \f[ \mathbf{u[j]} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{X[i,j]} \f]

__Calculate the deviations from the mean__

Mean subtraction is an integral part of the solution towards finding a principal component basis that minimizes the mean square error of approximating the data. Hence, we proceed by centering the data as follows:

- Subtract the empirical mean vector __u__ from each row of the data matrix __X__.

- Store mean-subtracted data in the \f$ n\times p \f$ matrix __B__.

  \f[ \mathbf{B} = \mathbf{X} - \mathbf{h}\mathbf{u^{T}} \f]

  where __h__ is an \f$ n\times 1 \f$ column vector of all 1s:

  \f[ h[i] = 1, i = 1, ..., n \f]

__Find the covariance matrix__

- Find the \f$ p\times p \f$ empirical covariance matrix __C__ from the outer product of matrix __B__ with itself:

  \f[ \mathbf{C} = \frac{1}{n-1} \mathbf{B^{*}} \cdot \mathbf{B} \f]

  where * is the conjugate transpose operator. Note that if B consists entirely of real numbers, which is the case in many applications, the "conjugate transpose" is the same as the regular transpose.

__Find the eigenvectors and eigenvalues of the covariance matrix__

- Compute the matrix __V__ of eigenvectors which diagonalizes the covariance matrix __C__:

  \f[ \mathbf{V^{-1}} \mathbf{C} \mathbf{V} = \mathbf{D} \f]

  where __D__ is the diagonal matrix of eigenvalues of __C__.

- Matrix __D__ will take the form of an \f$ p \times p \f$ diagonal matrix:

  \f[ D[k,l] = \left\{\begin{matrix} \lambda_k, k = l \\ 0, k \neq l \end{matrix}\right. \f]

  here, \f$ \lambda_j \f$ is the _j_-th eigenvalue of the covariance matrix __C__

- Matrix __V__, also of dimension _p_ x _p_, contains _p_ column vectors, each of length _p_, which represent the _p_ eigenvectors of the covariance matrix __C__.
- The eigenvalues and eigenvectors are ordered and paired. The _j_ th eigenvalue corresponds to the _j_ th eigenvector.

@note sources [[1]](https://robospace.wordpress.com/2013/10/09/object-orientation-principal-component-analysis-opencv/), [[2]](http://en.wikipedia.org/wiki/Principal_component_analysis) and special thanks to Svetlin Penkov for the original tutorial.

Source Code
-----------

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp
@end_toggle

@add_toggle_java
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java)

-   **Code at glance:**
    @include samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py
@end_toggle

@note Another example using PCA for dimensionality reduction while maintaining an amount of variance can be found at [opencv_source_code/samples/cpp/pca.cpp](https://github.com/opencv/opencv/tree/5.x/samples/cpp/pca.cpp)

Explanation
-----------

-   __Read image and convert it to binary__

Here we apply the necessary pre-processing procedures in order to be able to detect the objects of interest.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp pre-process
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java pre-process
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py pre-process
@end_toggle

-   __Extract objects of interest__

Then find and filter contours by size and obtain the orientation of the remaining ones.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp contours
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java contours
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py contours
@end_toggle

-   __Extract orientation__

Orientation is extracted by the call of getOrientation() function, which performs all the PCA procedure.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp pca
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java pca
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py pca
@end_toggle

First the data need to be arranged in a matrix with size n x 2, where n is the number of data points we have. Then we can perform that PCA analysis. The calculated mean (i.e. center of mass) is stored in the _cntr_ variable and the eigenvectors and eigenvalues are stored in the corresponding std::vector’s.

-   __Visualize result__

The final result is visualized through the drawAxis() function, where the principal components are drawn in lines, and each eigenvector is multiplied by its eigenvalue and translated to the mean position.

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp visualization
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java visualization
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py visualization
@end_toggle

@add_toggle_cpp
@snippet samples/cpp/tutorial_code/ml/introduction_to_pca/introduction_to_pca.cpp visualization1
@end_toggle

@add_toggle_java
@snippet samples/java/tutorial_code/ml/introduction_to_pca/IntroductionToPCADemo.java visualization1
@end_toggle

@add_toggle_python
@snippet samples/python/tutorial_code/ml/introduction_to_pca/introduction_to_pca.py visualization1
@end_toggle

Results
-------

The code opens an image, finds the orientation of the detected objects of interest and then visualizes the result by drawing the contours of the detected objects of interest, the center point, and the x-axis, y-axis regarding the extracted orientation.

![](images/pca_test1.jpg)

![](images/pca_output.png)
