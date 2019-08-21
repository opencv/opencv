OCR of Hand-written Data using SVM {#tutorial_py_svm_opencv}
==================================

Goal
----

In this chapter

-   We will revisit the hand-written data OCR, but, with SVM instead of kNN.

OCR of Hand-written Digits
--------------------------

In kNN, we directly used pixel intensity as the feature vector. This time we will use [Histogram of
Oriented Gradients](http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (HOG) as feature
vectors.

Here, before finding the HOG, we deskew the image using its second order moments. So we first define
a function **deskew()** which takes a digit image and deskew it. Below is the deskew() function:

@snippet samples/python/tutorial_code/ml/py_svm_opencv/hogsvm.py deskew

Below image shows above deskew function applied to an image of zero. Left image is the original
image and right image is the deskewed image.

![image](images/deskew.jpg)

Next we have to find the HOG Descriptor of each cell. For that, we find Sobel derivatives of each
cell in X and Y direction. Then find their magnitude and direction of gradient at each pixel. This
gradient is quantized to 16 integer values. Divide this image to four sub-squares. For each
sub-square, calculate the histogram of direction (16 bins) weighted with their magnitude. So each
sub-square gives you a vector containing 16 values. Four such vectors (of four sub-squares) together
gives us a feature vector containing 64 values. This is the feature vector we use to train our data.

@snippet samples/python/tutorial_code/ml/py_svm_opencv/hogsvm.py hog

Finally, as in the previous case, we start by splitting our big dataset into individual cells. For
every digit, 250 cells are reserved for training data and remaining 250 data is reserved for
testing. Full code is given below, you also can download it from [here](https://github.com/opencv/opencv/tree/3.4/samples/python/tutorial_code/ml/py_svm_opencv/hogsvm.py):

@include samples/python/tutorial_code/ml/py_svm_opencv/hogsvm.py

This particular technique gave me nearly 94% accuracy. You can try different values for various
parameters of SVM to check if higher accuracy is possible. Or you can read technical papers on this
area and try to implement them.

Additional Resources
--------------------

-#  [Histograms of Oriented Gradients Video](https://www.youtube.com/watch?v=0Zib1YEE4LU)

Exercises
---------

-#  OpenCV samples contain digits.py which applies a slight improvement of the above method to get
    improved result. It also contains the reference. Check it and understand it.
