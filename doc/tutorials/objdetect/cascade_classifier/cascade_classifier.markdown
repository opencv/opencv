Cascade Classifier {#tutorial_cascade_classifier}
==================

Goal
----

In this tutorial,

-   We will learn how the Haar cascade object detection works.
-   We will see the basics of face detection and eye detection using the Haar Feature-based Cascade Classifiers
-   We will use the @ref cv::CascadeClassifier class to detect objects in a video stream. Particularly, we
    will use the functions:
    -   @ref cv::CascadeClassifier::load to load a .xml classifier file. It can be either a Haar or a LBP classifier
    -   @ref cv::CascadeClassifier::detectMultiScale to perform the detection.

Theory
------

Object Detection using Haar feature-based cascade classifiers is an effective object detection
method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a
Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade
function is trained from a lot of positive and negative images. It is then used to detect objects in
other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images
(images of faces) and negative images (images without faces) to train the classifier. Then we need
to extract features from it. For this, Haar features shown in the below image are used. They are just
like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels
under the white rectangle from sum of pixels under the black rectangle.

![image](images/haar_features.jpg)

Now, all possible sizes and locations of each kernel are used to calculate lots of features. (Just
imagine how much computation it needs? Even a 24x24 window results over 160000 features). For each
feature calculation, we need to find the sum of the pixels under white and black rectangles. To solve
this, they introduced the integral image. However large your image, it reduces the calculations for a
given pixel to an operation involving just four pixels. Nice, isn't it? It makes things super-fast.

But among all these features we calculated, most of them are irrelevant. For example, consider the
image below. The top row shows two good features. The first feature selected seems to focus on the
property that the region of the eyes is often darker than the region of the nose and cheeks. The
second feature selected relies on the property that the eyes are darker than the bridge of the nose.
But the same windows applied to cheeks or any other place is irrelevant. So how do we select the
best features out of 160000+ features? It is achieved by **Adaboost**.

![image](images/haar.png)

For this, we apply each and every feature on all the training images. For each feature, it finds the
best threshold which will classify the faces to positive and negative. Obviously, there will be
errors or misclassifications. We select the features with minimum error rate, which means they are
the features that most accurately classify the face and non-face images. (The process is not as simple as
this. Each image is given an equal weight in the beginning. After each classification, weights of
misclassified images are increased. Then the same process is done. New error rates are calculated.
Also new weights. The process is continued until the required accuracy or error rate is achieved or
the required number of features are found).

The final classifier is a weighted sum of these weak classifiers. It is called weak because it alone
can't classify the image, but together with others forms a strong classifier. The paper says even
200 features provide detection with 95% accuracy. Their final setup had around 6000 features.
(Imagine a reduction from 160000+ features to 6000 features. That is a big gain).

So now you take an image. Take each 24x24 window. Apply 6000 features to it. Check if it is face or
not. Wow.. Isn't it a little inefficient and time consuming? Yes, it is. The authors have a good
solution for that.

In an image, most of the image is non-face region. So it is a better idea to have a simple
method to check if a window is not a face region. If it is not, discard it in a single shot, and don't
process it again. Instead, focus on regions where there can be a face. This way, we spend more time
checking possible face regions.

For this they introduced the concept of **Cascade of Classifiers**. Instead of applying all 6000
features on a window, the features are grouped into different stages of classifiers and applied one-by-one.
(Normally the first few stages will contain very many fewer features). If a window fails the first
stage, discard it. We don't consider the remaining features on it. If it passes, apply the second stage
of features and continue the process. The window which passes all stages is a face region. How is
that plan!

The authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in the first five
stages. (The two features in the above image are actually obtained as the best two features from
Adaboost). According to the authors, on average 10 features out of 6000+ are evaluated per
sub-window.

So this is a simple intuitive explanation of how Viola-Jones face detection works. Read the paper for
more details or check out the references in the Additional Resources section.

Haar-cascade Detection in OpenCV
--------------------------------
OpenCV provides a training method (see @ref tutorial_traincascade) or pretrained models, that can be read using the @ref cv::CascadeClassifier::load method.
The pretrained models are located in the data folder in the OpenCV installation or can be found [here](https://github.com/opencv/opencv/tree/master/data).

The following code example will use pretrained Haar cascade models to detect faces and eyes in an image.
First, a @ref cv::CascadeClassifier is created and the necessary XML file is loaded using the @ref cv::CascadeClassifier::load method.
Afterwards, the detection is done using the @ref cv::CascadeClassifier::detectMultiScale method, which returns boundary rectangles for the detected faces or eyes.

@add_toggle_cpp
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/objectDetection/objectDetection.cpp)
@include samples/cpp/tutorial_code/objectDetection/objectDetection.cpp
@end_toggle

@add_toggle_java
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/java/tutorial_code/objectDetection/cascade_classifier/ObjectDetectionDemo.java)
@include samples/java/tutorial_code/objectDetection/cascade_classifier/ObjectDetectionDemo.java
@end_toggle

@add_toggle_python
This tutorial code's is shown lines below. You can also download it from
[here](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/objectDetection/cascade_classifier/objectDetection.py)
@include samples/python/tutorial_code/objectDetection/cascade_classifier/objectDetection.py
@end_toggle

Result
------

-#  Here is the result of running the code above and using as input the video stream of a built-in
    webcam:

    ![](images/Cascade_Classifier_Tutorial_Result_Haar.jpg)

    Be sure the program will find the path of files *haarcascade_frontalface_alt.xml* and
    *haarcascade_eye_tree_eyeglasses.xml*. They are located in
    *opencv/data/haarcascades*

-#  This is the result of using the file *lbpcascade_frontalface.xml* (LBP trained) for the face
    detection. For the eyes we keep using the file used in the tutorial.

    ![](images/Cascade_Classifier_Tutorial_Result_LBP.jpg)

Additional Resources
--------------------

-#  Paul Viola and Michael J. Jones. Robust real-time face detection. International Journal of Computer Vision, 57(2):137–154, 2004. @cite Viola04
-#  Rainer Lienhart and Jochen Maydt. An extended set of haar-like features for rapid object detection. In Image Processing. 2002. Proceedings. 2002 International Conference on, volume 1, pages I–900. IEEE, 2002. @cite Lienhart02
-#  Video Lecture on [Face Detection and Tracking](https://www.youtube.com/watch?v=WfdYYNamHZ8)
-#  An interesting interview regarding Face Detection by [Adam
    Harvey](https://web.archive.org/web/20171204220159/http://www.makematics.com/research/viola-jones/)
-#  [OpenCV Face Detection: Visualized](https://vimeo.com/12774628) on Vimeo by Adam Harvey
