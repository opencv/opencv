Face Detection using Haar Cascades {#tutorial_py_face_detection}
==================================

Goal
----

In this session,

-   We will see the basics of face detection using Haar Feature-based Cascade Classifiers
-   We will extend the same for eye detection etc.

Basics
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

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any
object like car, planes etc. you can use OpenCV to create one. Its full details are given here:
[Cascade Classifier Training](@ref tutorial_traincascade).

Here we will deal with detection. OpenCV already contains many pre-trained classifiers for face,
eyes, smiles, etc. Those XML files are stored in the opencv/data/haarcascades/ folder. Let's create a
face and eye detector with OpenCV.

First we need to load the required XML classifiers. Then load our input image (or video) in
grayscale mode.
@code{.py}
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('sachin.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
@endcode
Now we find the faces in the image. If faces are found, it returns the positions of detected faces
as Rect(x,y,w,h). Once we get these locations, we can create a ROI for the face and apply eye
detection on this ROI (since eyes are always on the face !!! ).
@code{.py}
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
@endcode
Result looks like below:

![image](images/face.jpg)

Additional Resources
--------------------

-#  Video Lecture on [Face Detection and Tracking](http://www.youtube.com/watch?v=WfdYYNamHZ8)
2.  An interesting interview regarding Face Detection by [Adam
    Harvey](http://www.makematics.com/research/viola-jones/)

Exercises
---------
