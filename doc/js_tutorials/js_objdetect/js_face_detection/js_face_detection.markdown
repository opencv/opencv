Face Detection using Haar Cascades {#tutorial_js_face_detection}
==================================

Goal
----

-   learn the basics of face detection using Haar Feature-based Cascade Classifiers
-   extend the same for eye detection etc.

Basics
------

Object Detection using Haar feature-based cascade classifiers is an effective method proposed by Paul Viola and Michael Jones in the 2001 paper, "Rapid Object Detection using a
Boosted Cascade of Simple Features". It is a machine learning based approach in which a cascade
function is trained from a lot of positive and negative images. It is then used to detect objects in
other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images
(images of faces) and negative images (images without faces) to train the classifier. Then we need
to extract features from it. For this, Haar features shown in below image are used. They are just
like our convolutional kernel. Each feature is a single value obtained by subtracting the sum of pixels
under the white rectangle from the sum of pixels under the black rectangle.

![image](images/haar_features.jpg)

Now all possible sizes and locations of each kernel are used to calculate plenty of features. For each
feature calculation, we need to find the sum of the pixels under the white and black rectangles. To solve this,
they introduced the integral images. It simplifies calculation of the sum of the pixels, how large may be
the number of pixels, to an operation involving just four pixels.

But among all these features we calculated, most of them are irrelevant. For example, consider the
image below. Top row shows two good features. The first feature selected seems to focus on the
property that the region of the eyes is often darker than the region of the nose and cheeks. The
second feature selected relies on the property that the eyes are darker than the bridge of the nose.
But the same windows applying on cheeks or any other place is irrelevant. So how do we select the
best features out of 160000+ features? It is achieved by **Adaboost**.

![image](images/haar.png)

For this, we apply each and every feature on all the training images. For each feature, it finds the
best threshold which will classify the faces to positive and negative. But obviously, there will be
errors or misclassifications. We select the features with minimum error rate, which means they are
the features that best classifies the face and non-face images. (The process is not as simple as
this. Each image is given an equal weight in the beginning. After each classification, weights of
misclassified images are increased. Then again same process is done. New error rates are calculated.
Also new weights. The process is continued until required accuracy or error rate is achieved or
required number of features are found).

Final classifier is a weighted sum of these weak classifiers. It is called weak because it alone
can't classify the image, but together with others forms a strong classifier. The paper says even
200 features provide detection with 95% accuracy. Their final setup had around 6000 features.
(Imagine a reduction from 160000+ features to 6000 features. That is a big gain).

So now you take an image. Take each 24x24 window. Apply 6000 features to it. Check if it is face or
not. Wow.. Wow.. Isn't it a little inefficient and time consuming? Yes, it is. Authors have a good
solution for that.

In an image, most of the image region is non-face region. So it is a better idea to have a simple
method to check if a window is not a face region. If it is not, discard it in a single shot. Don't
process it again. Instead focus on region where there can be a face. This way, we can find more time
to check a possible face region.

For this they introduced the concept of **Cascade of Classifiers**. Instead of applying all the 6000
features on a window, group the features into different stages of classifiers and apply one-by-one.
(Normally first few stages will contain very less number of features). If a window fails the first
stage, discard it. We don't consider remaining features on it. If it passes, apply the second stage
of features and continue the process. The window which passes all stages is a face region. How is
the plan !!!

Authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in first five
stages. (Two features in the above image is actually obtained as the best two features from
Adaboost). According to authors, on an average, 10 features out of 6000+ are evaluated per
sub-window.

So this is a simple intuitive explanation of how Viola-Jones face detection works. Read paper for
more details.

Haar-cascade Detection in OpenCV
--------------------------------

Here we will deal with detection. OpenCV already contains many pre-trained classifiers for face,
eyes, smile etc. Those XML files are stored in opencv/data/haarcascades/ folder. Let's create a face
and eye detector with OpenCV.

We use the function: **detectMultiScale (image, objects, scaleFactor = 1.1, minNeighbors = 3, flags = 0, minSize = new cv.Size(0, 0), maxSize = new cv.Size(0, 0))** 

@param image               matrix of the type CV_8U containing an image where objects are detected. 
@param objects             vector of rectangles where each rectangle contains the detected object. The rectangles may be partially outside the original image.    
@param scaleFactor         parameter specifying how much the image size is reduced at each image scale.
@param minNeighbors        parameter specifying how many neighbors each candidate rectangle should have to retain it.
@param flags               parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
@param minSize             minimum possible object size. Objects smaller than this are ignored.
@param maxSize             maximum possible object size. Objects larger than this are ignored. If maxSize == minSize model is evaluated on single scale.

@note Don't forget to delete CascadeClassifier and RectVector!

Try it
------

Try this demo using the code above. Canvas elements named haarCascadeDetectionCanvasInput and haarCascadeDetectionCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
<head>
<style>
canvas {
    border: 1px solid black;
}
.err {
    color: red;
}
</style>
</head>
<body>
<div id="haarCascadeDetectionCodeArea">
<h2>Input your code</h2>
<button id="haarCascadeDetectionTryIt" disabled="true" onclick="haarCascadeDetectionExecuteCode()">Try it</button><br>
<textarea rows="17" cols="80" id="haarCascadeDetectionTestCode" spellcheck="false">
let src = cv.imread("haarCascadeDetectionCanvasInput");
let gray = new cv.Mat();
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
let faces = new cv.RectVector(), eyes = new cv.RectVector();
let face_cascade = new cv.CascadeClassifier(), eye_cascade = new cv.CascadeClassifier();
// load pre-trained classifiers
face_cascade.load("haarcascade_frontalface_default.xml");
eye_cascade.load("haarcascade_eye.xml");
// detect faces 
let msize = new cv.Size(0, 0);
face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);
for(let i = 0; i < faces.size(); ++i)
{
    let roiGray = gray.roi(faces.get(i));
    let roiSrc = src.roi(faces.get(i));
    let point1 = new cv.Point(faces.get(i).x, faces.get(i).y);
    let point2 = new cv.Point(faces.get(i).x + faces.get(i).width, faces.get(i).y + faces.get(i).height);
    cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
    // detect eyes in face ROI
    eye_cascade.detectMultiScale(roiGray, eyes);
    for (let j = 0; j < eyes.size(); ++j)
    {
        let point1 = new cv.Point(eyes.get(j).x, eyes.get(j).y);
        let point2 = new cv.Point(eyes.get(j).x + eyes.get(j).width, eyes.get(j).y + eyes.get(i).height);
        cv.rectangle(roiSrc, point1, point2, [0, 0, 255, 255]);
    }
    roiGray.delete(); roiSrc.delete();
}
cv.imshow("haarCascadeDetectionCanvasOutput", src);
src.delete(); gray.delete(); face_cascade.delete(); eye_cascade.delete(); faces.delete(); eyes.delete();
</textarea>
<p class="err" id="haarCascadeDetectionErr"></p>
</div>
<div id="haarCascadeDetectionShowcase">
    <div>
        <canvas id="haarCascadeDetectionCanvasInput"></canvas>
        <canvas id="haarCascadeDetectionCanvasOutput"></canvas>
    </div>
    <input type="file" id="haarCascadeDetectionInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function haarCascadeDetectionExecuteCode() {
    let haarCascadeDetectionText = document.getElementById("haarCascadeDetectionTestCode").value;
    try {
        eval(haarCascadeDetectionText);
        document.getElementById("haarCascadeDetectionErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("haarCascadeDetectionErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "haarCascadeDetectionCanvasInput");
let haarCascadeDetectionInputElement = document.getElementById("haarCascadeDetectionInput");
haarCascadeDetectionInputElement.addEventListener("change", haarCascadeDetectionHandleFiles, false);
function haarCascadeDetectionHandleFiles(e) {
    let haarCascadeDetectionUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(haarCascadeDetectionUrl, "haarCascadeDetectionCanvasInput");
}

function onReady() {
    document.getElementById("haarCascadeDetectionTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}

let Module = {
preRun: [function() {
	Module.FS_createPreloadedFile('/', 'haarcascade_eye.xml', 'haarcascade_eye.xml', true, false);
	Module.FS_createPreloadedFile('/', 'haarcascade_frontalface_default.xml', 'haarcascade_frontalface_default.xml', true, false);
	}],
};
</script>
</body>
\endhtmlonly