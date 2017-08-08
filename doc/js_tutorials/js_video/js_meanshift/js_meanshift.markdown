Meanshift and Camshift {#tutorial_js_meanshift}
======================

Goal
----

-   We will learn about Meanshift and Camshift algorithms to find and track objects in videos.

Meanshift
---------

The intuition behind the meanshift is simple. Consider you have a set of points. (It can be a pixel
distribution like histogram backprojection). You are given a small window ( may be a circle) and you
have to move that window to the area of maximum pixel density (or maximum number of points). It is
illustrated in the simple image given below:

![image](images/meanshift_basics.jpg)

The initial window is shown in blue circle with the name "C1". Its original center is marked in blue
rectangle, named "C1_o". But if you find the centroid of the points inside that window, you will
get the point "C1_r" (marked in small blue circle) which is the real centroid of window. Surely
they don't match. So move your window such that circle of the new window matches with previous
centroid. Again find the new centroid. Most probably, it won't match. So move it again, and continue
the iterations such that center of window and its centroid falls on the same location (or with a
small desired error). So finally what you obtain is a window with maximum pixel distribution. It is
marked with green circle, named "C2". As you can see in image, it has maximum number of points. The
whole process is demonstrated on a static image below:

![image](images/meanshift_face.gif)

So we normally pass the histogram backprojected image and initial target location. When the object
moves, obviously the movement is reflected in histogram backprojected image. As a result, meanshift
algorithm moves our window to the new location with maximum density.

### Meanshift in OpenCV.js

To use meanshift in OpenCV.js, first we need to setup the target, find its histogram so that we can
backproject the target on each frame for calculation of meanshift. We also need to provide initial
location of window. For histogram, only Hue is considered here. Also, to avoid false values due to
low light, low light values are discarded using **cv.inRange()** function.

### Try it

Here is the demo for cv.meanShift. Some core code is in the textbox, and you can click `try it` to 
investigate more.

\htmlonly
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

<div id="CodeArea">
<h3>Input your code</h3>
<textarea rows="30" cols="90" id="msTestCode" spellcheck="false">
// Mats used in the loop are all declared and deleted elsewhere
// take first frame of the video
msFrame = new cv.Mat(msHeight, msWidth, cv.CV_8UC4);
let cap = new cv.VideoCapture("msVideo"); // "msVideo" is the id of the video tag
cap.read(msFrame);

// setup initial location of window
let trackWindow = new cv.Rect(300, 120, 125, 250); // simply hardcoded the values 300, 120, 125, 250

// set up the ROI for tracking
let msRoi = msFrame.getRoiRect(trackWindow);
let msHsvRoi = new cv.Mat();
cv.cvtColor(msRoi, msHsvRoi, cv.COLOR_RGBA2RGB);
cv.cvtColor(msHsvRoi, msHsvRoi, cv.COLOR_RGB2HSV);
let mask = new cv.Mat();
let lowScalar = new cv.Scalar(30, 30, 0);
let highScalar = new cv.Scalar(180, 180, 180);
let low = new cv.Mat(msHsvRoi.rows, msHsvRoi.cols, msHsvRoi.type(), lowScalar);
let high = new cv.Mat(msHsvRoi.rows, msHsvRoi.cols, msHsvRoi.type(), highScalar);
cv.inRange(msHsvRoi, low, high, mask);
msRoiHist = new cv.Mat();
let msHsvRoiVec = new cv.MatVector();
msHsvRoiVec.push_back(msHsvRoi);
cv.calcHist(msHsvRoiVec, [0], mask, msRoiHist, [180], [0,180]);
cv.normalize(msRoiHist, msRoiHist, 0, 255, cv.NORM_MINMAX);

// delete useless mats.
msRoi.delete(); msHsvRoi.delete(); mask.delete(); low.delete(); high.delete(); msHsvRoiVec.delete();

// Setup the termination criteria, either 10 iteration or move by atleast 1 pt
let termCrit = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1);

msHsv = new cv.Mat(msHeight, msWidth, cv.CV_8UC3);
msDst = new cv.Mat();
msHsvVec = new cv.MatVector();
msHsvVec.push_back(msHsv);
msLoopIndex = setInterval(
    function() {
        if(msVideo.ended) {
            msStopVideo();
            return;
        }       
        cap.read(msFrame);
        cv.cvtColor(msFrame, msHsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(msHsv, msHsv, cv.COLOR_RGB2HSV);
        cv.calcBackProject(msHsvVec, [0], msRoiHist, msDst, [0,180], 1);

        // Apply meanshift to get the new location
        // and it also returns number of iterations meanShift took to converge, 
        // which is useless in this demo.
        [ , trackWindow] = cv.meanShift(msDst, trackWindow, termCrit);

        // Draw it on image
        let [x,y,w,h] = [trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height];
        cv.rectangle(msFrame, new cv.Point(x, y), new cv.Point(x+w, y+h), [255, 0, 0, 255], 2);
        cv.imshow("msCanvasOutput", msFrame);
    }, 33);
</textarea>
<p class="err" id="msErr"></p>
</div> 
<div id="contentarea">
    <button id="msStartup" disabled="true" onclick="msStartup()">try it</button>
    <button id="msStop" disabled="true" onclick="msStopVideo()">stop</button><br>
    <video id="msVideo" src="cup.mp4" width="640" muted hidden>Your browser does not support the video tag.</video>
    <canvas id="msCanvasOutput"></canvas>
</div>
<script async src="opencv.js" id="opencvjs"></script>
<script>
// ms means Meanshift
// Some HTML elements we need to configure.
let msVideo = document.getElementById("msVideo");
let msStop = document.getElementById("msStop");

// In this case, We set width 640, and the height will be computed based on the input video.
let msWidth = msVideo.width;
let msHeight = null;
let msLoopIndex = null;
let msFrame = null;
let msDst = null;
let msHsvVec = null;
let msRoiHist = null;

msVideo.oncanplay = function() {
    msVideo.setAttribute("height", msVideo.videoHeight/msVideo.videoWidth*msVideo.width);
    msHeight = msVideo.height;
};

msVideo.onended = msStopVideo;

function msStartup() {
    if(msVideo.readyState !== 4)
        msVideo.load();
    msVideo.play();
    msStop.disabled = false;

    let msTestCode = document.getElementById("msTestCode").value;
    try {
        eval(msTestCode);
        document.getElementById("msErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("msErr").innerHTML = err;
    }
    document.getElementById("msStartup").disabled = true;
}

function msStopVideo() {
    clearInterval(msLoopIndex);
    if (msFrame != null && !msFrame.isDeleted()) {
        msFrame.delete();
        msFrame = null;
    }
    if (msDst != null && !msDst.isDeleted()) {
        msDst.delete();
        msDst = null;
    }
    if (msHsvVec != null && !msHsvVec.isDeleted()) {
        msHsvVec.delete();
        msHsvVec = null;
    }
    if (msRoiHist != null && !msRoiHist.isDeleted()) {
        msRoiHist.delete();
        msRoiHist = null;
    }
    if (msHsv != null && !msHsv.isDeleted()) {
        msHsv.delete();
        msHsv = null;
    }
    //document.getElementById("msCanvasOutput").getContext("2d").clearRect(0, 0, msWidth, msHeight);
    msVideo.pause();
    msVideo.currentTime = 0;
    document.getElementById("msStartup").disabled = false;
}
</script>
</body>
\endhtmlonly

Camshift
--------

Did you closely watch the last result? There is a problem. Our window always has the same size when
car is farther away and it is very close to camera. That is not good. We need to adapt the window
size with size and rotation of the target. Once again, the solution came from "OpenCV Labs" and it
is called CAMshift (Continuously Adaptive Meanshift) published by Gary Bradsky in his paper
"Computer Vision Face Tracking for Use in a Perceptual User Interface" in 1988.

It applies meanshift first. Once meanshift converges, it updates the size of the window as,
\f$s = 2 \times \sqrt{\frac{M_{00}}{256}}\f$. It also calculates the orientation of best fitting ellipse
to it. Again it applies the meanshift with new scaled search window and previous window location.
The process is continued until required accuracy is met.

![image](images/camshift_face.gif)

### Camshift in OpenCV.js

It is almost same as meanshift, but it returns a rotated rectangle (that is our result) and box
parameters (used to be passed as search window in next iteration). 


### Try it

Here is the demo for cv.CamShift. Some core code is in the textbox, and you can click `try it` to 
investigate more.

\htmlonly
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

<div id="CodeArea">
<h3>Input your code</h3>
<textarea rows="30" cols="90" id="csTestCode" spellcheck="false">
// Mats used in the loop are all declared and deleted elsewhere
// take first frame of the video
csFrame = new cv.Mat(csHeight, csWidth, cv.CV_8UC4);
let cap = new cv.VideoCapture("csVideo"); // "csVideo" is the id of the video tag
cap.read(csFrame);

// setup initial location of window
let trackWindow = new cv.Rect(300, 120, 125, 250); // simply hardcoded the values 300, 120, 125, 250

// set up the ROI for tracking
let csRoi = csFrame.getRoiRect(trackWindow);
let csHsvRoi = new cv.Mat();
cv.cvtColor(csRoi, csHsvRoi, cv.COLOR_RGBA2RGB);
cv.cvtColor(csHsvRoi, csHsvRoi, cv.COLOR_RGB2HSV);
let mask = new cv.Mat();
let lowScalar = new cv.Scalar(30, 30, 0);
let highScalar = new cv.Scalar(180, 180, 180);
let low = new cv.Mat(csHsvRoi.rows, csHsvRoi.cols, csHsvRoi.type(), lowScalar);
let high = new cv.Mat(csHsvRoi.rows, csHsvRoi.cols, csHsvRoi.type(), highScalar);
cv.inRange(csHsvRoi, low, high, mask);
csRoiHist = new cv.Mat();
let csHsvRoiVec = new cv.MatVector();
csHsvRoiVec.push_back(csHsvRoi);
cv.calcHist(csHsvRoiVec, [0], mask, csRoiHist, [180], [0,180]);
cv.normalize(csRoiHist, csRoiHist, 0, 255, cv.NORM_MINMAX);

// delete useless mats.
csRoi.delete(); csHsvRoi.delete(); mask.delete(); low.delete(); high.delete(); csHsvRoiVec.delete();

// Setup the termination criteria, either 10 iteration or move by atleast 1 pt
let termCrit = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1);

csHsv = new cv.Mat(csHeight, csWidth, cv.CV_8UC3);
csHsvVec = new cv.MatVector();
csHsvVec.push_back(csHsv);
csDst = new cv.Mat();
let trackBox = null;
csLoopIndex = setInterval(
    function() {
        if(csVideo.ended) {
            csStopVideo();
            return;
        }
        cap.read(csFrame);
        cv.cvtColor(csFrame, csHsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(csHsv, csHsv, cv.COLOR_RGB2HSV);
        cv.calcBackProject(csHsvVec, [0], csRoiHist, csDst, [0,180], 1);

        // apply camshift to get the new location
        [trackBox, trackWindow] = cv.CamShift(csDst, trackWindow, termCrit);

        // Draw it on image
        let pts = cv.rotatedRectPoints(trackBox);
        cv.line(csFrame, pts[0], pts[1], [255, 0, 0, 255], 3);
        cv.line(csFrame, pts[1], pts[2], [255, 0, 0, 255], 3);
        cv.line(csFrame, pts[2], pts[3], [255, 0, 0, 255], 3);
        cv.line(csFrame, pts[3], pts[0], [255, 0, 0, 255], 3);
        cv.imshow("csCanvasOutput", csFrame);
    }, 33);  
</textarea>
<p class="err" id="csErr"></p>
</div>
<div id="contentarea">
    <button id="csStartup" disabled="true" onclick="csStartup()">try it</button>
    <button id="csStop" disabled="true" onclick="csStopVideo()">stop</button><br>
    <video id="csVideo" src="cup.mp4" width="640" muted hidden>Your browser does not support the video tag.</video>
    <canvas id="csCanvasOutput"></canvas>
</div>
<script>
// cs means Camshift
// Some HTML elements we need to configure.
let csVideo = document.getElementById("csVideo");
let csStop = document.getElementById("csStop");

// In this case, We set width 640, and the height will be computed based on the input video.
let csWidth = csVideo.width;
let csHeight = null;
let csLoopIndex = null;
let csFrame = null;
let csDst = null;
let csHsvVec = null;
let csHsv = null;
let csRoiHist = null;

csVideo.oncanplay = function() {
    csVideo.setAttribute("height", csVideo.videoHeight/csVideo.videoWidth*csVideo.width);
    csHeight = csVideo.height;
};

csVideo.onended = csStopVideo;

function csStartup() {
    if(csVideo.readyState !== 4)
        csVideo.load();
    csVideo.play();
    csStop.disabled = false;
    let csTestCode = document.getElementById("csTestCode").value;
    try {
        eval(csTestCode);
        document.getElementById("csErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("csErr").innerHTML = err;
    }
    document.getElementById("csStartup").disabled = true;
}

function csStopVideo() {
    clearInterval(csLoopIndex);
    if (csFrame != null && !csFrame.isDeleted()) {
        csFrame.delete();
        csFrame = null;
    }
    if (csDst != null && !csDst.isDeleted()) {
        csDst.delete();
        csDst = null;
    }
    if (csHsvVec != null && !csHsvVec.isDeleted()) {
        csHsvVec.delete();
        csHsvVec = null;
    }
    if (csRoiHist != null && !csRoiHist.isDeleted()) {
        csRoiHist.delete();
        csRoiHist = null;
    }
    if (csHsv != null && !csHsv.isDeleted()) {
        csHsv.delete();
        csHsv = null;
    }
    //document.getElementById("csCanvasOutput").getContext("2d").clearRect(0, 0, csWidth, csHeight);
    csVideo.pause();
    csVideo.currentTime = 0;
    document.getElementById("csStartup").disabled = false;
}

function onReady() {
    document.getElementById("msStartup").disabled = false;
    document.getElementById("csStartup").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly

Additional Resources
--------------------

-#  French Wikipedia page on [Camshift](http://fr.wikipedia.org/wiki/Camshift). (The two animations
    are taken from here)
2.  Bradski, G.R., "Real time face and object tracking as a component of a perceptual user
    interface," Applications of Computer Vision, 1998. WACV '98. Proceedings., Fourth IEEE Workshop
    on , vol., no., pp.214,219, 19-21 Oct 1998
