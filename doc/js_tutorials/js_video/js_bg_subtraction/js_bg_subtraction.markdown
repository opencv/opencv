Background Subtraction {#tutorial_js_bg_subtraction}
======================

Goal
----

In this chapter,

-   We will familiarize with the background subtraction methods available in OpenCV.

Basics
------

Background subtraction is a major preprocessing steps in many vision based applications. For
example, consider the cases like visitor counter where a static camera takes the number of visitors
entering or leaving the room, or a traffic camera extracting information about the vehicles etc. In
all these cases, first you need to extract the person or vehicles alone. Technically, you need to
extract the moving foreground from static background.

If you have an image of background alone, like image of the room without visitors, image of the road
without vehicles etc, it is an easy job. Just subtract the new image from the background. You get
the foreground objects alone. But in most of the cases, you may not have such an image, so we need
to extract the background from whatever images we have. It become more complicated when there is
shadow of the vehicles. Since shadow is also moving, simple subtraction will mark that also as
foreground. It complicates things.

OpenCV-JavaScript has implemented one algorithm for this purpose, which is very easy to use.

BackgroundSubtractorMOG2
------------------------

It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. It is based on two
papers by Z.Zivkovic, "Improved adaptive Gausian mixture model for background subtraction" in 2004
and "Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction"
in 2006. One important feature of this algorithm is that it selects the appropriate number of
gaussian distribution for each pixel. It provides better adaptibility to varying scenes due illumination
changes etc.

While coding, we need to create a background object of cv.BackgroundSubtractorMOG2. It has some optional 
parameters like length of history, number of gaussian mixtures, threshold etc. It is all set to some 
default values. Then inside the video loop, use apply() method to get the 
foreground mask. And, you have an option of selecting whether shadow to be detected or not. If 
detectShadows = True (which is so by default), it detects and marks shadows, but decreases the speed. 
Shadows will be marked in gray color.

Try it
------

Here is the demo for cv.BackgroundSubtractorMOG2. Some core code is in the textbox, and you can click 
`try it` to investigate more.

\htmlonly
<head>
<style>
video {
    border: 1px solid black;
}
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
<textarea rows="15" cols="90" id="bgsTestCode" spellcheck="false">
// frame and fgmask are declared and deleted elsewhere
frame = new cv.Mat(bgsHeight, bgsWidth, cv.CV_8UC4);
fgmask = new cv.Mat(bgsHeight, bgsWidth, cv.CV_8UC1);
let fgbg = new cv.BackgroundSubtractorMOG2(500, 16, true);

bgsLoopIndex = setInterval(
    function() {
        if(bgsVideo.ended) bgsStopVideo();
        context.drawImage(bgsVideo, 0, 0, bgsWidth, bgsHeight);
        frame.data().set(context.getImageData(0, 0, bgsWidth, bgsHeight).data);

        fgbg.apply(frame, fgmask);

        cv.imshow("bgsCanvasOutput", fgmask);

    }, 33);  
</textarea>
<p class="err" id="bgsErr"></p>
</div> 
<canvas id="bgsCanvasFrame" hidden></canvas>
<div id="contentarea">
    <button id="bgsStartup" disabled="true" onclick="bgsStartup()">try it</button>
    <button id="bgsStop" disabled="true" onclick="bgsStopVideo()">stop</button><br>
    <video id="bgsVideo" src="box.mp4" width="320">Your browser does not support the video tag.</video>
    <canvas id="bgsCanvasOutput"></canvas>
</div>
<script src="adapter.js"></script>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
// bgs means BackgroundSubtractorMOG2
// Some HTML elements we need to configure.
let bgsVideo = document.getElementById("bgsVideo");
let bgsCanvasFrame = document.getElementById("bgsCanvasFrame");
let bgsStop = document.getElementById("bgsStop");

// In this case, We set width 320, and the height will be computed based on the input video.
let bgsWidth = bgsVideo.width;
let bgsHeight = null;
let bgsLoopIndex = null;
let frame = null;
let fgmask = null;

bgsVideo.oncanplay = function() {
    bgsVideo.setAttribute("height", bgsVideo.videoHeight/bgsVideo.videoWidth*bgsVideo.width);
    bgsHeight = bgsVideo.height;
    bgsCanvasFrame.setAttribute("width", bgsWidth);
    bgsCanvasFrame.setAttribute("height", bgsHeight);
};

bgsVideo.onended = bgsStopVideo;

function bgsStartup() {
    if(bgsVideo.readyState !== 4)
        bgsVideo.load();
    bgsVideo.play();
    bgsStop.disabled = false;

    let context = bgsCanvasFrame.getContext("2d");
    let bgsTestCode = document.getElementById("bgsTestCode").value;
    try {
        eval(bgsTestCode);
        document.getElementById("bgsErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("bgsErr").innerHTML = err;
    }
    document.getElementById("bgsStartup").disabled = true;
}

function bgsStopVideo() {
    clearInterval(bgsLoopIndex);
    if (frame != null && !frame.isDeleted()) {
        frame.delete();
        frame = null;
    }
    if (fgmask != null && !fgmask.isDeleted()) {
        fgmask.delete();
        fgmask = null;
    }
    //document.getElementById("bgsCanvasOutput").getContext("2d").clearRect(0, 0, bgsWidth, bgsHeight);
    bgsVideo.pause();
    bgsVideo.currentTime = 0;
    document.getElementById("bgsStartup").disabled = false;
}

function onReady() {
    document.getElementById("bgsStartup").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
