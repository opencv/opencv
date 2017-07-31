Getting Started with Videos {#tutorial_js_video_display}
===========================

Goal
----

-   Learn to capture video from Camera and display it.

Capture video from camera
-------------------------

Often, we have to capture live stream with camera. In OpenCV-JavaScript, we use [WebRTC](https://webrtc.org/) 
and HTML canvas element to implement this.
Let's capture a video from the camera (either an in-built camera of your laptop or a usb camera 
is ok), convert it into grayscale video and display it. Just a simple task to get started.

To capture a video, you need to add some HTML elements in the web.
- a &lt;video&gt; to display video from camera directly
- a &lt;canvas&gt; to transfer video to canvas ImageData frame-by-frame
- another &lt;canvas&gt; to display the video OpenCV-JavaScript gets

Fisrt, we use WebRTC navigator.mediaDevices.getUserMedia to get the media stream.
@code{.js}
var video = document.getElementById("video"); // video is the id of <video>
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occured! " + err);
});
@endcode

@note This function is needless when you just capture video from a video file. But notice that 
<video> tag only supports video formats of Ogg(Theora), WebM(VP8/VP9) or MP4(H.264).

Playing video
-------------
Now, the browser gets the camera stream. Then we use CanvasRenderingContext2D.drawImage() method 
of the Canvas 2D API to draw video onto the canvas. Finally, we can use the method in @ref tutorial_js_image_display
 to read and display image in canvas. For playing video, cv.imshow() should be executed every delay 
milliseconds. We recommend setInterval() method. And if the video is 30fps, the delay milliseconds 
should be 33.
@code{.js}
var canvasFrame = document.getElementById("canvasFrame"); // canvasFrame is the id of <canvas>
var context = canvasFrame.getContext("2d");
var src = new cv.Mat(height, width, cv.CV_8UC4);
var dst = new cv.Mat(height, width, cv.CV_8UC1);
var loopIndex = setInterval(
    function() {
        context.drawImage(video, 0, 0, width, height);
        src.data().set(context.getImageData(0, 0, width, height).data);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY, 0);
        cv.imshow("canvasOutput", dst); // canvasOutput is the id of another <canvas>;
    }, 33);
@endcode
In addition, remember delete src and dst after clearInterval(loopIndex).

Try it
------

Here is the demo for above code. Click `start` to start your camera and paly it. The left video is from 
your camera directly, and the right one is from OpenCV-JavaScript. Click `processing` to gray the video. 
Here we set the video width as 320, and the height will be computed based on the input stream. Another 
tip is that the &lt;canvas&gt; used to draw video stream should be hidden. Some core code is in the 
textbox, and you can change it to investigate more.

\htmlonly
<head>
<style>
canvas {
    border: 1px solid black;
}
video {
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
<textarea rows="8" cols="70" id="TestCode" spellcheck="false">
// src is cv.CV_8UC4 and dst is cv.CV_8UC1
context.drawImage(video, 0, 0, width, height);
src.data().set(context.getImageData(0, 0, width, height).data);
cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY, 0);
cv.imshow("canvasOutput", dst);
</textarea>
<p class="err" id="vdErr"></p>
</div>
<canvas id="canvasFrame" hidden></canvas>

<div id="contentarea">
    <button id="startup" disabled="true" onclick="startup()">start</button>
    <input type="checkbox" id="checkbox" disabled="true" onchange="checkboxChange()">processing</input>
    <button id="stop" disabled="true" onclick="stopCamera()">stop</button><br>
    <video id="video">Your browser does not support the video tag.</video>
    <canvas id="canvasOutput"></canvas>
</div>
<script src="adapter.js"></script>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
// In this case, We set width 320, and the height will be computed based on the input stream.
var width = 320;
var height = 0;

// whether streaming video from the camera.
var streaming = false;

// Some HTML elements we need to configure.
var video = null;
var canvasFrame = null;
var checkbox = null;
var stop = null;
var stream = null;

function startup() {
    video = document.getElementById("video");
    canvasFrame = document.getElementById("canvasFrame");
    checkbox = document.getElementById("checkbox");
    stop = document.getElementById("stop");

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(s) {
            stream = s;
            video.srcObject = stream;
            video.play();
        })
        .catch(function(err) {
            console.log("An error occured! " + err);
    });

    video.addEventListener("canplay", function(ev){
        if (!streaming) {
            height = video.videoHeight / (video.videoWidth/width);
            video.setAttribute("width", width);
            video.setAttribute("height", height);
            canvasFrame.setAttribute("width", width);
            canvasFrame.setAttribute("height", height);
            streaming = true;
        }
        checkbox.disabled = false;
        stop.disabled = false;
        checkboxChange();
    }, false);
}

var loopIndex = null;
var src = null;
var dst = null;

function checkboxChange() {
    if (checkbox.checked) playProcessedVideo();
    else playVideo();
}

function playVideo() {
    if (!streaming) { console.warn("Please startup your webcam"); return; }
    stopLastVideo();
    var context = canvasFrame.getContext("2d");
    src = new cv.Mat(height, width, cv.CV_8UC4);
    loopIndex = setInterval(
        function() {
            context.drawImage(video, 0, 0, width, height);
            src.data().set(context.getImageData(0, 0, width, height).data);
            cv.imshow("canvasOutput", src);
        }, 33);
}

function playProcessedVideo() {
    if (!streaming) { console.warn("Please startup your webcam"); return; }
    stopLastVideo();
    var context = canvasFrame.getContext("2d");
    src = new cv.Mat(height, width, cv.CV_8UC4);
    dst = new cv.Mat(height, width, cv.CV_8UC1);
    loopIndex = setInterval(
        function() {
            var text = document.getElementById("TestCode").value;
            try {
                eval(text);
                document.getElementById("vdErr").innerHTML = " ";
            } catch(err) {
                document.getElementById("vdErr").innerHTML = err;
            }
        }, 33);
}

function stopLastVideo() {
    clearInterval(loopIndex);
    if (src != null && !src.isDeleted()) src.delete();
    if (dst != null && !dst.isDeleted()) dst.delete();
}

function stopCamera() {
    stopLastVideo();
    document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, width, height);
    checkbox.disabled = true;
    video.pause();
    video.srcObject=null;
    stream.getVideoTracks()[0].stop();
}

function onReady() {
    document.getElementById("startup").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly