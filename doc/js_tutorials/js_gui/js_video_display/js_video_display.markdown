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
let video = document.getElementById("video"); // video is the id of <video>
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
let canvasFrame = document.getElementById("canvasFrame"); // canvasFrame is the id of <canvas>
let context = canvasFrame.getContext("2d");
let src = new cv.Mat(height, width, cv.CV_8UC4);
let dst = new cv.Mat(height, width, cv.CV_8UC1);
let loopIndex = setInterval(
    function() {
        context.drawImage(video, 0, 0, width, height);
        src.data().set(context.getImageData(0, 0, width, height).data);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        cv.imshow("canvasOutput", dst); // canvasOutput is the id of another <canvas>;
    }, 33);
@endcode

In addition, OpenCV-JavaScript implements cv.VideoCapture using the above method. You need not to  
add the hidden canvas tag manually. The above code of playing video could be simplified as below.
@code{.js}
let src = new cv.Mat(height, width, cv.CV_8UC4);
let dst = new cv.Mat(height, width, cv.CV_8UC1);

// "video" is the id of the video tag
let cap = new cv.VideoCapture("video");
let loopIndex = setInterval(
    function() {
        // the read() function need a mat with cv.CV_8UC4 type and size same as the video
        cap.read(src);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        cv.imshow("canvasOutput", dst);
    }, 33);
@endcode

@note Remember delete src and dst after clearInterval(loopIndex).

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
<textarea rows="16" cols="70" id="TestCode" spellcheck="false">
// src and dst are declared and deleted elsewhere
src = new cv.Mat(height, width, cv.CV_8UC4);
dst = new cv.Mat(height, width, cv.CV_8UC1);

// "video" is the id of the video tag
let cap = new cv.VideoCapture("video");
loopIndex = setInterval(
    function(){
        cap.read(src);
        if (checkbox.checked) {
            cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
            cv.imshow("canvasOutput", dst);
        }
        else
            cv.imshow("canvasOutput", src);
    }, 33);
</textarea>
<p class="err" id="vdErr"></p>
</div>
<div id="contentarea">
    <button id="startup" disabled="true" onclick="startup()">start</button>
    <input type="checkbox" id="checkbox" disabled="true"">processing</input>
    <button id="stop" disabled="true" onclick="stopCamera()">stop</button><br>
    <video id="video">Your browser does not support the video tag.</video>
    <canvas id="canvasOutput"></canvas>
</div>
<script src="adapter.js"></script>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
// In this case, We set width 320, and the height will be computed based on the input stream.
let width = 320;
let height = 0;

// whether streaming video from the camera.
let streaming = false;

// Some HTML elements we need to configure.
let video = null;
let checkbox = null;
let start = null;
let stop = null;
let stream = null;

let loopIndex = 0;
let src = null;
let dst = null;

function initVideo(ev){
    if (!streaming) {
        height = video.videoHeight / (video.videoWidth/width);
        video.setAttribute("width", width);
        video.setAttribute("height", height);
        streaming = true;
    }
    checkbox.disabled = false;
    stop.disabled = false;
    playVideo();
}

function startup() {
    video = document.getElementById("video");
    checkbox = document.getElementById("checkbox");
    start = document.getElementById("startup");
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

    video.addEventListener("canplay", initVideo, false);
}

function playVideo() {
    if (!streaming) {
        console.warn("Please startup your webcam");
        return;
    }
    let text = document.getElementById("TestCode").value;
    try {
        eval(text);
        document.getElementById("vdErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("vdErr").innerHTML = err;
    }
    start.disabled = true;
}

function stopCamera() {
    clearInterval(loopIndex);
    if (src != null && !src.isDeleted()) {
        src.delete();
        src = null;
    }
    if (dst != null && !dst.isDeleted()) {
        dst.delete();
        dst = null;
    }
    document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, width, height);
    video.pause();
    video.srcObject = null;
    stream.getVideoTracks()[0].stop();
    start.disabled = false;
    video.removeEventListener("canplay", initVideo);
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