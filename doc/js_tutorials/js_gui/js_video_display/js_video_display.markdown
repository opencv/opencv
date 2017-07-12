Getting Started with Videos {#tutorial_js_video_display}
===========================

Goal
----

-   Learn to capture from Camera and display it.

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
        cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
        cv.imshow("canvasOutput", dst); // canvasOutput is the id of another <canvas>;
    }, 33);
@endcode
In addition, remember delete src and dst after clearInterval(loopIndex).

Try it
------

Here is the demo for above code. We add three buttons here. Click `start webcam` to start your camera 
and dispaly it. Click `graying` to transfer the video to OpenCV-JavaScript and display the 
grayscale video. Here we set the video width as 320, and the height will be computed based on 
the input stream. Another tip is that the &lt;canvas&gt; used to draw video stream should be hidden.
Some core code is in the textbox, and you can change it to investigate more.

\htmlonly
<head>
<style>
.hiddenCanvas {
    display:none;
}
.contentarea {
    display:inline
}
</style>
</head>
<body>

<div id="CodeArea">
<h2>Input your code</h2>
<textarea rows="6" cols="70" id="TestCode" spellcheck="false">
context.drawImage(video, 0, 0, width, height);
src.data().set(context.getImageData(0, 0, width, height).data);
cv.cvtColor(src, dst, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
cv.imshow("canvasOutput", dst);
</textarea>
</div><br>

<div class="hiddenCanvas">
<canvas id="canvasFrame"></canvas>
</div>
<div id="contentarea">
    <button id="startup" onclick="startup()">start webcam</button>
    <button id="startDisplay" disabled="true" onclick="startDisplay()">graying</button> 
    <button id="stopDisplay" disabled="true" onclick="stopDisplay()">stop</button><br>
    <video id="video">Click startup to open webcam</video>
    <canvas id="canvasOutput"></canvas>
</div>
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

function startup() {
    video = document.getElementById("video");
    canvasFrame = document.getElementById("canvasFrame");

    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(stream) {
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
    }, false);
}

var loopIndex = null;
var src = null;
var dst = null;

function startDisplay() {
    if (!streaming) { console.warn("Please startup your webcam"); return; }
    var context = canvasFrame.getContext("2d");
    src = new cv.Mat(height, width, cv.CV_8UC4);
    dst = new cv.Mat(height, width, cv.CV_8UC1);
    loopIndex = setInterval(
        function() {
            var text = document.getElementById("TestCode").value;
            eval(text);
        }, 33);
    document.getElementById("stopDisplay").disabled = false;
    document.getElementById("startDisplay").disabled = true;
}

function stopDisplay() {
    document.getElementById("stopDisplay").disabled = true;
    document.getElementById("startDisplay").disabled = false;
    clearInterval(loopIndex);
    src.delete();
    dst.delete();
}

document.getElementById("opencvjs").onload = function() {
    document.getElementById("startDisplay").disabled = false;
};
</script>
</body>
\endhtmlonly