Face Detection in Video Capture {#tutorial_js_face_detection_camera}
==================================

Goal
----

-   learn how to detect faces in video capture. 

@note  If you don't know how to capture video from camera, please review @ref tutorial_js_video_display.

\htmlonly
<!DOCTYPE html>
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
let src = new cv.Mat(height, width, cv.CV_8UC4);
let dst = new cv.Mat(height, width, cv.CV_8UC4);
let gray = new cv.Mat();
let faceFrontal = new cv.RectVector();
let faceFrontalCascadeFrontal = new cv.CascadeClassifier(); 

// load pre-trained classifiers
faceFrontalCascadeFrontal.load("haarcascade_frontalface_default.xml");

// "video" is the id of the video tag
let cap = new cv.VideoCapture("video");
loopIndex = setInterval(
    function(){
        if (checkbox.checked) {
            cap.read(src);
            src.copyTo(dst);
            cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
            // detect faceFrontal 
            faceFrontalCascadeFrontal.detectMultiScale(gray, faceFrontal, 1.1, 3, 0, {width : 0, height : 0}, {width : 0, height : 0});
            for (let i = 0; i < faceFrontal.size(); ++i) {
                let point1 = new cv.Point(faceFrontal.get(i).x, faceFrontal.get(i).y);
                let point2 = new cv.Point(faceFrontal.get(i).x + faceFrontal.get(i).width, faceFrontal.get(i).y + faceFrontal.get(i).height);
                cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
            }
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
let gray = null;
let faceFrontal = null;
let faceFrontalCascadeFrontal = null; 

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
    if (src != null && !src.isDeleted()) {
        src.delete();
        src = null;
    }
    if (dst != null && !dst.isDeleted()) {
        dst.delete();
        dst = null;
    }
    if (faceFrontalCascadeFrontal != null && !faceFrontalCascadeFrontal.isDeleted()) {
        faceFrontalCascadeFrontal.delete();
        faceFrontalCascadeFrontal = null;
    }
    if (faceFrontal != null && !faceFrontal.isDeleted()) {
        faceFrontal.delete();
        faceFrontal = null;
    }
    if (gray != null && !gray.isDeleted()) {
        gray.delete();
        gray = null;
    }
    clearInterval(loopIndex);
    document.getElementById("canvasOutput").getContext("2d").clearRect(0, 0, width, height);
    video.pause();
    video.srcObject = null;
    stream.getVideoTracks()[0].stop();
    start.disabled = false;
    video.removeEventListener("canplay", initVideo);
}

let Module = {
preRun: [function() {
    Module.FS_createPreloadedFile('/', 'haarcascade_eye.xml', 'haarcascade_eye.xml', true, false);
    Module.FS_createPreloadedFile('/', 'haarcascade_frontalface_default.xml', 'haarcascade_frontalface_default.xml', true, false);
    }],
};

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