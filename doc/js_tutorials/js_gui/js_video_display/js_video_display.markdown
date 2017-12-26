Getting Started with Videos {#tutorial_js_video_display}
===========================

Goal
----

-   Learn to capture video from a camera and display it.

Capture video from camera
-------------------------

Often, we have to capture live stream with a camera. In OpenCV.js, we use [WebRTC](https://webrtc.org/)
and HTML canvas element to implement this. Let's capture a video from the camera(built-in
or a usb), convert it into grayscale video and display it.

To capture a video, you need to add some HTML elements to the web page:
- a &lt;video&gt; to display video from camera directly
- a &lt;canvas&gt; to transfer video to canvas ImageData frame-by-frame
- another &lt;canvas&gt; to display the video OpenCV.js gets

First, we use WebRTC navigator.mediaDevices.getUserMedia to get the media stream.
@code{.js}
let video = document.getElementById("videoInput"); // video is the id of video tag
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occured! " + err);
    });
@endcode

@note This function is unnecessary when you capture video from a video file. But notice that
HTML video element only supports video formats of Ogg(Theora), WebM(VP8/VP9) or MP4(H.264).

Playing video
-------------
Now, the browser gets the camera stream. Then, we use CanvasRenderingContext2D.drawImage() method
of the Canvas 2D API to draw video onto the canvas. Finally, we can use the method in @ref tutorial_js_image_display
 to read and display image in canvas. For playing video, cv.imshow() should be executed every delay
milliseconds. We recommend setTimeout() method. And if the video is 30fps, the delay milliseconds
should be (1000/30 - processing_time).
@code{.js}
let canvasFrame = document.getElementById("canvasFrame"); // canvasFrame is the id of <canvas>
let context = canvasFrame.getContext("2d");
let src = new cv.Mat(height, width, cv.CV_8UC4);
let dst = new cv.Mat(height, width, cv.CV_8UC1);

const FPS = 30;
function processVideo() {
    let begin = Date.now();
    context.drawImage(video, 0, 0, width, height);
    src.data.set(context.getImageData(0, 0, width, height).data);
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
    cv.imshow("canvasOutput", dst); // canvasOutput is the id of another <canvas>;
    // schedule next one.
    let delay = 1000/FPS - (Date.now() - begin);
    setTimeout(processVideo, delay);
}

// schedule first one.
setTimeout(processVideo, 0);
@endcode

OpenCV.js implements **cv.VideoCapture (videoSource)** using the above method. You need not to
add the hidden canvas element manually.
@param videoSource   the video id or element.
@return              cv.VideoCapture instance

We use **read (image)** to get one frame of the video. For performance reasons, the image should be
constructed with cv.CV_8UC4 type and same size as the video.
@param image         image with cv.CV_8UC4 type and same size as the video.

The above code of playing video could be simplified as below.
@code{.js}
let src = new cv.Mat(height, width, cv.CV_8UC4);
let dst = new cv.Mat(height, width, cv.CV_8UC1);
let cap = new cv.VideoCapture(videoSource);

const FPS = 30;
function processVideo() {
    let begin = Date.now();
    cap.read(src);
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
    cv.imshow("canvasOutput", dst);
    // schedule next one.
    let delay = 1000/FPS - (Date.now() - begin);
    setTimeout(processVideo, delay);
}

// schedule first one.
setTimeout(processVideo, 0);
@endcode

@note Remember to delete src and dst after when stop.

Try it
------

\htmlonly
<iframe src="../../js_video_display.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly