Optical Flow {#tutorial_js_lucas_kanade}
============

Goal
----

-   We will understand the concepts of optical flow and its estimation using Lucas-Kanade
    method.
-   We will use functions like **cv.calcOpticalFlowPyrLK()** to track feature points in a
    video.

Optical Flow
------------

Optical flow is the pattern of apparent motion of image objects between two consecutive frames
caused by the movemement of object or camera. It is 2D vector field where each vector is a
displacement vector showing the movement of points from first frame to second. Consider the image
below (Image Courtesy: [Wikipedia article on Optical
Flow](http://en.wikipedia.org/wiki/Optical_flow)).

![image](images/optical_flow_basic1.jpg)

It shows a ball moving in 5 consecutive frames. The arrow shows its displacement vector. Optical
flow has many applications in areas like :

-   Structure from Motion
-   Video Compression
-   Video Stabilization ...

Optical flow works on several assumptions:

-#  The pixel intensities of an object do not change between consecutive frames.
2.  Neighbouring pixels have similar motion.

Consider a pixel \f$I(x,y,t)\f$ in first frame (Check a new dimension, time, is added here. Earlier we
were working with images only, so no need of time). It moves by distance \f$(dx,dy)\f$ in next frame
taken after \f$dt\f$ time. So since those pixels are the same and intensity does not change, we can say,

\f[I(x,y,t) = I(x+dx, y+dy, t+dt)\f]

Then take taylor series approximation of right-hand side, remove common terms and divide by \f$dt\f$ to
get the following equation:

\f[f_x u + f_y v + f_t = 0 \;\f]

where:

\f[f_x = \frac{\partial f}{\partial x} \; ; \; f_y = \frac{\partial f}{\partial y}\f]\f[u = \frac{dx}{dt} \; ; \; v = \frac{dy}{dt}\f]

Above equation is called Optical Flow equation. In it, we can find \f$f_x\f$ and \f$f_y\f$, they are image
gradients. Similarly \f$f_t\f$ is the gradient along time. But \f$(u,v)\f$ is unknown. We cannot solve this
one equation with two unknown variables. So several methods are provided to solve this problem and
one of them is Lucas-Kanade.

### Lucas-Kanade method

We have seen an assumption before, that all the neighbouring pixels will have similar motion.
Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion. We
can find \f$(f_x, f_y, f_t)\f$ for these 9 points. So now our problem becomes solving 9 equations with
two unknown variables which is over-determined. A better solution is obtained with least square fit
method. Below is the final solution which is two equation-two unknown problem and solve to get the
solution.

\f[\begin{bmatrix} u \\ v \end{bmatrix} =
\begin{bmatrix}
    \sum_{i}{f_{x_i}}^2  &  \sum_{i}{f_{x_i} f_{y_i} } \\
    \sum_{i}{f_{x_i} f_{y_i}} & \sum_{i}{f_{y_i}}^2
\end{bmatrix}^{-1}
\begin{bmatrix}
    - \sum_{i}{f_{x_i} f_{t_i}} \\
    - \sum_{i}{f_{y_i} f_{t_i}}
\end{bmatrix}\f]

( Check similarity of inverse matrix with Harris corner detector. It denotes that corners are better
points to be tracked.)

So from user point of view, idea is simple, we give some points to track, we receive the optical
flow vectors of those points. But again there are some problems. Until now, we were dealing with
small motions. So it fails when there is large motion. So again we go for pyramids. When we go up in
the pyramid, small motions are removed and large motions becomes small motions. So applying
Lucas-Kanade there, we get optical flow along with the scale.

Lucas-Kanade Optical Flow in OpenCV.js
-----------------------------------

OpenCV.js provides all these in a single function, **cv.calcOpticalFlowPyrLK()**. Here, we create a
simple application which tracks some points in a video. To decide the points, we use
**cv.goodFeaturesToTrack()**. We take the first frame, detect some Shi-Tomasi corner points in it,
then we iteratively track those points using Lucas-Kanade optical flow. For the function
**cv.calcOpticalFlowPyrLK()** we pass the previous frame, previous points and next frame. It
returns next points along with some status numbers which has a value of 1 if next point is found,
else zero. We iteratively pass these next points as previous points in next step. See the code demo
below.

### Try it

Here is the demo. Some core code is in the textbox, and you can click `try it` to 
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
<textarea rows="30" cols="90" id="lkofTestCode" spellcheck="false">
// Mats used in the loop are all declared and deleted elsewhere
//  params for ShiTomasi corner detection
let [maxCorners, qualityLevel, minDistance, blockSize] = [30, 0.3, 7, 7];

// Parameters for lucas kanade optical flow
let winSize  = new cv.Size(15,15);
let maxLevel = 2;
let criteria = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03);

// Create some random colors
let color = [];
for(let i = 0; i < maxCorners; i++)
    color.push(new cv.Scalar(parseInt(Math.random()*255), parseInt(Math.random()*255), parseInt(Math.random()*255), 255));

// Take first frame and find corners in it
let oldFrame = new cv.Mat(lkofHeight, lkofWidth, cv.CV_8UC4);
let cap = new cv.VideoCapture("lkofVideo"); // "lkofVideo" is the id of the video tag
cap.read(oldFrame);
oldGray = new cv.Mat();
cv.cvtColor(oldFrame, oldGray, cv.COLOR_RGB2GRAY);
p0 = new cv.Mat();
let none = new cv.Mat();
cv.goodFeaturesToTrack(oldGray, p0, maxCorners, qualityLevel, minDistance, none, blockSize);

// Create a mask image for drawing purposes
let zeroEle = new cv.Scalar(0, 0, 0, 255);
mask = new cv.Mat(oldFrame.rows, oldFrame.cols, oldFrame.type(), zeroEle);

frame = new cv.Mat(lkofHeight, lkofWidth, cv.CV_8UC4);
frameGray = new cv.Mat();
p1 = new cv.Mat();
st = new cv.Mat();
err = new cv.Mat();
lkofLoopIndex = setInterval(
    function() {
        if(lkofVideo.ended) {
            lkofStopVideo();
            return;
        }
        cap.read(frame);
        cv.cvtColor(frame, frameGray, cv.COLOR_RGBA2GRAY);

        // calculate optical flow
        cv.calcOpticalFlowPyrLK(oldGray, frameGray, p0, p1, st, err, winSize, maxLevel, criteria);

        // Select good points
        let goodNew = [];
        let goodOld = [];
        for(let i = 0; i < st.rows; i++) {
            if(st.data[i] === 1) {
                goodNew.push(new cv.Point(p1.data32F[i*2], p1.data32F[i*2+1]));
                goodOld.push(new cv.Point(p0.data32F[i*2], p0.data32F[i*2+1]));
            }
        }

        // draw the tracks
        for(let i = 0; i < goodNew.length; i++) {
            cv.line(mask, goodNew[i], goodOld[i], color[i], 2);
            cv.circle(frame, goodNew[i], 5, color[i],-1);
        }
        cv.add(frame, mask, frame);

        cv.imshow("lkofCanvasOutput", frame);

        // Now update the previous frame and previous points
        frameGray.copyTo(oldGray);
        p0.delete(); p0 = null;
        p0 = new cv.Mat(goodNew.length, 1, cv.CV_32FC2);
        for(let i = 0; i < goodNew.length; i++) {
            p0.data32F[i*2] = goodNew[i].x;
            p0.data32F[i*2+1] = goodNew[i].y;
        }
    }, 33); 
</textarea>
<p class="err" id="lkofErr"></p>
</div> 
<div id="contentarea">
    <button id="lkofStartup" disabled="true" onclick="lkofStartup()">try it</button>
    <button id="lkofStop" disabled="true" onclick="lkofStopVideo()">stop</button><br>
    <video id="lkofVideo" src="box.mp4" width="640" muted hidden>Your browser does not support the video tag.</video>
    <canvas id="lkofCanvasOutput"></canvas>
</div>
<script async src="opencv.js" id="opencvjs"></script>
<script>
// lkof means Lucas-Kanade Optical Flow
// Some HTML elements we need to configure.
let lkofVideo = document.getElementById("lkofVideo");
let lkofStop = document.getElementById("lkofStop");

// In this case, We set width 640, and the height will be computed based on the input video.
let lkofWidth = lkofVideo.width;
let lkofHeight = null;
let lkofLoopIndex = null;
let frame = null;
let oldGray = null;
let frameGray = null;
let p0 = null;
let p1 = null;
let st = null;
let err = null;
let mask = null;
let color = null;

lkofVideo.oncanplay = function() {
    lkofVideo.setAttribute("height", lkofVideo.videoHeight/lkofVideo.videoWidth*lkofVideo.width);
    lkofHeight = lkofVideo.height;
};

lkofVideo.onended = lkofStopVideo;

function lkofStartup() {
    if(lkofVideo.readyState !== 4)
        lkofVideo.load();
    lkofVideo.play();
    lkofStop.disabled = false;

    let lkofTestCode = document.getElementById("lkofTestCode").value;
    try {
        eval(lkofTestCode);
        document.getElementById("lkofErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("lkofErr").innerHTML = err;
    }   
    document.getElementById("lkofStartup").disabled = true;
}

function lkofStopVideo() {
    clearInterval(lkofLoopIndex);
    if (frame != null && !frame.isDeleted()) {
        frame.delete();
        frame = null;
    }
    if (oldGray != null && !oldGray.isDeleted()) {
        oldGray.delete();
        oldGray = null;
    }
    if (frameGray != null && !frameGray.isDeleted()) {
        frameGray.delete();
        frameGray = null;
    }
    if (p0 != null && !p0.isDeleted()) {
        p0.delete();
        p0 = null;
    }
    if (p1 != null && !p1.isDeleted()) {
        p1.delete();
        p1 = null;
    }
    if (st != null && !st.isDeleted()) {
        st.delete();
        st = null;
    }
    if (err != null && !err.isDeleted()) {
        err.delete();
        err = null;
    }
    if (mask != null && !mask.isDeleted()) {
        mask.delete();
        mask = null;
    }
    //document.getElementById("lkofCanvasOutput").getContext("2d").clearRect(0, 0, lkofWidth, lkofHeight);
    lkofVideo.pause();
    lkofVideo.currentTime = 0;
    document.getElementById("lkofStartup").disabled = false;
}
</script>
</body>
\endhtmlonly

(This code doesn't check how correct are the next keypoints. So even if any feature point disappears
in image, there is a chance that optical flow finds the next point which may look close to it. So
actually for a robust tracking, corner points should be detected in particular intervals.)

Dense Optical Flow in OpenCV.js
----------------------------

Lucas-Kanade method computes optical flow for a sparse feature set (in our example, corners detected
using Shi-Tomasi algorithm). OpenCV.js provides another algorithm to find the dense optical flow. It
computes the optical flow for all the points in the frame. It is based on Gunner Farneback's
algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by
Gunner Farneback in 2003.

Below sample shows how to find the dense optical flow using above algorithm, and the function is 
**cv.calcOpticalFlowFarneback()**. We get a 2-channel array with optical flow vectors, \f$(u,v)\f$. 
We find their magnitude and direction. We color code the result for better visualization. Direction 
corresponds to Hue value of the image. Magnitude corresponds to Value plane. See the code demo below.

### Try it

Here is the demo. Some core code is in the textbox, and you can click `try it` to 
investigate more.

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
<textarea rows="30" cols="90" id="dofTestCode" spellcheck="false">
// Mats used in the loop are all declared and deleted elsewhere
// take first frame of the video
let frame1 = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC4);
let cap = new cv.VideoCapture("dofVideo"); // "dofVideo" is the id of the video tag
cap.read(frame1);

prvs = new cv.Mat();
cv.cvtColor(frame1, prvs, cv.COLOR_RGBA2GRAY);
frame1.delete();
hsv = new cv.Mat();
hsv0 = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC1);
hsv1 = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC1, new cv.Scalar(255));
hsv2 = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC1);
hsvVec = new cv.MatVector();
hsvVec.push_back(hsv0); hsvVec.push_back(hsv1); hsvVec.push_back(hsv2);

frame2 = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC4);
next = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC1);
flow = new cv.Mat(dofHeight, dofWidth, cv.CV_32FC2);
flowVec = new cv.MatVector();
mag = new cv.Mat(dofHeight, dofWidth, cv.CV_32FC1);
ang = new cv.Mat(dofHeight, dofWidth, cv.CV_32FC1);
rgb = new cv.Mat(dofHeight, dofWidth, cv.CV_8UC3);
dofLoopIndex = setInterval(
    function() {
        if(dofVideo.ended) {
            dofStopVideo();
            return;
        }
        cap.read(frame2);
        cv.cvtColor(frame2, next, cv.COLOR_RGBA2GRAY);
        cv.calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        cv.split(flow, flowVec);
        let flow0 = flowVec.get(0);
        let flow1 = flowVec.get(1);
        cv.cartToPolar(flow0, flow1, mag, ang);
        flow0.delete(); flow1.delete();
        ang.convertTo(hsv0, cv.CV_8UC1, 180/Math.PI/2);
        cv.normalize(mag, hsv2, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1);
        cv.merge(hsvVec, hsv);
        cv.cvtColor(hsv, rgb, cv.COLOR_HSV2RGB);
        cv.imshow("dofCanvasOutput", rgb);
        next.copyTo(prvs);
    }, 33);    
</textarea>
<p class="err" id="dofErr"></p>
</div>
<div id="contentarea">
    <button id="dofStartup" disabled="true" onclick="dofStartup()">try it</button>
    <button id="dofStop" disabled="true" onclick="dofStopVideo()">stop</button><br>
    <video id="dofVideo" src="box.mp4" width="320" muted>Your browser does not support the video tag.</video>
    <canvas id="dofCanvasOutput"></canvas>
</div>
<script>
// dof means Dense Optical Flow
// Some HTML elements we need to configure.
let dofVideo = document.getElementById("dofVideo");
let dofStop = document.getElementById("dofStop");

// In this case, We set width 320, and the height will be computed based on the input video.
let dofWidth = dofVideo.width;
let dofHeight = null;
let dofLoopIndex = null;
let prvs = null;
let hsv = null;
let hsv0 = null;
let hsv1 = null;
let hsv2 = null;
let hsvVec = null;
let frame2 = null;
let next = null;
let flow = null;
let flowVec = null;
let mag = null;
let ang = null;
let rgb = null;

dofVideo.oncanplay = function() {
    dofVideo.setAttribute("height", dofVideo.videoHeight/dofVideo.videoWidth*dofVideo.width);
    dofHeight = dofVideo.height;
};

dofVideo.onended = dofStopVideo;

function dofStartup() {
    if(dofVideo.readyState !== 4)
        dofVideo.load();
    dofVideo.play();
    dofStop.disabled = false;
    let dofTestCode = document.getElementById("dofTestCode").value;

    try {
        eval(dofTestCode);
        document.getElementById("dofErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("dofErr").innerHTML = err;
    }
    document.getElementById("dofStartup").disabled = true;
}

function dofStopVideo() {
    clearInterval(dofLoopIndex);
    if (prvs != null && !prvs.isDeleted()) {
        prvs.delete();
        prvs = null;
    }
    if (hsv != null && !hsv.isDeleted()) {
        hsv.delete();
        hsv = null;
    }
    if (hsv0 != null && !hsv0.isDeleted()) {
        hsv0.delete();
        hsv0 = null;
    }
    if (hsv1 != null && !hsv1.isDeleted()) {
        hsv1.delete();
        hsv1 = null;
    }
    if (hsv2 != null && !hsv2.isDeleted()) {
        hsv2.delete();
        hsv2 = null;
    }
    if (hsvVec != null && !hsvVec.isDeleted()) {
        hsvVec.delete();
        hsvVec = null;
    }
    if (frame2 != null && !frame2.isDeleted()) {
        frame2.delete();
        frame2 = null;
    }
    if (flow != null && !flow.isDeleted()) {
        flow.delete();
        flow = null;
    }
    if (flowVec != null && !flowVec.isDeleted()) {
        flowVec.delete();
        flowVec = null;
    }
    if (next != null && !next.isDeleted()) {
        next.delete();
        next = null;
    }
    if (mag != null && !mag.isDeleted()) {
        mag.delete();
        mag = null;
    }
    if (ang != null && !ang.isDeleted()) {
        ang.delete();
        ang = null;
    }
    if (rgb != null && !rgb.isDeleted()) {
        rgb.delete();
        rgb = null;
    }
    //document.getElementById("dofCanvasOutput").getContext("2d").clearRect(0, 0, dofWidth, dofHeight);
    dofVideo.pause();
    dofVideo.currentTime = 0;
    document.getElementById("dofStartup").disabled = false;
}


function onReady() {
    document.getElementById("lkofStartup").disabled = false;
    document.getElementById("dofStartup").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly
