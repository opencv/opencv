Add a Trackbar to Your Application {#tutorial_js_trackbar}
=============================

Goal
----

-   Add a Trackbar to your application by using HTML DOM Input Range Object

Code Demo
---------

Here we will create a simple application which blends two images. We will let the user enter the
weight by using the trackbar.

First, we need to creat three canvas elements, two for input and one for output. Please refer to 
the tutorial @ref tutorial_js_image_display.
@code{.js}
var src1 = cv.imread("canvas1");
var src2 = cv.imread("canvas2");
@endcode

Then we use HTML DOM Input Range Object to implement the trackbar, which is shown as below. 
![](images/Trackbar_Tutorial_Range.png)

@note &lt;input&gt; elements with type="range" are not supported in Internet Explorer 9 and earlier versions.

You can create an &lt;input&gt; element with type="range" by using the document.createElement() method:
@code{.js}
var x = document.createElement("INPUT");
x.setAttribute("type", "range");
@endcode

You can access an &lt;input&gt; element with type="range" by using getElementById():
@code{.js}
var x = document.getElementById("myRange");
@endcode

As a trackbar, the range element need a trackbar name, the default value, minimum value, maximum value, 
step and the callback function which is executed everytime trackbar value changes. The callback function 
always has a default argument which is the trackbar position. Additionally, a text element to display the trackbar 
value is fine. In our case, we can create the trackbar as below:
@code{.html}
Weight: <input type="range" id="trackbar" value="50" min="0" max="100" step="1" oninput="addWeighted(this.value)">
<input type="text" id="weightValue" size="3" value="50"/>
@endcode

Finally, we can use the trackbar value in the callback function, blend the two images and display the result.
@code{.js}
function addWeighted(value) {
    document.getElementById("weightValue").value = value;
    var alpha = value/document.getElementById("trackbar").max;
    var beta = ( 1.0 - alpha );
    var dst = new cv.Mat();
    cv.addWeighted( src1, alpha, src2, beta, 0.0, dst, -1);
    cv.imshow("canvas3", dst);
    dst.delete();
}
@endcode

@sa cv.addWeighted

Try it
------

Here is the demo for above code. Trackbar and input images are ready. Slide the trackbar to see the result. 
And you can change the callback function and investigate more.

\htmlonly
<head>
<style>
canvas {
    border: 1px solid black;
}
</style>
</head>
<body>
<div id="CodeArea">
<h2>Input your code</h2>
<textarea rows="8" cols="70" id="TestCode" spellcheck="false">
var alpha = value/document.getElementById("trackbar").max;
var beta = ( 1.0 - alpha );
var dst = new cv.Mat();
cv.addWeighted( src1, alpha, src2, beta, 0.0, dst, -1);
cv.imshow("canvas3", dst);
dst.delete();
</textarea>
</div>
<div id="showcase">
    <div>
        <canvas id="canvas1"></canvas>
        <canvas id="canvas2"></canvas>
    </div>
    Weight: <input type="range"  id="trackbar" disabled="true" value="50" min="0" max="100" step="1" 
    oninput="addWeighted(this.value)"><input type="text" id="weightValue" size="3" value="50"><br>
    <canvas id="canvas3"></canvas>
</div>
<script async src="opencv.js"  id="opencvjs"></script>
<script>
function addWeighted(value) {
    document.getElementById("weightValue").value = value;    
    var text = document.getElementById("TestCode").value;
    eval(text);
}

var canvas1 = document.getElementById("canvas1");
var ctx1 = canvas1.getContext("2d");
var url1 = "LinuxLogo.jpg";
var img1 = new Image();
img1.onload = function() {
    canvas1.width = img1.width;
    canvas1.height = img1.height;
    ctx1.drawImage(img1,0,0,img1.width,img1.height);
}
img1.src = url1;

var canvas2 = document.getElementById("canvas2");
var ctx2 = canvas2.getContext("2d");
var url2 = "WindowsLogo.jpg";
var img2 = new Image();
img2.onload = function() {
    canvas2.width = img2.width;
    canvas2.height = img2.height;
    ctx2.drawImage(img2,0,0,img2.width,img2.height);
}
img2.src = url2;

var src1, src2;
document.getElementById("opencvjs").onload = function() {
    src1 = cv.imread("canvas1");
    src2 = cv.imread("canvas2");
    var value = document.getElementById("trackbar").value;
    var text = document.getElementById("TestCode").value;
    eval(text);
    document.getElementById("trackbar").disabled = false;
};
</script>
</body>
\endhtmlonly