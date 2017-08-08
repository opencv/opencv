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
let src1 = cv.imread("canvasInput1");
let src2 = cv.imread("canvasInput2");
@endcode

Then we use HTML DOM Input Range Object to implement the trackbar, which is shown as below. 
![](images/Trackbar_Tutorial_Range.png)

@note &lt;input&gt; elements with type="range" are not supported in Internet Explorer 9 and earlier versions.

You can create an &lt;input&gt; element with type="range" by using the document.createElement() method:
@code{.js}
let x = document.createElement("INPUT");
x.setAttribute("type", "range");
@endcode

You can access an &lt;input&gt; element with type="range" by using getElementById():
@code{.js}
let x = document.getElementById("myRange");
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
    let alpha = value/document.getElementById("trackbar").max;
    let beta = ( 1.0 - alpha );
    let dst = new cv.Mat();
    cv.addWeighted( src1, alpha, src2, beta, 0.0, dst, -1);
    cv.imshow("canvasOutput", dst);
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
.err {
    color: red;
}
</style>
</head>
<body>
<div id="CodeArea">
<h2>Input your code</h2>
<textarea rows="8" cols="70" id="TestCode" spellcheck="false">
let alpha = value/trackbar.max;
let beta = ( 1.0 - alpha );
let dst = new cv.Mat();
cv.addWeighted( src1, alpha, src2, beta, 0.0, dst, -1);
cv.imshow("canvasOutput", dst);
dst.delete();
</textarea>
<p class="err" id="tbErr"></p>
</div>
<div id="showcase">
    <div>
        <canvas id="canvasInput1"></canvas>
        <canvas id="canvasInput2"></canvas>
    </div>
    Weight: <input type="range" id="trackbar" disabled="true" value="50" min="0" max="100" step="1" 
    oninput="addWeighted(this.value)"><input type="text" id="weightValue" size="3" value="50"><br>
    <canvas id="canvasOutput"></canvas>
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
let weightValue = document.getElementById('weightValue');
let trackbar = document.getElementById('trackbar');

function addWeighted(value) {
    weightValue.value = value;    
    let text = document.getElementById("TestCode").value;
    try {
        eval(text);
        document.getElementById("tbErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("tbErr").innerHTML = err;
    }
}

loadImageToCanvas("apple.jpg", "canvasInput1");
loadImageToCanvas("orange.jpg", "canvasInput2");

let src1, src2;
function onReady() {
    src1 = cv.imread("canvasInput1");
    src2 = cv.imread("canvasInput2");
    addWeighted(trackbar.value);
    trackbar.disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly