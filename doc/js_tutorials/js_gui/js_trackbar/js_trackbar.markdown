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
var src1 = imread("canvas1");
var src2 = imread("canvas2");
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
    imshow("canvas3", dst);
    dst.delete();
}
@endcode

@sa cv.addWeighted

Try it
------

The result of the code demo as below:
![](images/Trackbar_Tutorial_Result.png)

And there is an interactive webpage for this tutorial, [createTrackbar](tutorial_js_interactive_trackbar.html). 
You can change the callback function and investigate more.
