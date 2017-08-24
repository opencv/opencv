Fourier Transform {#tutorial_js_fourier_transform}
=================

Goal
----

-   To find the Fourier Transform of images using OpenCV
-   Some applications of Fourier Transform
-   We will learn following functions : **cv.dft()** etc

Theory
------

Fourier Transform is used to analyze the frequency characteristics of various filters. For images,
**2D Discrete Fourier Transform (DFT)** is used to find the frequency domain. A fast algorithm
called **Fast Fourier Transform (FFT)** is used for calculation of DFT. Details about these can be
found in any image processing or signal processing textbooks.

For a sinusoidal signal, \f$x(t) = A \sin(2 \pi ft)\f$, we can say \f$f\f$ is the frequency of signal, and
if its frequency domain is taken, we can see a spike at \f$f\f$. If signal is sampled to form a discrete
signal, we get the same frequency domain, but is periodic in the range \f$[- \pi, \pi]\f$ or \f$[0,2\pi]\f$
(or \f$[0,N]\f$ for N-point DFT). You can consider an image as a signal which is sampled in two
directions. So taking fourier transform in both X and Y directions gives you the frequency
representation of image.

More intuitively, for the sinusoidal signal, if the amplitude varies so fast in short time, you can
say it is a high frequency signal. If it varies slowly, it is a low frequency signal. You can extend
the same idea to images. Where does the amplitude varies drastically in images ? At the edge points,
or noises. So we can say, edges and noises are high frequency contents in an image. If there is no
much changes in amplitude, it is a low frequency component.

Performance of DFT calculation is better for some array size. It is fastest when array size is power
of two. The arrays whose size is a product of 2’s, 3’s, and 5’s are also processed quite
efficiently. So if you are worried about the performance of your code, you can modify the size of
the array to any optimal size (by padding zeros) before finding DFT. OpenCV provides a function, **cv.getOptimalDFTSize()** for this. 

Now we will see how to find the Fourier Transform.

Fourier Transform in OpenCV
---------------------------

Performance of DFT calculation is better for some array size. It is fastest when array size is power of two. The arrays whose size is a product of 2’s, 3’s, and 5’s are also processed quite efficiently. So if you are worried about the performance of your code, you can modify the size of the array to any optimal size (by padding zeros). So how do we find this optimal size ? OpenCV provides a function, cv.getOptimalDFTSize() for this.

We use the functions: **cv.dft (src, dst, flags = 0, nonzeroRows = 0)** 

@param src           input array that could be real or complex.
@param dst           output array whose size and type depends on the flags.
@param flags         transformation flags, representing a combination of the cv.DftFlags
@param nonzeroRows   when the parameter is not zero, the function assumes that only the first nonzeroRows rows of the input array (DFT_INVERSE is not set) or only the first nonzeroRows of the output array (DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the rows more efficiently and save some time; this technique is very useful for calculating array cross-correlation or convolution using DFT.

**cv.getOptimalDFTSize (vecsize)**

@param vecsize   vector size.

**cv.copyMakeBorder (src, dst, top, bottom, left, right, borderType, value = new cv.Scalar())**

@param src           input array that could be real or complex.
@param dst           output array whose size and type depends on the flags.
@param top           parameter specifying how many top pixels in each direction from the source image rectangle to extrapolate. 
@param bottom        parameter specifying how many bottom pixels in each direction from the source image rectangle to extrapolate. 
@param left          parameter specifying how many left pixels in each direction from the source image rectangle to extrapolate. 
@param right         parameter specifying how many right pixels in each direction from the source image rectangle to extrapolate. 
@param borderType    border type.        
@param value         border value if borderType == cv.BORDER_CONSTANT.

**cv.magnitude (x, y, magnitude)**

@param x          floating-point array of x-coordinates of the vectors.
@param y          floating-point array of y-coordinates of the vectors; it must have the same size as x.        
@param magnitude  output array of the same size and type as x.

**cv.split (m, mv)**

@param m     input multi-channel array.
@param mv    output vector of arrays; the arrays themselves are reallocated, if needed.

**cv.merge (mv, dst)**

@param mv      input vector of matrices to be merged; all the matrices in mv must have the same size and the same depth.
@param dst     output array of the same size and the same depth as mv[0]; The number of channels will be the total number of channels in the matrix array.

Try it
------

Try this demo using the code above. Canvas elements named dftCanvasInput and dftCanvasOutput have been prepared. Choose an image and
click `Try it` to see the result. You can change the code in the textbox to investigate more.

\htmlonly
<!DOCTYPE html>
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
<div id="dftCodeArea">
<h2>Input your code</h2>
<button id="dftTryIt" disabled="true" onclick="dftExecuteCode()">Try it</button><br>
<textarea rows="17" cols="90" id="dftTestCode" spellcheck="false">
let src = cv.imread("dftCanvasInput");
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);

// get optimal size of DFT
let optimalRows = cv.getOptimalDFTSize(src.rows);
let optimalCols = cv.getOptimalDFTSize(src.cols);
let s0 = cv.Scalar.all(0);
let padded = new cv.Mat()
cv.copyMakeBorder(src, padded, 0, optimalRows - src.rows, 0, optimalCols - src.cols, cv.BORDER_CONSTANT, s0);

// use cv.MatVector to distribute space for real part and imaginary part
let plane0 = new cv.Mat()
padded.convertTo(plane0, cv.CV_32F);
let planes = new cv.MatVector();
let complexI = new cv.Mat();
let plane1 = new cv.Mat.zeros(padded.cols, padded.rows, cv.CV_32F);
planes.push_back(plane0); planes.push_back(plane1);
cv.merge(planes, complexI);

// in-place dft transfrom
cv.dft(complexI, complexI);

// compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
cv.split(complexI, planes);
cv.magnitude(planes.get(0), planes.get(1), planes.get(0));
let mag = planes.get(0);
let m1 = new cv.Mat.ones(mag.rows, mag.cols, mag.type());
cv.add(mag, m1, mag);
cv.log(mag, mag);

// crop the spectrum, if it has an odd number of rows or columns
let rect = new cv.Rect(0, 0, mag.cols & -2, mag.rows & -2);
mag = mag.roi(rect);

// rearrange the quadrants of Fourier image
// so that the origin is at the image center
let cx = mag.cols / 2;
let cy = mag.rows / 2;
let tmp = new cv.Mat();

let rect0 = new cv.Rect(0, 0, cx, cy);
let rect1 = new cv.Rect(cx, 0, cx, cy);
let rect2 = new cv.Rect(0, cy, cx, cy);
let rect3 = new cv.Rect(cx, cy, cx, cy);

let q0 = mag.roi(rect0);
let q1 = mag.roi(rect1);
let q2 = mag.roi(rect2);
let q3 = mag.roi(rect3);

// exchange 1 and 4 quadrants
q0.copyTo(tmp);
q3.copyTo(q0);
tmp.copyTo(q3);

// exchange 2 and 3 quadrants
q1.copyTo(tmp);
q2.copyTo(q1);
tmp.copyTo(q2);

// The pixel value of cv.CV_32S type image ranges from 0 to 1.
cv.normalize(mag, mag, 0, 1, cv.NORM_MINMAX);

cv.imshow("dftCanvasOutput", mag);
src.delete(); padded.delete(); planes.delete(); complexI.delete(); m1.delete(); tmp.delete(); 
</textarea>
<p class="err" id="dftErr"></p>
</div>
<div id="dftShowcase">
    <div>
        <canvas id="dftCanvasInput"></canvas>
        <canvas id="dftCanvasOutput"></canvas>
    </div>
    <input type="file" id="dftInput" name="file" />
</div>
<script src="utils.js"></script>
<script async src="opencv.js" id="opencvjs"></script>
<script>
function dftExecuteCode() {
    let dftText = document.getElementById("dftTestCode").value;
    try {
        eval(dftText);
        document.getElementById("dftErr").innerHTML = " ";
    } catch(err) {
        document.getElementById("dftErr").innerHTML = err;
    }
}

loadImageToCanvas("lena.jpg", "dftCanvasInput");
let dftInputElement = document.getElementById("dftInput");
dftInputElement.addEventListener("change", dftHandleFiles, false);
function dftHandleFiles(e) {
    let dftUrl = URL.createObjectURL(e.target.files[0]);
    loadImageToCanvas(dftUrl, "dftCanvasInput");
}

function onReady() {
    document.getElementById("dftTryIt").disabled = false;
}
if (typeof cv !== 'undefined') {
    onReady();
} else {
    document.getElementById("opencvjs").onload = onReady;
}
</script>
</body>
\endhtmlonly