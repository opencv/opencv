Barcode Recognition    {#tutorial_barcode_detect_and_decode}
===================

@tableofcontents

@prev_tutorial{tutorial_aruco_faq}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 4.8 |

Goal
----

In this chapter we will familiarize with the barcode detection and decoding methods available in OpenCV.

Basics
----

Barcode is major technique to identify commodity in real life. A common barcode is a pattern of parallel lines arranged by black bars and white bars with vastly different reflectivity. Barcode recognition is to scan the barcode in the horizontal direction to get a string of binary codes composed of bars of different widths and colors, that is, the code information of the barcode. The content of barcode can be decoded by matching with various barcode encoding methods. Currently, we support EAN-8, EAN-13, UPC-A and UPC-E standards.

See https://en.wikipedia.org/wiki/Universal_Product_Code and https://en.wikipedia.org/wiki/International_Article_Number

Related papers: @cite Xiangmin2015research , @cite kass1987analyzing , @cite bazen2002systematic

Code example
------------

### Main class
Several algorithms were introduced for barcode recognition.

While coding, we firstly need to create a cv::barcode::BarcodeDetector object. It has mainly three member functions, which will be introduced in the following.

#### Initialization

Optionally user can construct barcode detector with super resolution model which should be downloaded from https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode (`sr.caffemodel`, `sr.prototxt`).

@snippet cpp/barcode.cpp initialize

We need to create variables to store the outputs.

@snippet cpp/barcode.cpp output

#### Detecting

cv::barcode::BarcodeDetector::detect method uses an algorithm based on directional coherence. First, we compute the average squared gradients of every pixel, @cite bazen2002systematic . Then we divide an image into square patches and compute the **gradient orientation coherence** and **mean gradient direction** of each patch. Then, we connect all patches that have **high gradient orientation coherence** and **similar gradient direction**. At this stage we use multiscale patches to capture the gradient distribution of multi-size barcodes, and apply non-maximum suppression to filter duplicate proposals. At last, we use cv::minAreaRect to bound the ROI, and output the corners of the rectangles.

Detect codes in the input image, and output the corners of detected rectangles:

@snippet cpp/barcode.cpp detect

#### Decoding

cv::barcode::BarcodeDetector::decode method first super-scales the image (_optionally_) if it is smaller than threshold, sharpens the image and then binaries it by OTSU or local binarization. Then it reads the contents of the barcode by matching the similarity of the specified barcode pattern.

#### Detecting and decoding

cv::barcode::BarcodeDetector::detectAndDecode combines `detect` and `decode` in a single call. A simple example below shows how to use this function:

@snippet cpp/barcode.cpp detectAndDecode

Visualize the results:

@snippet cpp/barcode.cpp visualize

Results
-------

Original image:

![image](images/barcode_book.jpg)

After detection:

![image](images/barcode_book_res.jpg)
