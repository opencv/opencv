Barcode Recognition    {#tutorial_barcode_detect_and_decode}
===================

@tableofcontents

@prev_tutorial{tutorial_traincascade}
@next_tutorial{tutorial_introduction_to_svm}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 4.8 |

Goal
----

In this chapter,

-   We will familiarize with the bar code detection and decoding methods available in OpenCV.

Basics
----

Bar code is major technique to identify commodity in real life.  A common bar code is a pattern of parallel lines arranged by black bars and white bars with vastly different reflectivity. Bar code recognition is to scan the bar code in the horizontal direction to get a string of binary codes composed of bars of different widths and colors, that is, the code information of the bar code. The content of bar code can be decoded by matching with various bar code encoding methods. For current work, we only support EAN13 encoding method.

### EAN 13

The EAN-13 bar code is based on the UPC-A standard, which was first implemented in Europe by the International Item Coding Association and later gradually spread worldwide. Most of the common goods in life use EAN-13 barcode.

for more detail see [EAN - Wikipedia](https://en.wikipedia.org/wiki/International_Article_Number)

### BarcodeDetector
Several algorithms were introduced for bar code recognition.

While coding, we firstly need to create a **cv::barcode::BarcodeDetector** object.  It has mainly three member functions, which will be introduced in the following.

#### Initilization

User can construct BarcodeDetector with super resolution model which should be downloaded automatically to `<opencv_build_dir>/downloads/barcode`. If not, please download them from `https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode`

or choose not to use super resolution.

@snippet ./samples/barcode.cpp initialize
We need create variables to store the outputs.
@snippet ./samples/barcode.cpp output

#### detect

It is a algorithm based on directional coherence. First of all, we compute the average squared gradients of every pixels. It was proposed in the paper "Systematic methods for the computation of the directional fields and singular points of fingerprints" by A.M. Bazen and S.H. Gerez in 2002. Then we divide the image into some square patches and compute the **gradient orientation coherence** and **mean gradient direction** of each patch. At last we connected the patches that have **high gradient orientation coherence** and **similar gradient direction**. In this stage, we use multi-scale patches to capture the gradient distribution of multi-size bar codes, and apply non-maximum suppression to filter duplicate proposals. A last, we use minAreaRect() to bound the ROI, and output the corners of the rectangles.

Detect codes in the input image, and output the corners of detected rectangles:

@snippet ./samples/barcode.cpp detect

#### decode

This function first super-scales the image if it is smaller than threshold, sharpens the image and then binaries it by OTSU or local-binarization. At last reads the contents of the barcode by matching the similarity of the specified barcode pattern. Only EAN-13 barcode currently supported.

You can find more information in **cv::barcode::BarcodeDetector::decode()**.

#### detectAndDecode

This function combines `detect`  and `decode`.  A simple example below to use this function showing recognized bar codes.

@snippet ./samples/barcode.cpp detectAndDecode

Visualize the results:
@snippet ./samples/barcode.cpp visualize

Results
-------

**Original Image**

Below image shows four EAN 13 bar codes photoed by a smart phone.

![image](images/barcode_book.jpg)

**Result of detectAndDecode**

Bar codes are bounded by green box, and decoded numbers are lying on the boxes.

![image](images/barcode_book_res.jpg)
