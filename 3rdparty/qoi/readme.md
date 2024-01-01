qoi.h extemds for OpenCV
========================
This document is appendix described what is qoi.h customized for OpenCV.
When updating qoi.h embedded in OpenCV, please refer this document.

Base version
------------
The base qoi.h has been downloaded with following command.

https://raw.githubusercontent.com/phoboslab/qoi/36190eb07dc5d85f408d998d1884eb69573adf68/qoi.h

namespace cv
------------
The qoi_* functions are moved into cv namespace instead of extern "C".
In OpenCV5, std liibrary is collesions with other stb user applications.

QOI_EXT_ENCODE_ORDER_BGRA
-------------------------
- If defined, the order of input data to encode is BGR/BGRA.
- If not, it is RGB/RGBA order.

union qoi_rgba_t
----------------
Before r,g,b,a are invidual parameters. However it is not suitable for optimization.
After r,g,b,a are combied into one array.
