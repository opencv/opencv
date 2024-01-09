Extended qoi.h for OpenCV
=========================
This document is appendix described what is qoi.h extended for OpenCV.
When updating qoi.h embedded in OpenCV, please refer this document.

The base qoi.h has been downloaded from here.

https://github.com/phoboslab/qoi/tree/827a7e4418ca75553e4396ca4e8c508c9dc07048

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

add static_cast<>()
-------------------
To suppress warning C4244, add static_cast<>() to convert unsigned int to unsigned char.

replace int to size_t
---------------------
To suppress warning C4267, use size_t instead of int.
