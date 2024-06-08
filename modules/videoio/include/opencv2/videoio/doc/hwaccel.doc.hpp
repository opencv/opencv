// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

//
// This file should not be used with compiler (documentation only)
//

namespace cv {
/** @addtogroup videoio_hwaccel
This section contains information about API to control Hardware-accelerated video decoding and encoding.

@note Check [Wiki page](https://github.com/opencv/opencv/wiki/Video-IO-hardware-acceleration)
for description of supported hardware / software configurations and available benchmarks

cv::VideoCapture properties:
- #CAP_PROP_HW_ACCELERATION (as #VideoAccelerationType)
- #CAP_PROP_HW_DEVICE

cv::VideoWriter properties:
- #VIDEOWRITER_PROP_HW_ACCELERATION (as #VideoAccelerationType)
- #VIDEOWRITER_PROP_HW_DEVICE

Properties are supported by these backends:

- #CAP_FFMPEG
- #CAP_GSTREAMER
- #CAP_MSMF (Windows)

@{
 */

/** @} */
}  // namespace
