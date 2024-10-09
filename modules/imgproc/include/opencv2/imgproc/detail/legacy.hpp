// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_IMGPROC_DETAIL_LEGACY_HPP
#define OPENCV_IMGPROC_DETAIL_LEGACY_HPP

#include "opencv2/imgproc.hpp"

namespace cv {

#ifdef __OPENCV_BUILD

CV_EXPORTS void findContours_legacy(InputArray _image,
                                    OutputArrayOfArrays _contours,
                                    OutputArray _hierarchy,
                                    int mode,
                                    int method,
                                    Point offset = Point());
CV_EXPORTS void findContours_legacy(InputArray image,
                                    OutputArrayOfArrays contours,
                                    int mode,
                                    int method,
                                    Point offset = Point());

CV_EXPORTS float EMD_legacy( InputArray _signature1, InputArray _signature2,
               int distType, InputArray _cost,
               float* lowerBound, OutputArray _flow );

CV_EXPORTS float wrapperEMD_legacy(InputArray _signature1, InputArray _signature2,
               int distType, InputArray _cost,
               Ptr<float> lowerBound, OutputArray _flow);

#endif

}  // namespace cv

#endif  // OPENCV_IMGPROC_DETAIL_LEGACY_HPP
