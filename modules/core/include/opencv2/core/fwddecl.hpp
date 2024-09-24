// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_FWDDECL_HPP
#define OPENCV_CORE_FWDDECL_HPP

#include "opencv2/core/cvdef.h"

namespace cv {

//! @cond IGNORED

////////////////// forward declarations for important OpenCV types //////////////////

template<typename _Tp, int cn> class Vec;
template<typename _Tp, int m, int n> class Matx;

template<typename _Tp> class Complex;
template<typename _Tp> class Point_;
template<typename _Tp> class Point3_;
template<typename _Tp> class Size_;
template<typename _Tp> class Rect_;
template<typename _Tp> class Scalar_;

class CV_EXPORTS RotatedRect;
class CV_EXPORTS Range;
class CV_EXPORTS TermCriteria;
class CV_EXPORTS KeyPoint;
class CV_EXPORTS DMatch;
class CV_EXPORTS RNG;

class CV_EXPORTS Mat;
class CV_EXPORTS MatExpr;

class CV_EXPORTS UMat;

class CV_EXPORTS SparseMat;
typedef Mat MatND;

template<typename _Tp> class Mat_;
template<typename _Tp> class SparseMat_;

class CV_EXPORTS MatConstIterator;
class CV_EXPORTS SparseMatIterator;
class CV_EXPORTS SparseMatConstIterator;
template<typename _Tp> class MatIterator_;
template<typename _Tp> class MatConstIterator_;
template<typename _Tp> class SparseMatIterator_;
template<typename _Tp> class SparseMatConstIterator_;

namespace ogl
{
    class CV_EXPORTS Buffer;
    class CV_EXPORTS Texture2D;
    class CV_EXPORTS Arrays;
}

namespace cuda
{
    class CV_EXPORTS GpuMat;
    class CV_EXPORTS HostMem;
    class CV_EXPORTS Stream;
    class CV_EXPORTS Event;
}

namespace cudev
{
    template <typename _Tp> class GpuMat_;
}

//! @endcond

} // cv::

#endif // OPENCV_CORE_FWDDECL_HPP
