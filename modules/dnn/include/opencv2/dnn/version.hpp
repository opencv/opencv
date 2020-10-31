// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_VERSION_HPP
#define OPENCV_DNN_VERSION_HPP

/// Use with major OpenCV version only.
#define OPENCV_DNN_API_VERSION 20201117

#if !defined CV_DOXYGEN && !defined CV_STATIC_ANALYSIS && !defined CV_DNN_DONT_ADD_INLINE_NS
#define CV__DNN_INLINE_NS __CV_CAT(dnn4_v, OPENCV_DNN_API_VERSION)
#define CV__DNN_INLINE_NS_BEGIN namespace CV__DNN_INLINE_NS {
#define CV__DNN_INLINE_NS_END }
namespace cv { namespace dnn { namespace CV__DNN_INLINE_NS { } using namespace CV__DNN_INLINE_NS; }}
#else
#define CV__DNN_INLINE_NS_BEGIN
#define CV__DNN_INLINE_NS_END
#endif

#endif  // OPENCV_DNN_VERSION_HPP
