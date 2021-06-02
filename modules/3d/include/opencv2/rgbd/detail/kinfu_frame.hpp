// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include <opencv2/core/affine.hpp>

namespace cv {
namespace detail {

CV_EXPORTS_W void renderPointsNormals(InputArray _points, InputArray _normals, OutputArray image, cv::Affine3f lightPose);
CV_EXPORTS_W void renderPointsNormalsColors(InputArray _points, InputArray _normals, InputArray _colors, OutputArray image, Affine3f lightPose);

//TODO: no dependencies outside already, can be removed from public interface to internal one
CV_EXPORTS_W void makeFrameFromDepth(InputArray depth, OutputArray pyrPoints, OutputArray pyrNormals,
                                     const Matx33f intr, int levels, float depthFactor,
                                     float sigmaDepth, float sigmaSpatial, int kernelSize,
                                     float truncateThreshold);
CV_EXPORTS_W void makeColoredFrameFromDepth(InputArray _depth, InputArray _rgb,
                                            OutputArray pyrPoints, OutputArray pyrNormals, OutputArray pyrColors,
                                            const Matx33f intr, const Matx33f rgb_intr, int levels, float depthFactor,
                                            float sigmaDepth, float sigmaSpatial, int kernelSize,
                                            float truncateThreshold);
CV_EXPORTS_W void buildPyramidPointsNormals(InputArray _points, InputArray _normals,
                                            OutputArrayOfArrays pyrPoints, OutputArrayOfArrays pyrNormals,
                                            int levels);
//TODO: up to this

} // namespace detail
} // namespace cv
#endif
