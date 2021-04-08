// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include <opencv2/core/affine.hpp>
#include "utils.hpp"

namespace cv {

template<> class DataType<cv::Point3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 3,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<cv::Vec3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 3,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<cv::Vec4f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 4,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

namespace kinfu {

typedef cv::Vec4f ptype;
inline cv::Vec3f fromPtype(const ptype& x)
{
    return cv::Vec3f(x[0], x[1], x[2]);
}

inline ptype toPtype(const cv::Vec3f& x)
{
    return ptype(x[0], x[1], x[2], 0);
}

enum
{
    DEPTH_TYPE = DataType<depthType>::type,
    POINT_TYPE = DataType<ptype    >::type,
    COLOR_TYPE = DataType<ptype    >::type
};

typedef cv::Mat_< ptype > Points;
typedef Points Normals;
typedef Points Colors;

typedef cv::Mat_< depthType > Depth;

void renderPointsNormals(InputArray _points, InputArray _normals, OutputArray image, cv::Affine3f lightPose);
void renderPointsNormalsColors(InputArray _points, InputArray _normals, InputArray _colors, OutputArray image, Affine3f lightPose);
void makeFrameFromDepth(InputArray depth, OutputArray pyrPoints, OutputArray pyrNormals,
                        const Intr intr, int levels, float depthFactor,
                        float sigmaDepth, float sigmaSpatial, int kernelSize,
                        float truncateThreshold);
void makeColoredFrameFromDepth(InputArray _depth, InputArray _rgb,
                        OutputArray pyrPoints, OutputArray pyrNormals, OutputArray pyrColors,
                        const Intr intr, const Intr rgb_intr, int levels, float depthFactor,
                        float sigmaDepth, float sigmaSpatial, int kernelSize,
                        float truncateThreshold);
void buildPyramidPointsNormals(InputArray _points, InputArray _normals,
                               OutputArrayOfArrays pyrPoints, OutputArrayOfArrays pyrNormals,
                               int levels);

} // namespace kinfu
} // namespace cv
#endif
