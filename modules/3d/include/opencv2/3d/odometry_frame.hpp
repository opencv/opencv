// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_ODOMETRY_FRAME_HPP
#define OPENCV_3D_ODOMETRY_FRAME_HPP

#include <opencv2/core.hpp>

namespace cv
{
/** Indicates what pyramid is to access using getPyramidAt() method:
*/

enum OdometryFramePyramidType
{
    PYR_IMAGE = 0,    //!< The pyramid of grayscale images
    PYR_DEPTH = 1,    //!< The pyramid of depth images
    PYR_MASK = 2,     //!< The pyramid of masks
    PYR_CLOUD = 3,    //!< The pyramid of point clouds, produced from the pyramid of depths
    PYR_DIX = 4,      //!< The pyramid of dI/dx derivative images
    PYR_DIY = 5,      //!< The pyramid of dI/dy derivative images
    PYR_TEXMASK = 6,  //!< The pyramid of "textured" masks (i.e. additional masks for normals or grayscale images)
    PYR_NORM = 7,     //!< The pyramid of normals
    PYR_NORMMASK = 8, //!< The pyramid of normals masks
    N_PYRAMIDS
};

class CV_EXPORTS_W OdometryFrame
{
public:
    //TODO: add to docs: check image channels, if 3 or 4 then do cvtColor(BGR(A)2GRAY)
    OdometryFrame(InputArray image = noArray(), InputArray depth = noArray(), InputArray mask = noArray(), InputArray normals = noArray());
    ~OdometryFrame() {};

    void getImage(OutputArray image) const;
    void getGrayImage(OutputArray image) const;
    void getDepth(OutputArray depth) const;
    void getScaledDepth(OutputArray depth) const;
    void getMask(OutputArray mask) const;
    void getNormals(OutputArray normals) const;

    size_t getPyramidLevels(OdometryFramePyramidType oftype) const;
    void getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) const;

    class Impl;
    Ptr<Impl> impl;
};
}
#endif // !OPENCV_3D_ODOMETRY_FRAME_HPP
