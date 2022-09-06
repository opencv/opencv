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

/**
 * @brief An object that keeps per-frame data for Odometry algorithms from user-provided images to algorithm-specific precalculated data.
 * When not empty, it contains a depth image, a mask of valid pixels and a set of pyramids generated from that data.
 * A BGR/Gray image and normals are optional.
 * OdometryFrame is made to be used together with Odometry class to reuse precalculated data between Rt data calculations.
 * A correct way to do that is to call Odometry::prepareFrames() on prev and next frames and then pass them to Odometry::compute() method.
 */
class CV_EXPORTS_W_SIMPLE OdometryFrame
{
public:
    /**
     * @brief Construct a new OdometryFrame object. All non-empty images should have the same size.
     *
     * @param depth A depth image, should be CV_8UC1
     * @param image An BGR or grayscale image (or noArray() if it's not required for used ICP algorithm).
     * Should be CV_8UC3 or CV_8C4 if it's BGR image or CV_8UC1 if it's grayscale. If it's BGR then it's converted to grayscale
     * image automatically.
     * @param mask A user-provided mask of valid pixels, should be CV_8UC1
     * @param normals A user-provided normals to the depth surface, should be CV_32FC4
     */
    CV_WRAP explicit OdometryFrame(InputArray depth = noArray(), InputArray image = noArray(), InputArray mask = noArray(), InputArray normals = noArray());
    ~OdometryFrame() {};

    /**
     * @brief Get the original user-provided BGR/Gray image
     *
     * @param image Output image
     */
    CV_WRAP void getImage(OutputArray image) const;
    /**
     * @brief Get the gray image generated from the user-provided BGR/Gray image
     *
     * @param image Output image
     */
    CV_WRAP void getGrayImage(OutputArray image) const;
    /**
     * @brief Get the original user-provided depth image
     *
     * @param depth Output image
     */
    CV_WRAP void getDepth(OutputArray depth) const;
    /**
     * @brief Get the depth image generated from the user-provided one after conversion, rescale or filtering for ICP algorithm needs
     *
     * @param depth Output image
     */
    CV_WRAP void getProcessedDepth(OutputArray depth) const;
    /**
     * @brief Get the valid pixels mask generated for the ICP calculations intersected with the user-provided mask
     *
     * @param mask Output image
     */
    CV_WRAP void getMask(OutputArray mask) const;
    /**
     * @brief Get the normals image either generated for the ICP calculations or user-provided
     *
     * @param normals Output image
     */
    CV_WRAP void getNormals(OutputArray normals) const;

    /**
     * @brief Get the amount of levels in pyramids (all of them if not empty should have the same number of levels)
     * or 0 if no pyramids were prepared yet
     */
    CV_WRAP int getPyramidLevels() const;
    /**
     * @brief Get the image generated for the ICP calculations from one of the pyramids specified by pyrType. Returns empty image if
     * the pyramid is empty or there's no such pyramid level
     *
     * @param img Output image
     * @param pyrType Type of pyramid
     * @param level Level in the pyramid
     */
    CV_WRAP void getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) const;

    class Impl;
    Ptr<Impl> impl;
};
}
#endif // !OPENCV_3D_ODOMETRY_FRAME_HPP
