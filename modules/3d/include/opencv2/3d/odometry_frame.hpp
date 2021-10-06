#ifndef ODOMETRY_FRAME_HPP
#define ODOMETRY_FRAME_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>

namespace cv
{
/** Indicates what pyramid is to access using get/setPyramid... methods:
* @param PYR_IMAGE The pyramid of RGB images
* @param PYR_DEPTH The pyramid of depth images
* @param PYR_MASK  The pyramid of masks
* @param PYR_CLOUD The pyramid of point clouds, produced from the pyramid of depths
* @param PYR_DIX   The pyramid of dI/dx derivative images
* @param PYR_DIY   The pyramid of dI/dy derivative images
* @param PYR_TEXMASK The pyramid of textured masks
* @param PYR_NORM  The pyramid of normals
* @param PYR_NORMMASK The pyramid of normals masks
*/

enum OdometryFramePyramidType
{
    PYR_IMAGE = 0, PYR_DEPTH = 1, PYR_MASK = 2, PYR_CLOUD = 3, PYR_DIX = 4, PYR_DIY = 5, PYR_TEXMASK = 6, PYR_NORM = 7, PYR_NORMMASK = 8,
    N_PYRAMIDS
};

enum OdometryFrameStoreType
{
    MAT  = 0,
    UMAT = 1
};

class OdometryFrameImpl
{
public:
    OdometryFrameImpl() {};
    ~OdometryFrameImpl() {};
    virtual void setImage(InputArray  image) = 0;
    virtual void getImage(OutputArray image) = 0;
    virtual void getGrayImage(OutputArray image) = 0;
    virtual void setDepth(InputArray  depth) = 0;
    virtual void getDepth(OutputArray depth) = 0;
    virtual void setMask(InputArray  mask) = 0;
    virtual void getMask(OutputArray mask) = 0;
    virtual void setNormals(InputArray  normals) = 0;
    virtual void getNormals(OutputArray normals) = 0;
    virtual void   setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype) = 0;
    virtual void   setPyramidLevels(size_t _nLevels) = 0;
    virtual size_t getPyramidLevels(OdometryFramePyramidType oftype) = 0;
    virtual void setPyramidAt(InputArray  img,
        OdometryFramePyramidType pyrType, size_t level) = 0;
    virtual void getPyramidAt(OutputArray img,
        OdometryFramePyramidType pyrType, size_t level) = 0;
};

class OdometryFrame
{
private:
    Ptr<OdometryFrameImpl> odometryFrame;
public:
    OdometryFrame(OdometryFrameStoreType matType);
    ~OdometryFrame() {};
    void setImage(InputArray  image) { this->odometryFrame->setImage(image); }
    void getImage(OutputArray image) const { this->odometryFrame->getImage(image); }
    void getGrayImage(OutputArray image) const { this->odometryFrame->getGrayImage(image); }
    void setDepth(InputArray  depth) { this->odometryFrame->setDepth(depth); }
    void getDepth(OutputArray depth) const { this->odometryFrame->getDepth(depth); }
    void setMask(InputArray  mask) { this->odometryFrame->setMask(mask); }
    void getMask(OutputArray mask) const { this->odometryFrame->getMask(mask); }
    void setNormals(InputArray  normals) { this->odometryFrame->setNormals(normals); }
    void getNormals(OutputArray normals) const { this->odometryFrame->getNormals(normals); }
    void setPyramidLevel(size_t _nLevels, OdometryFramePyramidType oftype)
    {
        this->odometryFrame->setPyramidLevel(_nLevels, oftype);
    }
    void setPyramidLevels(size_t _nLevels) { this->odometryFrame->setPyramidLevels(_nLevels); }
    size_t getPyramidLevels(OdometryFramePyramidType oftype) const { return this->odometryFrame->getPyramidLevels(oftype); }
    void setPyramidAt(InputArray  img, OdometryFramePyramidType pyrType, size_t level)
    {
        this->odometryFrame->setPyramidAt(img, pyrType, level);
    }
    void getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) const
    {
        this->odometryFrame->getPyramidAt(img, pyrType, level);
    }
};
}
#endif // !ODOMETRY_FRAME_HPP
