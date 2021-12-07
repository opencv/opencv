// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#ifndef OPENCV_3D_VOLUME_SETTINGS_HPP
#define OPENCV_3D_VOLUME_SETTINGS_HPP

#include <opencv2/core/cvstd.hpp>

namespace cv
{

class CV_EXPORTS_W VolumeSettings
{
public:
    VolumeSettings();
    ~VolumeSettings();

    void  setVoxelSize(float  val);
    float getVoxelSize() const;

    void  setRaycastStepFactor(float val);
    float getRaycastStepFactor() const;

    void  setTruncDist(float val);
    float getTruncDist() const;

    void  setDepthFactor(float val);
    float getDepthFactor() const;

    void setMaxWeight(int val);
    int  getMaxWeight() const;

    void setZFirstMemOrder(bool val);
    bool getZFirstMemOrder() const;

    void setPose(InputArray val);
    void getPose(OutputArray val) const;

    void setResolution(InputArray val);
    void getResolution(OutputArray val) const;

    void setIntrinsics(InputArray val);
    void getIntrinsics(OutputArray val) const;

    class Impl;
private:
    Ptr<Impl> impl;
};

}

#endif // !OPENCV_3D_VOLUME_SETTINGS_HPP
