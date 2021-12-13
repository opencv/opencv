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


    void  setWidth(int val);
    int   getWidth() const;
    void  setHeight(int val);
    int   getHeight() const;
    void  setDepthFactor(float val);
    float getDepthFactor() const;
    void  setVoxelSize(float val);
    float getVoxelSize() const;
    void  setTruncatedDistance(float val);
    float getTruncatedDistance() const;
    void  setMaxWeight(int val);
    int   getMaxWeight() const;
    void  setRaycastStepFactor(float val);
    float getRaycastStepFactor() const;
    void  setZFirstMemOrder(bool val);
    bool  getZFirstMemOrder() const;

    void setVolumePose(InputArray val);
    void getVolumePose(OutputArray val) const;
    void setVolumeResolution(InputArray val);
    void getVolumeResolution(OutputArray val) const;
    void setVolumeDimentions(InputArray val);
    void getVolumeDimentions(OutputArray val) const;
    void setCameraIntrinsics(InputArray val);
    void getCameraIntrinsics(OutputArray val) const;


    class Impl;
private:
    Ptr<Impl> impl;
};

}

#endif // !OPENCV_3D_VOLUME_SETTINGS_HPP
