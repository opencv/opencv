// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "../precomp.hpp"

namespace cv
{
class VolumeSettings::Impl
{
public:
    Impl() {};
    virtual ~Impl() {};

    virtual void  setWidth(int  val) = 0;
    virtual int   getWidth() const = 0;
    virtual void  setHeight(int  val) = 0;
    virtual int   getHeight() const = 0;
    virtual void  setDepthFactor(float val) = 0;
    virtual float getDepthFactor() const = 0;
    virtual void  setVoxelSize(float  val) = 0;
    virtual float getVoxelSize() const = 0;
    virtual void  setTruncatedDistance(float val) = 0;
    virtual float getTruncatedDistance() const = 0;
    virtual void  setMaxWeight(int val) = 0;
    virtual int   getMaxWeight() const = 0;
    virtual void  setRaycastStepFactor(float val) = 0;
    virtual float getRaycastStepFactor() const = 0;
    virtual void  setZFirstMemOrder(bool val) = 0;
    virtual bool  getZFirstMemOrder() const = 0;

    virtual void setVolumePose(InputArray val) = 0;
    virtual void getVolumePose(OutputArray val) const = 0;
    virtual void setVolumeResolution(InputArray val) = 0;
    virtual void getVolumeResolution(OutputArray val) const = 0;
    virtual void setCameraIntrinsics(InputArray val) = 0;
    virtual void getCameraIntrinsics(OutputArray val) const = 0;
};

class VolumeSettingsImpl : public VolumeSettings::Impl
{
public:
    VolumeSettingsImpl();
    ~VolumeSettingsImpl();

    virtual void  setWidth(int  val) override;
    virtual int   getWidth() const override;
    virtual void  setHeight(int  val) override;
    virtual int   getHeight() const override;
    virtual void  setDepthFactor(float val) override;
    virtual float getDepthFactor() const override;
    virtual void  setVoxelSize(float  val) override;
    virtual float getVoxelSize() const override;
    virtual void  setTruncatedDistance(float val) override;
    virtual float getTruncatedDistance() const override;
    virtual void  setMaxWeight(int val) override;
    virtual int   getMaxWeight() const override;
    virtual void  setRaycastStepFactor(float val) override;
    virtual float getRaycastStepFactor() const override;
    virtual void  setZFirstMemOrder(bool val) override;
    virtual bool  getZFirstMemOrder() const override;

    virtual void setVolumePose(InputArray val) override;
    virtual void getVolumePose(OutputArray val) const override;
    virtual void setVolumeResolution(InputArray val) override;
    virtual void getVolumeResolution(OutputArray val) const override;
    virtual void setCameraIntrinsics(InputArray val) override;
    virtual void getCameraIntrinsics(OutputArray val) const override;

private:
    int   width;
    int   height;
    float depthFactor;
    float voxelSize;
    float truncatedDistance;
    int   maxWeight;
    float raycastStepFactor;
    bool  zFirstMemOrder;

    Matx44f volumePose;
    Point3i volumeResolution;
    Matx33f cameraIntrinsics;

public:
    // duplicate classes for all volumes
    class DefaultSets {
    public:
        static const int width  = 640;
        static const int height = 480;
        static constexpr float fx = 525.f; // focus point x axis
        static constexpr float fy = 525.f; // focus point y axis
        static constexpr float cx = float(width) / 2.f - 0.5f;  // central point x axis
        static constexpr float cy = float(height) / 2.f - 0.5f; // central point y axis
        static constexpr float depthFactor = 5000.f; // 5000 for the 16-bit PNG files, 1 for the 32-bit float images in the ROS bag files
        static constexpr float volumeSize = 3.f; // meters
        static constexpr float voxelSize = volumeSize / 128.f; //meters
        static constexpr float truncatedDistance = 2 * voxelSize;
        static const int maxWeight = 64; // number of frames
        static constexpr float raycastStepFactor = 0.75f;
        static const bool zFirstMemOrder = true; // order of voxels in volume

        const Matx33f  cameraIntrinsics = Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1); // camera settings
        const Affine3f volumePose = Affine3f().translate(Vec3f(-volumeSize / 2.f, -volumeSize / 2.f, 0.5f));
        const Matx44f  volumePoseMatrix = volumePose.matrix;
        const Point3i  volumeResolution = Vec3i::all(128); //number of voxels
    };
};


VolumeSettings::VolumeSettings()
{
    this->impl = makePtr<VolumeSettingsImpl>();
}

VolumeSettings::~VolumeSettings() {}

void  VolumeSettings::setWidth(int val) { this->impl->setWidth(val); };
int   VolumeSettings::getWidth() const { return this->impl->getWidth(); };
void  VolumeSettings::setHeight(int val) { this->impl->setHeight(val); };
int   VolumeSettings::getHeight() const { return this->impl->getHeight(); };
void  VolumeSettings::setVoxelSize(float val) { this->impl->setVoxelSize(val); };
float VolumeSettings::getVoxelSize() const { return this->impl->getVoxelSize(); };
void  VolumeSettings::setRaycastStepFactor(float val) { this->impl->setRaycastStepFactor(val); };
float VolumeSettings::getRaycastStepFactor() const { return this->impl->getRaycastStepFactor(); };
void  VolumeSettings::setTruncatedDistance(float val) { this->impl->setTruncatedDistance(val); };
float VolumeSettings::getTruncatedDistance() const { return this->impl->getTruncatedDistance(); };
void  VolumeSettings::setDepthFactor(float val) { this->impl->setDepthFactor(val); };
float VolumeSettings::getDepthFactor() const { return this->impl->getDepthFactor(); };
void  VolumeSettings::setMaxWeight(int val) { this->impl->setMaxWeight(val); };
int   VolumeSettings::getMaxWeight() const { return this->impl->getMaxWeight(); };
void  VolumeSettings::setZFirstMemOrder(bool val) { this->impl->setZFirstMemOrder(val); };
bool  VolumeSettings::getZFirstMemOrder() const { return this->impl->getZFirstMemOrder(); };

void VolumeSettings::setVolumePose(InputArray val) { this->impl->setVolumePose(val); };
void VolumeSettings::getVolumePose(OutputArray val) const { this->impl->getVolumePose(val); };
void VolumeSettings::setVolumeResolution(InputArray val) { this->impl->setVolumeResolution(val); };
void VolumeSettings::getVolumeResolution(OutputArray val) const { this->impl->getVolumeResolution(val); };
void VolumeSettings::setCameraIntrinsics(InputArray val) { this->impl->setCameraIntrinsics(val); };
void VolumeSettings::getCameraIntrinsics(OutputArray val) const { this->impl->getCameraIntrinsics(val); };


VolumeSettingsImpl::VolumeSettingsImpl()
{
    DefaultSets ds;
    this->width = ds.width;
    this->height = ds.height;
    this->depthFactor = ds.depthFactor;
    this->voxelSize = ds.voxelSize;
    this->truncatedDistance = ds.truncatedDistance;
    this->maxWeight = ds.maxWeight;
    this->raycastStepFactor = ds.raycastStepFactor;
    this->zFirstMemOrder = ds.zFirstMemOrder;

    this->volumePose = ds.volumePoseMatrix;
    this->volumeResolution = ds.volumeResolution;
    this->cameraIntrinsics = ds.cameraIntrinsics;

}
VolumeSettingsImpl::~VolumeSettingsImpl() {}


void VolumeSettingsImpl::setWidth(int val)
{
    this->width = val;
}

int VolumeSettingsImpl::getWidth() const
{
    return this->width;
}

void VolumeSettingsImpl::setHeight(int val)
{
    this->height = val;
}

int VolumeSettingsImpl::getHeight() const
{
    return this->height;
}

void VolumeSettingsImpl::setDepthFactor(float val)
{
    this->depthFactor = val;
}

float VolumeSettingsImpl::getDepthFactor() const
{
    return this->depthFactor;
}

void VolumeSettingsImpl::setVoxelSize(float  val)
{
    this->voxelSize = val;
}

float VolumeSettingsImpl::getVoxelSize() const
{
    return this->voxelSize;
}

void VolumeSettingsImpl::setTruncatedDistance(float val)
{
    this->truncatedDistance = val;
}

float VolumeSettingsImpl::getTruncatedDistance() const
{
    return this->truncatedDistance;
}

void VolumeSettingsImpl::setMaxWeight(int val)
{
    this->maxWeight = val;
}

int VolumeSettingsImpl::getMaxWeight() const
{
    return this->maxWeight;
}

void VolumeSettingsImpl::setRaycastStepFactor(float val)
{
    this->raycastStepFactor = val;
}

float VolumeSettingsImpl::getRaycastStepFactor() const
{
    return this->raycastStepFactor;
}

void VolumeSettingsImpl::setZFirstMemOrder(bool val)
{
    this->zFirstMemOrder = val;
}

bool VolumeSettingsImpl::getZFirstMemOrder() const
{
    return this->zFirstMemOrder;
}


void VolumeSettingsImpl::setVolumePose(InputArray val)
{
    if (!val.empty())
    {
        val.copyTo(this->volumePose);
    }
}

void VolumeSettingsImpl::getVolumePose(OutputArray val) const
{
    Mat(this->volumePose).copyTo(val);
}

void VolumeSettingsImpl::setVolumeResolution(InputArray val)
{
    if (!val.empty())
    {
        this->volumeResolution = Point3i(val.getMat());
    }
}

void VolumeSettingsImpl::getVolumeResolution(OutputArray val) const
{
    Mat(this->volumeResolution).copyTo(val);
}

void VolumeSettingsImpl::setCameraIntrinsics(InputArray val)
{
    if (!val.empty())
    {
        this->cameraIntrinsics = Matx33f(val.getMat());
    }
}

void VolumeSettingsImpl::getCameraIntrinsics(OutputArray val) const
{
    Mat(this->cameraIntrinsics).copyTo(val);
}


}
