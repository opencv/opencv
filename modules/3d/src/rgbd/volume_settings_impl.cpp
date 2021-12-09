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

    virtual void  setVoxelSize(float  val) = 0;
    virtual float getVoxelSize() const = 0;

    virtual void  setRaycastStepFactor(float val) = 0;
    virtual float getRaycastStepFactor() const = 0;

    virtual void  setTruncDist(float val) = 0;
    virtual float getTruncDist() const = 0;

    virtual void  setDepthFactor(float val) = 0;
    virtual float getDepthFactor() const = 0;

    virtual void setMaxWeight(int val) = 0;
    virtual int  getMaxWeight() const = 0;

    virtual void setZFirstMemOrder(bool val) = 0;
    virtual bool getZFirstMemOrder() const = 0;

    virtual void setPose(InputArray val) = 0;
    virtual void getPose(OutputArray val) const = 0;

    virtual void setResolution(InputArray val) = 0;
    virtual void getResolution(OutputArray val) const = 0;

    virtual void setIntrinsics(InputArray val) = 0;
    virtual void getIntrinsics(OutputArray val) const = 0;


};

class VolumeSettingsImpl : public VolumeSettings::Impl
{
public:
    VolumeSettingsImpl();
    ~VolumeSettingsImpl();

    virtual void  setVoxelSize(float  val) override;
    virtual float getVoxelSize() const override;

    virtual void  setRaycastStepFactor(float val) override;
    virtual float getRaycastStepFactor() const override;

    virtual void  setTruncDist(float val) override;
    virtual float getTruncDist() const override;

    virtual void  setDepthFactor(float val) override;
    virtual float getDepthFactor() const override;

    virtual void setMaxWeight(int val) override;
    virtual int  getMaxWeight() const override;

    virtual void setZFirstMemOrder(bool val) override;
    virtual bool getZFirstMemOrder() const override;

    virtual void setPose(InputArray val) override;
    virtual void getPose(OutputArray val) const override;

    virtual void setResolution(InputArray val) override;
    virtual void getResolution(OutputArray val) const override;

    virtual void setIntrinsics(InputArray val) override;
    virtual void getIntrinsics(OutputArray val) const override;

private:
    float   voxelSize;
    Matx44f pose;
    float   raycastStepFactor;
    float   truncDist;
    int     maxWeight;
    Point3i resolution;
    bool    zFirstMemOrder;
    Matx33f intrinsics;
    float   depthFactor;
public:
    // duplicate classes for all volumes
    class DefaultSets {
    public:
        static const int width  = 640;
        static const int height = 480;
        static constexpr float fx = 525.f;
        static constexpr float fy = 525.f;
        static constexpr float cx = float(width) / 2.f - 0.5f;
        static constexpr float cy = float(height) / 2.f - 0.5f;

        static constexpr float volSize = 3.f;

        const Matx33f intr = Matx33f(fx, 0, cx, 0, fy, cy, 0, 0, 1);
        const Affine3f volumePose = Affine3f().translate(Vec3f(-volSize / 2.f, -volSize / 2.f, 0.5f));
        const Matx44f pose = volumePose.matrix;
        const Point3i resolution = Vec3i::all(128); //number of voxels
        // 5000 for the 16-bit PNG files, 1 for the 32-bit float images in the ROS bag files
        static constexpr float depthFactor = 5000.f;

        static constexpr float voxelSize = volSize / 128.f; //meters
        static constexpr float raycastStepFactor = 0.75f;
        static constexpr float truncDist = 2 * voxelSize;
        static const int maxWeight = 64; //frames
        static const bool zFirstMemOrder = true;
    };
};


VolumeSettings::VolumeSettings()
{
    this->impl = makePtr<VolumeSettingsImpl>();
}

VolumeSettings::~VolumeSettings() {}

void  VolumeSettings::setVoxelSize(float val) { this->impl->setVoxelSize(val); };
float VolumeSettings::getVoxelSize() const { return this->impl->getVoxelSize(); };
void  VolumeSettings::setRaycastStepFactor(float val) { this->impl->setRaycastStepFactor(val); };
float VolumeSettings::getRaycastStepFactor() const { return this->impl->getRaycastStepFactor(); };
void  VolumeSettings::setTruncDist(float val) { this->impl->setTruncDist(val); };
float VolumeSettings::getTruncDist() const { return this->impl->getTruncDist(); };
void  VolumeSettings::setDepthFactor(float val) { this->impl->setDepthFactor(val); };
float VolumeSettings::getDepthFactor() const { return this->impl->getDepthFactor(); };
void VolumeSettings::setMaxWeight(int val) { this->impl->setMaxWeight(val); };
int  VolumeSettings::getMaxWeight() const { return this->impl->getMaxWeight(); };
void VolumeSettings::setZFirstMemOrder(bool val) { this->impl->setZFirstMemOrder(val); };
bool VolumeSettings::getZFirstMemOrder() const { return this->impl->getZFirstMemOrder(); };
void VolumeSettings::setPose(InputArray val) { this->impl->setPose(val); };
void VolumeSettings::getPose(OutputArray val) const { this->impl->getPose(val); };
void VolumeSettings::setResolution(InputArray val) { this->impl->setResolution(val); };
void VolumeSettings::getResolution(OutputArray val) const { this->impl->getResolution(val); };
void VolumeSettings::setIntrinsics(InputArray val) { this->impl->setIntrinsics(val); };
void VolumeSettings::getIntrinsics(OutputArray val) const { this->impl->getIntrinsics(val); };


VolumeSettingsImpl::VolumeSettingsImpl()
{
    DefaultSets ds;
    this->voxelSize = ds.voxelSize;
    this->pose = ds.pose;
    this->raycastStepFactor = ds.raycastStepFactor;
    this->truncDist = ds.truncDist;
    this->maxWeight = ds.maxWeight;
    this->resolution = ds.resolution;
    this->zFirstMemOrder = ds.zFirstMemOrder;
    this->intrinsics = ds.intr;
    this->depthFactor = ds.depthFactor;

}
VolumeSettingsImpl::~VolumeSettingsImpl() {}


void VolumeSettingsImpl::setVoxelSize(float  val)
{
    this->voxelSize = val;
}

float VolumeSettingsImpl::getVoxelSize() const
{
    return this->voxelSize;
}

void VolumeSettingsImpl::setRaycastStepFactor(float val)
{
    this->raycastStepFactor = val;
}

float VolumeSettingsImpl::getRaycastStepFactor() const
{
    return this->raycastStepFactor;
}

void VolumeSettingsImpl::setTruncDist(float val)
{
    this->truncDist = val;
}

float VolumeSettingsImpl::getTruncDist() const
{
    return this->truncDist;
}

void VolumeSettingsImpl::setDepthFactor(float val)
{
    this->depthFactor = val;
}

float VolumeSettingsImpl::getDepthFactor() const
{
    return this->depthFactor;
}

void VolumeSettingsImpl::setMaxWeight(int val)
{
    this->maxWeight = val;
}

int VolumeSettingsImpl::getMaxWeight() const
{
    return this->maxWeight;
}

void VolumeSettingsImpl::setZFirstMemOrder(bool val)
{
    this->zFirstMemOrder = val;
}

bool VolumeSettingsImpl::getZFirstMemOrder() const
{
    return this->zFirstMemOrder;
}

void VolumeSettingsImpl::setPose(InputArray val)
{
    if (!val.empty())
    {
        val.copyTo(this->pose);
    }
}

void VolumeSettingsImpl::getPose(OutputArray val) const
{
    Mat(this->pose).copyTo(val);
}

void VolumeSettingsImpl::setResolution(InputArray val)
{
    if (!val.empty())
    {
        this->resolution = Point3i(val.getMat());
    }
}

void VolumeSettingsImpl::getResolution(OutputArray val) const
{
    Mat(this->resolution).copyTo(val);
}

void VolumeSettingsImpl::setIntrinsics(InputArray val)
{
    if (!val.empty())
    {
        this->intrinsics = Matx33f(val.getMat());
    }
}

void VolumeSettingsImpl::getIntrinsics(OutputArray val) const
{
    Mat(this->intrinsics).copyTo(val);
}


}
