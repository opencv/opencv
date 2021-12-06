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

    virtual void setMaxWeight(int val) = 0;
    virtual int  getMaxWeight() const = 0;

    virtual void setZFirstMemOrder(bool val) = 0;
    virtual bool getZFirstMemOrder() const = 0;

    virtual void setPose(InputArray val) = 0;
    virtual void getPose(OutputArray val) const = 0;

    virtual void setResolution(InputArray val) = 0;
    virtual void getResolution(OutputArray val) const = 0;
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

    virtual void setMaxWeight(int val) override;
    virtual int  getMaxWeight() const override;

    virtual void setZFirstMemOrder(bool val) override;
    virtual bool getZFirstMemOrder() const override;

    virtual void setPose(InputArray val) override;
    virtual void getPose(OutputArray val) const override;

    virtual void setResolution(InputArray val) override;
    virtual void getResolution(OutputArray val) const override;

private:
    float   voxelSize;
    Matx44f pose;
    float   raycastStepFactor;
    float   truncDist;
    int     maxWeight;
    Point3i resolution;
    bool    zFirstMemOrder;

public:
    class DefaultSets {
    public:
        const Affine3f volumePose = Affine3f().translate(Vec3f(-volSize / 2.f, -volSize / 2.f, 0.5f));
        const Matx44f pose = volumePose.matrix;
        const Point3i resolution = Vec3i::all(128);
        static constexpr float volSize = 3.f;
        static constexpr float voxelSize = volSize / 128.f; //meters
        static constexpr float raycastStepFactor = 0.75f;
        static constexpr float truncDist = 2 * voxelSize;
        static const int maxWeight = 64;
        static const bool zFirstMemOrder = true;
    };
};

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

}
VolumeSettingsImpl::~VolumeSettingsImpl() {}


VolumeSettings::VolumeSettings()
{
    this->impl = makePtr<VolumeSettingsImpl>();
}

VolumeSettings::~VolumeSettings() {}

void  VolumeSettings::setVoxelSize(float val) { this->setVoxelSize(val); };
float VolumeSettings::getVoxelSize() const { return this->getVoxelSize(); };
void  VolumeSettings::setRaycastStepFactor(float val) { this->setRaycastStepFactor(val); };
float VolumeSettings::getRaycastStepFactor() const { return this->getRaycastStepFactor(); };
void  VolumeSettings::setTruncDist(float val) { this->setTruncDist(val); };
float VolumeSettings::getTruncDist() const { return this->getTruncDist(); };
void VolumeSettings::setMaxWeight(int val) { this->setMaxWeight(val); };
int  VolumeSettings::getMaxWeight() const { return this->getMaxWeight(); };
void VolumeSettings::setZFirstMemOrder(bool val) { this->setZFirstMemOrder(val); };
bool VolumeSettings::getZFirstMemOrder() const { return this->getZFirstMemOrder(); };
void VolumeSettings::setPose(InputArray val) { this->setPose(val); };
void VolumeSettings::getPose(OutputArray val) const { this->getPose(val); };
void VolumeSettings::setResolution(InputArray val) { this->setResolution(val); };
void VolumeSettings::getResolution(OutputArray val) const { this->getResolution(val); };


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
        resolution = Point3i(val.getMat());
    }
}

void VolumeSettingsImpl::getResolution(OutputArray val) const
{
    Mat(this->resolution).copyTo(val);
}


}
