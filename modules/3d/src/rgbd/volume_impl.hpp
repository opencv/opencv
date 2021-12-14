// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_VOLUME_IMPL_HPP
#define OPENCV_3D_VOLUME_IMPL_HPP

#include <iostream>

#include "../precomp.hpp"

namespace cv
{

class Volume::Impl
{
private:
    // TODO: make debug function, which show histogram of volume points values
    // make this function run with debug lvl == 10
public:
    Impl(const VolumeSettings& settings);
    virtual ~Impl() {};

    virtual void integrate(const OdometryFrame& frame, InputArray pose) = 0;
    virtual void integrate(InputArray frame, InputArray pose) = 0;
    virtual void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const = 0;
    virtual void raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const = 0;

    virtual void fetchNormals() const = 0;
    virtual void fetchPointsNormals() const = 0;

    virtual void reset() = 0;
    virtual int getVisibleBlocks() const = 0;
    virtual size_t getTotalVolumeUnits() const = 0;

public:
    const VolumeSettings& settings;
};


class TsdfVolume : public Volume::Impl
{
public:
    TsdfVolume(const VolumeSettings& settings);
    ~TsdfVolume();

    virtual void integrate(const OdometryFrame& frame, InputArray pose) override;
    virtual void integrate(InputArray frame, InputArray pose) override;
    virtual void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const override;
    virtual void raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const override;

    virtual void fetchNormals() const override;
    virtual void fetchPointsNormals() const override;

    virtual void reset() override;
    virtual int getVisibleBlocks() const override;
    virtual size_t getTotalVolumeUnits() const override;

public:
    Vec6f frameParams;
#ifndef HAVE_OPENCL
    Mat pixNorms;
    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Consist of Voxel elements
    Mat volume;
#else
    //temporary solution
    Mat cpu_pixNorms;
    Mat cpu_volume;
    UMat pixNorms;
    UMat volume;
#endif
};


class HashTsdfVolume : public Volume::Impl
{
public:
    HashTsdfVolume(const VolumeSettings& settings);
    ~HashTsdfVolume();

    virtual void integrate(const OdometryFrame& frame, InputArray pose) override;
    virtual void integrate(InputArray frame, InputArray pose) override;
    virtual void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const override;
    virtual void raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const override;

    virtual void fetchNormals() const override;
    virtual void fetchPointsNormals() const override;

    virtual void reset() override;
    virtual int getVisibleBlocks() const override;
    virtual size_t getTotalVolumeUnits() const override;
private:
};


class ColorTsdfVolume : public Volume::Impl
{
public:
    ColorTsdfVolume(const VolumeSettings& settings);
    ~ColorTsdfVolume();

    virtual void integrate(const OdometryFrame& frame, InputArray pose) override;
    virtual void integrate(InputArray frame, InputArray pose) override;
    virtual void raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const override;
    virtual void raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const override;

    virtual void fetchNormals() const override;
    virtual void fetchPointsNormals() const override;

    virtual void reset() override;
    virtual int getVisibleBlocks() const override;
    virtual size_t getTotalVolumeUnits() const override;
private:
};


Volume::Volume()
{
    VolumeSettings settings;
    this->impl = makePtr<TsdfVolume>(settings);
}
Volume::Volume(VolumeType vtype, const VolumeSettings& settings)
{
    std::cout << "Volume::Volume()" << std::endl;

    switch (vtype)
    {
    case VolumeType::TSDF:
        std::cout << "case VolumeType::TSDF" << std::endl;
        this->impl = makePtr<TsdfVolume>(settings);
        break;
    case VolumeType::HashTSDF:
        this->impl = makePtr<HashTsdfVolume>(settings);
        break;
    case VolumeType::ColorTSDF:
        this->impl = makePtr<ColorTsdfVolume>(settings);
        break;
    default:
        //CV_Error(Error::StsInternal,
        //	"Incorrect OdometryType, you are able to use only { ICP, RGB, RGBD }");
        std::cout << "Incorrect OdometryType, you are able to use only { ICP, RGB, RGBD }" << std::endl;
        break;
    }
}
Volume::~Volume() {}

void Volume::integrate(const OdometryFrame& frame, InputArray pose) { this->impl->integrate(frame, pose); }
void Volume::integrate(InputArray frame, InputArray pose) { this->impl->integrate(frame, pose); }
void Volume::raycast(InputArray cameraPose, int height, int width, OdometryFrame& outFrame) const { this->impl->raycast(cameraPose, height, width, outFrame); }
void Volume::raycast(InputArray cameraPose, int height, int width, OutputArray _points, OutputArray _normals) const { this->impl->raycast(cameraPose, height, width, _points, _normals); }
void Volume::fetchNormals() const { this->impl->fetchNormals(); }
void Volume::fetchPointsNormals() const { this->impl->fetchPointsNormals(); }
void Volume::reset() { this->impl->reset(); }
int Volume::getVisibleBlocks() const { return this->impl->getVisibleBlocks(); }
size_t Volume::getTotalVolumeUnits() const { return this->impl->getTotalVolumeUnits(); }


}

#endif // !OPENCV_3D_VOLUME_IMPL_HPP
