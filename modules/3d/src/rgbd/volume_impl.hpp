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

    public:
        Impl() {};
        virtual ~Impl() {};

        virtual void integrate() = 0;
        virtual void raycast() const = 0;

        virtual void fetchNormals() const = 0;
        virtual void fetchPointsNormals() const = 0;

        virtual void reset() = 0;
        virtual int getVisibleBlocks() const = 0;
        virtual size_t getTotalVolumeUnits() const = 0;
    };


    class TsdfVolume : public Volume::Impl
    {
    public:
        TsdfVolume(VolumeSettings settings);
        ~TsdfVolume();

        virtual void integrate() override;
        virtual void raycast() const override;

        virtual void fetchNormals() const override;
        virtual void fetchPointsNormals() const override;

        virtual void reset() override;
        virtual int getVisibleBlocks() const override;
        virtual size_t getTotalVolumeUnits() const override;
    private:
        VolumeSettings settings;
    };


    class HashTsdfVolume : public Volume::Impl
    {
    public:
        HashTsdfVolume(VolumeSettings settings);
        ~HashTsdfVolume();

        virtual void integrate() override;
        virtual void raycast() const override;

        virtual void fetchNormals() const override;
        virtual void fetchPointsNormals() const override;

        virtual void reset() override;
        virtual int getVisibleBlocks() const override;
        virtual size_t getTotalVolumeUnits() const override;
    private:
        VolumeSettings settings;
    };


    class ColorTsdfVolume : public Volume::Impl
    {
    public:
        ColorTsdfVolume(VolumeSettings settings);
        ~ColorTsdfVolume();

        virtual void integrate() override;
        virtual void raycast() const override;

        virtual void fetchNormals() const override;
        virtual void fetchPointsNormals() const override;

        virtual void reset() override;
        virtual int getVisibleBlocks() const override;
        virtual size_t getTotalVolumeUnits() const override;
    private:
        VolumeSettings settings;
    };


    Volume::Volume()
    {
        VolumeSettings settings;
        this->impl = makePtr<TsdfVolume>(settings);
    }
    Volume::Volume(VolumeType vtype, VolumeSettings settings)
    {
        switch (vtype)
        {
        case VolumeType::TSDF:
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
            break;
        }
    }
    Volume::~Volume() {}

    void Volume::integrate() { this->impl->integrate(); }
    void Volume::raycast() const { this->impl->raycast(); }
    void Volume::fetchNormals() const { this->impl->fetchNormals(); }
    void Volume::fetchPointsNormals() const { this->impl->fetchPointsNormals(); }
    void Volume::reset() { this->impl->reset(); }
    int Volume::getVisibleBlocks() const { return this->impl->getVisibleBlocks(); }
    size_t Volume::getTotalVolumeUnits() const { return this->impl->getTotalVolumeUnits(); }


}

#endif // !OPENCV_3D_VOLUME_IMPL_HPP
