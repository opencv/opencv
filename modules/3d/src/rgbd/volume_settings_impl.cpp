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

    private:

    };

    class VolumeSettingsImpl : public VolumeSettings::Impl
    {
    public:
        VolumeSettingsImpl();
        ~VolumeSettingsImpl();

    private:

    };

    VolumeSettingsImpl::VolumeSettingsImpl() {}
    VolumeSettingsImpl::~VolumeSettingsImpl() {}


    VolumeSettings::VolumeSettings()
    {
        this->impl = makePtr<VolumeSettingsImpl>();
    }

    VolumeSettings::~VolumeSettings()
    {
    }

}
