// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include <iostream>
#include "volume_impl.hpp"

namespace cv
{

    TsdfVolume::TsdfVolume(VolumeSettings settings) { this->settings = settings; }
    TsdfVolume::~TsdfVolume() {}

    void TsdfVolume::integrate() { std::cout << "TsdfVolume::integrate()" << std::endl; }
    void TsdfVolume::raycast() const { std::cout << "TsdfVolume::raycast()" << std::endl; }

    void TsdfVolume::fetchNormals() const {}
    void TsdfVolume::fetchPointsNormals() const {}

    void TsdfVolume::reset() {}
    int TsdfVolume::getVisibleBlocks() const { return 1; }
    size_t TsdfVolume::getTotalVolumeUnits() const { return 1; }


    HashTsdfVolume::HashTsdfVolume(VolumeSettings settings) { this->settings = settings; }
    HashTsdfVolume::~HashTsdfVolume() {}

    void HashTsdfVolume::integrate() { std::cout << "HashTsdfVolume::integrate()" << std::endl; }
    void HashTsdfVolume::raycast() const { std::cout << "HashTsdfVolume::raycast()" << std::endl; }

    void HashTsdfVolume::fetchNormals() const {}
    void HashTsdfVolume::fetchPointsNormals() const {}

    void HashTsdfVolume::reset() {}
    int HashTsdfVolume::getVisibleBlocks() const { return 1; }
    size_t HashTsdfVolume::getTotalVolumeUnits() const { return 1; }


    ColorTsdfVolume::ColorTsdfVolume(VolumeSettings settings) { this->settings = settings; }
    ColorTsdfVolume::~ColorTsdfVolume() {}

    void ColorTsdfVolume::integrate() { std::cout << "ColorTsdfVolume::integrate()" << std::endl; }
    void ColorTsdfVolume::raycast() const { std::cout << "ColorTsdfVolume::raycast()" << std::endl; }

    void ColorTsdfVolume::fetchNormals() const {}
    void ColorTsdfVolume::fetchPointsNormals() const {}

    void ColorTsdfVolume::reset() {}
    int ColorTsdfVolume::getVisibleBlocks() const { return 1; }
    size_t ColorTsdfVolume::getTotalVolumeUnits() const { return 1; }

}
