// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"

#include <opencv2/core/ocl.hpp>

#include "utils.hpp"

namespace cv
{

OdometryFrame::OdometryFrame(InputArray depth, InputArray image, InputArray mask, InputArray normals)
{
    this->impl = makePtr<OdometryFrame::Impl>();
    if (!image.empty())
    {
        image.copyTo(this->impl->image);
    }
    if (!depth.empty())
    {
        depth.copyTo(this->impl->depth);
    }
    if (!mask.empty())
    {
        mask.copyTo(this->impl->mask);
    }
    if (!normals.empty())
    {
        normals.copyTo(this->impl->normals);
    }
}

void OdometryFrame::getImage(OutputArray image) const { this->impl->getImage(image); }
void OdometryFrame::getGrayImage(OutputArray image) const { this->impl->getGrayImage(image); }
void OdometryFrame::getDepth(OutputArray depth) const { this->impl->getDepth(depth); }
void OdometryFrame::getProcessedDepth(OutputArray depth) const { this->impl->getProcessedDepth(depth); }
void OdometryFrame::getMask(OutputArray mask) const { this->impl->getMask(mask); }
void OdometryFrame::getNormals(OutputArray normals) const { this->impl->getNormals(normals); }

int OdometryFrame::getPyramidLevels() const { return this->impl->getPyramidLevels(); }

void OdometryFrame::getPyramidAt(OutputArray img, OdometryFramePyramidType pyrType, size_t level) const
{
    this->impl->getPyramidAt(img, pyrType, level);
}

void OdometryFrame::Impl::getImage(OutputArray _image) const
{
    _image.assign(this->image);
}

void OdometryFrame::Impl::getGrayImage(OutputArray _image) const
{
    _image.assign(this->imageGray);
}

void OdometryFrame::Impl::getDepth(OutputArray _depth) const
{
    _depth.assign(this->depth);
}

void OdometryFrame::Impl::getProcessedDepth(OutputArray _depth) const
{
    _depth.assign(this->scaledDepth);
}

void OdometryFrame::Impl::getMask(OutputArray _mask) const
{
    _mask.assign(this->mask);
}

void OdometryFrame::Impl::getNormals(OutputArray _normals) const
{
    _normals.assign(this->normals);
}

int OdometryFrame::Impl::getPyramidLevels() const
{
    // all pyramids should have the same size
    for (const auto& p : this->pyramids)
    {
        if (!p.empty())
            return (int)(p.size());
    }
    return 0;
}


void OdometryFrame::Impl::getPyramidAt(OutputArray _img, OdometryFramePyramidType pyrType, size_t level) const
{
    CV_Assert(pyrType < OdometryFramePyramidType::N_PYRAMIDS);
    if (level < pyramids[pyrType].size())
        _img.assign(pyramids[pyrType][level]);
    else
        _img.clear();
}

}
