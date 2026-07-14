// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _CAP_LIBREALSENE_HPP_
#define _CAP_LIBREALSENE_HPP_

#ifdef HAVE_LIBREALSENSE

#include <librealsense2/rs.hpp>

namespace cv
{

class VideoCapture_LibRealsense : public IVideoCapture
{
public:
    VideoCapture_LibRealsense(int index);
    virtual ~VideoCapture_LibRealsense();

    virtual double getProperty(int propIdx) const CV_OVERRIDE;
    virtual bool setProperty(int propIdx, double propVal) CV_OVERRIDE;

    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int outputType, OutputArray frame) CV_OVERRIDE;
    virtual int getCaptureDomain() CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE;

protected:
    rs2::pipeline mPipe;
    rs2::frameset mData;
    rs2::align    mAlign;

    double getDepthGeneratorProperty(int propIdx) const;
    double getImageGeneratorProperty(int propIdx) const;
    double getIrGeneratorProperty(int propIdx) const;
    double getCommonProperty(int propIdx) const;
};

}

#endif
#endif
