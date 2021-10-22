// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef ODOMETRY_RGBD_HPP
#define ODOMETRY_RGBD_HPP

#include "../precomp.hpp"
#include "utils.hpp"
namespace cv
{
class OdometryRGBD : public OdometryImpl
{
private:
	OdometrySettings settings;
    OdometryAlgoType algtype;

public:
	OdometryRGBD(OdometrySettings settings, OdometryAlgoType algtype);
	~OdometryRGBD();

    virtual OdometryFrame createOdometryFrame();
    virtual void prepareFrame(OdometryFrame frame);
	virtual void prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};
}
#endif //ODOMETRY_RGBD_HPP
