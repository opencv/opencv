// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef ODOMETRY_RGB_HPP
#define ODOMETRY_RGB_HPP

#include "../precomp.hpp"
#include "utils.hpp"
namespace cv
{
class OdometryRGB : public OdometryImpl
{
private:
	OdometrySettings settings;
    OdometryAlgoType algtype;

public:
	OdometryRGB(OdometrySettings settings, OdometryAlgoType algtype);
	~OdometryRGB();

    virtual OdometryFrame createOdometryFrame();
    virtual void prepareFrame(OdometryFrame frame);
	virtual void prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};
}
#endif //ODOMETRY_RGB_HPP
