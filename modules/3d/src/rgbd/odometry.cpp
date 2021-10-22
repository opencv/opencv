// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "utils.hpp"
#include "opencv2/3d/odometry.hpp"

#include "odometry_icp.hpp"
#include "odometry_rgb.hpp"
#include "odometry_rgbd.hpp"

namespace cv
{

Odometry::Odometry()
{
    OdometrySettings settings;
	this->odometry = makePtr<OdometryICP>(settings, OdometryAlgoType::COMMON);
}

Odometry::Odometry(OdometryType otype, OdometrySettings settings, OdometryAlgoType algtype)
{
	switch (otype)
	{
	case OdometryType::DEPTH:
		this->odometry = makePtr<OdometryICP>(settings, algtype);
		break;
	case OdometryType::RGB:
		this->odometry = makePtr<OdometryRGB>(settings, algtype);
		break;
	case OdometryType::RGB_DEPTH:
		this->odometry = makePtr<OdometryRGBD>(settings, algtype);
		break;
	default:
		CV_Error(Error::StsInternal,
			"Incorrect OdometryType, you are able to use only { ICP, RGB, RGBD }");
		break;
	}
	
}

Odometry::~Odometry()
{
}

OdometryFrame Odometry::createOdometryFrame()
{
    return this->odometry->createOdometryFrame();
}

OdometryFrame Odometry::createOdometryFrame(OdometryFrameStoreType matType)
{
    return OdometryFrame(matType);
}

void Odometry::prepareFrame(OdometryFrame frame)
{
	this->odometry->prepareFrame(frame);
}

void Odometry::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	this->odometry->prepareFrames(srcFrame, dstFrame);
}

bool Odometry::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt)
{
    this->prepareFrames(srcFrame, dstFrame);
	return this->odometry->compute(srcFrame, dstFrame, Rt);
}

}
