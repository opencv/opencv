#include "../precomp.hpp"
#include "utils.hpp"
#include "opencv2/3d/odometry.hpp"

#include "odometry_icp.hpp"
#include "odometry_rgb.hpp"
#include "odometry_rgbd.hpp"

namespace cv
{

Odometry::Odometry(OdometryType otype, OdometrySettings settings, OdometryAlgoType algtype)
{
	switch (otype)
	{
	case OdometryType::ICP:
		this->odometry = makePtr<OdometryICP>(settings, algtype);
		break;
	case OdometryType::RGB:
		this->odometry = makePtr<OdometryRGB>(settings, algtype);
		break;
	case OdometryType::RGBD:
		this->odometry = makePtr<OdometryRGBD>(settings, algtype);
		break;
	default:
		//CV_Error(Error::StsInternal,
		//	"Incorrect OdometryType, you are able to use only { ICP, RGB, RGBD }");
		break;
	}
	
}

Odometry::~Odometry()
{
}

bool Odometry::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	this->odometry->prepareFrames(srcFrame, dstFrame);
	return true;
}

bool Odometry::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt)
{
	this->odometry->compute(srcFrame, dstFrame, Rt);
	return true;
}

}
