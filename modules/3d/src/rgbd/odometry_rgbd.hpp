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

	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};
}
#endif //ODOMETRY_RGBD_HPP
