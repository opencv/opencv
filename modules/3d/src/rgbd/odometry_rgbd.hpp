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

public:
	OdometryRGBD(OdometrySettings settings);
	~OdometryRGBD();

	virtual OdometryFrame createOdometryFrame() override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt, OdometryAlgoType algtype) const override;
};
}
#endif //ODOMETRY_RGBD_HPP
