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

public:
	OdometryRGB(OdometrySettings settings);
	~OdometryRGB();

	virtual OdometryFrame createOdometryFrame() override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt, OdometryAlgoType algtype) const override;
};
}
#endif //ODOMETRY_RGB_HPP
