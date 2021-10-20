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

    virtual bool prepareFrame(OdometryFrame frame);
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame);
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const override;
};
}
#endif //ODOMETRY_RGB_HPP
