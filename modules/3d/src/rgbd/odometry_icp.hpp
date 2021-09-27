#ifndef ODOMETRY_ICP_HPP
#define ODOMETRY_ICP_HPP

#include "../precomp.hpp"
#include "utils.hpp"
namespace cv
{
class OdometryICP : public OdometryImpl
{
private:
	OdometrySettings settings;

public:
	OdometryICP(OdometrySettings settings);
	~OdometryICP();

	virtual OdometryFrame createOdometryFrame() override;
	virtual bool prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame) override;
	virtual bool compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt, OdometryAlgoType algtype) const override;
};
}
#endif //ODOMETRY_ICP_HPP
