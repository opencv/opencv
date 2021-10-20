#include "odometry_icp.hpp"
#include "odometry_functions.hpp"

namespace cv
{

OdometryICP::OdometryICP(OdometrySettings _settings, OdometryAlgoType _algtype)
{
	this->settings = _settings;
    this->algtype = _algtype;
}

OdometryICP::~OdometryICP()
{
}

bool OdometryICP::prepareFrame(OdometryFrame frame)
{
	return prepareICPFrame(frame, frame, this->settings);
}

bool OdometryICP::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	return prepareICPFrame(srcFrame, dstFrame, this->settings);
}

bool OdometryICP::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
	//std::cout << "OdometryICP::compute()" << std::endl;
	Matx33f cameraMatrix;
	settings.getCameraMatrix(cameraMatrix);
	std::vector<int> iterCounts;
	Mat miterCounts;
	settings.getIterCounts(miterCounts);
	for (int i = 0; i < miterCounts.size().height; i++)
		iterCounts.push_back(miterCounts.at<int>(i));
	bool isCorrect = RGBDICPOdometryImpl(Rt, Mat(), srcFrame, dstFrame, cameraMatrix,
		this->settings.getMaxDepthDiff(), this->settings.getAngleThreshold(),
		iterCounts, this->settings.getMaxTranslation(),
		this->settings.getMaxRotation(), settings.getSobelScale(),
		OdometryType::DEPTH, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
    return isCorrect;
}

}
