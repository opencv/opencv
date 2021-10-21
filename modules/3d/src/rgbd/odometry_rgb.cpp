#include "odometry_functions.hpp"
#include "odometry_rgb.hpp"

namespace cv
{

OdometryRGB::OdometryRGB(OdometrySettings _settings, OdometryAlgoType _algtype)
{
	this->settings = _settings;
    this->algtype = _algtype;
}

OdometryRGB::~OdometryRGB()
{
}

OdometryFrame OdometryRGB::createOdometryFrame()
{
    return OdometryFrame(OdometryFrameStoreType::MAT);
}

bool OdometryRGB::prepareFrame(OdometryFrame frame)
{
	return prepareRGBFrame(frame, frame, this->settings);
}

bool OdometryRGB::prepareFrames(OdometryFrame srcFrame, OdometryFrame dstFrame)
{
	return prepareRGBFrame(srcFrame, dstFrame, this->settings);
}

bool OdometryRGB::compute(OdometryFrame srcFrame, OdometryFrame dstFrame, OutputArray Rt) const
{
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
		OdometryType::RGB, OdometryTransformType::RIGID_TRANSFORMATION, this->algtype);
	return isCorrect;
}

}
