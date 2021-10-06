#include "../precomp.hpp"
#include "utils.hpp"

#include <iostream>

namespace cv
{

const cv::Matx33f defaultCameraMatrix =
{ /* fx, 0, cx*/ 0, 0, 0,
  /* 0, fy, cy*/ 0, 0, 0,
  /* 0,  0,  0*/ 0, 0, 0 };
const std::vector<int> defaultIterCounts = { 7, 7, 7, 10 };

const float defaultMinDepth = 0.f;
const float defaultMaxDepth = 4.f;
const float defaultMaxDepthDiff = 0.07f;
const float defaultMaxPointsPart = 0.07f;

static const int defaultSobelSize = 3;
static const double defaultSobelScale = 1. / 8.;
static const int defaultNormalWinSize = 5;

const float defaultAngleThreshold = (float)(30. * CV_PI / 180.);
const float defaultMaxTranslation = 0.15f;
const float defaultMaxRotation = 15.f;

const float defaultMinGradientMagnitude = 10.f;
std::vector<float> defaultMinGradientMagnitudes = std::vector<float>(defaultIterCounts.size(), defaultMinGradientMagnitude);


class OdometrySettingsImplCommon : public OdometrySettingsImpl
{
public:
    OdometrySettingsImplCommon();
    ~OdometrySettingsImplCommon() {};
	virtual void setCameraMatrix(InputArray val);
	virtual void getCameraMatrix(OutputArray val) const;
	virtual void setIterCounts(InputArray val);
	virtual void getIterCounts(OutputArray val) const;

	virtual void  setMinDepth(float val);
	virtual float getMinDepth() const;
	virtual void  setMaxDepth(float val);
	virtual float getMaxDepth() const;
	virtual void  setMaxDepthDiff(float val);
	virtual float getMaxDepthDiff() const;
	virtual void  setMaxPointsPart(float val);
	virtual float getMaxPointsPart() const;

	virtual void setSobelSize(int val);
	virtual int  getSobelSize() const;
	virtual void   setSobelScale(double val);
	virtual double getSobelScale() const;
	virtual void setNormalWinSize(int val);
	virtual int  getNormalWinSize() const;

	virtual void  setAngleThreshold(float val);
	virtual float getAngleThreshold() const;
	virtual void  setMaxTranslation(float val);
	virtual float getMaxTranslation() const;
	virtual void  setMaxRotation(float val);
	virtual float getMaxRotation() const;

	virtual void  setMinGradientMagnitude(float val);
	virtual float getMinGradientMagnitude() const;
	virtual void setMinGradientMagnitudes(InputArray val);
	virtual void getMinGradientMagnitudes(OutputArray val) const;

private:
    Matx33f cameraMatrix;
	std::vector<int> iterCounts;

	float minDepth;
	float maxDepth;
	float maxDepthDiff;
	float maxPointsPart;

	int sobelSize;
	double sobelScale;
	int normalWinSize;

	float angleThreshold;
	float maxTranslation;
	float maxRotation;

	float minGradientMagnitude;
	std::vector<float> minGradientMagnitudes;
};


OdometrySettings::OdometrySettings()
{
	this->odometrySettings = makePtr<OdometrySettingsImplCommon>();
}

OdometrySettingsImplCommon::OdometrySettingsImplCommon()
{
    this->cameraMatrix = defaultCameraMatrix;
	this->iterCounts = defaultIterCounts;

	this->minDepth = defaultMinDepth;
	this->maxDepth = defaultMaxDepth;
	this->maxDepthDiff = defaultMaxDepthDiff;
	this->maxPointsPart = defaultMaxPointsPart;

	this->sobelSize = defaultSobelSize;
	this->sobelScale = defaultSobelScale;
	this->normalWinSize = defaultNormalWinSize;

    this->angleThreshold = defaultAngleThreshold;
	this->maxTranslation = defaultMaxTranslation;
	this->maxRotation = defaultMaxRotation;

	this->minGradientMagnitude = defaultMinGradientMagnitude;
	this->minGradientMagnitudes = defaultMinGradientMagnitudes;
}

void OdometrySettingsImplCommon::setCameraMatrix(InputArray val)
{
    if (!val.empty())
    {
        CV_Assert(val.rows() == 3 && val.cols() == 3 && val.channels() == 1);
        CV_Assert(val.type() == CV_32F);
        val.copyTo(cameraMatrix);
    }
	else
	{
		this->cameraMatrix = defaultCameraMatrix;
	}
}

void OdometrySettingsImplCommon::getCameraMatrix(OutputArray val) const
{
    Mat(this->cameraMatrix).copyTo(val);
}

void OdometrySettingsImplCommon::setIterCounts(InputArray val)
{
	if (!val.empty())
	{
		size_t nLevels = val.size(-1).width;
		std::vector<Mat> pyramids;
		val.getMatVector(pyramids);
		this->iterCounts.clear();
		for (int i = 0; i < nLevels; i++)
			this->iterCounts.push_back(pyramids[i].at<int>(0));
	}
	else
	{
		this->iterCounts = defaultIterCounts;
	}
}

void OdometrySettingsImplCommon::getIterCounts(OutputArray val) const
{
	Mat(defaultIterCounts).copyTo(val);
}

void  OdometrySettingsImplCommon::setMinDepth(float val)
{
	this->minDepth = val;
}
float OdometrySettingsImplCommon::getMinDepth() const
{
	return this->minDepth;
}
void  OdometrySettingsImplCommon::setMaxDepth(float val)
{
	this->maxDepth = val;
}
float OdometrySettingsImplCommon::getMaxDepth() const
{
	return this->maxDepth;
}
void  OdometrySettingsImplCommon::setMaxDepthDiff(float val)
{
	this->maxDepthDiff = val;
}
float OdometrySettingsImplCommon::getMaxDepthDiff() const
{
	return this->maxDepthDiff;
}
void  OdometrySettingsImplCommon::setMaxPointsPart(float val)
{
	this->maxPointsPart = val;
}
float OdometrySettingsImplCommon::getMaxPointsPart() const
{
	return this->maxPointsPart;
}

void OdometrySettingsImplCommon::setSobelSize(int val)
{
	this->sobelSize = val;
}
int  OdometrySettingsImplCommon::getSobelSize() const
{
	return this->sobelScale;
}
void   OdometrySettingsImplCommon::setSobelScale(double val)
{
	this->sobelScale = val;
}
double OdometrySettingsImplCommon::getSobelScale() const
{
	return this->sobelScale;
}
void OdometrySettingsImplCommon::setNormalWinSize(int val)
{
	this->normalWinSize = val;
}
int  OdometrySettingsImplCommon::getNormalWinSize() const
{
	return this->normalWinSize;
}

void  OdometrySettingsImplCommon::setAngleThreshold(float val)
{
	this->angleThreshold = val;
}
float OdometrySettingsImplCommon::getAngleThreshold() const
{
	return this->angleThreshold;
}
void  OdometrySettingsImplCommon::setMaxTranslation(float val)
{
	this->maxTranslation = val;
}
float OdometrySettingsImplCommon::getMaxTranslation() const
{
	return this->maxTranslation;
}
void  OdometrySettingsImplCommon::setMaxRotation(float val)
{
	this->maxRotation = val;
}
float OdometrySettingsImplCommon::getMaxRotation() const
{
	return this->maxRotation;
}

void  OdometrySettingsImplCommon::setMinGradientMagnitude(float val)
{
	this->minGradientMagnitude = val;
}
float OdometrySettingsImplCommon::getMinGradientMagnitude() const
{
	return this->minGradientMagnitude;
}
void OdometrySettingsImplCommon::setMinGradientMagnitudes(InputArray val)
{
	if (!val.empty())
	{
		size_t nLevels = val.size(-1).width;
		std::vector<Mat> pyramids;
		val.getMatVector(pyramids);
		this->minGradientMagnitudes.clear();
		for (int i = 0; i < nLevels; i++)
			this->minGradientMagnitudes.push_back(pyramids[i].at<int>(0));
	}
	else
	{
		this->minGradientMagnitudes = defaultMinGradientMagnitudes;
	}
}
void OdometrySettingsImplCommon::getMinGradientMagnitudes(OutputArray val) const
{
	Mat(this->minGradientMagnitudes).copyTo(val);
}

}
