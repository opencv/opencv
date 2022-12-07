// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "utils.hpp"

namespace cv
{

class OdometrySettings::Impl
{
public:
    Impl() {};
    virtual ~Impl() {};
    virtual void setCameraMatrix(InputArray val) = 0;
    virtual void getCameraMatrix(OutputArray val) const = 0;
    virtual void setIterCounts(InputArray val) = 0;
    virtual void getIterCounts(OutputArray val) const = 0;

    virtual void  setMinDepth(float val) = 0;
    virtual float getMinDepth() const = 0;
    virtual void  setMaxDepth(float val) = 0;
    virtual float getMaxDepth() const = 0;
    virtual void  setMaxDepthDiff(float val) = 0;
    virtual float getMaxDepthDiff() const = 0;
    virtual void  setMaxPointsPart(float val) = 0;
    virtual float getMaxPointsPart() const = 0;

    virtual void setSobelSize(int val) = 0;
    virtual int  getSobelSize() const = 0;
    virtual void   setSobelScale(double val) = 0;
    virtual double getSobelScale() const = 0;

    virtual void setNormalWinSize(int val) = 0;
    virtual int  getNormalWinSize() const = 0;
    virtual void setNormalDiffThreshold(float val) = 0;
    virtual float getNormalDiffThreshold() const = 0;
    virtual void setNormalMethod(RgbdNormals::RgbdNormalsMethod nm) = 0;
    virtual RgbdNormals::RgbdNormalsMethod getNormalMethod() const = 0;

    virtual void  setAngleThreshold(float val) = 0;
    virtual float getAngleThreshold() const = 0;
    virtual void  setMaxTranslation(float val) = 0;
    virtual float getMaxTranslation() const = 0;
    virtual void  setMaxRotation(float val) = 0;
    virtual float getMaxRotation() const = 0;

    virtual void  setMinGradientMagnitude(float val) = 0;
    virtual float getMinGradientMagnitude() const = 0;
    virtual void setMinGradientMagnitudes(InputArray val) = 0;
    virtual void getMinGradientMagnitudes(OutputArray val) const = 0;
};

class OdometrySettingsImplCommon : public OdometrySettings::Impl
{
public:
    OdometrySettingsImplCommon();
    ~OdometrySettingsImplCommon() {};
    virtual void setCameraMatrix(InputArray val) override;
    virtual void getCameraMatrix(OutputArray val) const override;
    virtual void setIterCounts(InputArray val) override;
    virtual void getIterCounts(OutputArray val) const override;

    virtual void  setMinDepth(float val) override;
    virtual float getMinDepth() const override;
    virtual void  setMaxDepth(float val) override;
    virtual float getMaxDepth() const override;
    virtual void  setMaxDepthDiff(float val) override;
    virtual float getMaxDepthDiff() const override;
    virtual void  setMaxPointsPart(float val) override;
    virtual float getMaxPointsPart() const override;

    virtual void setSobelSize(int val) override;
    virtual int  getSobelSize() const override;
    virtual void   setSobelScale(double val) override;
    virtual double getSobelScale() const override;

    virtual void setNormalWinSize(int val) override;
    virtual int  getNormalWinSize() const override;
    virtual void setNormalDiffThreshold(float val) override;
    virtual float getNormalDiffThreshold() const override;
    virtual void setNormalMethod(RgbdNormals::RgbdNormalsMethod nm) override;
    virtual RgbdNormals::RgbdNormalsMethod getNormalMethod() const override;

    virtual void  setAngleThreshold(float val) override;
    virtual float getAngleThreshold() const override;
    virtual void  setMaxTranslation(float val) override;
    virtual float getMaxTranslation() const override;
    virtual void  setMaxRotation(float val) override;
    virtual float getMaxRotation() const override;

    virtual void  setMinGradientMagnitude(float val) override;
    virtual float getMinGradientMagnitude() const override;
    virtual void setMinGradientMagnitudes(InputArray val) override;
    virtual void getMinGradientMagnitudes(OutputArray val) const override;

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
    float normalDiffThreshold;
    RgbdNormals::RgbdNormalsMethod normalMethod;

    float angleThreshold;
    float maxTranslation;
    float maxRotation;

    float minGradientMagnitude;
    std::vector<float> minGradientMagnitudes;

public:
    class DefaultSets {
    public:
        static const int width  = 640;
        static const int height = 480;
        static constexpr float fx = 525.f;
        static constexpr float fy = 525.f;
        static constexpr float cx = float(width) / 2.f - 0.5f;
        static constexpr float cy = float(height) / 2.f - 0.5f;
        const cv::Matx33f defaultCameraMatrix = { fx, 0, cx, 0, fy, cy, 0, 0, 1 };
        const std::vector<int> defaultIterCounts = { 7, 7, 7, 10 };

        static constexpr float defaultMinDepth = 0.f;
        static constexpr float defaultMaxDepth = 4.f;
        static constexpr float defaultMaxDepthDiff = 0.07f;
        static constexpr float defaultMaxPointsPart = 0.07f;

        static const int defaultSobelSize = 3;
        static constexpr double defaultSobelScale = 1. / 8.;

        static const int defaultNormalWinSize = 5;
        static const RgbdNormals::RgbdNormalsMethod defaultNormalMethod = RgbdNormals::RGBD_NORMALS_METHOD_FALS;
        static constexpr float defaultNormalDiffThreshold = 50.f;

        static constexpr float defaultAngleThreshold = (float)(30. * CV_PI / 180.);
        static constexpr float defaultMaxTranslation = 0.15f;
        static constexpr float defaultMaxRotation = 15.f;

        static constexpr float defaultMinGradientMagnitude = 10.f;
        const std::vector<float> defaultMinGradientMagnitudes = std::vector<float>(defaultIterCounts.size(), 10.f /*defaultMinGradientMagnitude*/);
    };
};


OdometrySettings::OdometrySettings()
{
    this->impl = makePtr<OdometrySettingsImplCommon>();
}

OdometrySettings::OdometrySettings(const OdometrySettings& s)
{
    this->impl = makePtr<OdometrySettingsImplCommon>(*s.impl.dynamicCast<OdometrySettingsImplCommon>());
}

OdometrySettings& OdometrySettings::operator=(const OdometrySettings& s)
{
    this->impl = makePtr<OdometrySettingsImplCommon>(*s.impl.dynamicCast<OdometrySettingsImplCommon>());
    return *this;
}

void OdometrySettings::setCameraMatrix(InputArray val) { this->impl->setCameraMatrix(val); }
void OdometrySettings::getCameraMatrix(OutputArray val) const { this->impl->getCameraMatrix(val); }
void OdometrySettings::setIterCounts(InputArray val) { this->impl->setIterCounts(val); }
void OdometrySettings::getIterCounts(OutputArray val) const { this->impl->getIterCounts(val); }

void  OdometrySettings::setMinDepth(float val) { this->impl->setMinDepth(val); }
float OdometrySettings::getMinDepth() const { return this->impl->getMinDepth(); }
void  OdometrySettings::setMaxDepth(float val) { this->impl->setMaxDepth(val); }
float OdometrySettings::getMaxDepth() const { return this->impl->getMaxDepth(); }
void  OdometrySettings::setMaxDepthDiff(float val) { this->impl->setMaxDepthDiff(val); }
float OdometrySettings::getMaxDepthDiff() const { return this->impl->getMaxDepthDiff(); }
void  OdometrySettings::setMaxPointsPart(float val) { this->impl->setMaxPointsPart(val); }
float OdometrySettings::getMaxPointsPart() const { return this->impl->getMaxPointsPart(); }

void OdometrySettings::setSobelSize(int val) { this->impl->setSobelSize(val); }
int  OdometrySettings::getSobelSize() const { return this->impl->getSobelSize(); }
void   OdometrySettings::setSobelScale(double val) { this->impl->setSobelScale(val); }
double OdometrySettings::getSobelScale() const { return this->impl->getSobelScale(); }

void  OdometrySettings::setNormalWinSize(int val) { this->impl->setNormalWinSize(val); }
int   OdometrySettings::getNormalWinSize() const { return this->impl->getNormalWinSize(); }
void  OdometrySettings::setNormalDiffThreshold(float val) { this->impl->setNormalDiffThreshold(val); }
float OdometrySettings::getNormalDiffThreshold() const { return this->impl->getNormalDiffThreshold(); }
void  OdometrySettings::setNormalMethod(RgbdNormals::RgbdNormalsMethod nm) { this->impl->setNormalMethod(nm); }
RgbdNormals::RgbdNormalsMethod OdometrySettings::getNormalMethod() const { return this->impl->getNormalMethod(); }

void  OdometrySettings::setAngleThreshold(float val) { this->impl->setAngleThreshold(val); }
float OdometrySettings::getAngleThreshold() const { return this->impl->getAngleThreshold(); }
void  OdometrySettings::setMaxTranslation(float val) { this->impl->setMaxTranslation(val); }
float OdometrySettings::getMaxTranslation() const { return this->impl->getMaxTranslation(); }
void  OdometrySettings::setMaxRotation(float val) { this->impl->setMaxRotation(val); }
float OdometrySettings::getMaxRotation() const { return this->impl->getMaxRotation(); }

void  OdometrySettings::setMinGradientMagnitude(float val) { this->impl->setMinGradientMagnitude(val); }
float OdometrySettings::getMinGradientMagnitude() const { return this->impl->getMinGradientMagnitude(); }
void OdometrySettings::setMinGradientMagnitudes(InputArray val) { this->impl->setMinGradientMagnitudes(val); }
void OdometrySettings::getMinGradientMagnitudes(OutputArray val) const { this->impl->getMinGradientMagnitudes(val); }


OdometrySettingsImplCommon::OdometrySettingsImplCommon()
{
    DefaultSets ds;
    this->cameraMatrix = ds.defaultCameraMatrix;
    this->iterCounts = ds.defaultIterCounts;

    this->minDepth = ds.defaultMinDepth;
    this->maxDepth = ds.defaultMaxDepth;
    this->maxDepthDiff = ds.defaultMaxDepthDiff;
    this->maxPointsPart = ds.defaultMaxPointsPart;

    this->sobelSize = ds.defaultSobelSize;
    this->sobelScale = ds.defaultSobelScale;

    this->normalWinSize = ds.defaultNormalWinSize;
    this->normalDiffThreshold = ds.defaultNormalDiffThreshold;
    this->normalMethod = ds.defaultNormalMethod;

    this->angleThreshold = ds.defaultAngleThreshold;
    this->maxTranslation = ds.defaultMaxTranslation;
    this->maxRotation = ds.defaultMaxRotation;

    this->minGradientMagnitude = ds.defaultMinGradientMagnitude;
    this->minGradientMagnitudes = ds.defaultMinGradientMagnitudes;
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
        DefaultSets ds;
        this->cameraMatrix = ds.defaultCameraMatrix;
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
        for (size_t i = 0; i < nLevels; i++)
            this->iterCounts.push_back(pyramids[i].at<int>(0));
    }
    else
    {
        DefaultSets ds;
        this->iterCounts = ds.defaultIterCounts;
    }
}

void OdometrySettingsImplCommon::getIterCounts(OutputArray val) const
{
    Mat(this->iterCounts).copyTo(val);
}

void OdometrySettingsImplCommon::setMinDepth(float val)
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
    return this->sobelSize;
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
void OdometrySettingsImplCommon::setNormalDiffThreshold(float val)
{
    this->normalDiffThreshold = val;
}
float OdometrySettingsImplCommon::getNormalDiffThreshold() const
{
    return this->normalDiffThreshold;
}
void OdometrySettingsImplCommon::setNormalMethod(RgbdNormals::RgbdNormalsMethod nm)
{
    this->normalMethod = nm;
}
RgbdNormals::RgbdNormalsMethod OdometrySettingsImplCommon::getNormalMethod() const
{
    return this->normalMethod;
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
        for (size_t i = 0; i < nLevels; i++)
            this->minGradientMagnitudes.push_back(pyramids[i].at<float>(0));
    }
    else
    {
        DefaultSets ds;
        this->minGradientMagnitudes = ds.defaultMinGradientMagnitudes;
    }
}
void OdometrySettingsImplCommon::getMinGradientMagnitudes(OutputArray val) const
{
    Mat(this->minGradientMagnitudes).copyTo(val);
}

}
