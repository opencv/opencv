#ifndef ODOMETRY_SETTINGS_HPP
#define ODOMETRY_SETTINGS_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cvstd.hpp>

namespace cv
{
class OdometrySettingsImpl
{
public:
    OdometrySettingsImpl() {};
    ~OdometrySettingsImpl() {};
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

private:
    Matx33f cameraMatrix;
};

class OdometrySettings
{
public:
    OdometrySettings();
    ~OdometrySettings() {};
    void setCameraMatrix(InputArray val) { this->odometrySettings->setCameraMatrix(val); }
    void getCameraMatrix(OutputArray val) const { this->odometrySettings->getCameraMatrix(val); }
    void setIterCounts(InputArray val) { this->odometrySettings->setIterCounts(val); }
    void getIterCounts(OutputArray val) const { this->odometrySettings->getIterCounts(val); }

    void  setMinDepth(float val) { this->odometrySettings->setMinDepth(val); };
    float getMinDepth() const { return this->odometrySettings->getMinDepth(); };
    void  setMaxDepth(float val) { this->odometrySettings->setMaxDepth(val); };
    float getMaxDepth() const { return this->odometrySettings->getMaxDepth(); };
    void  setMaxDepthDiff(float val) { this->odometrySettings->setMaxDepthDiff(val); };
    float getMaxDepthDiff() const { return this->odometrySettings->getMaxDepthDiff(); };
    void  setMaxPointsPart(float val) { this->odometrySettings->setMaxPointsPart(val); };
    float getMaxPointsPart() const { return this->odometrySettings->getMaxPointsPart(); };

    void setSobelSize(int val) { this->odometrySettings->setSobelSize(val); };
    int  getSobelSize() const { return this->odometrySettings->getSobelSize(); };
    void   setSobelScale(double val) { this->odometrySettings->setSobelScale(val); };
    double getSobelScale() const { return this->odometrySettings->getSobelScale(); };
    void setNormalWinSize(int val) { this->odometrySettings->setNormalWinSize(val); };
    int  getNormalWinSize() const { return this->odometrySettings->getNormalWinSize(); };

    void  setAngleThreshold(float val) { this->odometrySettings->setAngleThreshold(val); };
    float getAngleThreshold() const { return this->odometrySettings->getAngleThreshold(); };
    void  setMaxTranslation(float val) { this->odometrySettings->setMaxTranslation(val); };
    float getMaxTranslation() const { return this->odometrySettings->getMaxTranslation(); };
    void  setMaxRotation(float val) { this->odometrySettings->setMaxRotation(val); };
    float getMaxRotation() const { return this->odometrySettings->getMaxRotation(); };

    void  setMinGradientMagnitude(float val) { this->odometrySettings->setMinGradientMagnitude(val); };
    float getMinGradientMagnitude() const { return this->odometrySettings->getMinGradientMagnitude(); };
    void setMinGradientMagnitudes(InputArray val) { this->odometrySettings->setMinGradientMagnitudes(val); };
    void getMinGradientMagnitudes(OutputArray val) const { this->odometrySettings->getMinGradientMagnitudes(val); };

private:
    Ptr<OdometrySettingsImpl> odometrySettings;
};

}
#endif //ODOMETRY_SETTINGS_HPP
