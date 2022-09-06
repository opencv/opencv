// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_ODOMETRY_SETTINGS_HPP
#define OPENCV_3D_ODOMETRY_SETTINGS_HPP

#include <opencv2/core.hpp>

namespace cv
{

class CV_EXPORTS_W_SIMPLE OdometrySettings
{
public:
    CV_WRAP OdometrySettings();
    OdometrySettings(const OdometrySettings&);
    OdometrySettings& operator=(const OdometrySettings&);
    ~OdometrySettings() {};
    CV_WRAP void setCameraMatrix(InputArray val);
    CV_WRAP void getCameraMatrix(OutputArray val) const;
    CV_WRAP void setIterCounts(InputArray val);
    CV_WRAP void getIterCounts(OutputArray val) const;

    CV_WRAP void  setMinDepth(float val);
    CV_WRAP float getMinDepth() const;
    CV_WRAP void  setMaxDepth(float val);
    CV_WRAP float getMaxDepth() const;
    CV_WRAP void  setMaxDepthDiff(float val);
    CV_WRAP float getMaxDepthDiff() const;
    CV_WRAP void  setMaxPointsPart(float val);
    CV_WRAP float getMaxPointsPart() const;

    CV_WRAP void setSobelSize(int val);
    CV_WRAP int  getSobelSize() const;
    CV_WRAP void   setSobelScale(double val);
    CV_WRAP double getSobelScale() const;

    CV_WRAP void setNormalWinSize(int val);
    CV_WRAP int  getNormalWinSize() const;
    CV_WRAP void setNormalDiffThreshold(float val);
    CV_WRAP float getNormalDiffThreshold() const;
    CV_WRAP void setNormalMethod(RgbdNormals::RgbdNormalsMethod nm);
    CV_WRAP RgbdNormals::RgbdNormalsMethod getNormalMethod() const;

    CV_WRAP void  setAngleThreshold(float val);
    CV_WRAP float getAngleThreshold() const;
    CV_WRAP void  setMaxTranslation(float val);
    CV_WRAP float getMaxTranslation() const;
    CV_WRAP void  setMaxRotation(float val);
    CV_WRAP float getMaxRotation() const;

    CV_WRAP void  setMinGradientMagnitude(float val);
    CV_WRAP float getMinGradientMagnitude() const;
    CV_WRAP void setMinGradientMagnitudes(InputArray val);
    CV_WRAP void getMinGradientMagnitudes(OutputArray val) const;

    class Impl;

private:
    Ptr<Impl> impl;
};

}
#endif //OPENCV_3D_ODOMETRY_SETTINGS_HPP
