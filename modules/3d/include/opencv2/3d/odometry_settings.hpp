// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_ODOMETRY_SETTINGS_HPP
#define OPENCV_3D_ODOMETRY_SETTINGS_HPP

#include <opencv2/core.hpp>

namespace cv
{

class CV_EXPORTS_W OdometrySettings
{
public:
    OdometrySettings();
    ~OdometrySettings() {};
    void setCameraMatrix(InputArray val);
    void getCameraMatrix(OutputArray val) const;
    void setIterCounts(InputArray val);
    void getIterCounts(OutputArray val) const;

    void  setMinDepth(float val);
    float getMinDepth() const;
    void  setMaxDepth(float val);
    float getMaxDepth() const;
    void  setMaxDepthDiff(float val);
    float getMaxDepthDiff() const;
    void  setMaxPointsPart(float val);
    float getMaxPointsPart() const;

    void setSobelSize(int val);
    int  getSobelSize() const;
    void   setSobelScale(double val);
    double getSobelScale() const;
    void setNormalWinSize(int val);
    int  getNormalWinSize() const;

    void  setAngleThreshold(float val);
    float getAngleThreshold() const;
    void  setMaxTranslation(float val);
    float getMaxTranslation() const;
    void  setMaxRotation(float val);
    float getMaxRotation() const;

    void  setMinGradientMagnitude(float val);
    float getMinGradientMagnitude() const;
    void setMinGradientMagnitudes(InputArray val);
    void getMinGradientMagnitudes(OutputArray val) const;

    class Impl;

private:
    Ptr<Impl> impl;
};

}
#endif //OPENCV_3D_ODOMETRY_SETTINGS_HPP
