#ifndef CALIB_CONTROLLER_HPP
#define CALIB_CONTROLLER_HPP

#include "calibCommon.hpp"
#include <stack>
#include <string>
#include <ostream>

namespace calib {

    class calibController
    {
    protected:
        cv::Ptr<calibrationData> mCalibData;
        int mCalibFlags;
        unsigned mMinFramesNum;
        bool mNeedTuning;
        bool mConfIntervalsState;
        bool mCoverageQualityState;

        double estimateCoverageQuality();
    public:
        calibController();
        calibController(cv::Ptr<calibrationData> data, int initialFlags, bool autoTuning,
                        int minFramesNum);

        void updateState();

        bool getCommonCalibrationState() const;

        bool getFramesNumberState() const;
        bool getConfidenceIntrervalsState() const;
        bool getRMSState() const;
        bool getPointsCoverageState() const;
        int getNewFlags() const;
    };

    class calibDataController
    {
    protected:
        cv::Ptr<calibrationData> mCalibData;
        std::stack<cameraParameters> mParamsStack;
        std::string mParamsFileName;
        unsigned mMaxFramesNum;
        double mAlpha;

        double estimateGridSubsetQuality(size_t excludedIndex);
    public:
        calibDataController(cv::Ptr<calibrationData> data, int maxFrames, double convParameter);
        calibDataController();

        void filterFrames();
        void setParametersFileName(const std::string& name);
        void deleteLastFrame();
        void rememberCurrentParameters();
        void deleteAllData();
        bool saveCurrentCameraParameters() const;
        void printParametersToConsole(std::ostream &output) const;
        void updateUndistortMap();
    };

}

#endif
