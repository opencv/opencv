#include "rotationConverters.hpp"

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#define CALIB_PI 3.14159265358979323846
#define CALIB_PI_2 1.57079632679489661923

void calib::Euler(const cv::Mat& src, cv::Mat& dst, int argType)
{
    if((src.rows == 3) && (src.cols == 3))
    {
        //convert rotaion matrix to 3 angles (pitch, yaw, roll)
        dst = cv::Mat(3, 1, CV_64F);
        double pitch, yaw, roll;

        if(src.at<double>(0,2) < -0.998)
        {
            pitch = -atan2(src.at<double>(1,0), src.at<double>(1,1));
            yaw = -CALIB_PI_2;
            roll = 0.;
        }
        else if(src.at<double>(0,2) > 0.998)
        {
            pitch = atan2(src.at<double>(1,0), src.at<double>(1,1));
            yaw = CALIB_PI_2;
            roll = 0.;
        }
        else
        {
            pitch = atan2(-src.at<double>(1,2), src.at<double>(2,2));
            yaw = asin(src.at<double>(0,2));
            roll = atan2(-src.at<double>(0,1), src.at<double>(0,0));
        }

        if(argType == CALIB_DEGREES)
        {
            pitch *= 180./CALIB_PI;
            yaw *= 180./CALIB_PI;
            roll *= 180./CALIB_PI;
        }
        else if(argType != CALIB_RADIANS)
            CV_Error(cv::Error::StsBadFlag, "Invalid argument type");

        dst.at<double>(0,0) = pitch;
        dst.at<double>(1,0) = yaw;
        dst.at<double>(2,0) = roll;
    }
    else if( (src.cols == 1 && src.rows == 3) ||
             (src.cols == 3 && src.rows == 1 ) )
    {
        //convert vector which contains 3 angles (pitch, yaw, roll) to rotaion matrix
        double pitch, yaw, roll;
        if(src.cols == 1 && src.rows == 3)
        {
            pitch = src.at<double>(0,0);
            yaw = src.at<double>(1,0);
            roll = src.at<double>(2,0);
        }
        else{
            pitch = src.at<double>(0,0);
            yaw = src.at<double>(0,1);
            roll = src.at<double>(0,2);
        }

        if(argType == CALIB_DEGREES)
        {
            pitch *= CALIB_PI / 180.;
            yaw *= CALIB_PI / 180.;
            roll *= CALIB_PI / 180.;
        }
        else if(argType != CALIB_RADIANS)
            CV_Error(cv::Error::StsBadFlag, "Invalid argument type");

        dst = cv::Mat(3, 3, CV_64F);
        cv::Mat M(3, 3, CV_64F);
        cv::Mat i = cv::Mat::eye(3, 3, CV_64F);
        i.copyTo(dst);
        i.copyTo(M);

        double* pR = dst.ptr<double>();
        pR[4] = cos(pitch);
        pR[7] = sin(pitch);
        pR[8] = pR[4];
        pR[5] = -pR[7];

        double* pM = M.ptr<double>();
        pM[0] = cos(yaw);
        pM[2] = sin(yaw);
        pM[8] = pM[0];
        pM[6] = -pM[2];

        dst *= M;
        i.copyTo(M);
        pM[0] = cos(roll);
        pM[3] = sin(roll);
        pM[4] = pM[0];
        pM[1] = -pM[3];

        dst *= M;
    }
    else
        CV_Error(cv::Error::StsBadFlag, "Input matrix must be 1x3, 3x1 or 3x3" );
}

void calib::RodriguesToEuler(const cv::Mat& src, cv::Mat& dst, int argType)
{
    CV_Assert((src.cols == 1 && src.rows == 3) || (src.cols == 3 && src.rows == 1));
    cv::Mat R;
    cv::Rodrigues(src, R);
    Euler(R, dst, argType);
}

void calib::EulerToRodrigues(const cv::Mat& src, cv::Mat& dst, int argType)
{
    CV_Assert((src.cols == 1 && src.rows == 3) || (src.cols == 3 && src.rows == 1));
    cv::Mat R;
    Euler(src, R, argType);
    cv::Rodrigues(R, dst);
}
