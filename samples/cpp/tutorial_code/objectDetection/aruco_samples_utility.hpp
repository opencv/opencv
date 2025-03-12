#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <ctime>

namespace {
inline static bool readCameraParameters(const std::string& filename, cv::Mat &camMatrix, cv::Mat &distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

inline static bool saveCameraParams(const std::string &filename, cv::Size imageSize, float aspectRatio, int flags,
                                    const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, double totalAvgErr) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        return false;

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if (flags & cv::CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
        snprintf(buf, sizeof(buf), "flags: %s%s%s%s",
                flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    }
    fs << "flags" << flags;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "avg_reprojection_error" << totalAvgErr;
    return true;
}

inline static cv::aruco::DetectorParameters readDetectorParamsFromCommandLine(cv::CommandLineParser &parser) {
    cv::aruco::DetectorParameters detectorParams;
    if (parser.has("dp")) {
        cv::FileStorage fs(parser.get<std::string>("dp"), cv::FileStorage::READ);
        bool readOk = detectorParams.readDetectorParameters(fs.root());
        if(!readOk) {
            throw std::runtime_error("Invalid detector parameters file\n");
        }
    }
    return detectorParams;
}

inline static void readCameraParamsFromCommandLine(cv::CommandLineParser &parser, cv::Mat& camMatrix, cv::Mat& distCoeffs) {
    //! [camDistCoeffs]
    if(parser.has("c")) {
        bool readOk = readCameraParameters(parser.get<std::string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            throw std::runtime_error("Invalid camera file\n");
        }
    }
    //! [camDistCoeffs]
}

inline static cv::aruco::Dictionary readDictionatyFromCommandLine(cv::CommandLineParser &parser) {
    cv::aruco::Dictionary dictionary;
    if (parser.has("cd")) {
        cv::FileStorage fs(parser.get<std::string>("cd"), cv::FileStorage::READ);
        bool readOk = dictionary.readDictionary(fs.root());
        if(!readOk) {
            throw std::runtime_error("Invalid dictionary file\n");
        }
    }
    else {
        int dictionaryId = parser.has("d") ? parser.get<int>("d"): cv::aruco::DICT_4X4_50;
        if (!parser.has("d")) {
            std::cout << "The default DICT_4X4_50 dictionary has been selected, you could "
                         "select the specific dictionary using flags -d or -cd." << std::endl;
        }
        dictionary = cv::aruco::getPredefinedDictionary(dictionaryId);
    }
    return dictionary;
}

}
