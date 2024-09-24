// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <vector>
#include <string>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <fstream>

// ! [detectPointsAndCalibrate_signature]
static void detectPointsAndCalibrate (cv::Size pattern_size, float pattern_distance, const std::string &pattern_type,
           const std::vector<bool> &is_fisheye, const std::vector<std::string> &filenames,
           const cv::String* dict_path=nullptr)
// ! [detectPointsAndCalibrate_signature]
{
// ! [calib_init]
    std::vector<cv::Point3f> board (pattern_size.area());
    const int num_cameras = (int)is_fisheye.size();
    std::vector<std::vector<cv::Mat>> image_points_all;
    std::vector<cv::Size> image_sizes;
    std::vector<cv::Mat> Ks, distortions, Ts, Rs;
    cv::Mat rvecs0, tvecs0, errors_mat, output_pairs;
    if (pattern_type == "checkerboard" || pattern_type == "charuco") {
        for (int i = 0; i < pattern_size.height; i++) {
            for (int j = 0; j < pattern_size.width; j++) {
                board[i*pattern_size.width+j] = cv::Point3f((float)j, (float)i, 0.f) * pattern_distance;
            }
        }
    } else if (pattern_type == "circles") {
        for (int i = 0; i < pattern_size.height; i++) {
            for (int j = 0; j < pattern_size.width; j++) {
                board[i*pattern_size.width+j] = cv::Point3f((float)j, (float)i, 0.f) * pattern_distance;
            }
        }
    } else if (pattern_type == "acircles") {
        for (int i = 0; i < pattern_size.height; i++) {
            for (int j = 0; j < pattern_size.width; j++) {
                if (i % 2 == 1) {
                    board[i*pattern_size.width+j] = cv::Point3f((j + .5f)*pattern_distance, (i/2 + .5f) * pattern_distance, 0.f);
                } else{
                    board[i*pattern_size.width+j] = cv::Point3f(j*pattern_distance, (i/2)*pattern_distance, 0);
                }
            }
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented, "pattern_type is not implemented!");
    }
// ! [calib_init]
// ! [charuco_detector]
    cv::Ptr<cv::aruco::CharucoDetector> detector;
    if (pattern_type == "charuco") {
        CV_Assert(dict_path != nullptr);
        cv::FileStorage fs(*dict_path, cv::FileStorage::READ);
        CV_Assert(fs.isOpened());

        int dict_int;
        double square_size, marker_size;
        fs["dictionary"] >> dict_int;
        fs["square_size"] >> square_size;
        fs["marker_size"] >> marker_size;

        auto dictionary = cv::aruco::getPredefinedDictionary(dict_int);
        // For charuco board, the size is defined to be the number of box (not inner corner)
        auto charuco_board = cv::aruco::CharucoBoard(
            cv::Size(pattern_size.width+1, pattern_size.height+1),
            static_cast<float>(square_size), static_cast<float>(marker_size), dictionary);

        // It is suggested to use refinement in detecting charuco board
        auto detector_params = cv::aruco::DetectorParameters();
        auto charuco_params = cv::aruco::CharucoParameters();
        charuco_params.tryRefineMarkers = true;
        detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
        detector = cv::makePtr<cv::aruco::CharucoDetector>(charuco_board, charuco_params, detector_params);
    }
// ! [charuco_detector]
// ! [detect_pattern]
    int num_frames = -1;
    for (const auto &filename : filenames) {
        std::fstream file(filename);
        CV_Assert(file.is_open());
        std::string img_file;
        std::vector<cv::Mat> image_points_cameras;
        bool save_img_size = true;
        while (std::getline(file, img_file)) {
            if (img_file.empty()){
                image_points_cameras.emplace_back(cv::Mat());
                continue;
            }
            cv::Mat img = cv::imread(img_file), corners;
            if (save_img_size) {
                image_sizes.emplace_back(cv::Size(img.cols, img.rows));
                save_img_size = false;
            }

            bool success = false;
            if (pattern_type == "checkerboard") {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                success = cv::findChessboardCorners(img, pattern_size, corners);
            }
            else if (pattern_type == "circles")
            {
                success = cv::findCirclesGrid(img, pattern_size, corners, cv::CALIB_CB_SYMMETRIC_GRID);
            }
            else if (pattern_type == "acircles")
            {
                success = cv::findCirclesGrid(img, pattern_size, corners, cv::CALIB_CB_ASYMMETRIC_GRID);
            }
            else if (pattern_type == "charuco")
            {
                std::vector<int> ids; cv::Mat corners_sub;
                detector->detectBoard(img, corners_sub, ids);
                corners.create(static_cast<int>(board.size()), 2, CV_32F);
                if (ids.size() < 4)
                    success = false;
                else {
                    success = true;
                    int head = 0;
                    for (int i = 0; i < static_cast<int>(board.size()); i++) {
                        if (head < static_cast<int>(ids.size()) && ids[head] == i) {
                            corners.at<float>(i, 0) = corners_sub.at<float>(head, 0);
                            corners.at<float>(i, 1) = corners_sub.at<float>(head, 1);
                            head++;
                        } else {
                            // points outside of frame border are dropped by calibrateMultiview
                            corners.at<float>(i, 0) = -1.;
                            corners.at<float>(i, 1) = -1.;
                        }
                    }
                }
            }

            cv::Mat corners2;
            corners.convertTo(corners2, CV_32FC2);

            if (success && corners.rows == pattern_size.area())
                image_points_cameras.emplace_back(corners2);
            else
                image_points_cameras.emplace_back(cv::Mat());
        }
        if (num_frames == -1)
            num_frames = (int)image_points_cameras.size();
        else
            CV_Assert(num_frames == (int)image_points_cameras.size());
        image_points_all.emplace_back(image_points_cameras);
    }
// ! [detect_pattern]
// ! [detection_matrix]
    cv::Mat visibility(num_cameras, num_frames, CV_8UC1);
    for (int i = 0; i < num_cameras; i++) {
        for (int j = 0; j < num_frames; j++) {
            visibility.at<unsigned char>(i,j) = image_points_all[i][j].empty() ? 0 : 1;
        }
    }
// ! [detection_matrix]
    CV_Assert(num_frames != -1);

    std::vector<std::vector<cv::Point3f>> objPoints(num_frames, board);

// ! [multiview_calib]
    const double rmse = calibrateMultiview(objPoints, image_points_all, image_sizes, visibility,
                                           Rs, Ts, Ks, distortions, cv::noArray(), cv::noArray(),
                                           is_fisheye, errors_mat, output_pairs);
// ! [multiview_calib]
    std::cout << "average RMSE over detection mask " << rmse << "\n";
    for (int c = 0; c < (int)Rs.size(); c++) {
        std::cout << "camera " << c << '\n';
        std::cout << "rotation\n" << Rs[c] << "\n";
        std::cout << "translation\n" << Ts[c] << "\n";
        std::cout << "intrinsic matrix\n" << Ks[c] << "\n";
        std::cout << "distortion\n" << distortions[c] << "\n";
    }
}

int main (int argc, char **argv) {
    cv::String keys =
            "{help h usage ? || print help }"
            "{pattern_size || (inner) grid width, (inner) grid height }"
            "{pattern_distance || pattern scale}"
            "{pattern_type |  checkerboard  | pattern type, e.g., checkerboard or acircles or charuco (recommended)}"
            "{is_fisheye || cameras type fisheye (1), pinhole(0), separated by comma (no space)}"
            "{filenames || files containing path to image names separated by comma (no space)}"
            "{board_dict_path || file containing dictionary information (required field: dictionary, square_size, marker_size). Needed if pattern_type is charuco.}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    CV_Assert(parser.has("pattern_size") && parser.has("pattern_type") &&
        parser.has("is_fisheye") && parser.has("filenames"));
    CV_Assert(parser.get<cv::String>("pattern_type") == "checkerboard" ||
              parser.get<cv::String>("pattern_type") == "circles" ||
              parser.get<cv::String>("pattern_type") == "acircles" ||
              parser.get<cv::String>("pattern_type") == "charuco"
              );
    if (parser.get<cv::String>("pattern_type") == "charuco")
        CV_Assert(parser.has("board_dict_path"));

    cv::Size pattern_size;
    const cv::String pattern_size_str = parser.get<cv::String>("pattern_size");
    std::string temp_str;
    int pattern_size_count = 0;
    for (char i : pattern_size_str) {
        if (i == ',') {
            if (pattern_size_count == 0)
                pattern_size.width = std::stoi(temp_str);
            pattern_size_count++;
            temp_str = "";
        } else {
            temp_str += i;
        }
    }
    CV_Assert(pattern_size_count == 1);
    pattern_size.height = std::stoi(temp_str);

    std::vector<bool> is_fisheye;
    const cv::String is_fisheye_str = parser.get<cv::String>("is_fisheye");
    for (char i : is_fisheye_str) {
        if (i == '0') {
            is_fisheye.push_back(false);
        } else if (i == '1') {
            is_fisheye.push_back(true);
        }
    }
    const cv::String filenames_str = parser.get<cv::String>("filenames");
    std::vector<std::string> filenames;
    temp_str = "";
    for (char i : filenames_str) {
        if (i == ',') {
            filenames.emplace_back(temp_str);
            temp_str = "";
        } else {
            temp_str += i;
        }
    }
    filenames.emplace_back(temp_str);
    CV_CheckEQ(filenames.size(), is_fisheye.size(), "filenames size must be equal to number of cameras!");

    if (parser.has("board_dict_path")) {
        cv::String board_dict_path = parser.get<cv::String>("board_dict_path");
        detectPointsAndCalibrate (pattern_size, parser.get<float>("pattern_distance"), parser.get<cv::String>("pattern_type"), is_fisheye, filenames, &board_dict_path);
    } else {
        detectPointsAndCalibrate (pattern_size, parser.get<float>("pattern_distance"), parser.get<cv::String>("pattern_type"), is_fisheye, filenames);
    }
    return 0;
}
