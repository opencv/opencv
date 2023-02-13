// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <vector>
#include <string>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

static void detectPointsAndCalibrate (cv::Size pattern_size, double pattern_scale, const std::string &pattern_type,
           const std::vector<bool> &is_fisheye, const std::vector<std::string> &filenames) {
    std::vector<cv::Vec3f> board (pattern_size.area());
    const int num_cameras = (int)is_fisheye.size();
    std::vector<std::vector<cv::Mat>> image_points_all;
    std::vector<cv::Size> image_sizes;
    std::vector<cv::Mat> Ks, distortions, Ts, Rs;
    cv::Mat rvecs0, tvecs0, errors_mat, output_pairs;
    if (pattern_type == "checkerboard") {
        for (int i = 0; i < pattern_size.height; i++) {
            for (int j = 0; j < pattern_size.width; j++) {
                board[i*pattern_size.width+j] = cv::Vec3f((float)j, (float)i, 0) * pattern_scale;
            }
        }
    } else {
        CV_Error(cv::Error::StsNotImplemented, "pattern_type is not implemented!");
    }
    int num_frames = -1;
    for (const auto &filename : filenames) {
        std::fstream file(filename);
        CV_Assert(file.is_open());
        std::string img_file;
        std::vector<cv::Mat> image_points_cameras;
        bool save_img_size = true;
        while (std::getline(file, img_file)) {
            if (img_file.empty())
                break;
            std::cout << img_file << "\n";
            cv::Mat img = cv::imread(img_file), corners;
            if (save_img_size) {
                image_sizes.emplace_back(cv::Size(img.cols, img.rows));
                save_img_size = false;
            }
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            bool success = false;
            if (pattern_type == "checkerboard") {
                success = cv::findChessboardCorners(img, pattern_size, corners);
            }
            if (success && corners.rows == pattern_size.area())
                image_points_cameras.emplace_back(corners);
            else
                image_points_cameras.emplace_back(cv::Mat());
        }
        if (num_frames == -1)
            num_frames = (int)image_points_cameras.size();
        else
            CV_Assert(num_frames == (int)image_points_cameras.size());
        image_points_all.emplace_back(image_points_cameras);
    }

    cv::Mat visibility = cv::Mat_<int>(num_cameras, num_frames);
    for (int i = 0; i < num_cameras; i++) {
        for (int j = 0; j < num_frames; j++) {
            visibility.at<int>(i,j) = (int)(!image_points_all[i][j].empty());
        }
    }
    CV_Assert(num_frames != -1);
    std::vector<std::vector<cv::Vec3f>> objPoints(num_frames, board);
    const double rmse = calibrateMultiview (objPoints, image_points_all, image_sizes, visibility,
       Rs, Ts, Ks, distortions, cv::noArray(), cv::noArray(), is_fisheye, errors_mat, output_pairs, false/*use intrinsics guess*/);
    std::cout << "average RMSE over detection mask " << rmse << "\n";
    for (int c = 0; c < (int)Rs.size(); c++) {
        std::cout << "camera " << c << '\n';
        std::cout << Rs[c] << " rotation\n";
        std::cout << Ts[c] << " translation\n";
        std::cout << Ks[c] << " intrinsic matrix\n";
        std::cout << distortions[c] << " distortion\n";
    }
}

int main (int argc, char **argv) {
    cv::String keys =
            "{help h usage ? || print help }"
            "{pattern_width || pattern grid width}"
            "{pattern_height || pattern grid height}"
            "{pattern_scale || pattern scale}"
            "{pattern_type |  checkerboard  | pattern type, e.g., checkerboard}"
            "{is_fisheye || cameras type fisheye (1), pinhole(0), separated by comma (no space)}"
            "{files_with_images || files containing path to image names separated by comma (no space)}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    CV_Assert(parser.has("pattern_width") && parser.has("pattern_width") && parser.has("pattern_height") && parser.has("pattern_type") &&
        parser.has("is_fisheye") && parser.has("files_with_images"));
    const cv::Size pattern_size (parser.get<int>("pattern_width"), parser.get<int>("pattern_height"));
    std::vector<bool> is_fisheye;
    const cv::String is_fisheye_str = parser.get<cv::String>("is_fisheye");
    for (char i : is_fisheye_str) {
        if (i == '0') {
            is_fisheye.push_back(false);
        } else if (i == '1') {
            is_fisheye.push_back(true);
        }
    }
    const cv::String files_with_images_str = parser.get<cv::String>("files_with_images");
    std::vector<std::string> filenames;
    std::string temp_str;
    for (char i : files_with_images_str) {
        if (i == ',') {
            filenames.emplace_back(temp_str);
            temp_str = "";
        } else {
            temp_str += i;
        }
    }
    filenames.emplace_back(temp_str);
    CV_CheckEQ(filenames.size(), is_fisheye.size(), "filenames size must be equal to number of cameras!");
    detectPointsAndCalibrate (pattern_size, parser.get<double>("pattern_scale"), parser.get<cv::String>("pattern_type"), is_fisheye, filenames);
    return 0;
}