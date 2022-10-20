#include <vector>
#include <string>
#include <cassert>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

void detectPointsAndCalibrate (cv::Size pattern_size, double pattern_scale, const std::string &pattern_type, const std::vector<bool> &is_fisheye, const std::vector<std::string> &filenames) {
    std::vector<cv::Vec3f> board (pattern_size.area());
    const int num_cameras = (int)is_fisheye.size();
    std::vector<std::vector<cv::Mat>> image_points_all;
    std::vector<cv::Size> image_sizes;
    std::vector<std::vector<cv::Vec3f>> objPoints(num_frames, board);
    std::vector<cv::Mat> Ks, distortions, Ts, Rs;
    cv::Mat rvecs0, tvecs0, errors_mat, output_pairs;
    if (pattern_type == "checkerboard") {
        for (int i = 0; i < pattern_size.height; i++) {
            for (int j = 0; j < pattern_size.width; j++) {
                board[i*pattern_size.width+j] = cv::Vec3f((float)j, (float)i, 0) * pattern_scale;
            }
        }
    } else {
        assert(false && "not implemented!");
    }
    int num_frames = -1;
    for (const auto &filename : filenames) {
        std::fstream file(filename);
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
            assert(num_frames == (int)image_points_cameras.size());
        image_points_all.emplace_back(image_points_cameras);
    }

    cv::Mat visibility = cv::Mat_<int>(num_cameras, num_frames);
    for (int i = 0; i < num_cameras; i++) {
        for (int j = 0; j < num_frames; j++) {
            visibility.at<int>(i,j) = (int)(!image_points_all[i][j].empty());
        }
    }
    bool ret = calibrateMultiview (objPoints, image_points_all, image_sizes, visibility,
       Rs, Ts, Ks, distortions, cv::noArray(), cv::noArray(), is_fisheye, errors_mat, output_pairs, false/*use intrinsics guess*/));
    assert(ret);
    for (int c = 0; c < (int)Rs.size(); c++) {
        std::cout << "camera " << c << '\n';
        std::cout << Rs[c] << " rotation\n";
        std::cout << Ts[c] << " translation\n";
        std::cout << Ks[c] << " intrinsic matrix\n";
        std::cout << distortions[c] << " distortion\n";
    }
}

int main (int argc, char **argv) {
    const cv::Size pattern_size (18, 13);
    const double pattern_scale = 0.01;
    const int num_cameras = 10;
    const std::vector<bool> is_fisheye(num_cameras, false);
    const std::string pattern_type = "checkerboard", files_folder = "multiview_calibration_images/";
    std::vector<std::string> filenames(num_cameras);
    for (int i = 0; i < num_cameras; i++)
        filenames[i] = files_folder + "cam_"+std::to_string(i)+".txt";
    detectPointsAndCalibrate (pattern_size, pattern_scale, pattern_type, is_fisheye, filenames);
    return 0;
}