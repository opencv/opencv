#include "precomp.hpp"

#include <iostream>

namespace cv {

bool CalibrationResult::loadFromFile(const String& filename) {
    try {
        FileStorage fs(filename, FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Trying to open: " << filename << std::endl;
            return false;
        }
        
        fs["degree"] >> degree;
        FileNode red_node = fs["red_channel"];
        poly_red.degree = degree;
        red_node["coeffs_x"] >> poly_red.coeffs_x;
        red_node["coeffs_y"] >> poly_red.coeffs_y;
        red_node["mean_x"] >> poly_red.mean_x;
        red_node["mean_y"] >> poly_red.mean_y;
        red_node["std_x"] >> poly_red.std_x;
        red_node["std_y"] >> poly_red.std_y;
        
        FileNode blue_node = fs["blue_channel"];
        poly_blue.degree = degree;
        blue_node["coeffs_x"] >> poly_blue.coeffs_x;
        blue_node["coeffs_y"] >> poly_blue.coeffs_y;
        blue_node["mean_x"] >> poly_blue.mean_x;
        blue_node["mean_y"] >> poly_blue.mean_y;
        blue_node["std_x"] >> poly_blue.std_x;
        blue_node["std_y"] >> poly_blue.std_y;
        
        fs.release();
        return true;
    } catch (const Exception& e) {
        std::cerr << "Trying to open: " << filename << std::endl;
        std::cout << "Error loading calibration file: " << e.what() << "\n";
        return false;
    }
}

void Polynomial2D::computeDeltas(const Mat& X, const Mat& Y, Mat& dx, Mat& dy) const {
    CV_Assert(X.type() == CV_32F && Y.type() == CV_32F && X.size() == Y.size());

    const int h = X.rows, w = X.cols, D = degree;
    dx.create(X.size(), CV_32F);
    dy.create(Y.size(), CV_32F);

    const double inv_std_x = 1.0 / std_x;
    const double inv_std_y = 1.0 / std_y;

    parallel_for_( Range(0, h),
        [&](const Range& rows)
    {
        std::vector<double> x_pow(D + 1);
        std::vector<double> y_pow(D + 1);

        for (int y = rows.start; y < rows.end; ++y)
        {
            const float* XR = X.ptr<float>(y);
            const float* YR = Y.ptr<float>(y);
            float* DX = dx.ptr<float>(y);
            float* DY = dy.ptr<float>(y);

            for (int x = 0; x < w; ++x)
            {
                const double xn = (XR[x] - mean_x) * inv_std_x;
                const double yn = (YR[x] - mean_y) * inv_std_y;

                x_pow[0] = y_pow[0] = 1.0;
                for (int k = 1; k <= D; ++k)
                {
                    x_pow[k] = x_pow[k - 1] * xn;
                    y_pow[k] = y_pow[k - 1] * yn;
                }

                double dx_val = 0.0, dy_val = 0.0;
                std::size_t idx = 0;

                for (int total = 0; total <= D; ++total)
                {
                    for (int i = 0; i <= total; ++i)
                    {
                        const int j = total - i;
                        const double term = x_pow[i] * y_pow[j];
                        dx_val += coeffs_x[idx] * term;
                        dy_val += coeffs_y[idx] * term;
                        ++idx;
                    }
                }

                DX[x] = static_cast<float>(dx_val);
                DY[x] = static_cast<float>(dy_val);
            }
        }
    } );
}

void ChromaticAberrationCorrector::buildRemaps(int height, int width, const Polynomial2D& poly, 
                                                 Mat& map_x, Mat& map_y) {
    Mat X, Y;
    Mat x_coords = Mat::zeros(1, width, CV_32F);
    Mat y_coords = Mat::zeros(height, 1, CV_32F);
    
    for (int i = 0; i < width; ++i) {
        x_coords.at<float>(0, i) = static_cast<float>(i);
    }
    for (int i = 0; i < height; ++i) {
        y_coords.at<float>(i, 0) = static_cast<float>(i);
    }
    
    repeat(x_coords, height, 1, X);
    repeat(y_coords, 1, width, Y);
    
    Mat dx, dy;
    poly.computeDeltas(X, Y, dx, dy);
    
    map_x = X - dx;
    map_y = Y - dy;
}

bool ChromaticAberrationCorrector::loadCalibration(const String& calibration_file) {
    return calib_result_.loadFromFile(calibration_file);
}

Mat ChromaticAberrationCorrector::correctImage(InputArray input_image) {
    Mat image = input_image.getMat();
    CV_Assert(image.channels() == 3);
    
    const int height = image.rows;
    const int width = image.cols;
    
    std::vector<Mat> channels;
    split(image, channels);
    Mat b = channels[0], g = channels[1], r = channels[2];
    
    Mat map_x_r, map_y_r, map_x_b, map_y_b;
    buildRemaps(height, width, calib_result_.poly_red, map_x_r, map_y_r);
    buildRemaps(height, width, calib_result_.poly_blue, map_x_b, map_y_b);
    
    Mat r_corr, b_corr, g_corr;
    remap(r, r_corr, map_x_r, map_y_r, INTER_LINEAR, BORDER_REPLICATE);
    remap(b, b_corr, map_x_b, map_y_b, INTER_LINEAR, BORDER_REPLICATE);
    
    Mat map_x_g, map_y_g;
    Mat x_coords = Mat::zeros(1, width, CV_32F);
    Mat y_coords = Mat::zeros(height, 1, CV_32F);
    
    for (int i = 0; i < width; ++i) {
        x_coords.at<float>(0, i) = static_cast<float>(i);
    }
    for (int i = 0; i < height; ++i) {
        y_coords.at<float>(i, 0) = static_cast<float>(i);
    }
    
    repeat(x_coords, height, 1, map_x_g);
    repeat(y_coords, 1, width, map_y_g);
    
    g_corr = g;

    std::vector<Mat> corrected_channels = {b_corr, g_corr, r_corr};
    Mat corrected_image;
    merge(corrected_channels, corrected_image);
    
    return corrected_image;
}

Mat correctChromaticAberration(InputArray image, const String& calibration_file) {
    ChromaticAberrationCorrector corrector;
    CV_Assert(corrector.loadCalibration(calibration_file));
    return corrector.correctImage(image);
}

}
