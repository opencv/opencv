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
    dx = Mat::zeros(X.size(), CV_32F);
    dy = Mat::zeros(Y.size(), CV_32F);
    
    const int height = X.rows;
    const int width = X.cols;

    // std::cout << height << " " << width << "\n";
    
    for (int y = 0; y < height; ++y) {
        const float* x_row = X.ptr<float>(y);
        const float* y_row = Y.ptr<float>(y);
        float* dx_row = dx.ptr<float>(y);
        float* dy_row = dy.ptr<float>(y);
        
        for (int x = 0; x < width; ++x) {
            double x_norm = (x_row[x] - mean_x) / std_x;
            double y_norm = (y_row[x] - mean_y) / std_y;
            
            // Compute monomial terms and polynomial evaluation
            double delta_x = 0.0, delta_y = 0.0;
            size_t term_idx = 0;
            
            for (int total = 0; total <= degree && term_idx < coeffs_x.size(); ++total){
                for (int i = 0; i <= total && term_idx < coeffs_x.size(); ++i) { // i grows first
                    int  j    = total - i;
                    double t  = std::pow(x_norm, i) * std::pow(y_norm, j);
                    delta_x  += coeffs_x[term_idx] * t;
                    delta_y  += coeffs_y[term_idx] * t;
                    ++term_idx;
                }
            }
            dx_row[x] = static_cast<float>(delta_x);
            dy_row[x] = static_cast<float>(delta_y);
        }
    }
}

std::vector<double> ChromaticAberrationCorrector::computeMonomialTerms(double x, double y, int degree) const {
    std::vector<double> terms;
    terms.reserve((degree + 1) * (degree + 2) / 2);
    
    for (int total = 0; total <= degree; ++total) {
        for (int i = 0; i <= total; ++i) {
            int j = total - i;
            terms.push_back(std::pow(x, i) * std::pow(y, j));
        }
    }
    return terms;
}

void ChromaticAberrationCorrector::buildRemaps(int height, int width, const Polynomial2D& poly, 
                                                 Mat& map_x, Mat& map_y) {
    // Create coordinate grids
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
    
    // Compute polynomial deltas
    Mat dx, dy;
    poly.computeDeltas(X, Y, dx, dy);
    
    // Build remap matrices
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
    
    // Split channels
    std::vector<Mat> channels;
    split(image, channels);
    Mat b = channels[0], g = channels[1], r = channels[2];
    
    // Build remap maps for red and blue channels
    Mat map_x_r, map_y_r, map_x_b, map_y_b;
    buildRemaps(height, width, calib_result_.poly_red, map_x_r, map_y_r);
    buildRemaps(height, width, calib_result_.poly_blue, map_x_b, map_y_b);
    
    // Apply corrections
    Mat r_corr, b_corr, g_corr;
    remap(r, r_corr, map_x_r, map_y_r, INTER_LINEAR, BORDER_REPLICATE);
    remap(b, b_corr, map_x_b, map_y_b, INTER_LINEAR, BORDER_REPLICATE);
    
    // Green channel - identity mapping (no correction needed)
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
