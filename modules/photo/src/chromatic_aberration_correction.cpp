#include "precomp.hpp"

#include <iostream>

namespace cv {

bool CalibrationResult::loadFromFile(const String& filename)
{

    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()){
        CV_Error_(Error::StsError,
                    ("Cannot open calibration file: %s", filename.c_str()));
        return false;
    }


    int imgW = 0, imgH = 0;
    fs["image_width"]  >> imgW;
    fs["image_height"] >> imgH;
    if (imgW <= 0 || imgH <= 0) {
        CV_Error(cv::Error::StsBadArg,
                    "image_width and image_height must be positive");
        return false;
    }
    auto readChannel = [&](const char* key, Polynomial2D& poly)
    {
        FileNode ch = fs[key];
        if (ch.empty())
            CV_Error_(cv::Error::StsParseError,
                        ("Missing channel \"%s\"", key));

        ch["coeffs_x"] >> poly.coeffs_x;
        ch["coeffs_y"] >> poly.coeffs_y;

        if (poly.coeffs_x.empty() || poly.coeffs_y.empty())
            CV_Error_(Error::StsParseError,
                        ("%s: coeffs_x/coeffs_y missing", key));

        if (poly.coeffs_x.size() != poly.coeffs_y.size())
            CV_Error_(Error::StsBadSize,
                        ("%s: coeffs_x (%zu) vs coeffs_y (%zu)",
                        key, poly.coeffs_x.size(), poly.coeffs_y.size()));

        if (!cv::checkRange(poly.coeffs_x, true) ||
            !cv::checkRange(poly.coeffs_y, true))
            CV_Error_(Error::StsBadArg,
                        ("%s: coefficient array contains NaN/Inf", key));
        size_t m = poly.coeffs_x.size();
        double n_float = (std::sqrt(1.0 + 8.0 * m) - 3.0) / 2.0;
        int deg = static_cast<int>(std::round(n_float));
        size_t expected_m = static_cast<size_t>((deg + 1) * (deg + 2) / 2);
        if (m != expected_m){
            CV_Error_(Error::StsBadArg,
                    ("Coefficient count %zu is not triangular for degree %d "
                    "(expected %zu)", m, deg, expected_m));
        }
        poly.degree = deg;
    };

    readChannel("red_channel",  poly_red);
    readChannel("blue_channel", poly_blue);

    fs["red_channel"]["rms"] >> rms_red;
    fs["blue_channel"]["rms"] >> rms_blue;
    if (poly_red.coeffs_x.size() != poly_blue.coeffs_x.size()){
        CV_Error_(cv::Error::StsBadSize,
                    ("Red (%zu) and blue (%zu) coefficient counts differ",
                    poly_red.coeffs_x.size(), poly_blue.coeffs_x.size()));
        return false;
    }

    width = imgW;
    height = imgH;
    degree = poly_red.degree;

    return true;

}

void Polynomial2D::computeDeltas(const Mat& X, const Mat& Y, Mat& dx, Mat& dy) const {
    CV_Assert(X.type() == CV_32F && Y.type() == CV_32F && X.size() == Y.size());

    const int h = X.rows, w = X.cols, D = degree;
    dx.create(X.size(), CV_32F);
    dy.create(Y.size(), CV_32F);

    const double mean_x = w * 0.5;
    const double mean_y = h * 0.5;
    const double inv_std_x = 1.0 / mean_x;
    const double inv_std_y = 1.0 / mean_y;

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
    if (!corrector.loadCalibration(calibration_file)) {
        CV_Error_(Error::StsError, ("Failed to load chromatic-aberration calibration file: %s", calibration_file.c_str()));
    }
    return corrector.correctImage(image);
}

}
