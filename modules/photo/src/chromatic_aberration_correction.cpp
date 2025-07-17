#include "precomp.hpp"

#include <iostream>

namespace cv {

bool loadCalibrationResultFromFile(const String& calibration_file, Mat& coeffMat, // mterms x 4, [Bx,By,Rx,Ry]
                                   int& degree,
                                   int& width,
                                   int& height) {
    FileStorage fs(calibration_file, FileStorage::READ);
    if (!fs.isOpened()){
        CV_Error_(Error::StsError,
                    ("Cannot open calibration file: %s", calibration_file.c_str()));
        return false;
    }




    int imgW = 0, imgH = 0;
    fs["image_width"]  >> imgW;
    fs["image_height"] >> imgH;
    if (imgW <= 0 || imgH <= 0) {
        CV_Error(Error::StsBadArg,
                    "image_width and image_height must be positive");
        return false;
    }
    auto readChannel = [&](const char* key, 
                            std::vector<double>& coeffs_x,
                           std::vector<double>& coeffs_y,
                           int& deg_out)
    {
        FileNode ch = fs[key];
        if (ch.empty())
            CV_Error_(Error::StsParseError,
                        ("Missing channel \"%s\"", key));

        ch["coeffs_x"] >> coeffs_x;
        ch["coeffs_y"] >> coeffs_y;

        if (coeffs_x.empty() || coeffs_y.empty())
            CV_Error_(Error::StsParseError,
                        ("%s: coeffs_x/coeffs_y missing", key));

        if (coeffs_x.size() != coeffs_y.size())
            CV_Error_(Error::StsBadSize,
                        ("%s: coeffs_x (%zu) vs coeffs_y (%zu)",
                        key, coeffs_x.size(), coeffs_y.size()));

        if (!checkRange(coeffs_x, true) ||
            !checkRange(coeffs_y, true))
            CV_Error_(Error::StsBadArg,
                        ("%s: coefficient array contains NaN/Inf", key));
        size_t m = coeffs_x.size();
        double n_float = (std::sqrt(1.0 + 8.0 * m) - 3.0) / 2.0;
        int deg = static_cast<int>(std::round(n_float));
        size_t expected_m = static_cast<size_t>((deg + 1) * (deg + 2) / 2);
        if (m != expected_m){
            CV_Error_(Error::StsBadArg,
                    ("Coefficient count %zu is not triangular for degree %d "
                    "(expected %zu)", m, deg, expected_m));
        }
        deg_out = deg;
    };

    std::vector<double> red_x, red_y, blue_x, blue_y;
    int deg_red = 0, deg_blue = 0;
    readChannel("red_channel",  red_x,  red_y,  deg_red);
    readChannel("blue_channel", blue_x, blue_y, deg_blue);


    if (red_x.size() != blue_x.size() || deg_red != deg_blue){
        CV_Error_(Error::StsBadSize,
                    ("Red (%zu) and blue (%zu) coefficient counts differ",
                    red_x.size(), blue_x.size()));
        return false;
    }

    const int mterms = (int)red_x.size();

    coeffMat.create(4, mterms, CV_32F);  // rows=4 components, cols=mterms

    float* Bx = coeffMat.ptr<float>(0);
    float* By = coeffMat.ptr<float>(1);
    float* Rx = coeffMat.ptr<float>(2);
    float* Ry = coeffMat.ptr<float>(3);

    for (int i = 0; i < mterms; ++i) {
        Bx[i] = static_cast<float>(blue_x[i]);
        By[i] = static_cast<float>(blue_y[i]);
        Rx[i] = static_cast<float>(red_x[i]);
        Ry[i] = static_cast<float>(red_y[i]);
    }

    width = imgW;
    height = imgH;
    degree = deg_red;

    return true;
}

void ChromaticAberrationCorrector::buildRemapsFromCoeffMat(int height, int width,
                             const Mat& coeffs,
                             int degree,
                             int rowX, int rowY,
                             Mat& map_x, Mat& map_y)
{
    CV_Assert(coeffs.type() == CV_32F);
    CV_Assert(rowX >= 0 && rowX < coeffs.cols);
    CV_Assert(rowY >= 0 && rowY < coeffs.cols);
    const int expected_rows = (degree + 1) * (degree + 2) / 2;
    CV_Assert(coeffs.cols == expected_rows);
    CV_Assert(coeffs.rows == 4);

    Mat X(1, width, CV_32F), Y(height, 1, CV_32F);
    for (int i = 0; i < width;  ++i) X.at<float>(0,i) = (float)i;
    for (int j = 0; j < height; ++j) Y.at<float>(j,0) = (float)j;

    Mat Xgrid, Ygrid;
    repeat(X, height, 1, Xgrid);
    repeat(Y, 1, width,  Ygrid);

    Mat dx(height, width, CV_32F);
    Mat dy(height, width, CV_32F);

    const double mean_x    = width  * 0.5;
    const double mean_y    = height * 0.5;
    const double inv_std_x = 1.0 / mean_x;
    const double inv_std_y = 1.0 / mean_y;

    const float* base = coeffs.ptr<float>(0);
    const int stride  = coeffs.cols;

    const float* Cx = coeffs.ptr<float>(rowX);
    const float* Cy = coeffs.ptr<float>(rowY);


    parallel_for_(Range(0, height), [&](const Range& rows){
        std::vector<double> x_pow(degree + 1);
        std::vector<double> y_pow(degree + 1);
        for (int y = rows.start; y < rows.end; ++y) {
            const float* XR = Xgrid.ptr<float>(y);
            const float* YR = Ygrid.ptr<float>(y);
            float* DX = dx.ptr<float>(y);
            float* DY = dy.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                const double xn = (XR[x] - mean_x) * inv_std_x;
                const double yn = (YR[x] - mean_y) * inv_std_y;

                x_pow[0] = y_pow[0] = 1.0;
                for (int k = 1; k <= degree; ++k) {
                    x_pow[k] = x_pow[k-1] * xn;
                    y_pow[k] = y_pow[k-1] * yn;
                }

                double dxv = 0.0, dyv = 0.0;
                int idx = 0;
                for (int t = 0; t <= degree; ++t){
                    for (int i = 0; i <= t; ++i){
                        const int j = t - i;
                        const double term = x_pow[i] * y_pow[j];
                        dxv += Cx[idx] * term;
                        dyv += Cy[idx] * term;
                        ++idx;
                    }
                }

                DX[x] = (float)dxv;
                DY[x] = (float)dyv;
            }
        }
    });

    map_x = Xgrid - dx;
    map_y = Ygrid - dy;
}


Mat ChromaticAberrationCorrector::correctImage(InputArray input_image) {
    Mat image = input_image.getMat();
    if(image.channels() != 3) {
        CV_Error_(Error::StsBadArg,
                    ("images need to have 3 channels"));
    
    }
    
    const int height = image.rows;
    const int width = image.cols;

    if (height != height_ || width != width_) {
        CV_Error_(Error::StsBadArg,
                    ("This corrector can only be used with the following image sizes: : %d and %d",
                    height_,
                    width_));
    }
    
    std::vector<Mat> channels;
    split(image, channels);
    Mat b = channels[0], g = channels[1], r = channels[2];
    
    Mat map_x_r, map_y_r, map_x_b, map_y_b;

    buildRemapsFromCoeffMat(height, width, coeffMat_, degree_, 2, 3, map_x_r, map_y_r);
    buildRemapsFromCoeffMat(height, width, coeffMat_, degree_, 0, 1, map_x_b, map_y_b);
    
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



ChromaticAberrationCorrector::ChromaticAberrationCorrector(const String& calibration_file) {
    if (!loadCalibrationResultFromFile(calibration_file, coeffMat_, degree_, width_, height_)) {
        CV_Error_(Error::StsError, ("Failed to load chromatic-aberration calibration file: %s", calibration_file.c_str()));
    }
}

Mat correctChromaticAberration(InputArray image, const String& calibration_file) {
    ChromaticAberrationCorrector corrector(calibration_file);
    return corrector.correctImage(image);
}

}
