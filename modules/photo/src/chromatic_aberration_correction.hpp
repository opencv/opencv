#ifndef OPENCV_PHOTO_CHROMATIC_ABERRATION_CORRECTION_HPP
#define OPENCV_PHOTO_CHROMATIC_ABERRATION_CORRECTION_HPP

#include "precomp.hpp"

namespace cv {

struct CV_EXPORTS Polynomial2D {
    std::vector<double> coeffs_x;
    std::vector<double> coeffs_y;
    int degree;
    double mean_x;
    double mean_y;
    double std_x;
    double std_y;

    
    Polynomial2D() : degree(0) {}
    
    void computeDeltas(const Mat& X, const Mat& Y, Mat& dx, Mat& dy) const;
};

struct CV_EXPORTS CalibrationResult {
    int degree;
    Polynomial2D poly_red;
    Polynomial2D poly_blue;
    int width;
    int height;
    double rms_red;
    double rms_blue;
    
    CalibrationResult() : degree(0) {}
    
    bool loadFromFile(const String& filename);
};

class CV_EXPORTS ChromaticAberrationCorrector {
public:
    ChromaticAberrationCorrector() = default;
    
    bool loadCalibration(const String& calibration_file);
    Mat correctImage(InputArray input_image);
    
private:
    CalibrationResult calib_result_;
    
    void buildRemaps(int height, int width, const Polynomial2D& poly, 
                       Mat& map_x, Mat& map_y);
    std::vector<double> computeMonomialTerms(double x, double y, int degree) const;
};

CV_EXPORTS Mat correctChromaticAberration(InputArray image, const String& calibration_file);

}

#endif // OPENCV_PHOTO_CHROMATIC_ABERRATION_CORRECTION_HPP