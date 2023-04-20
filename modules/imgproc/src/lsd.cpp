/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <vector>

/////////////////////////////////////////////////////////////////////////////////////////
// Default LSD parameters
// SIGMA_SCALE 0.6    - Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
// QUANT       2.0    - Bound to the quantization error on the gradient norm.
// ANG_TH      22.5   - Gradient angle tolerance in degrees.
// LOG_EPS     0.0    - Detection threshold: -log10(NFA) > log_eps
// DENSITY_TH  0.7    - Minimal density of region points in rectangle.
// N_BINS      1024   - Number of bins in pseudo-ordering of gradient modulus.

#define M_3_2_PI    (3 * CV_PI) / 2   // 3/2 pi
#define M_2__PI     (2 * CV_PI)         // 2 pi

#ifndef M_LN10
#define M_LN10      2.30258509299404568402
#endif

#define NOTDEF      double(-1024.0) // Label for pixels with undefined gradient.

#define NOTUSED     0   // Label for pixels not used in yet.
#define USED        1   // Label for pixels already used in detection.

#define RELATIVE_ERROR_FACTOR 100.0

const double DEG_TO_RADS = CV_PI / 180;

#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/////////////////////////////////////////////////////////////////////////////////////////

inline double distSq(const double x1, const double y1,
                     const double x2, const double y2)
{
    return (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
}

inline double dist(const double x1, const double y1,
                   const double x2, const double y2)
{
    return sqrt(distSq(x1, y1, x2, y2));
}

// Signed angle difference
inline double angle_diff_signed(const double& a, const double& b)
{
    double diff = a - b;
    while(diff <= -CV_PI) diff += M_2__PI;
    while(diff >   CV_PI) diff -= M_2__PI;
    return diff;
}

// Absolute value angle difference
inline double angle_diff(const double& a, const double& b)
{
    return std::fabs(angle_diff_signed(a, b));
}

// Compare doubles by relative error.
inline bool double_equal(const double& a, const double& b)
{
    // trivial case
    if(a == b) return true;

    double abs_diff = fabs(a - b);
    double aa = fabs(a);
    double bb = fabs(b);
    double abs_max = (aa > bb)? aa : bb;

    if(abs_max < DBL_MIN) abs_max = DBL_MIN;

    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}

// function to sort points by y and then by x
inline bool AsmallerB_YoverX(const cv::Point2d &a, const cv::Point2d &b) {
    if (a.y == b.y) return a.x < b.x;
    else return a.y < b.y;
}

// function to get the slope of the rectangle for a specific row
inline double get_slope(cv::Point2d p1, cv::Point2d p2) {
    return ((int) ceil(p2.y) != (int) ceil(p1.y)) ? (p2.x - p1.x) / (p2.y - p1.y) : 0;
}

// function to get the limit of the rectangle for a specific row
inline double get_limit(cv::Point2d p, int row, double slope) {
    return p.x + (row - p.y) * slope;
}

/**
 *   Computes the natural logarithm of the absolute value of
 *   the gamma function of x using Windschitl method.
 *   See http://www.rskey.org/gamma.htm
 */
inline double log_gamma_windschitl(const double& x)
{
    return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log(x*sinh(1/x) + 1/(810.0*pow(x, 6.0)));
}

/**
 *   Computes the natural logarithm of the absolute value of
 *   the gamma function of x using the Lanczos approximation.
 *   See http://www.rskey.org/gamma.htm
 */
inline double log_gamma_lanczos(const double& x)
{
    static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
    double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
    double b = 0;
    for(int n = 0; n < 7; ++n)
    {
        a -= log(x + double(n));
        b += q[n] * pow(x, double(n));
    }
    return a + log(b);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv{

class LineSegmentDetectorImpl CV_FINAL : public LineSegmentDetector
{
public:

/**
 * Create a LineSegmentDetectorImpl object. Specifying scale, number of subdivisions for the image, should the lines be refined and other constants as follows:
 *
 * @param _refine       How should the lines found be refined?
 *                      LSD_REFINE_NONE - No refinement applied.
 *                      LSD_REFINE_STD  - Standard refinement is applied. E.g. breaking arches into smaller line approximations.
 *                      LSD_REFINE_ADV  - Advanced refinement. Number of false alarms is calculated,
 *                                    lines are refined through increase of precision, decrement in size, etc.
 * @param _scale        The scale of the image that will be used to find the lines. Range (0..1].
 * @param _sigma_scale  Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
 * @param _quant        Bound to the quantization error on the gradient norm.
 * @param _ang_th       Gradient angle tolerance in degrees.
 * @param _log_eps      Detection threshold: -log10(NFA) > _log_eps
 * @param _density_th   Minimal density of aligned region points in rectangle.
 * @param _n_bins       Number of bins in pseudo-ordering of gradient modulus.
 */
    LineSegmentDetectorImpl(int _refine = LSD_REFINE_STD, double _scale = 0.8,
        double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
        double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

/**
 * Detect lines in the input image.
 *
 * @param _image    A grayscale(CV_8UC1) input image.
 *                  If only a roi needs to be selected, use
 *                  lsd_ptr->detect(image(roi), ..., lines);
 *                  lines += Scalar(roi.x, roi.y, roi.x, roi.y);
 * @param _lines    Return: A vector of Vec4f elements specifying the beginning and ending point of a line.
 *                          Where Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
 *                          Returned lines are strictly oriented depending on the gradient.
 * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param prec      Return: Vector of precisions with which the lines are found.
 * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%.
 *                          The bigger the value, logarithmically better the detection.
 *                              * -1 corresponds to 10 mean false alarms
 *                              * 0 corresponds to 1 mean false alarm
 *                              * 1 corresponds to 0.1 mean false alarms
 *                          This vector will be calculated _only_ when the objects type is REFINE_ADV
 */
    void detect(InputArray _image, OutputArray _lines,
                OutputArray width = noArray(), OutputArray prec = noArray(),
                OutputArray nfa = noArray()) CV_OVERRIDE;

/**
 * Draw lines on the given canvas.
 *
 * @param image     The image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @param lines     The lines that need to be drawn
 */
    void drawSegments(InputOutputArray _image, InputArray lines) CV_OVERRIDE;

/**
 * Draw both vectors on the image canvas. Uses blue for lines 1 and red for lines 2.
 *
 * @param size      The size of the image, where lines1 and lines2 were found.
 * @param lines1    The first lines that need to be drawn. Color - Blue.
 * @param lines2    The second lines that need to be drawn. Color - Red.
 * @param image     An optional image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @return          The number of mismatching pixels between lines1 and lines2.
 */
    int compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image = noArray()) CV_OVERRIDE;

private:
    Mat image;
    Mat scaled_image;
    Mat_<double> angles;     // in rads
    Mat_<double> modgrad;
    Mat_<uchar> used;

    int img_width;
    int img_height;
    double LOG_NT;

    bool w_needed;
    bool p_needed;
    bool n_needed;

    const double SCALE;
    const int doRefine;
    const double SIGMA_SCALE;
    const double QUANT;
    const double ANG_TH;
    const double LOG_EPS;
    const double DENSITY_TH;
    const int N_BINS;

    struct RegionPoint {
        int x;
        int y;
        uchar* used;
        double angle;
        double modgrad;
    };

    struct normPoint
    {
        Point2i p;
        int norm;
    };

    std::vector<normPoint> ordered_points;

    struct rect
    {
        double x1, y1, x2, y2;    // first and second point of the line segment
        double width;             // rectangle width
        double x, y;              // center of the rectangle
        double theta;             // angle
        double dx,dy;             // (dx,dy) is vector oriented as the line segment
        double prec;              // tolerance angle
        double p;                 // probability of a point with angle within 'prec'
    };

    LineSegmentDetectorImpl& operator= (const LineSegmentDetectorImpl&); // to quiet MSVC

/**
 * Detect lines in the whole input image.
 *
 * @param lines         Return: A vector of Vec4f elements specifying the beginning and ending point of a line.
 *                              Where Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
 *                              Returned lines are strictly oriented depending on the gradient.
 * @param widths        Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param precisions    Return: Vector of precisions with which the lines are found.
 * @param nfas          Return: Vector containing number of false alarms in the line region, with precision of 10%.
 *                              The bigger the value, logarithmically better the detection.
 *                                  * -1 corresponds to 10 mean false alarms
 *                                  * 0 corresponds to 1 mean false alarm
 *                                  * 1 corresponds to 0.1 mean false alarms
 */
    void flsd(std::vector<Vec4f>& lines,
              std::vector<double>& widths, std::vector<double>& precisions,
              std::vector<double>& nfas);

/**
 * Finds the angles and the gradients of the image. Generates a list of pseudo ordered points.
 *
 * @param threshold      The minimum value of the angle that is considered defined, otherwise NOTDEF
 * @param n_bins         The number of bins with which gradients are ordered by, using bucket sort.
 * @param ordered_points Return: Vector of coordinate points that are pseudo ordered by magnitude.
 *                       Pixels would be ordered by norm value, up to a precision given by max_grad/n_bins.
 */
    void ll_angle(const double& threshold, const unsigned int& n_bins);

/**
 * Grow a region starting from point s with a defined precision,
 * returning the containing points size and the angle of the gradients.
 *
 * @param s         Starting point for the region.
 * @param reg       Return: Vector of points, that are part of the region
 * @param reg_angle Return: The mean angle of the region.
 * @param prec      The precision by which each region angle should be aligned to the mean.
 */
    void region_grow(const Point2i& s, std::vector<RegionPoint>& reg,
                     double& reg_angle, const double& prec);

/**
 * Finds the bounding rotated rectangle of a region.
 *
 * @param reg       The region of points, from which the rectangle to be constructed from.
 * @param reg_angle The mean angle of the region.
 * @param prec      The precision by which points were found.
 * @param p         Probability of a point with angle within 'prec'.
 * @param rec       Return: The generated rectangle.
 */
    void region2rect(const std::vector<RegionPoint>& reg, const double reg_angle,
                     const double prec, const double p, rect& rec) const;

/**
 * Compute region's angle as the principal inertia axis of the region.
 * @return          Regions angle.
 */
    double get_theta(const std::vector<RegionPoint>& reg, const double& x,
                     const double& y, const double& reg_angle, const double& prec) const;

/**
 * An estimation of the angle tolerance is performed by the standard deviation of the angle at points
 * near the region's starting point. Then, a new region is grown starting from the same point, but using the
 * estimated angle tolerance. If this fails to produce a rectangle with the right density of region points,
 * 'reduce_region_radius' is called to try to satisfy this condition.
 */
    bool refine(std::vector<RegionPoint>& reg, double reg_angle,
                const double prec, double p, rect& rec, const double& density_th);

/**
 * Reduce the region size, by elimination the points far from the starting point, until that leads to
 * rectangle with the right density of region points or to discard the region if too small.
 */
    bool reduce_region_radius(std::vector<RegionPoint>& reg, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th);

/**
 * Try some rectangles variations to improve NFA value. Only if the rectangle is not meaningful (i.e., log_nfa <= log_eps).
 * @return      The new NFA value.
 */
    double rect_improve(rect& rec) const;

/**
 * Calculates the number of correctly aligned points within the rectangle.
 * @return      The new NFA value.
 */
    double rect_nfa(const rect& rec) const;

/**
 * Computes the NFA values based on the total number of points, points that agree.
 * n, k, p are the binomial parameters.
 * @return      The new NFA value.
 */
    double nfa(const int& n, const int& k, const double& p) const;

/**
 * Is the point at place 'address' aligned to angle theta, up to precision 'prec'?
 * @return      Whether the point is aligned.
 */
    bool isAligned(int x, int y, const double& theta, const double& prec) const;

public:
    // Compare norm
    static inline bool compare_norm( const normPoint& n1, const normPoint& n2 )
    {
        return (n1.norm > n2.norm);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////

CV_EXPORTS Ptr<LineSegmentDetector> createLineSegmentDetector(
        int _refine, double _scale, double _sigma_scale, double _quant, double _ang_th,
        double _log_eps, double _density_th, int _n_bins)
{
    return makePtr<LineSegmentDetectorImpl>(
            _refine, _scale, _sigma_scale, _quant, _ang_th,
            _log_eps, _density_th, _n_bins);
}

/////////////////////////////////////////////////////////////////////////////////////////

LineSegmentDetectorImpl::LineSegmentDetectorImpl(int _refine, double _scale, double _sigma_scale, double _quant,
        double _ang_th, double _log_eps, double _density_th, int _n_bins)
        : img_width(0), img_height(0), LOG_NT(0), w_needed(false), p_needed(false), n_needed(false),
          SCALE(_scale), doRefine(_refine), SIGMA_SCALE(_sigma_scale), QUANT(_quant),
          ANG_TH(_ang_th), LOG_EPS(_log_eps), DENSITY_TH(_density_th), N_BINS(_n_bins)
{
    CV_Assert(_scale > 0 && _sigma_scale > 0 && _quant >= 0 &&
              _ang_th > 0 && _ang_th < 180 && _density_th >= 0 && _density_th < 1 &&
              _n_bins > 0);
}

void LineSegmentDetectorImpl::detect(InputArray _image, OutputArray _lines,
                OutputArray _width, OutputArray _prec, OutputArray _nfa)
{
    CV_INSTRUMENT_REGION();

    image = _image.getMat();
    CV_Assert(!image.empty() && image.type() == CV_8UC1);

    std::vector<Vec4f> lines;
    std::vector<double> w, p, n;
    w_needed = _width.needed();
    p_needed = _prec.needed();
    if (doRefine < LSD_REFINE_ADV)
        n_needed = false;
    else
        n_needed = _nfa.needed();

    flsd(lines, w, p, n);

    Mat(lines).copyTo(_lines);
    if(w_needed) Mat(w).copyTo(_width);
    if(p_needed) Mat(p).copyTo(_prec);
    if(n_needed) Mat(n).copyTo(_nfa);

    // Clear used structures
    ordered_points.clear();
}

void LineSegmentDetectorImpl::flsd(std::vector<Vec4f>& lines,
    std::vector<double>& widths, std::vector<double>& precisions,
    std::vector<double>& nfas)
{
    // Angle tolerance
    const double prec = CV_PI * ANG_TH / 180;
    const double p = ANG_TH / 180;
    const double rho = QUANT / sin(prec);    // gradient magnitude threshold

    if(SCALE != 1)
    {
        Mat gaussian_img;
        const double sigma = (SCALE < 1)?(SIGMA_SCALE / SCALE):(SIGMA_SCALE);
        const double sprec = 3;
        const unsigned int h =  (unsigned int)(ceil(sigma * sqrt(2 * sprec * log(10.0))));
        Size ksize(1 + 2 * h, 1 + 2 * h); // kernel size
        GaussianBlur(image, gaussian_img, ksize, sigma);
        // Scale image to needed size
        resize(gaussian_img, scaled_image, Size(), SCALE, SCALE, INTER_LINEAR_EXACT);
        ll_angle(rho, N_BINS);
    }
    else
    {
        scaled_image = image;
        ll_angle(rho, N_BINS);
    }

    LOG_NT = 5 * (log10(double(img_width)) + log10(double(img_height))) / 2 + log10(11.0);
    const size_t min_reg_size = size_t(-LOG_NT/log10(p)); // minimal number of points in region that can give a meaningful event

    // // Initialize region only when needed
    // Mat region = Mat::zeros(scaled_image.size(), CV_8UC1);
    used = Mat_<uchar>::zeros(scaled_image.size()); // zeros = NOTUSED
    std::vector<RegionPoint> reg;

    // Search for line segments
    for(size_t i = 0, points_size = ordered_points.size(); i < points_size; ++i)
    {
        const Point2i& point = ordered_points[i].p;
        if((used.at<uchar>(point) == NOTUSED) && (angles.at<double>(point) != NOTDEF))
        {
            double reg_angle;
            region_grow(ordered_points[i].p, reg, reg_angle, prec);

            // Ignore small regions
            if(reg.size() < min_reg_size) { continue; }

            // Construct rectangular approximation for the region
            rect rec;
            region2rect(reg, reg_angle, prec, p, rec);

            double log_nfa = -1;
            if(doRefine > LSD_REFINE_NONE)
            {
                // At least REFINE_STANDARD lvl.
                if(!refine(reg, reg_angle, prec, p, rec, DENSITY_TH)) { continue; }

                if(doRefine >= LSD_REFINE_ADV)
                {
                    // Compute NFA
                    log_nfa = rect_improve(rec);
                    if(log_nfa <= LOG_EPS) { continue; }
                }
            }
            // Found new line

            // Add the offset
            rec.x1 += 0.5; rec.y1 += 0.5;
            rec.x2 += 0.5; rec.y2 += 0.5;

            // scale the result values if a sub-sampling was performed
            if(SCALE != 1)
            {
                rec.x1 /= SCALE; rec.y1 /= SCALE;
                rec.x2 /= SCALE; rec.y2 /= SCALE;
                rec.width /= SCALE;
            }

            //Store the relevant data
            lines.push_back(Vec4f(float(rec.x1), float(rec.y1), float(rec.x2), float(rec.y2)));
            if(w_needed) widths.push_back(rec.width);
            if(p_needed) precisions.push_back(rec.p);
            if(n_needed && doRefine >= LSD_REFINE_ADV) nfas.push_back(log_nfa);
        }
    }
}

void LineSegmentDetectorImpl::ll_angle(const double& threshold,
                                   const unsigned int& n_bins)
{
    //Initialize data
    angles = Mat_<double>(scaled_image.size());
    modgrad = Mat_<double>(scaled_image.size());

    img_width = scaled_image.cols;
    img_height = scaled_image.rows;

    // Undefined the down and right boundaries
    angles.row(img_height - 1).setTo(NOTDEF);
    angles.col(img_width - 1).setTo(NOTDEF);

    // Computing gradient for remaining pixels
    double max_grad = -1;
    for(int y = 0; y < img_height - 1; ++y)
    {
        const uchar* scaled_image_row = scaled_image.ptr<uchar>(y);
        const uchar* next_scaled_image_row = scaled_image.ptr<uchar>(y+1);
        double* angles_row = angles.ptr<double>(y);
        double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width-1; ++x)
        {
            int DA = next_scaled_image_row[x + 1] - scaled_image_row[x];
            int BC = scaled_image_row[x + 1] - next_scaled_image_row[x];
            int gx = DA + BC;    // gradient x component
            int gy = DA - BC;    // gradient y component
            double norm = std::sqrt((gx * gx + gy * gy) / 4.0); // gradient norm

            modgrad_row[x] = norm;    // store gradient

            if (norm <= threshold)  // norm too small, gradient no defined
            {
                angles_row[x] = NOTDEF;
            }
            else
            {
                angles_row[x] = fastAtan2(float(gx), float(-gy)) * DEG_TO_RADS;  // gradient angle computation
                if (norm > max_grad) { max_grad = norm; }
            }

        }
    }

    // Compute histogram of gradient values
    double bin_coef = (max_grad > 0) ? double(n_bins - 1) / max_grad : 0; // If all image is smooth, max_grad <= 0
    for(int y = 0; y < img_height - 1; ++y)
    {
        const double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width - 1; ++x)
        {
            normPoint _point;
            int i = int(modgrad_row[x] * bin_coef);
            _point.p = Point(x, y);
            _point.norm = i;
            ordered_points.push_back(_point);
        }
    }

    // Use stable sort to ensure deterministic region growing and thus overall LSD result determinism.
    std::stable_sort(ordered_points.begin(), ordered_points.end(), compare_norm);
}

void LineSegmentDetectorImpl::region_grow(const Point2i& s, std::vector<RegionPoint>& reg,
                                      double& reg_angle, const double& prec)
{
    reg.clear();

    // Point to this region
    RegionPoint seed;
    seed.x = s.x;
    seed.y = s.y;
    seed.used = &used.at<uchar>(s);
    reg_angle = angles.at<double>(s);
    seed.angle = reg_angle;
    seed.modgrad = modgrad.at<double>(s);
    reg.push_back(seed);

    float sumdx = float(std::cos(reg_angle));
    float sumdy = float(std::sin(reg_angle));
    *seed.used = USED;

    //Try neighboring regions
    for (size_t i = 0;i<reg.size();i++)
    {
        const RegionPoint& rpoint = reg[i];
        int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
        int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);
        for(int yy = yy_min; yy <= yy_max; ++yy)
        {
            uchar* used_row = used.ptr<uchar>(yy);
            const double* angles_row = angles.ptr<double>(yy);
            const double* modgrad_row = modgrad.ptr<double>(yy);
            for(int xx = xx_min; xx <= xx_max; ++xx)
            {
                uchar& is_used = used_row[xx];
                if(is_used != USED &&
                   (isAligned(xx, yy, reg_angle, prec)))
                {
                    const double& angle = angles_row[xx];
                    // Add point
                    is_used = USED;
                    RegionPoint region_point;
                    region_point.x = xx;
                    region_point.y = yy;
                    region_point.used = &is_used;
                    region_point.modgrad = modgrad_row[xx];
                    region_point.angle = angle;
                    reg.push_back(region_point);

                    // Update region's angle
                    sumdx += cos(float(angle));
                    sumdy += sin(float(angle));
                    // reg_angle is used in the isAligned, so it needs to be updates?
                    reg_angle = fastAtan2(sumdy, sumdx) * DEG_TO_RADS;
                }
            }
        }
    }
}

void LineSegmentDetectorImpl::region2rect(const std::vector<RegionPoint>& reg,
                                      const double reg_angle, const double prec, const double p, rect& rec) const
{
    double x = 0, y = 0, sum = 0;
    for(size_t i = 0; i < reg.size(); ++i)
    {
        const RegionPoint& pnt = reg[i];
        const double& weight = pnt.modgrad;
        x += double(pnt.x) * weight;
        y += double(pnt.y) * weight;
        sum += weight;
    }

    // Weighted sum must differ from 0
    CV_Assert(sum > 0);

    x /= sum;
    y /= sum;

    double theta = get_theta(reg, x, y, reg_angle, prec);

    // Find length and width
    double dx = cos(theta);
    double dy = sin(theta);
    double l_min = 0, l_max = 0, w_min = 0, w_max = 0;

    for(size_t i = 0; i < reg.size(); ++i)
    {
        double regdx = double(reg[i].x) - x;
        double regdy = double(reg[i].y) - y;

        double l = regdx * dx + regdy * dy;
        double w = -regdx * dy + regdy * dx;

        if(l > l_max) l_max = l;
        else if(l < l_min) l_min = l;
        if(w > w_max) w_max = w;
        else if(w < w_min) w_min = w;
    }

    // Store values
    rec.x1 = x + l_min * dx;
    rec.y1 = y + l_min * dy;
    rec.x2 = x + l_max * dx;
    rec.y2 = y + l_max * dy;
    rec.width = w_max - w_min;
    rec.x = x;
    rec.y = y;
    rec.theta = theta;
    rec.dx = dx;
    rec.dy = dy;
    rec.prec = prec;
    rec.p = p;

    // Min width of 1 pixel
    if(rec.width < 1.0) rec.width = 1.0;
}

double LineSegmentDetectorImpl::get_theta(const std::vector<RegionPoint>& reg, const double& x,
                                      const double& y, const double& reg_angle, const double& prec) const
{
    double Ixx = 0.0;
    double Iyy = 0.0;
    double Ixy = 0.0;

    // Compute inertia matrix
    for(size_t i = 0; i < reg.size(); ++i)
    {
        const double& regx = reg[i].x;
        const double& regy = reg[i].y;
        const double& weight = reg[i].modgrad;
        double dx = regx - x;
        double dy = regy - y;
        Ixx += dy * dy * weight;
        Iyy += dx * dx * weight;
        Ixy -= dx * dy * weight;
    }

    // Check if inertia matrix is null
    CV_Assert(!(double_equal(Ixx, 0) && double_equal(Iyy, 0) && double_equal(Ixy, 0)));

    // Compute smallest eigenvalue
    double lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

    // Compute angle
    double theta = (fabs(Ixx)>fabs(Iyy))?
                    double(fastAtan2(float(lambda - Ixx), float(Ixy))):
                    double(fastAtan2(float(Ixy), float(lambda - Iyy))); // in degs
    theta *= DEG_TO_RADS;

    // Correct angle by 180 deg if necessary
    if(angle_diff(theta, reg_angle) > prec) { theta += CV_PI; }

    return theta;
}

bool LineSegmentDetectorImpl::refine(std::vector<RegionPoint>& reg, double reg_angle,
                                 const double prec, double p, rect& rec, const double& density_th)
{
    double density = double(reg.size()) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density >= density_th) { return true; }

    // Try to reduce angle tolerance
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    const double& ang_c = reg[0].angle;
    double sum = 0, s_sum = 0;
    int n = 0;

    for (size_t i = 0; i < reg.size(); ++i)
    {
        *(reg[i].used) = NOTUSED;
        if (dist(xc, yc, reg[i].x, reg[i].y) < rec.width)
        {
            const double& angle = reg[i].angle;
            double ang_d = angle_diff_signed(angle, ang_c);
            sum += ang_d;
            s_sum += ang_d * ang_d;
            ++n;
        }
    }
    CV_Assert(n > 0);
    double mean_angle = sum / double(n);
    // 2 * standard deviation
    double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / double(n) + mean_angle * mean_angle);

    // Try new region
    region_grow(Point(reg[0].x, reg[0].y), reg, reg_angle, tau);

    if (reg.size() < 2) { return false; }

    region2rect(reg, reg_angle, prec, p, rec);
    density = double(reg.size()) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density < density_th)
    {
        return reduce_region_radius(reg, reg_angle, prec, p, rec, density, density_th);
    }
    else
    {
        return true;
    }
}

bool LineSegmentDetectorImpl::reduce_region_radius(std::vector<RegionPoint>& reg, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th)
{
    // Compute region's radius
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    double radSq1 = distSq(xc, yc, rec.x1, rec.y1);
    double radSq2 = distSq(xc, yc, rec.x2, rec.y2);
    double radSq = radSq1 > radSq2 ? radSq1 : radSq2;

    while(density < density_th)
    {
        radSq *= 0.75*0.75; // Reduce region's radius to 75% of its value
        // Remove points from the region and update 'used' map
        for (size_t i = 0; i < reg.size(); ++i)
        {
            if(distSq(xc, yc, double(reg[i].x), double(reg[i].y)) > radSq)
            {
                // Remove point from the region
                *(reg[i].used) = NOTUSED;
                std::swap(reg[i], reg[reg.size() - 1]);
                reg.pop_back();
                --i; // To avoid skipping one point
            }
        }

        if(reg.size() < 2) { return false; }

        // Re-compute rectangle
        region2rect(reg ,reg_angle, prec, p, rec);

        // Re-compute region points density
        density = double(reg.size()) /
                  (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);
    }

    return true;
}

double LineSegmentDetectorImpl::rect_improve(rect& rec) const
{
    double delta = 0.5;
    double delta_2 = delta / 2.0;

    double log_nfa = rect_nfa(rec);

    if(log_nfa > LOG_EPS) return log_nfa; // Good rectangle

    // Try to improve
    // Finer precision
    rect r = rect(rec); // Copy
    for(int n = 0; n < 5; ++n)
    {
        r.p /= 2;
        r.prec = r.p * CV_PI;
        double log_nfa_new = rect_nfa(r);
        if(log_nfa_new > log_nfa)
        {
            log_nfa = log_nfa_new;
            rec = rect(r);
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    // Try to reduce width
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    // Try to reduce one side of rectangle
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.x1 += -r.dy * delta_2;
            r.y1 +=  r.dx * delta_2;
            r.x2 += -r.dy * delta_2;
            r.y2 +=  r.dx * delta_2;
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    // Try to reduce other side of rectangle
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.x1 -= -r.dy * delta_2;
            r.y1 -=  r.dx * delta_2;
            r.x2 -= -r.dy * delta_2;
            r.y2 -=  r.dx * delta_2;
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    // Try finer precision
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.p /= 2;
            r.prec = r.p * CV_PI;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }

    return log_nfa;
}

double LineSegmentDetectorImpl::rect_nfa(const rect& rec) const
{
    int total_pts = 0, alg_pts = 0;
    double half_width = rec.width / 2.0;
    double dyhw = rec.dy * half_width;
    double dxhw = rec.dx * half_width;

    cv::Point2d v_tmp[4];
    v_tmp[0] = cv::Point2d(rec.x1 - dyhw, rec.y1 + dxhw);
    v_tmp[1] = cv::Point2d(rec.x2 - dyhw, rec.y2 + dxhw);
    v_tmp[2] = cv::Point2d(rec.x2 + dyhw, rec.y2 - dxhw);
    v_tmp[3] = cv::Point2d(rec.x1 + dyhw, rec.y1 - dxhw);

    // Find the vertex with the smallest y coordinate (or the smallest x if there is a tie).
    int offset = 0;
    for (int i = 1; i < 4; ++i) {
        if (AsmallerB_YoverX(v_tmp[i], v_tmp[offset])){
            offset = i;
        }
    }

    // Rotate the vertices so that the first one is the one with the smallest y coordinate (or the smallest x if there is a tie).
    // The rest will be then ordered counterclockwise.
    cv::Point2d ordered_y[4];
    for (int i = 0; i < 4; ++i) {
        ordered_y[i] = v_tmp[(i + offset) % 4];
    }

    double flstep = get_slope(ordered_y[0], ordered_y[1]); //first left step
    double slstep = get_slope(ordered_y[1], ordered_y[2]); //second left step

    double frstep = get_slope(ordered_y[0], ordered_y[3]); //first right step
    double srstep = get_slope(ordered_y[3], ordered_y[2]); //second right step

    double top_y = ordered_y[0].y, bottom_y = ordered_y[2].y;

    // Loop around all points in the region and count those that are aligned.
    std::vector<cv::Point> points;
    double left_limit, right_limit;
    for(int y = (int) ceil(top_y); y <= (int) ceil(bottom_y); ++y)
    {
        if (y < 0 || y >= img_height) continue;

        if(y <= int(ceil(ordered_y[1].y)))
            left_limit = get_limit(ordered_y[0], y, flstep);
        else
            left_limit = get_limit(ordered_y[1], y, slstep);

        if(y < int(ceil(ordered_y[3].y)))
            right_limit = get_limit(ordered_y[0], y, frstep);
        else
            right_limit = get_limit(ordered_y[3], y, srstep);

        for(int x = (int) ceil(left_limit); x <= (int)(right_limit); ++x) {
            if (x < 0 || x >= img_width) continue;

            ++total_pts;
            if(isAligned(x, y, rec.theta, rec.prec))
            {
                ++alg_pts;
            }
        }
    }

    return nfa(total_pts, alg_pts, rec.p);
}

double LineSegmentDetectorImpl::nfa(const int& n, const int& k, const double& p) const
{
    // Trivial cases
    if(n == 0 || k == 0) { return -LOG_NT; }
    if(n == k) { return -LOG_NT - double(n) * log10(p); }

    double p_term = p / (1 - p);

    double log1term = log_gamma(double(n) + 1) - log_gamma(double(k) + 1)
                - log_gamma(double(n-k) + 1)
                + double(k) * log(p) + double(n-k) * log(1.0 - p);
    double term = exp(log1term);

    if(double_equal(term, 0))
    {
        if(k > n * p) return -log1term / M_LN10 - LOG_NT;
        else return -LOG_NT;
    }

    // Compute more terms if needed
    double bin_tail = term;
    double tolerance = 0.1; // an error of 10% in the result is accepted
    for(int i = k + 1; i <= n; ++i)
    {
        double bin_term = double(n - i + 1) / double(i);
        double mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;
        if(bin_term < 1)
        {
            double err = term * ((1 - pow(mult_term, double(n-i+1))) / (1 - mult_term) - 1);
            if(err < tolerance * fabs(-log10(bin_tail) - LOG_NT) * bin_tail) break;
        }

    }
    return -log10(bin_tail) - LOG_NT;
}

inline bool LineSegmentDetectorImpl::isAligned(int x, int y, const double& theta, const double& prec) const
{
    if(x < 0 || y < 0 || x >= angles.cols || y >= angles.rows) { return false; }
    const double& a = angles.at<double>(y, x);
    if(a == NOTDEF) { return false; }

    // It is assumed that 'theta' and 'a' are in the range [-pi,pi]
    double n_theta = theta - a;
    if(n_theta < 0) { n_theta = -n_theta; }
    if(n_theta > M_3_2_PI)
    {
        n_theta -= M_2__PI;
        if(n_theta < 0) n_theta = -n_theta;
    }

    return n_theta <= prec;
}


void LineSegmentDetectorImpl::drawSegments(InputOutputArray _image, InputArray lines)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_image.empty() && (_image.channels() == 1 || _image.channels() == 3));

    if (_image.channels() == 1)
    {
        cvtColor(_image, _image, COLOR_GRAY2BGR);
    }

    Mat _lines = lines.getMat();
    const int N = _lines.checkVector(4);

    CV_Assert(_lines.depth() == CV_32F || _lines.depth() == CV_32S);

    // Draw segments
    if (_lines.depth() == CV_32F)
    {
        for (int i = 0; i < N; ++i)
        {
            const Vec4f& v = _lines.at<Vec4f>(i);
            const Point2f b(v[0], v[1]);
            const Point2f e(v[2], v[3]);
            line(_image, b, e, Scalar(0, 0, 255), 1);
        }
    }
    else
    {
        for (int i = 0; i < N; ++i)
        {
            const Vec4i& v = _lines.at<Vec4i>(i);
            const Point2i b(v[0], v[1]);
            const Point2i e(v[2], v[3]);
            line(_image, b, e, Scalar(0, 0, 255), 1);
        }
    }
}


int LineSegmentDetectorImpl::compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image)
{
    CV_INSTRUMENT_REGION();

    Size sz = size;
    if (_image.needed() && _image.size() != size) sz = _image.size();
    CV_Assert(!sz.empty());

    Mat_<uchar> I1 = Mat_<uchar>::zeros(sz);
    Mat_<uchar> I2 = Mat_<uchar>::zeros(sz);

    Mat _lines1 = lines1.getMat();
    Mat _lines2 = lines2.getMat();
    const int N1 = _lines1.checkVector(4);
    const int N2 = _lines2.checkVector(4);

    CV_Assert(_lines1.depth() == CV_32F || _lines1.depth() == CV_32S);
    CV_Assert(_lines2.depth() == CV_32F || _lines2.depth() == CV_32S);

    if (_lines1.depth() == CV_32S)
        _lines1.convertTo(_lines1, CV_32F);
    if (_lines2.depth() == CV_32S)
        _lines2.convertTo(_lines2, CV_32F);

    // Draw segments
    for(int i = 0; i < N1; ++i)
    {
        const Point2f b(_lines1.at<Vec4f>(i)[0], _lines1.at<Vec4f>(i)[1]);
        const Point2f e(_lines1.at<Vec4f>(i)[2], _lines1.at<Vec4f>(i)[3]);
        line(I1, b, e, Scalar::all(255), 1);
    }
    for(int i = 0; i < N2; ++i)
    {
        const Point2f b(_lines2.at<Vec4f>(i)[0], _lines2.at<Vec4f>(i)[1]);
        const Point2f e(_lines2.at<Vec4f>(i)[2], _lines2.at<Vec4f>(i)[3]);
        line(I2, b, e, Scalar::all(255), 1);
    }

    // Count the pixels that don't agree
    Mat Ixor;
    bitwise_xor(I1, I2, Ixor);
    int N = countNonZero(Ixor);

    if (_image.needed())
    {
        CV_Assert(_image.channels() == 3);
        Mat img = _image.getMatRef();
        CV_Assert(img.isContinuous() && I1.isContinuous() && I2.isContinuous());

        for (unsigned int i = 0; i < I1.total(); ++i)
        {
            uchar i1 = I1.ptr()[i];
            uchar i2 = I2.ptr()[i];
            if (i1 || i2)
            {
                unsigned int base_idx = i * 3;
                if (i1) img.ptr()[base_idx] = 255;
                else img.ptr()[base_idx] = 0;
                img.ptr()[base_idx + 1] = 0;
                if (i2) img.ptr()[base_idx + 2] = 255;
                else img.ptr()[base_idx + 2] = 0;
            }
        }
    }

    return N;
}

} // namespace cv
