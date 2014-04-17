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

struct edge
{
    cv::Point p;
    bool taken;
};

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

inline bool AsmallerB_XoverY(const edge& a, const edge& b)
{
    if (a.p.x == b.p.x) return a.p.y < b.p.y;
    else return a.p.x < b.p.x;
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

class LineSegmentDetectorImpl : public LineSegmentDetector
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
 * @param _lines    Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                          Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
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
                OutputArray nfa = noArray());

/**
 * Draw lines on the given canvas.
 *
 * @param image     The image, where lines will be drawn.
 *                  Should have the size of the image, where the lines were found
 * @param lines     The lines that need to be drawn
 */
    void drawSegments(InputOutputArray _image, InputArray lines);

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
    int compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image = noArray());

private:
    Mat image;
    Mat_<double> scaled_image;
    double *scaled_image_data;
    Mat_<double> angles;     // in rads
    double *angles_data;
    Mat_<double> modgrad;
    double *modgrad_data;
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


    struct coorlist
    {
        Point2i p;
        struct coorlist* next;
    };

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
 * @param lines         Return: A vector of Vec4i elements specifying the beginning and ending point of a line.
 *                              Where Vec4i is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
 *                              Returned lines are strictly oriented depending on the gradient.
 * @param widths        Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
 * @param precisions    Return: Vector of precisions with which the lines are found.
 * @param nfas          Return: Vector containing number of false alarms in the line region, with precision of 10%.
 *                              The bigger the value, logarithmically better the detection.
 *                                  * -1 corresponds to 10 mean false alarms
 *                                  * 0 corresponds to 1 mean false alarm
 *                                  * 1 corresponds to 0.1 mean false alarms
 */
    void flsd(std::vector<Vec4i>& lines,
              std::vector<double>& widths, std::vector<double>& precisions,
              std::vector<double>& nfas);

/**
 * Finds the angles and the gradients of the image. Generates a list of pseudo ordered points.
 *
 * @param threshold The minimum value of the angle that is considered defined, otherwise NOTDEF
 * @param n_bins    The number of bins with which gradients are ordered by, using bucket sort.
 * @param list      Return: Vector of coordinate points that are pseudo ordered by magnitude.
 *                  Pixels would be ordered by norm value, up to a precision given by max_grad/n_bins.
 */
    void ll_angle(const double& threshold, const unsigned int& n_bins, std::vector<coorlist>& list);

/**
 * Grow a region starting from point s with a defined precision,
 * returning the containing points size and the angle of the gradients.
 *
 * @param s         Starting point for the region.
 * @param reg       Return: Vector of points, that are part of the region
 * @param reg_size  Return: The size of the region.
 * @param reg_angle Return: The mean angle of the region.
 * @param prec      The precision by which each region angle should be aligned to the mean.
 */
    void region_grow(const Point2i& s, std::vector<RegionPoint>& reg,
                     int& reg_size, double& reg_angle, const double& prec);

/**
 * Finds the bounding rotated rectangle of a region.
 *
 * @param reg       The region of points, from which the rectangle to be constructed from.
 * @param reg_size  The number of points in the region.
 * @param reg_angle The mean angle of the region.
 * @param prec      The precision by which points were found.
 * @param p         Probability of a point with angle within 'prec'.
 * @param rec       Return: The generated rectangle.
 */
    void region2rect(const std::vector<RegionPoint>& reg, const int reg_size, const double reg_angle,
                     const double prec, const double p, rect& rec) const;

/**
 * Compute region's angle as the principal inertia axis of the region.
 * @return          Regions angle.
 */
    double get_theta(const std::vector<RegionPoint>& reg, const int& reg_size, const double& x,
                     const double& y, const double& reg_angle, const double& prec) const;

/**
 * An estimation of the angle tolerance is performed by the standard deviation of the angle at points
 * near the region's starting point. Then, a new region is grown starting from the same point, but using the
 * estimated angle tolerance. If this fails to produce a rectangle with the right density of region points,
 * 'reduce_region_radius' is called to try to satisfy this condition.
 */
    bool refine(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                const double prec, double p, rect& rec, const double& density_th);

/**
 * Reduce the region size, by elimination the points far from the starting point, until that leads to
 * rectangle with the right density of region points or to discard the region if too small.
 */
    bool reduce_region_radius(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
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
    bool isAligned(const int& address, const double& theta, const double& prec) const;
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
        :SCALE(_scale), doRefine(_refine), SIGMA_SCALE(_sigma_scale), QUANT(_quant),
        ANG_TH(_ang_th), LOG_EPS(_log_eps), DENSITY_TH(_density_th), N_BINS(_n_bins)
{
    CV_Assert(_scale > 0 && _sigma_scale > 0 && _quant >= 0 &&
              _ang_th > 0 && _ang_th < 180 && _density_th >= 0 && _density_th < 1 &&
              _n_bins > 0);
}

void LineSegmentDetectorImpl::detect(InputArray _image, OutputArray _lines,
                OutputArray _width, OutputArray _prec, OutputArray _nfa)
{
    Mat_<double> img = _image.getMat();
    CV_Assert(!img.empty() && img.channels() == 1);

    // Convert image to double
    img.convertTo(image, CV_64FC1);

    std::vector<Vec4i> lines;
    std::vector<double> w, p, n;
    w_needed = _width.needed();
    p_needed = _prec.needed();
    n_needed = _nfa.needed();

    CV_Assert((!_nfa.needed()) ||                              // NFA InputArray will be filled _only_ when
              (_nfa.needed() && doRefine >= LSD_REFINE_ADV));  // REFINE_ADV type LineSegmentDetectorImpl object is created.

    flsd(lines, w, p, n);

    Mat(lines).copyTo(_lines);
    if(w_needed) Mat(w).copyTo(_width);
    if(p_needed) Mat(p).copyTo(_prec);
    if(n_needed) Mat(n).copyTo(_nfa);
}

void LineSegmentDetectorImpl::flsd(std::vector<Vec4i>& lines,
    std::vector<double>& widths, std::vector<double>& precisions,
    std::vector<double>& nfas)
{
    // Angle tolerance
    const double prec = CV_PI * ANG_TH / 180;
    const double p = ANG_TH / 180;
    const double rho = QUANT / sin(prec);    // gradient magnitude threshold

    std::vector<coorlist> list;
    if(SCALE != 1)
    {
        Mat gaussian_img;
        const double sigma = (SCALE < 1)?(SIGMA_SCALE / SCALE):(SIGMA_SCALE);
        const double sprec = 3;
        const unsigned int h =  (unsigned int)(ceil(sigma * sqrt(2 * sprec * log(10.0))));
        Size ksize(1 + 2 * h, 1 + 2 * h); // kernel size
        GaussianBlur(image, gaussian_img, ksize, sigma);
        // Scale image to needed size
        resize(gaussian_img, scaled_image, Size(), SCALE, SCALE);
        ll_angle(rho, N_BINS, list);
    }
    else
    {
        scaled_image = image;
        ll_angle(rho, N_BINS, list);
    }

    LOG_NT = 5 * (log10(double(img_width)) + log10(double(img_height))) / 2 + log10(11.0);
    const int min_reg_size = int(-LOG_NT/log10(p)); // minimal number of points in region that can give a meaningful event

    // // Initialize region only when needed
    // Mat region = Mat::zeros(scaled_image.size(), CV_8UC1);
    used = Mat_<uchar>::zeros(scaled_image.size()); // zeros = NOTUSED
    std::vector<RegionPoint> reg(img_width * img_height);

    // Search for line segments
    unsigned int ls_count = 0;
    for(size_t i = 0, list_size = list.size(); i < list_size; ++i)
    {
        unsigned int adx = list[i].p.x + list[i].p.y * img_width;
        if((used.data[adx] == NOTUSED) && (angles_data[adx] != NOTDEF))
        {
            int reg_size;
            double reg_angle;
            region_grow(list[i].p, reg, reg_size, reg_angle, prec);

            // Ignore small regions
            if(reg_size < min_reg_size) { continue; }

            // Construct rectangular approximation for the region
            rect rec;
            region2rect(reg, reg_size, reg_angle, prec, p, rec);

            double log_nfa = -1;
            if(doRefine > LSD_REFINE_NONE)
            {
                // At least REFINE_STANDARD lvl.
                if(!refine(reg, reg_size, reg_angle, prec, p, rec, DENSITY_TH)) { continue; }

                if(doRefine >= LSD_REFINE_ADV)
                {
                    // Compute NFA
                    log_nfa = rect_improve(rec);
                    if(log_nfa <= LOG_EPS) { continue; }
                }
            }
            // Found new line
            ++ls_count;

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
            lines.push_back(Vec4i(int(rec.x1), int(rec.y1), int(rec.x2), int(rec.y2)));
            if(w_needed) widths.push_back(rec.width);
            if(p_needed) precisions.push_back(rec.p);
            if(n_needed && doRefine >= LSD_REFINE_ADV) nfas.push_back(log_nfa);


            // //Add the linesID to the region on the image
            // for(unsigned int el = 0; el < reg_size; el++)
            // {
            //     region.data[reg[i].x + reg[i].y * width] = ls_count;
            // }
        }
    }
}

void LineSegmentDetectorImpl::ll_angle(const double& threshold,
                                   const unsigned int& n_bins,
                                   std::vector<coorlist>& list)
{
    //Initialize data
    angles = Mat_<double>(scaled_image.size());
    modgrad = Mat_<double>(scaled_image.size());

    angles_data = angles.ptr<double>(0);
    modgrad_data = modgrad.ptr<double>(0);
    scaled_image_data = scaled_image.ptr<double>(0);

    img_width = scaled_image.cols;
    img_height = scaled_image.rows;

    // Undefined the down and right boundaries
    angles.row(img_height - 1).setTo(NOTDEF);
    angles.col(img_width - 1).setTo(NOTDEF);

    // Computing gradient for remaining pixels
    CV_Assert(scaled_image.isContinuous() &&
              modgrad.isContinuous() &&
              angles.isContinuous());   // Accessing image data linearly

    double max_grad = -1;
    for(int y = 0; y < img_height - 1; ++y)
    {
        for(int addr = y * img_width, addr_end = addr + img_width - 1; addr < addr_end; ++addr)
        {
            double DA = scaled_image_data[addr + img_width + 1] - scaled_image_data[addr];
            double BC = scaled_image_data[addr + 1] - scaled_image_data[addr + img_width];
            double gx = DA + BC;    // gradient x component
            double gy = DA - BC;    // gradient y component
            double norm = std::sqrt((gx * gx + gy * gy) / 4); // gradient norm

            modgrad_data[addr] = norm;    // store gradient

            if (norm <= threshold)  // norm too small, gradient no defined
            {
                angles_data[addr] = NOTDEF;
            }
            else
            {
                angles_data[addr] = fastAtan2(float(gx), float(-gy)) * DEG_TO_RADS;  // gradient angle computation
                if (norm > max_grad) { max_grad = norm; }
            }

        }
    }

    // Compute histogram of gradient values
    list = std::vector<coorlist>(img_width * img_height);
    std::vector<coorlist*> range_s(n_bins);
    std::vector<coorlist*> range_e(n_bins);
    unsigned int count = 0;
    double bin_coef = (max_grad > 0) ? double(n_bins - 1) / max_grad : 0; // If all image is smooth, max_grad <= 0

    for(int y = 0; y < img_height - 1; ++y)
    {
        const double* norm = modgrad_data + y * img_width;
        for(int x = 0; x < img_width - 1; ++x, ++norm)
        {
            // Store the point in the right bin according to its norm
            int i = int((*norm) * bin_coef);
            if(!range_e[i])
            {
                range_e[i] = range_s[i] = &list[count];
                ++count;
            }
            else
            {
                range_e[i]->next = &list[count];
                range_e[i] = &list[count];
                ++count;
            }
            range_e[i]->p = Point(x, y);
            range_e[i]->next = 0;
        }
    }

    // Sort
    int idx = n_bins - 1;
    for(;idx > 0 && !range_s[idx]; --idx);
    coorlist* start = range_s[idx];
    coorlist* end = range_e[idx];
    if(start)
    {
        while(idx > 0)
        {
            --idx;
            if(range_s[idx])
            {
                end->next = range_s[idx];
                end = range_e[idx];
            }
        }
    }
}

void LineSegmentDetectorImpl::region_grow(const Point2i& s, std::vector<RegionPoint>& reg,
                                      int& reg_size, double& reg_angle, const double& prec)
{
    // Point to this region
    reg_size = 1;
    reg[0].x = s.x;
    reg[0].y = s.y;
    int addr = s.x + s.y * img_width;
    reg[0].used = used.data + addr;
    reg_angle = angles_data[addr];
    reg[0].angle = reg_angle;
    reg[0].modgrad = modgrad_data[addr];

    float sumdx = float(std::cos(reg_angle));
    float sumdy = float(std::sin(reg_angle));
    *reg[0].used = USED;

    //Try neighboring regions
    for(int i = 0; i < reg_size; ++i)
    {
        const RegionPoint& rpoint = reg[i];
        int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
        int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);
        for(int yy = yy_min; yy <= yy_max; ++yy)
        {
            int c_addr = xx_min + yy * img_width;
            for(int xx = xx_min; xx <= xx_max; ++xx, ++c_addr)
            {
                if((used.data[c_addr] != USED) &&
                   (isAligned(c_addr, reg_angle, prec)))
                {
                    // Add point
                    used.data[c_addr] = USED;
                    RegionPoint& region_point = reg[reg_size];
                    region_point.x = xx;
                    region_point.y = yy;
                    region_point.used = &(used.data[c_addr]);
                    region_point.modgrad = modgrad_data[c_addr];
                    const double& angle = angles_data[c_addr];
                    region_point.angle = angle;
                    ++reg_size;

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

void LineSegmentDetectorImpl::region2rect(const std::vector<RegionPoint>& reg, const int reg_size,
                                      const double reg_angle, const double prec, const double p, rect& rec) const
{
    double x = 0, y = 0, sum = 0;
    for(int i = 0; i < reg_size; ++i)
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

    double theta = get_theta(reg, reg_size, x, y, reg_angle, prec);

    // Find length and width
    double dx = cos(theta);
    double dy = sin(theta);
    double l_min = 0, l_max = 0, w_min = 0, w_max = 0;

    for(int i = 0; i < reg_size; ++i)
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

double LineSegmentDetectorImpl::get_theta(const std::vector<RegionPoint>& reg, const int& reg_size, const double& x,
                                      const double& y, const double& reg_angle, const double& prec) const
{
    double Ixx = 0.0;
    double Iyy = 0.0;
    double Ixy = 0.0;

    // Compute inertia matrix
    for(int i = 0; i < reg_size; ++i)
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

bool LineSegmentDetectorImpl::refine(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
                                 const double prec, double p, rect& rec, const double& density_th)
{
    double density = double(reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density >= density_th) { return true; }

    // Try to reduce angle tolerance
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    const double& ang_c = reg[0].angle;
    double sum = 0, s_sum = 0;
    int n = 0;

    for (int i = 0; i < reg_size; ++i)
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
    double mean_angle = sum / double(n);
    // 2 * standard deviation
    double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / double(n) + mean_angle * mean_angle);

    // Try new region
    region_grow(Point(reg[0].x, reg[0].y), reg, reg_size, reg_angle, tau);

    if (reg_size < 2) { return false; }

    region2rect(reg, reg_size, reg_angle, prec, p, rec);
    density = double(reg_size) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density < density_th)
    {
        return reduce_region_radius(reg, reg_size, reg_angle, prec, p, rec, density, density_th);
    }
    else
    {
        return true;
    }
}

bool LineSegmentDetectorImpl::reduce_region_radius(std::vector<RegionPoint>& reg, int& reg_size, double reg_angle,
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
        for(int i = 0; i < reg_size; ++i)
        {
            if(distSq(xc, yc, double(reg[i].x), double(reg[i].y)) > radSq)
            {
                // Remove point from the region
                *(reg[i].used) = NOTUSED;
                std::swap(reg[i], reg[reg_size - 1]);
                --reg_size;
                --i; // To avoid skipping one point
            }
        }

        if(reg_size < 2) { return false; }

        // Re-compute rectangle
        region2rect(reg, reg_size ,reg_angle, prec, p, rec);

        // Re-compute region points density
        density = double(reg_size) /
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

    std::vector<edge> ordered_x(4);
    edge* min_y = &ordered_x[0];
    edge* max_y = &ordered_x[0]; // Will be used for loop range

    ordered_x[0].p.x = int(rec.x1 - dyhw); ordered_x[0].p.y = int(rec.y1 + dxhw); ordered_x[0].taken = false;
    ordered_x[1].p.x = int(rec.x2 - dyhw); ordered_x[1].p.y = int(rec.y2 + dxhw); ordered_x[1].taken = false;
    ordered_x[2].p.x = int(rec.x2 + dyhw); ordered_x[2].p.y = int(rec.y2 - dxhw); ordered_x[2].taken = false;
    ordered_x[3].p.x = int(rec.x1 + dyhw); ordered_x[3].p.y = int(rec.y1 - dxhw); ordered_x[3].taken = false;

    std::sort(ordered_x.begin(), ordered_x.end(), AsmallerB_XoverY);

    // Find min y. And mark as taken. find max y.
    for(unsigned int i = 1; i < 4; ++i)
    {
        if(min_y->p.y > ordered_x[i].p.y) {min_y = &ordered_x[i]; }
        if(max_y->p.y < ordered_x[i].p.y) {max_y = &ordered_x[i]; }
    }
    min_y->taken = true;

    // Find leftmost untaken point;
    edge* leftmost = 0;
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(!ordered_x[i].taken)
        {
            if(!leftmost) // if uninitialized
            {
                leftmost = &ordered_x[i];
            }
            else if (leftmost->p.x > ordered_x[i].p.x)
            {
                leftmost = &ordered_x[i];
            }
        }
    }
    leftmost->taken = true;

    // Find rightmost untaken point;
    edge* rightmost = 0;
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(!ordered_x[i].taken)
        {
            if(!rightmost) // if uninitialized
            {
                rightmost = &ordered_x[i];
            }
            else if (rightmost->p.x < ordered_x[i].p.x)
            {
                rightmost = &ordered_x[i];
            }
        }
    }
    rightmost->taken = true;

    // Find last untaken point;
    edge* tailp = 0;
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(!ordered_x[i].taken)
        {
            if(!tailp) // if uninitialized
            {
                tailp = &ordered_x[i];
            }
            else if (tailp->p.x > ordered_x[i].p.x)
            {
                tailp = &ordered_x[i];
            }
        }
    }
    tailp->taken = true;

    double flstep = (min_y->p.y != leftmost->p.y) ?
                    (min_y->p.x - leftmost->p.x) / (min_y->p.y - leftmost->p.y) : 0; //first left step
    double slstep = (leftmost->p.y != tailp->p.x) ?
                    (leftmost->p.x - tailp->p.x) / (leftmost->p.y - tailp->p.x) : 0; //second left step

    double frstep = (min_y->p.y != rightmost->p.y) ?
                    (min_y->p.x - rightmost->p.x) / (min_y->p.y - rightmost->p.y) : 0; //first right step
    double srstep = (rightmost->p.y != tailp->p.x) ?
                    (rightmost->p.x - tailp->p.x) / (rightmost->p.y - tailp->p.x) : 0; //second right step

    double lstep = flstep, rstep = frstep;

    double left_x = min_y->p.x, right_x = min_y->p.x;

    // Loop around all points in the region and count those that are aligned.
    int min_iter = std::max(min_y->p.y, 0);
    int max_iter = std::min(max_y->p.y, img_height - 1);
    for(int y = min_iter; y <= max_iter; ++y)
    {
        int adx = y * img_width + int(left_x);
        for(int x = int(left_x); x <= int(right_x); ++x, ++adx)
        {
            ++total_pts;
            if(isAligned(adx, rec.theta, rec.prec))
            {
                ++alg_pts;
            }
        }

        if(y >= leftmost->p.y) { lstep = slstep; }
        if(y >= rightmost->p.y) { rstep = srstep; }

        left_x += lstep;
        right_x += rstep;
    }

    return nfa(total_pts, alg_pts, rec.p);
}

double LineSegmentDetectorImpl::nfa(const int& n, const int& k, const double& p) const
{
    // Trivial cases
    if(n == 0 || k == 0) { return -LOG_NT; }
    if(n == k) { return -LOG_NT - double(n) * log10(p); }

    double p_term = p / (1 - p);

    double log1term = (double(n) + 1) - log_gamma(double(k) + 1)
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

inline bool LineSegmentDetectorImpl::isAligned(const int& address, const double& theta, const double& prec) const
{
    if(address < 0) { return false; }
    const double& a = angles_data[address];
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
    CV_Assert(!_image.empty() && (_image.channels() == 1 || _image.channels() == 3));

    Mat gray;
    if (_image.channels() == 1)
    {
        gray = _image.getMatRef();
    }
    else if (_image.channels() == 3)
    {
        cvtColor(_image, gray, CV_BGR2GRAY);
    }

    // Create a 3 channel image in order to draw colored lines
    std::vector<Mat> planes;
    planes.push_back(gray);
    planes.push_back(gray);
    planes.push_back(gray);

    merge(planes, _image);

    Mat _lines;
    _lines = lines.getMat();

    // Draw segments
    for(int i = 0; i < _lines.size().width; ++i)
    {
        const Vec4i& v = _lines.at<Vec4i>(i);
        Point b(v[0], v[1]);
        Point e(v[2], v[3]);
        line(_image.getMatRef(), b, e, Scalar(0, 0, 255), 1);
    }
}


int LineSegmentDetectorImpl::compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image)
{
    Size sz = size;
    if (_image.needed() && _image.size() != size) sz = _image.size();
    CV_Assert(sz.area());

    Mat_<uchar> I1 = Mat_<uchar>::zeros(sz);
    Mat_<uchar> I2 = Mat_<uchar>::zeros(sz);

    Mat _lines1;
    Mat _lines2;
    _lines1 = lines1.getMat();
    _lines2 = lines2.getMat();
    // Draw segments
    for(int i = 0; i < _lines1.size().width; ++i)
    {
        Point b(_lines1.at<Vec4i>(i)[0], _lines1.at<Vec4i>(i)[1]);
        Point e(_lines1.at<Vec4i>(i)[2], _lines1.at<Vec4i>(i)[3]);
        line(I1, b, e, Scalar::all(255), 1);
    }
    for(int i = 0; i < _lines2.size().width; ++i)
    {
        Point b(_lines2.at<Vec4i>(i)[0], _lines2.at<Vec4i>(i)[1]);
        Point e(_lines2.at<Vec4i>(i)[2], _lines2.at<Vec4i>(i)[3]);
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
            uchar i1 = I1.data[i];
            uchar i2 = I2.data[i];
            if (i1 || i2)
            {
                unsigned int base_idx = i * 3;
                if (i1) img.data[base_idx] = 255;
                else img.data[base_idx] = 0;
                img.data[base_idx + 1] = 0;
                if (i2) img.data[base_idx + 2] = 255;
                else img.data[base_idx + 2] = 0;
            }
        }
    }

    return N;
}

} // namespace cv
