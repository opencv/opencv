// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include "opencv2/video/detail/tracking.detail.hpp"

#include "tracker_csrt_segmentation.hpp"
#include "tracker_csrt_utils.hpp"
#include "tracker_csrt_scale_estimation.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

/**
* \brief Implementation of TrackerModel for CSRT algorithm
*/
class TrackerCSRTModel CV_FINAL : public TrackerModel
{
public:
    TrackerCSRTModel(){}
    ~TrackerCSRTModel(){}
protected:
    void modelEstimationImpl(const std::vector<Mat>& /*responses*/) CV_OVERRIDE {}
    void modelUpdateImpl() CV_OVERRIDE {}
};

class TrackerCSRTImpl CV_FINAL : public TrackerCSRT
{
public:
    TrackerCSRTImpl(const Params &parameters = Params());

    Params params;

    Ptr<TrackerCSRTModel> model;

    double last_score;

    // Tracker API
    virtual void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    virtual bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    virtual void setInitialMask(InputArray mask) CV_OVERRIDE;
    virtual float getTrackingScore() const CV_OVERRIDE;

protected:
    void update_csr_filter(const Mat &image, const Mat &my_mask);
    void update_histograms(const Mat &image, const Rect &region);
    void extract_histograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb);
    std::vector<Mat> create_csr_filter(const std::vector<cv::Mat>
            img_features, const cv::Mat Y, const cv::Mat P);
    Mat calculate_response(const Mat &image, const std::vector<Mat> filter);
    Mat get_location_prior(const Rect roi, const Size2f target_size, const Size img_sz);
    Mat segment_region(const Mat &image, const Size &target_size, float scale_factor);
    Point2f estimate_new_position(const Mat &image);
    std::vector<Mat> get_features(const Mat &patch, const Size2i &feature_size);

    bool check_mask_area(const Mat &mat, const double obj_area);
    float current_scale_factor;
    Mat window;
    Mat yf;
    Rect2f bounding_box;
    std::vector<Mat> csr_filter;
    std::vector<float> filter_weights;
    Size2f original_target_size;
    Size2i image_size;
    Size2f template_size;
    Size2i rescaled_template_size;
    float rescale_ratio;
    Point2f object_center;
    DSST dsst;
    Histogram hist_foreground;
    Histogram hist_background;
    double p_b;
    Mat erode_element;
    Mat filter_mask;
    Mat preset_mask;
    Mat default_mask;
    float default_mask_area;
    int cell_size;
};

TrackerCSRTImpl::TrackerCSRTImpl(const TrackerCSRT::Params &parameters) :
    params(parameters)
{
    // nothing
}

void TrackerCSRTImpl::setInitialMask(InputArray mask)
{
    preset_mask = mask.getMat();
}

bool TrackerCSRTImpl::check_mask_area(const Mat &mat, const double obj_area)
{
    double threshold = 0.05;
    double mask_area= sum(mat)[0];
    if(mask_area < threshold*obj_area) {
        return false;
    }
    return true;
}

Mat TrackerCSRTImpl::calculate_response(const Mat &image, const std::vector<Mat> filter)
{
    Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
        cvFloor(current_scale_factor * template_size.height));
    resize(patch, patch, rescaled_template_size, 0, 0, INTER_CUBIC);

    std::vector<Mat> ftrs = get_features(patch, yf.size());
    std::vector<Mat> Ffeatures = fourier_transform_features(ftrs);
    Mat resp, res;
    if(params.use_channel_weights){
        res = Mat::zeros(Ffeatures[0].size(), CV_32FC2);
        Mat resp_ch;
        Mat mul_mat;
        for(size_t i = 0; i < Ffeatures.size(); ++i) {
            mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0, true);
            res += (resp_ch * filter_weights[i]);
        }
        idft(res, res, DFT_SCALE | DFT_REAL_OUTPUT);
    } else {
        res = Mat::zeros(Ffeatures[0].size(), CV_32FC2);
        Mat resp_ch;
        for(size_t i = 0; i < Ffeatures.size(); ++i) {
            mulSpectrums(Ffeatures[i], filter[i], resp_ch, 0 , true);
            res = res + resp_ch;
        }
        idft(res, res, DFT_SCALE | DFT_REAL_OUTPUT);
    }
    return res;
}

void TrackerCSRTImpl::update_csr_filter(const Mat &image, const Mat &mask)
{
    Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
        cvFloor(current_scale_factor * template_size.height));
    resize(patch, patch, rescaled_template_size, 0, 0, INTER_CUBIC);

    std::vector<Mat> ftrs = get_features(patch, yf.size());
    std::vector<Mat> Fftrs = fourier_transform_features(ftrs);
    std::vector<Mat> new_csr_filter = create_csr_filter(Fftrs, yf, mask);
    //calculate per channel weights
    if(params.use_channel_weights) {
        Mat current_resp;
        double max_val;
        float sum_weights = 0;
        std::vector<float> new_filter_weights = std::vector<float>(new_csr_filter.size());
        for(size_t i = 0; i < new_csr_filter.size(); ++i) {
            mulSpectrums(Fftrs[i], new_csr_filter[i], current_resp, 0, true);
            idft(current_resp, current_resp, DFT_SCALE | DFT_REAL_OUTPUT);
            minMaxLoc(current_resp, NULL, &max_val, NULL, NULL);
            sum_weights += static_cast<float>(max_val);
            new_filter_weights[i] = static_cast<float>(max_val);
        }
        //update filter weights with new values
        float updated_sum = 0;
        for(size_t i = 0; i < filter_weights.size(); ++i) {
            filter_weights[i] = filter_weights[i]*(1.0f - params.weights_lr) +
                params.weights_lr * (new_filter_weights[i] / sum_weights);
            updated_sum += filter_weights[i];
        }
        //normalize weights
        for(size_t i = 0; i < filter_weights.size(); ++i) {
            filter_weights[i] /= updated_sum;
        }
    }
    for(size_t i = 0; i < csr_filter.size(); ++i) {
        csr_filter[i] = (1.0f - params.filter_lr)*csr_filter[i] + params.filter_lr * new_csr_filter[i];
    }
    std::vector<Mat>().swap(ftrs);
    std::vector<Mat>().swap(Fftrs);
}


std::vector<Mat> TrackerCSRTImpl::get_features(const Mat &patch, const Size2i &feature_size)
{
    std::vector<Mat> features;
    if (params.use_hog) {
        std::vector<Mat> hog = get_features_hog(patch, cell_size);
        features.insert(features.end(), hog.begin(),
                hog.begin()+params.num_hog_channels_used);
    }
    // TODO: restore color_names feature mode
    // if (params.use_color_names) {
    //     std::vector<Mat> cn;
    //     cn = get_features_cn(patch, feature_size);
    //     features.insert(features.end(), cn.begin(), cn.end());
    // }
    if(params.use_gray) {
        Mat gray_m;
        cvtColor(patch, gray_m, COLOR_BGR2GRAY);
        resize(gray_m, gray_m, feature_size, 0, 0, INTER_CUBIC);
        gray_m.convertTo(gray_m, CV_32FC1, 1.0/255.0, -0.5);
        features.push_back(gray_m);
    }
    if(params.use_rgb) {
        std::vector<Mat> rgb_features = get_features_rgb(patch, feature_size);
        features.insert(features.end(), rgb_features.begin(), rgb_features.end());
    }

    for (size_t i = 0; i < features.size(); ++i) {
        features.at(i) = features.at(i).mul(window);
    }
    return features;
}

class ParallelCreateCSRFilter : public ParallelLoopBody {
public:
    ParallelCreateCSRFilter(
        const std::vector<cv::Mat> img_features_,
        const cv::Mat Y_,
        const cv::Mat P_,
        int admm_iterations_,
        std::vector<Mat> &result_filter_):
        result_filter(result_filter_)
    {
        this->img_features = img_features_;
        this->Y = Y_;
        this->P = P_;
        this->admm_iterations = admm_iterations_;
    }
    virtual void operator ()(const Range& range) const CV_OVERRIDE
    {
        for (int i = range.start; i < range.end; i++) {
            float mu = 5.0f;
            float beta = 3.0f;
            float mu_max = 20.0f;
            float lambda = mu / 100.0f;

            Mat F = img_features[i];

            Mat Sxy, Sxx;
            mulSpectrums(F, Y, Sxy, 0, true);
            mulSpectrums(F, F, Sxx, 0, true);

            Mat H;
            H = divide_complex_matrices(Sxy, (Sxx + lambda));
            idft(H, H, DFT_SCALE|DFT_REAL_OUTPUT);
            H = H.mul(P);
            dft(H, H, DFT_COMPLEX_OUTPUT);
            Mat L = Mat::zeros(H.size(), H.type()); //Lagrangian multiplier
            Mat G;
            for(int iteration = 0; iteration < admm_iterations; ++iteration) {
                G = divide_complex_matrices((Sxy + (mu * H) - L) , (Sxx + mu));
                idft((mu * G) + L, H, DFT_SCALE | DFT_REAL_OUTPUT);
                float lm = 1.0f / (lambda+mu);
                H = H.mul(P*lm);
                dft(H, H, DFT_COMPLEX_OUTPUT);

                //Update variables for next iteration
                L = L + mu * (G - H);
                mu = min(mu_max, beta*mu);
            }
            result_filter[i] = H;
        }
    }

    ParallelCreateCSRFilter& operator=(const ParallelCreateCSRFilter &) {
        return *this;
    }

private:
    int admm_iterations;
    Mat Y;
    Mat P;
    std::vector<Mat> img_features;
    std::vector<Mat> &result_filter;
};


std::vector<Mat> TrackerCSRTImpl::create_csr_filter(
        const std::vector<cv::Mat> img_features,
        const cv::Mat Y,
        const cv::Mat P)
{
    std::vector<Mat> result_filter;
    result_filter.resize(img_features.size());
    ParallelCreateCSRFilter parallelCreateCSRFilter(img_features, Y, P,
            params.admm_iterations, result_filter);
    parallel_for_(Range(0, static_cast<int>(result_filter.size())), parallelCreateCSRFilter);

    return result_filter;
}

Mat TrackerCSRTImpl::get_location_prior(
        const Rect roi,
        const Size2f target_size,
        const Size img_sz)
{
    int x1 = cvRound(max(min(roi.x-1, img_sz.width-1) , 0));
    int y1 = cvRound(max(min(roi.y-1, img_sz.height-1) , 0));

    int x2 = cvRound(min(max(roi.width-1, 0) , img_sz.width-1));
    int y2 = cvRound(min(max(roi.height-1, 0) , img_sz.height-1));

    Size target_sz;
    target_sz.width = target_sz.height = cvFloor(min(target_size.width, target_size.height));

    double cx = x1 + (x2-x1)/2.;
    double cy = y1 + (y2-y1)/2.;
    double kernel_size_width = 1.0/(0.5*static_cast<double>(target_sz.width)*1.4142+1);
    double kernel_size_height = 1.0/(0.5*static_cast<double>(target_sz.height)*1.4142+1);

    cv::Mat kernel_weight = Mat::zeros(1 + cvFloor(y2 - y1) , 1+cvFloor(-(x1-cx) + (x2-cx)), CV_64FC1);
    for (int y = y1; y < y2+1; ++y){
        double * weightPtr = kernel_weight.ptr<double>(y);
        double tmp_y = std::pow((cy-y)*kernel_size_height, 2);
        for (int x = x1; x < x2+1; ++x){
            weightPtr[x] = kernel_epan(std::pow((cx-x)*kernel_size_width,2) + tmp_y);
        }
    }

    double max_val;
    cv::minMaxLoc(kernel_weight, NULL, &max_val, NULL, NULL);
    Mat fg_prior = kernel_weight / max_val;
    fg_prior.setTo(0.5, fg_prior < 0.5);
    fg_prior.setTo(0.9, fg_prior > 0.9);
    return fg_prior;
}

Mat TrackerCSRTImpl::segment_region( const Mat &image, const Size &target_size, float scale_factor)
{
    Rect valid_pixels;
    Mat patch = get_subwindow(image, object_center, cvFloor(scale_factor * template_size.width),
        cvFloor(scale_factor * template_size.height), &valid_pixels);
    Size2f scaled_target = Size2f(target_size.width * scale_factor,
            target_size.height * scale_factor);
    Mat fg_prior = get_location_prior(
            Rect(0,0, patch.size().width, patch.size().height),
            scaled_target , patch.size());

    std::vector<Mat> img_channels;
    split(patch, img_channels);
    std::pair<Mat, Mat> probs = Segment::computePosteriors2(img_channels, 0, 0, patch.cols, patch.rows,
                    p_b, fg_prior, 1.0-fg_prior, hist_foreground, hist_background);

    Mat mask = Mat::zeros(probs.first.size(), probs.first.type());
    probs.first(valid_pixels).copyTo(mask(valid_pixels));
    double max_resp = get_max(mask);
    threshold(mask, mask, max_resp / 2.0, 1, THRESH_BINARY);
    mask.convertTo(mask, CV_32FC1, 1.0);
    return mask;
}


void TrackerCSRTImpl::extract_histograms(const Mat &image, cv::Rect region, Histogram &hf, Histogram &hb)
{
    // get coordinates of the region
    int x1 = std::min(std::max(0, region.x), image.cols-1);
    int y1 = std::min(std::max(0, region.y), image.rows-1);
    int x2 = std::min(std::max(0, region.x + region.width), image.cols-1);
    int y2 = std::min(std::max(0, region.y + region.height), image.rows-1);

    // calculate coordinates of the background region
    int offsetX = (x2-x1+1) / params.background_ratio;
    int offsetY = (y2-y1+1) / params.background_ratio;
    int outer_y1 = std::max(0, (int)(y1-offsetY));
    int outer_y2 = std::min(image.rows, (int)(y2+offsetY+1));
    int outer_x1 = std::max(0, (int)(x1-offsetX));
    int outer_x2 = std::min(image.cols, (int)(x2+offsetX+1));

    // calculate probability for the background
    p_b = 1.0 - ((x2-x1+1) * (y2-y1+1)) /
        ((double) (outer_x2-outer_x1+1) * (outer_y2-outer_y1+1));

    // split multi-channel image into the std::vector of matrices
    std::vector<Mat> img_channels(image.channels());
    split(image, img_channels);
    for(size_t k=0; k<img_channels.size(); k++) {
        img_channels.at(k).convertTo(img_channels.at(k), CV_8UC1);
    }

    hf.extractForegroundHistogram(img_channels, Mat(), false, x1, y1, x2, y2);
    hb.extractBackGroundHistogram(img_channels, x1, y1, x2, y2,
        outer_x1, outer_y1, outer_x2, outer_y2);
    std::vector<Mat>().swap(img_channels);
}

void TrackerCSRTImpl::update_histograms(const Mat &image, const Rect &region)
{
    // create temporary histograms
    Histogram hf(image.channels(), params.histogram_bins);
    Histogram hb(image.channels(), params.histogram_bins);
    extract_histograms(image, region, hf, hb);

    // get histogram vectors from temporary histograms
    std::vector<double> hf_vect_new = hf.getHistogramVector();
    std::vector<double> hb_vect_new = hb.getHistogramVector();
    // get histogram vectors from learned histograms
    std::vector<double> hf_vect = hist_foreground.getHistogramVector();
    std::vector<double> hb_vect = hist_background.getHistogramVector();

    // update histograms - use learning rate
    for(size_t i=0; i<hf_vect.size(); i++) {
        hf_vect_new[i] = (1-params.histogram_lr)*hf_vect[i] +
            params.histogram_lr*hf_vect_new[i];
        hb_vect_new[i] = (1-params.histogram_lr)*hb_vect[i] +
            params.histogram_lr*hb_vect_new[i];
    }

    // set learned histograms
    hist_foreground.setHistogramVector(&hf_vect_new[0]);
    hist_background.setHistogramVector(&hb_vect_new[0]);

    std::vector<double>().swap(hf_vect);
    std::vector<double>().swap(hb_vect);
}

Point2f TrackerCSRTImpl::estimate_new_position(const Mat &image)
{

    Mat resp = calculate_response(image, csr_filter);

    Point max_loc;
    minMaxLoc(resp, NULL, &last_score, NULL, &max_loc);
    if (last_score < params.psr_threshold)
        return Point2f(-1,-1); // target "lost"

    // take into account also subpixel accuracy
    float col = ((float) max_loc.x) + subpixel_peak(resp, "horizontal", max_loc);
    float row = ((float) max_loc.y) + subpixel_peak(resp, "vertical", max_loc);
    if(row + 1 > (float)resp.rows / 2.0f) {
        row = row - resp.rows;
    }
    if(col + 1 > (float)resp.cols / 2.0f) {
        col = col - resp.cols;
    }
    // calculate x and y displacements
    Point2f new_center = object_center + Point2f(current_scale_factor * (1.0f / rescale_ratio) *cell_size*(col),
            current_scale_factor * (1.0f / rescale_ratio) *cell_size*(row));
    //sanity checks
    if(new_center.x < 0)
        new_center.x = 0;
    if(new_center.x >= image_size.width)
        new_center.x = static_cast<float>(image_size.width - 1);
    if(new_center.y < 0)
        new_center.y = 0;
    if(new_center.y >= image_size.height)
        new_center.y = static_cast<float>(image_size.height - 1);

    return new_center;
}

// *********************************************************************
// *                        Update API function                        *
// *********************************************************************
bool TrackerCSRTImpl::update(InputArray image_, Rect& boundingBox)
{
    Mat image;
    if(image_.channels() == 1)    //treat gray image as color image
        cvtColor(image_, image, COLOR_GRAY2BGR);
    else
        image = image_.getMat();

    object_center = estimate_new_position(image);
    if (object_center.x < 0 && object_center.y < 0)
        return false;

    current_scale_factor = dsst.getScale(image, object_center);
    //update bouding_box according to new scale and location
    bounding_box.x = object_center.x - current_scale_factor * original_target_size.width / 2.0f;
    bounding_box.y = object_center.y - current_scale_factor * original_target_size.height / 2.0f;
    bounding_box.width = current_scale_factor * original_target_size.width;
    bounding_box.height = current_scale_factor * original_target_size.height;

    //update tracker
    if(params.use_segmentation) {
        Mat hsv_img = bgr2hsv(image);
        update_histograms(hsv_img, bounding_box);
        filter_mask = segment_region(hsv_img, original_target_size, current_scale_factor);
        resize(filter_mask, filter_mask, yf.size(), 0, 0, INTER_NEAREST);
        if(check_mask_area(filter_mask, default_mask_area)) {
            dilate(filter_mask , filter_mask, erode_element);
        } else {
            filter_mask = default_mask;
        }
    } else {
        filter_mask = default_mask;
    }
    update_csr_filter(image, filter_mask);
    dsst.update(image, object_center);
    boundingBox = bounding_box;
    return true;
}

float TrackerCSRTImpl::getTrackingScore() const
{
    return static_cast<float>(last_score);
}

// *********************************************************************
// *                        Init API function                          *
// *********************************************************************
void TrackerCSRTImpl::init(InputArray image_, const Rect& boundingBox)
{
    Mat image;
    if(image_.channels() == 1)    //treat gray image as color image
        cvtColor(image_, image, COLOR_GRAY2BGR);
    else
        image = image_.getMat();

    current_scale_factor = 1.0;
    image_size = image.size();
    bounding_box = boundingBox;
    cell_size = cvFloor(std::min(4.0, std::max(1.0, static_cast<double>(
        cvCeil((bounding_box.width * bounding_box.height)/400.0)))));
    original_target_size = Size(bounding_box.size());

    template_size.width = static_cast<float>(cvFloor(original_target_size.width + params.padding *
            sqrt(original_target_size.width * original_target_size.height)));
    template_size.height = static_cast<float>(cvFloor(original_target_size.height + params.padding *
            sqrt(original_target_size.width * original_target_size.height)));
    template_size.width = template_size.height =
        (template_size.width + template_size.height) / 2.0f;
    rescale_ratio = sqrt((params.template_size * params.template_size) / (template_size.width * template_size.height));
    if(rescale_ratio > 1)  {
        rescale_ratio = 1;
    }
    rescaled_template_size = Size2i(cvFloor(template_size.width * rescale_ratio),
            cvFloor(template_size.height * rescale_ratio));
    object_center = Point2f(static_cast<float>(boundingBox.x) + original_target_size.width / 2.0f,
            static_cast<float>(boundingBox.y) + original_target_size.height / 2.0f);

    yf = gaussian_shaped_labels(params.gsl_sigma,
            rescaled_template_size.width / cell_size, rescaled_template_size.height / cell_size);
    if(params.window_function.compare("hann") == 0) {
        window = get_hann_win(Size(yf.cols,yf.rows));
    } else if(params.window_function.compare("cheb") == 0) {
        window = get_chebyshev_win(Size(yf.cols,yf.rows), params.cheb_attenuation);
    } else if(params.window_function.compare("kaiser") == 0) {
        window = get_kaiser_win(Size(yf.cols,yf.rows), params.kaiser_alpha);
    } else {
        CV_Error(Error::StsBadArg, "Not a valid window function");
    }

    Size2i scaled_obj_size = Size2i(cvFloor(original_target_size.width * rescale_ratio / cell_size),
            cvFloor(original_target_size.height * rescale_ratio / cell_size));
    //set dummy mask and area;
    int x0 = std::max((yf.size().width - scaled_obj_size.width)/2 - 1, 0);
    int y0 = std::max((yf.size().height - scaled_obj_size.height)/2 - 1, 0);
    default_mask = Mat::zeros(yf.size(), CV_32FC1);
    default_mask(Rect(x0,y0,scaled_obj_size.width, scaled_obj_size.height)) = 1.0f;
    default_mask_area = static_cast<float>(sum(default_mask)[0]);

    //initalize segmentation
    if(params.use_segmentation) {
        Mat hsv_img = bgr2hsv(image);
        hist_foreground = Histogram(hsv_img.channels(), params.histogram_bins);
        hist_background = Histogram(hsv_img.channels(), params.histogram_bins);
        extract_histograms(hsv_img, bounding_box, hist_foreground, hist_background);
        filter_mask = segment_region(hsv_img, original_target_size, current_scale_factor);
        //update calculated mask with preset mask
        if(preset_mask.data){
            Mat preset_mask_padded = Mat::zeros(filter_mask.size(), filter_mask.type());
            int sx = std::max((int)cvFloor(preset_mask_padded.cols / 2.0f - preset_mask.cols / 2.0f) - 1, 0);
            int sy = std::max((int)cvFloor(preset_mask_padded.rows / 2.0f - preset_mask.rows / 2.0f) - 1, 0);
            preset_mask.copyTo(preset_mask_padded(
                        Rect(sx, sy, preset_mask.cols, preset_mask.rows)));
            filter_mask = filter_mask.mul(preset_mask_padded);
        }
        erode_element = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(1,1));
        resize(filter_mask, filter_mask, yf.size(), 0, 0, INTER_NEAREST);
        if(check_mask_area(filter_mask, default_mask_area)) {
            dilate(filter_mask , filter_mask, erode_element);
        } else {
            filter_mask = default_mask;
        }

    } else {
        filter_mask = default_mask;
    }

    //initialize filter
    Mat patch = get_subwindow(image, object_center, cvFloor(current_scale_factor * template_size.width),
        cvFloor(current_scale_factor * template_size.height));
    resize(patch, patch, rescaled_template_size, 0, 0, INTER_CUBIC);
    std::vector<Mat> patch_ftrs = get_features(patch, yf.size());
    std::vector<Mat> Fftrs = fourier_transform_features(patch_ftrs);
    csr_filter = create_csr_filter(Fftrs, yf, filter_mask);

    if(params.use_channel_weights) {
        Mat current_resp;
        filter_weights = std::vector<float>(csr_filter.size());
        float chw_sum = 0;
        for (size_t i = 0; i < csr_filter.size(); ++i) {
            mulSpectrums(Fftrs[i], csr_filter[i], current_resp, 0, true);
            idft(current_resp, current_resp, DFT_SCALE | DFT_REAL_OUTPUT);
            double max_val;
            minMaxLoc(current_resp, NULL, &max_val, NULL , NULL);
            chw_sum += static_cast<float>(max_val);
            filter_weights[i] = static_cast<float>(max_val);
        }
        for (size_t i = 0; i < filter_weights.size(); ++i) {
            filter_weights[i] /= chw_sum;
        }
    }

    //initialize scale search
    dsst = DSST(image, bounding_box, template_size, params.number_of_scales, params.scale_step,
            params.scale_model_max_area, params.scale_sigma_factor, params.scale_lr);

    model=makePtr<TrackerCSRTModel>();
}

}}}  // cv::detail::tracking

//==============================================================================

namespace cv {

TrackerCSRT::Params::Params()
{
    use_channel_weights = true;
    use_segmentation = true;
    use_hog = true;
    use_color_names = true;
    use_gray = true;
    use_rgb = false;
    window_function = "hann";
    kaiser_alpha = 3.75f;
    cheb_attenuation = 45;
    padding = 3.0f;
    template_size = 200;
    gsl_sigma = 1.0f;
    hog_orientations = 9;
    hog_clip = 0.2f;
    num_hog_channels_used = 18;
    filter_lr = 0.02f;
    weights_lr = 0.02f;
    admm_iterations = 4;
    number_of_scales = 33;
    scale_sigma_factor = 0.250f;
    scale_model_max_area = 512.0f;
    scale_lr = 0.025f;
    scale_step = 1.020f;
    histogram_bins = 16;
    background_ratio = 2;
    histogram_lr = 0.04f;
    psr_threshold = 0.035f;
}


TrackerCSRT::TrackerCSRT() { }

TrackerCSRT::~TrackerCSRT() { }

Ptr<TrackerCSRT> TrackerCSRT::create(const TrackerCSRT::Params &parameters)
{
    return makePtr<cv::detail::tracking::TrackerCSRTImpl>(parameters);
}

}  // cv::
