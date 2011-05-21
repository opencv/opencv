#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "blenders.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

static const float WEIGHT_EPS = 1e-5f;

Ptr<Blender> Blender::createDefault(int type)
{
    if (type == NO)
        return new Blender();
    if (type == FEATHER)
        return new FeatherBlender();
    if (type == MULTI_BAND)
        return new MultiBandBlender();
    CV_Error(CV_StsBadArg, "unsupported blending method");
    return NULL;
}


void Blender::prepare(const vector<Point> &corners, const vector<Size> &sizes)
{
    prepare(resultRoi(corners, sizes));
}


void Blender::prepare(Rect dst_roi)
{
    dst_.create(dst_roi.size(), CV_32FC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;
}


void Blender::feed(const Mat &img, const Mat &mask, Point tl) 
{
    CV_Assert(img.type() == CV_32FC3);
    CV_Assert(mask.type() == CV_8U);

    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3f *src_row = img.ptr<Point3f>(y);
        Point3f *dst_row = dst_.ptr<Point3f>(dy + y);

        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x]) 
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}


void Blender::blend(Mat &dst, Mat &dst_mask)
{
    dst_.setTo(Scalar::all(0), dst_mask_ == 0);
    dst = dst_;
    dst_mask = dst_mask_;
    dst_.release();
    dst_mask_.release();
}


void FeatherBlender::prepare(Rect dst_roi)
{
    Blender::prepare(dst_roi);
    dst_weight_map_.create(dst_roi.size(), CV_32F);
    dst_weight_map_.setTo(0);
}


void FeatherBlender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_32FC3);
    CV_Assert(mask.type() == CV_8U);

    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    createWeightMap(mask, sharpness_, weight_map_);

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3f* src_row = img.ptr<Point3f>(y);
        Point3f* dst_row = dst_.ptr<Point3f>(dy + y);

        const float* weight_row = weight_map_.ptr<float>(y);
        float* dst_weight_row = dst_weight_map_.ptr<float>(dy + y);

        for (int x = 0; x < img.cols; ++x)               
        {
            dst_row[dx + x] += src_row[x] * weight_row[x];
            dst_weight_row[dx + x] += weight_row[x];
        }
    }
}


void FeatherBlender::blend(Mat &dst, Mat &dst_mask)
{
    normalize(dst_weight_map_, dst_);
    dst_mask_ = dst_weight_map_ > WEIGHT_EPS;
    Blender::blend(dst, dst_mask);
}


void MultiBandBlender::prepare(Rect dst_roi)
{
    Blender::prepare(dst_roi);

    dst_pyr_laplace_.resize(num_bands_ + 1);
    dst_pyr_laplace_[0].create(dst_roi.size(), CV_32FC3);
    dst_pyr_laplace_[0].setTo(Scalar::all(0));

    dst_band_weights_.resize(num_bands_ + 1);
    dst_band_weights_[0].create(dst_roi.size(), CV_32F);
    dst_band_weights_[0].setTo(0);

    for (int i = 1; i <= num_bands_; ++i)
    {
        dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2, 
                                   (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_32FC3);
        dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                                    (dst_band_weights_[i - 1].cols + 1) / 2, CV_32F);
        dst_pyr_laplace_[i].setTo(Scalar::all(0));
        dst_band_weights_[i].setTo(0);
    }
}


void MultiBandBlender::feed(const Mat &img, const Mat &mask, Point tl)
{
    CV_Assert(img.type() == CV_32FC3);
    CV_Assert(mask.type() == CV_8U);

    int top = tl.y - dst_roi_.y;
    int left = tl.x - dst_roi_.x;
    int bottom = dst_roi_.br().y - tl.y - img.rows;
    int right = dst_roi_.br().x - tl.x - img.cols;

    // Create the source image Laplacian pyramid
    vector<Mat> src_pyr_gauss(num_bands_ + 1);
    copyMakeBorder(img, src_pyr_gauss[0], top, bottom, left, right, 
                   BORDER_REFLECT);
    for (int i = 0; i < num_bands_; ++i)
        pyrDown(src_pyr_gauss[i], src_pyr_gauss[i + 1]);
    vector<Mat> src_pyr_laplace;
    createLaplacePyr(src_pyr_gauss, src_pyr_laplace);
    src_pyr_gauss.clear();

    // Create the weight map Gaussian pyramid
    Mat weight_map;
    mask.convertTo(weight_map, CV_32F, 1./255.);
    vector<Mat> weight_pyr_gauss(num_bands_ + 1);
    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, 
                   BORDER_CONSTANT);
    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= num_bands_; ++i)
    {
        for (int y = 0; y < dst_pyr_laplace_[i].rows; ++y)
        {
            const Point3f* src_row = src_pyr_laplace[i].ptr<Point3f>(y);
            Point3f* dst_row = dst_pyr_laplace_[i].ptr<Point3f>(y);

            const float* weight_row = weight_pyr_gauss[i].ptr<float>(y);

            for (int x = 0; x < dst_pyr_laplace_[i].cols; ++x)               
                dst_row[x] += src_row[x] * weight_row[x];
        }
        dst_band_weights_[i] += weight_pyr_gauss[i];
    }    
}


void MultiBandBlender::blend(Mat &dst, Mat &dst_mask)
{
    for (int i = 0; i <= num_bands_; ++i)
        normalize(dst_band_weights_[i], dst_pyr_laplace_[i]);

    restoreImageFromLaplacePyr(dst_pyr_laplace_);

    dst_ = dst_pyr_laplace_[0];
    dst_mask_ = dst_band_weights_[0] > WEIGHT_EPS;
    dst_pyr_laplace_.clear();
    dst_band_weights_.clear();

    Blender::blend(dst, dst_mask);
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

Rect resultRoi(const vector<Point> &corners, const vector<Size> &sizes)
{
    Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
    Point br(numeric_limits<int>::min(), numeric_limits<int>::min());

    CV_Assert(sizes.size() == corners.size());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        tl.x = min(tl.x, corners[i].x);
        tl.y = min(tl.y, corners[i].y);
        br.x = max(br.x, corners[i].x + sizes[i].width);
        br.y = max(br.y, corners[i].y + sizes[i].height);
    }

    return Rect(tl, br);
}


void normalize(const Mat& weight, Mat& src)
{
    CV_Assert(weight.type() == CV_32F);
    CV_Assert(src.type() == CV_32FC3);
    for (int y = 0; y < src.rows; ++y)
    {
        Point3f *row = src.ptr<Point3f>(y);
        const float *weight_row = weight.ptr<float>(y);

        for (int x = 0; x < src.cols; ++x)
            row[x] *= 1.f / (weight_row[x] + WEIGHT_EPS);
    }
}


void createWeightMap(const Mat &mask, float sharpness, Mat &weight)
{
    CV_Assert(mask.type() == CV_8U);
    distanceTransform(mask, weight, CV_DIST_L1, 3);
    threshold(weight * sharpness, weight, 1.f, 1.f, THRESH_TRUNC);
}


void createLaplacePyr(const vector<Mat> &pyr_gauss, vector<Mat> &pyr_laplace)
{
    if (pyr_gauss.size() == 0)
        return;

    pyr_laplace.resize(pyr_gauss.size());

    Mat tmp;
    for (size_t i = 0; i < pyr_laplace.size() - 1; ++i)
    {
        pyrUp(pyr_gauss[i + 1], tmp, pyr_gauss[i].size());
        pyr_laplace[i] = pyr_gauss[i] - tmp;
    }
    pyr_laplace[pyr_laplace.size() - 1] = pyr_gauss[pyr_laplace.size() - 1].clone();
}


void restoreImageFromLaplacePyr(vector<Mat> &pyr)
{
    if (pyr.size() == 0)
        return;

    Mat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i)
    {
        pyrUp(pyr[i], tmp, pyr[i - 1].size());
        pyr[i - 1] += tmp;
    }
}
