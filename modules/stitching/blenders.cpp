#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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


Point Blender::operator ()(const vector<Mat> &src, const vector<Point> &corners, const vector<Mat> &masks,
                           Mat& dst)
{
    Mat dst_mask;
    return (*this)(src, corners, masks, dst, dst_mask);
}


Point Blender::operator ()(const vector<Mat> &src, const vector<Point> &corners, const vector<Mat> &masks,
                           Mat &dst, Mat &dst_mask)
{
    Point dst_tl = blend(src, corners, masks, dst, dst_mask);
    dst.setTo(Scalar::all(0), dst_mask == 0);
    return dst_tl;
}


Point Blender::blend(const vector<Mat> &src, const vector<Point> &corners, const vector<Mat> &masks,
                     Mat &dst, Mat &dst_mask)
{
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].type() == CV_32FC3);
        CV_Assert(masks[i].type() == CV_8U);
    }
    const int image_type = src[0].type();

    Rect dst_roi = resultRoi(src, corners);

    dst.create(dst_roi.size(), image_type);
    dst.setTo(Scalar::all(0));

    dst_mask.create(dst_roi.size(), CV_8U);
    dst_mask.setTo(Scalar::all(0));

    for (size_t i = 0; i < src.size(); ++i)
    {
        int dx = corners[i].x - dst_roi.x;
        int dy = corners[i].y - dst_roi.y;

        for (int y = 0; y < src[i].rows; ++y)
        {
            const Point3f *src_row = src[i].ptr<Point3f>(y);
            Point3f *dst_row = dst.ptr<Point3f>(dy + y);

            const uchar *mask_row = masks[i].ptr<uchar>(y);
            uchar *dst_mask_row = dst_mask.ptr<uchar>(dy + y);

            for (int x = 0; x < src[i].cols; ++x)
            {
                if (mask_row[x])
                    dst_row[dx + x] = src_row[x];
                dst_mask_row[dx + x] |= mask_row[x];
            }
        }
    }

    return dst_roi.tl();
}


Point FeatherBlender::blend(const vector<Mat> &src, const vector<Point> &corners, const vector<Mat> &masks,
                            Mat &dst, Mat &dst_mask)
{
    vector<Mat> weights(masks.size());
    for (size_t i = 0; i < weights.size(); ++i)
        createWeightMap(masks[i], sharpness_, weights[i]);

    Mat dst_weight;
    Point dst_tl = blendLinear(src, corners, weights, dst, dst_weight);
    dst_mask = dst_weight > WEIGHT_EPS;

    return dst_tl;
}


Point MultiBandBlender::blend(const vector<Mat> &src, const vector<Point> &corners, const vector<Mat> &masks,
                             Mat &dst, Mat &dst_mask)
{
    CV_Assert(src.size() == corners.size() && src.size() == masks.size());
    const int num_images = src.size();
    const int img_type = src[0].type();

    Rect dst_roi = resultRoi(src, corners);
    computeResultMask(masks, corners, dst_mask);

    vector<Mat> dst_pyr_laplace(num_bands_ + 1);
    dst_pyr_laplace[0].create(dst_roi.size(), img_type);
    dst_pyr_laplace[0].setTo(Scalar::all(0));

    vector<Mat> dst_band_weights(num_bands_ + 1);
    dst_band_weights[0].create(dst_roi.size(), CV_32F);
    dst_band_weights[0].setTo(0);

    for (int i = 1; i <= num_bands_; ++i)
    {
        dst_pyr_laplace[i].create((dst_pyr_laplace[i - 1].rows + 1) / 2, 
                                  (dst_pyr_laplace[i - 1].cols + 1) / 2, img_type);
        dst_pyr_laplace[i].setTo(Scalar::all(0));

        dst_band_weights[i].create((dst_band_weights[i - 1].rows + 1) / 2,
                                   (dst_band_weights[i - 1].cols + 1) / 2, CV_32F);
        dst_band_weights[i].setTo(0);
    }

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        int top = corners[img_idx].y - dst_roi.y;
        int bottom = dst_roi.br().y - corners[img_idx].y - src[img_idx].rows;
        int left = corners[img_idx].x - dst_roi.x;
        int right = dst_roi.br().x - corners[img_idx].x - src[img_idx].cols;

        vector<Mat> src_pyr_gauss(num_bands_ + 1);
        copyMakeBorder(src[img_idx], src_pyr_gauss[0], top, bottom, left, right, BORDER_REFLECT);
        for (int i = 0; i < num_bands_; ++i)
            pyrDown(src_pyr_gauss[i], src_pyr_gauss[i + 1]);

        vector<Mat> src_pyr_laplace;
        createLaplacePyr(src_pyr_gauss, src_pyr_laplace);

        vector<Mat> weight_pyr_gauss(num_bands_ + 1);
        Mat mask_f;
        masks[img_idx].convertTo(mask_f, CV_32F, 1./255.);
        copyMakeBorder(mask_f, weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT);
        for (int i = 0; i < num_bands_; ++i)
            pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

        for (int band_idx = 0; band_idx <= num_bands_; ++band_idx)
        {
            for (int y = 0; y < dst_pyr_laplace[band_idx].rows; ++y)
            {
                const Point3f* src_row = src_pyr_laplace[band_idx].ptr<Point3f>(y);
                const float* weight_row = weight_pyr_gauss[band_idx].ptr<float>(y);
                Point3f* dst_row = dst_pyr_laplace[band_idx].ptr<Point3f>(y);
                for (int x = 0; x < dst_pyr_laplace[band_idx].cols; ++x)               
                    dst_row[x] += src_row[x] * weight_row[x];
            }
            dst_band_weights[band_idx] += weight_pyr_gauss[band_idx];
        }
    }

    for (int band_idx = 0; band_idx <= num_bands_; ++band_idx)
        normalize(dst_band_weights[band_idx], dst_pyr_laplace[band_idx]);

    restoreImageFromLaplacePyr(dst_pyr_laplace);
    dst = dst_pyr_laplace[0];
    return dst_roi.tl();
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

Rect resultRoi(const vector<Mat> &src, const vector<Point> &corners)
{
    Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
    Point br(numeric_limits<int>::min(), numeric_limits<int>::min());

    CV_Assert(src.size() == corners.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        tl.x = min(tl.x, corners[i].x);
        tl.y = min(tl.y, corners[i].y);
        br.x = max(br.x, corners[i].x + src[i].cols);
        br.y = max(br.y, corners[i].y + src[i].rows);
    }

    return Rect(tl, br);
}


Point computeResultMask(const vector<Mat> &masks, const vector<Point> &corners, Mat &dst_mask)
{
    Rect dst_roi = resultRoi(masks, corners);

    dst_mask.create(dst_roi.size(), CV_8U);
    dst_mask.setTo(Scalar::all(0));

    for (size_t i = 0; i < masks.size(); ++i)
    {
        int dx = corners[i].x - dst_roi.x;
        int dy = corners[i].y - dst_roi.y;

        for (int y = 0; y < masks[i].rows; ++y)
        {
            const uchar *mask_row = masks[i].ptr<uchar>(y);
            uchar *dst_mask_row = dst_mask.ptr<uchar>(dy + y);

            for (int x = 0; x < masks[i].cols; ++x)
                dst_mask_row[dx + x] |= mask_row[x];
        }
    }

    return dst_roi.tl();
}


Point blendLinear(const vector<Mat> &src, const vector<Point> &corners, const vector<Mat> &weights,
                  Mat &dst, Mat& dst_weight)
{
    for (size_t i = 0; i < src.size(); ++i)
    {
        CV_Assert(src[i].type() == CV_32FC3);
        CV_Assert(weights[i].type() == CV_32F);
    }
    const int image_type = src[0].type();

    Rect dst_roi = resultRoi(src, corners);

    dst.create(dst_roi.size(), image_type);
    dst.setTo(Scalar::all(0));

    dst_weight.create(dst_roi.size(), CV_32F);
    dst_weight.setTo(Scalar::all(0));

    // Compute colors sums and weights
    for (size_t i = 0; i < src.size(); ++i)
    {
        int dx = corners[i].x - dst_roi.x;
        int dy = corners[i].y - dst_roi.y;

        for (int y = 0; y < src[i].rows; ++y)
        {
            const Point3f *src_row = src[i].ptr<Point3f>(y);
            Point3f *dst_row = dst.ptr<Point3f>(dy + y);

            const float *weight_row = weights[i].ptr<float>(y);
            float *dst_weight_row = dst_weight.ptr<float>(dy + y);

            for (int x = 0; x < src[i].cols; ++x)
            {
                dst_row[dx + x] += src_row[x] * weight_row[x];
                dst_weight_row[dx + x] += weight_row[x];
            }
        }
    }

    normalize(dst_weight, dst);

    return dst_roi.tl();
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
