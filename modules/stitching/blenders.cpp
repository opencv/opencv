#include <opencv2/imgproc/imgproc.hpp>
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

    Rect dst_roi = resultRoi(src, corners);

    vector<Mat> src_(num_images);
    vector<Point> corners_(num_images);
    vector<Mat> masks_(num_images);

    // TODO avoid creating extra border
    for (int i = 0; i < num_images; ++i)
    {
        copyMakeBorder(src[i], src_[i],
                       corners[i].y - dst_roi.y, dst_roi.br().y - corners[i].y - src[i].rows,
                       corners[i].x - dst_roi.x, dst_roi.br().x - corners[i].x - src[i].cols,
                       BORDER_REFLECT);
        copyMakeBorder(masks[i], masks_[i],
                       corners[i].y - dst_roi.y, dst_roi.br().y - corners[i].y - src[i].rows,
                       corners[i].x - dst_roi.x, dst_roi.br().x - corners[i].x - src[i].cols,
                       BORDER_CONSTANT);
        corners_[i] = Point(0, 0);
    }

    Mat weight_map;
    vector<Mat> src_pyr_gauss;
    vector< vector<Mat> > src_pyr_laplace(num_images);
    vector< vector<Mat> > weight_pyr_gauss(num_images);

    // Compute all pyramids
    for (int i = 0; i < num_images; ++i)
    {
        createGaussPyr(src_[i], num_bands_, src_pyr_gauss);
        createLaplacePyr(src_pyr_gauss, src_pyr_laplace[i]);

        masks_[i].convertTo(weight_map, CV_32F, 1. / 255.);
        createGaussPyr(weight_map, num_bands_, weight_pyr_gauss[i]);
    }

    computeResultMask(masks, corners, dst_mask);

    Mat dst_level_weight;
    vector<Mat> dst_pyr_laplace(num_bands_ + 1);
    vector<Mat> src_pyr_slice(num_images);
    vector<Mat> weight_pyr_slice(num_images);

    // Blend pyramids
    for (int level_id = 0; level_id <= num_bands_; ++level_id)
    {
        for (int i = 0; i < num_images; ++i)
        {
            src_pyr_slice[i] = src_pyr_laplace[i][level_id];
            weight_pyr_slice[i] = weight_pyr_gauss[i][level_id];
        }
        blendLinear(src_pyr_slice, corners_, weight_pyr_slice,
                    dst_pyr_laplace[level_id], dst_level_weight);
    }

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

    // Normalize sums
    for (int y = 0; y < dst.rows; ++y)
    {
        Point3f *dst_row = dst.ptr<Point3f>(y);
        float *dst_weight_row = dst_weight.ptr<float>(y);

        for (int x = 0; x < dst.cols; ++x)
        {
            dst_weight_row[x] += WEIGHT_EPS;
            dst_row[x] *= 1.f / dst_weight_row[x];
        }
    }

    return dst_roi.tl();
}


void createWeightMap(const Mat &mask, float sharpness, Mat &weight)
{
    CV_Assert(mask.type() == CV_8U);
    distanceTransform(mask, weight, CV_DIST_L1, 3);
    threshold(weight * sharpness, weight, 1.f, 1.f, THRESH_TRUNC);
}


void createGaussPyr(const Mat &img, int num_layers, vector<Mat> &pyr)
{
    pyr.resize(num_layers + 1);
    pyr[0] = img.clone();
    for (int i = 0; i < num_layers; ++i)
        pyrDown(pyr[i], pyr[i + 1]);
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
