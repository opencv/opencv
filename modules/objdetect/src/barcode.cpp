// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#include "precomp.hpp"
#include <opencv2/objdetect/barcode.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "barcode_decoder/ean13_decoder.hpp"
#include "barcode_decoder/ean8_decoder.hpp"
#include "barcode_detector/bardetect.hpp"
#include "barcode_decoder/common/super_scale.hpp"
#include "barcode_decoder/common/utils.hpp"
#include "graphical_code_detector_impl.hpp"

using std::string;
using std::vector;
using std::make_shared;
using std::array;
using std::shared_ptr;
using std::dynamic_pointer_cast;

namespace cv {
namespace barcode {

//==================================================================================================

static bool checkBarInputImage(InputArray img, Mat &gray)
{
    CV_Assert(!img.empty());
    CV_CheckDepthEQ(img.depth(), CV_8U, "");
    if (img.cols() <= 40 || img.rows() <= 40)
    {
        return false; // image data is not enough for providing reliable results
    }
    int incn = img.channels();
    CV_Check(incn, incn == 1 || incn == 3 || incn == 4, "");
    if (incn == 3 || incn == 4)
    {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = img.getMat();
    }
    return true;
}

static void updatePointsResult(OutputArray points_, const vector<Point2f> &points)
{
    if (points_.needed())
    {
        int N = int(points.size() / 4);
        if (N > 0)
        {
            Mat m_p(N, 4, CV_32FC2, (void *) &points[0]);
            int points_type = points_.fixedType() ? points_.type() : CV_32FC2;
            m_p.reshape(2, points_.rows()).convertTo(points_, points_type);  // Mat layout: N x 4 x 2cn
        }
        else
        {
            points_.release();
        }
    }
}

inline const array<shared_ptr<AbsDecoder>, 2> &getDecoders()
{
    //indicate Decoder
    static const array<shared_ptr<AbsDecoder>, 2> decoders{
            shared_ptr<AbsDecoder>(new Ean13Decoder()), shared_ptr<AbsDecoder>(new Ean8Decoder())};
    return decoders;
}

//==================================================================================================

class BarDecode
{
public:
    void init(const vector<Mat> &bar_imgs_);

    const vector<Result> &getDecodeInformation()
    { return result_info; }

    bool decodeMultiplyProcess();

private:
    vector<Mat> bar_imgs;
    vector<Result> result_info;
};

void BarDecode::init(const vector<Mat> &bar_imgs_)
{
    bar_imgs = bar_imgs_;
}

bool BarDecode::decodeMultiplyProcess()
{
    static float constexpr THRESHOLD_CONF = 0.6f;
    result_info.clear();
    result_info.resize(bar_imgs.size());
    parallel_for_(Range(0, int(bar_imgs.size())), [&](const Range &range) {
        for (int i = range.start; i < range.end; i++)
        {
            Mat bin_bar;
            Result max_res;
            float max_conf = -1.f;
            bool decoded = false;
            for (const auto &decoder:getDecoders())
            {
                if (decoded)
                { break; }
                for (const auto binary_type : binary_types)
                {
                    binarize(bar_imgs[i], bin_bar, binary_type);
                    auto cur_res = decoder->decodeROI(bin_bar);
                    if (cur_res.second > max_conf)
                    {
                        max_res = cur_res.first;
                        max_conf = cur_res.second;
                        if (max_conf > THRESHOLD_CONF)
                        {
                            // code decoded
                            decoded = true;
                            break;
                        }
                    }
                } //binary types
            } //decoder types

            result_info[i] = max_res;
        }
    });
    return !result_info.empty();
}

//==================================================================================================
// Private class definition and implementation (pimpl)

struct BarcodeImpl : public GraphicalCodeDetector::Impl
{
public:
    shared_ptr<SuperScale> sr;
    bool use_nn_sr = false;
    double detectorThrDownSample = 512.f;
    vector<float> detectorWindowSizes = {0.01f, 0.03f, 0.06f, 0.08f};
    double detectorThrGradMagnitude = 64.f;

public:
    //=================
    // own methods
    BarcodeImpl() {}

    vector<Mat> initDecode(const Mat &src, const vector<vector<Point2f>> &points) const;
    bool decodeWithType(InputArray img,
                     InputArray points,
                     vector<string> &decoded_info,
                     vector<string> &decoded_type) const;
    bool detectAndDecodeWithType(InputArray img,
                              vector<string> &decoded_info,
                              vector<string> &decoded_type,
                              OutputArray points_) const;

    //=================
    // implement interface
    ~BarcodeImpl() CV_OVERRIDE {}
    bool detect(InputArray img, OutputArray points) const CV_OVERRIDE;
    string decode(InputArray img, InputArray points, OutputArray straight_code) const CV_OVERRIDE;
    string detectAndDecode(InputArray img, OutputArray points, OutputArray straight_code) const CV_OVERRIDE;
    bool detectMulti(InputArray img, OutputArray points) const CV_OVERRIDE;
    bool decodeMulti(InputArray img, InputArray points, vector<string>& decoded_info, OutputArrayOfArrays straight_code) const CV_OVERRIDE;
    bool detectAndDecodeMulti(InputArray img, vector<string>& decoded_info, OutputArray points, OutputArrayOfArrays straight_code) const CV_OVERRIDE;
};

// return cropped and scaled bar img
vector<Mat> BarcodeImpl::initDecode(const Mat &src, const vector<vector<Point2f>> &points) const
{
    vector<Mat> bar_imgs;
    for (auto &corners : points)
    {
        Mat bar_img;
        cropROI(src, bar_img, corners);
//        sharpen(bar_img, bar_img);
        // empirical settings
        if (bar_img.cols < 320 || bar_img.cols > 640)
        {
            float scale = 560.0f / static_cast<float>(bar_img.cols);
            sr->processImageScale(bar_img, bar_img, scale, use_nn_sr);
        }
        bar_imgs.emplace_back(bar_img);
    }
    return bar_imgs;
}

bool BarcodeImpl::decodeWithType(InputArray img,
                              InputArray points,
                              vector<string> &decoded_info,
                              vector<string> &decoded_type) const
{
    Mat inarr;
    if (!checkBarInputImage(img, inarr))
    {
        return false;
    }
    CV_Assert(points.size().width > 0);
    CV_Assert((points.size().width % 4) == 0);
    vector<vector<Point2f>> src_points;
    Mat bar_points = points.getMat();
    bar_points = bar_points.reshape(2, 1);
    for (int i = 0; i < bar_points.size().width; i += 4)
    {
        vector<Point2f> tempMat = bar_points.colRange(i, i + 4);
        if (contourArea(tempMat) > 0.0)
        {
            src_points.push_back(tempMat);
        }
    }
    CV_Assert(!src_points.empty());
    vector<Mat> bar_imgs = initDecode(inarr, src_points);
    BarDecode bardec;
    bardec.init(bar_imgs);
    bardec.decodeMultiplyProcess();
    const vector<Result> info = bardec.getDecodeInformation();
    decoded_info.clear();
    decoded_type.clear();
    bool ok = false;
    for (const auto &res : info)
    {
        if (res.isValid())
        {
            ok = true;
        }

        decoded_info.emplace_back(res.result);
        decoded_type.emplace_back(res.typeString());
    }
    return ok;
}

bool BarcodeImpl::detectAndDecodeWithType(InputArray img,
                                       vector<string> &decoded_info,
                                       vector<string> &decoded_type,
                                       OutputArray points_) const
{
    Mat inarr;
    if (!checkBarInputImage(img, inarr))
    {
        points_.release();
        return false;
    }
    vector<Point2f> points;
    bool ok = this->detect(inarr, points);
    if (!ok)
    {
        points_.release();
        return false;
    }
    updatePointsResult(points_, points);
    decoded_info.clear();
    decoded_type.clear();
    ok = decodeWithType(inarr, points, decoded_info, decoded_type);
    return ok;
}

bool BarcodeImpl::detect(InputArray img, OutputArray points) const
{
    Mat inarr;
    if (!checkBarInputImage(img, inarr))
    {
        points.release();
        return false;
    }

    Detect bardet;
    bardet.init(inarr, detectorThrDownSample);
    bardet.localization(detectorWindowSizes, detectorThrGradMagnitude);
    if (!bardet.computeTransformationPoints())
    { return false; }
    vector<vector<Point2f>> pnts2f = bardet.getTransformationPoints();
    vector<Point2f> trans_points;
    for (auto &i : pnts2f)
    {
        for (const auto &j : i)
        {
            trans_points.push_back(j);
        }
    }
    updatePointsResult(points, trans_points);
    return true;
}

string BarcodeImpl::decode(InputArray img, InputArray points, OutputArray straight_code) const
{
    CV_UNUSED(straight_code);
    vector<string> decoded_info;
    vector<string> decoded_type;
    if (!decodeWithType(img, points, decoded_info, decoded_type))
        return string();
    if (decoded_info.size() < 1)
        return string();
    return decoded_info[0];
}

string BarcodeImpl::detectAndDecode(InputArray img, OutputArray points, OutputArray straight_code) const
{
    CV_UNUSED(straight_code);
    vector<string> decoded_info;
    vector<string> decoded_type;
    vector<Point2f> points_;
    if (!detectAndDecodeWithType(img, decoded_info, decoded_type, points_))
        return string();
    if (points_.size() < 4 || decoded_info.size() < 1)
        return string();
    points_.resize(4);
    updatePointsResult(points, points_);
    return decoded_info[0];
}

bool BarcodeImpl::detectMulti(InputArray img, OutputArray points) const
{
    return detect(img, points);
}

bool BarcodeImpl::decodeMulti(InputArray img, InputArray points, vector<string> &decoded_info, OutputArrayOfArrays straight_code) const
{
    CV_UNUSED(straight_code);
    vector<string> decoded_type;
    return decodeWithType(img, points, decoded_info, decoded_type);
}

bool BarcodeImpl::detectAndDecodeMulti(InputArray img, vector<string> &decoded_info, OutputArray points, OutputArrayOfArrays straight_code) const
{
    CV_UNUSED(straight_code);
    vector<string> decoded_type;
    return detectAndDecodeWithType(img, decoded_info, decoded_type, points);
}

//==================================================================================================
// Public class implementation

BarcodeDetector::BarcodeDetector()
    : BarcodeDetector(string(), string())
{
}

BarcodeDetector::BarcodeDetector(const string &prototxt_path, const string &model_path)
{
    Ptr<BarcodeImpl> p_ = new BarcodeImpl();
    p = p_;
    p_->sr = make_shared<SuperScale>();
    if (!prototxt_path.empty() && !model_path.empty())
    {
        CV_Assert(utils::fs::exists(prototxt_path));
        CV_Assert(utils::fs::exists(model_path));
        int res = p_->sr->init(prototxt_path, model_path);
        CV_Assert(res == 0);
        p_->use_nn_sr = true;
    }
}

BarcodeDetector::~BarcodeDetector() = default;

bool BarcodeDetector::decodeWithType(InputArray img, InputArray points, vector<string> &decoded_info, vector<string> &decoded_type) const
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);
    return p_->decodeWithType(img, points, decoded_info, decoded_type);
}

bool BarcodeDetector::detectAndDecodeWithType(InputArray img, vector<string> &decoded_info, vector<string> &decoded_type, OutputArray points_) const
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);
    return p_->detectAndDecodeWithType(img, decoded_info, decoded_type, points_);
}

double BarcodeDetector::getDownsamplingThreshold() const
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);

    return p_->detectorThrDownSample;
}

BarcodeDetector& BarcodeDetector::setDownsamplingThreshold(double thresh)
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);
    CV_Assert(thresh >= 64);

    p_->detectorThrDownSample = thresh;
    return *this;
}

void BarcodeDetector::getDetectorScales(CV_OUT std::vector<float>& sizes) const
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);

    sizes = p_->detectorWindowSizes;
}

BarcodeDetector& BarcodeDetector::setDetectorScales(const std::vector<float>& sizes)
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);
    CV_Assert(sizes.size() > 0 && sizes.size() <= 16);

    for (const float &size : sizes) {
        CV_Assert(size > 0 && size < 1);
    }

    p_->detectorWindowSizes = sizes;

    return *this;
}

double BarcodeDetector::getGradientThreshold() const
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);

    return p_->detectorThrGradMagnitude;
}

BarcodeDetector& BarcodeDetector::setGradientThreshold(double thresh)
{
    Ptr<BarcodeImpl> p_ = dynamic_pointer_cast<BarcodeImpl>(p);
    CV_Assert(p_);
    CV_Assert(thresh >= 0 && thresh < 1e4);

    p_->detectorThrGradMagnitude = thresh;
    return *this;
}

}// namespace barcode
} // namespace cv
