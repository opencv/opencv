#include "opencv2/core.hpp"
#include "qbardecoder.hpp"
#include <iostream>
#include <unordered_map>

#ifdef __NEON__
#include <arm_neon.h>
#endif

#if (defined WIN32 || defined _WIN32) && defined(_M_ARM)
#include <Intrin.h>
#include "arm_neon.h"
#define __NEON__ 1
#define CPU_HAS_NEON_FEATURE (true)
#elif defined(__ARM_NEON__)
#include <arm_neon.h>
#define __NEON__ 1
#define CPU_HAS_NEON_FEATURE (true)
#endif

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || \
defined(__MINGW32__) || defined(__BORLANDC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif
#define CLIP(x, x1, x2) (std::fmax<float>)(x1, (std::fmin<float>)(x, x2))
using namespace zxing::common;
using namespace zxing;
using namespace zxing::qrcode;
using namespace cv;
using namespace cv::dnn;

#ifdef _WIN32
#include <time.h>
#include <windows.h>
#include <wchar.h>

int gettimeofday(struct timeval *tp, void *tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year   = wtm.wYear - 1900;
    tm.tm_mon   = wtm.wMonth - 1;
    tm.tm_mday   = wtm.wDay;
    tm.tm_hour   = wtm.wHour;
    tm.tm_min   = wtm.wMinute;
    tm.tm_sec   = wtm.wSecond;
    tm.tm_isdst  = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#else
#include <sys/time.h>
#endif

namespace cv {
void QBarDecoder::Detect(Mat srcImage, std::vector<DetectInfo> &bboxes) {
    if(_init_detector_model_)
        detector_->Detect(srcImage, bboxes);
}

QBAR_RESULT QBarDecoder::Decode(Mat& srcCvImage)
{
    if (srcCvImage.data == nullptr || (srcCvImage.rows < 1) || (srcCvImage.cols < 1))
    {
        return QBAR_RESULT::MakeInvalid();
    }

    Mat img = srcCvImage;
    if (!img.isContinuous())
    {
        img = img.clone();
    }

    DecodeHints decodeHints;
    AddFormatsToDecodeHints(decodeHints);
    decodeHints.setTryHarder(1);
    decodeHints.setPureBarcode(true);
    decodeHints.qbar_points.resize(4);
    decodeHints.qbar_points[0].x = 0;
    decodeHints.qbar_points[0].y = 0;
    decodeHints.qbar_points[1].x = img.cols - 1;
    decodeHints.qbar_points[1].y = 0;
    decodeHints.qbar_points[2].x = img.cols - 1;
    decodeHints.qbar_points[2].y = img.rows - 1;
    decodeHints.qbar_points[3].x = 0;
    decodeHints.qbar_points[3].y = img.rows - 1;

    int width = img.cols;
    int height = img.rows;
    int comps = 1;
    int pixelStep = 1;
    int tryBinarizeTime = 3;
    Ref<Result> result;
    Ref<QBarSource> source;
    ErrorHandler err_handler;

    for (int tb = 0; tb < tryBinarizeTime; tb++)
    {
        err_handler.Reset();

        if (source == NULL || height * width > source->getMaxSize())
        {
            source = QBarSource::create(img.data, width, height, comps, pixelStep, err_handler);
            if (err_handler.ErrCode())
            {
                std::cout << "continue by errmsg " << err_handler.ErrMsg() << std::endl;
                continue;
            }
        }
        else
        {
            source->reset(img.data, width, height, comps, pixelStep, err_handler);
            if (err_handler.ErrCode())
            {
                std::cout << "continue by errmsg " << err_handler.ErrMsg() << std::endl;
                continue;
            }
        }

        int ret = Decode(source, result, decodeHints);
        if (ret == 0)
        {
            return ProcessResult(result);
        }

        binarizer_mgr_.SwitchBinarizer();
    }

    return QBAR_RESULT::MakeInvalid();
}

int QBarDecoder::Decode(Ref<LuminanceSource> source, Ref<Result> &result, DecodeHints& decodeHints)
{
    int res = -1;
    std::string cell_result;

    zxing::Ref<zxing::Binarizer> binarizer = binarizer_mgr_.Binarize(source);
    zxing::Ref<zxing::BinaryBitmap> binary_bitmap(new BinaryBitmap(binarizer));
    binary_bitmap->m_poUnicomBlock = new UnicomBlock(source->getWidth(), source->getHeight());

    try
    {
        result = reader_.decode(binary_bitmap, decodeHints);
        res = (result == NULL) ? 1 : 0;
    }
    catch (std::exception &e)
    {
        std::cout << "Decode exception: " << e.what() << std::endl;
        return -1;
    }

    if (res == 0)
    {
        result->setBinaryMethod(static_cast<int>(binarizer_mgr_.GetCurBinarizer()));
    }

    return res;
}

QBAR_RESULT QBarDecoder::ProcessResult(zxing::Result *zx_result)
{
    QBAR_RESULT result;

    result.typeID = static_cast<int>(zx_result->getBarcodeFormat());
    result.typeName = std::string(BarcodeFormat::barcodeFormatNames[result.typeID]);

    std::string toCharset = output_charset_;
    std::string rawString = zx_result->getText()->getText();

    char *rawChars = (char *)rawString.c_str();
    std::string fromCharset = StringUtils::guessEncoding(rawChars, rawString.size());

    // first copy rst
    result.charset = fromCharset;
    result.data = rawString;

#ifndef NO_ICONV
    if ((toCharset == "ANY") || (fromCharset == toCharset) || (fromCharset == "ANY"))
    {
        // No need to convert charset
        result.charset = fromCharset;
        result.data = rawString;
    }
    else
    {
        // Need to convert charset
        result.charset = toCharset;

        std::string toString = StringUtils::convertString(rawString.c_str(), rawString.size(), fromCharset.c_str(), toCharset.c_str());

        // Try convert to GBK again
        if (toString == "" && toCharset != StringUtils::GBK)
        {
            toCharset = StringUtils::GBK;
            toString = StringUtils::convertString(rawString.c_str(), rawString.size(), fromCharset.c_str(), toCharset.c_str());

            if (toString.size() > 0)
            {
                result.charset = toCharset;
            }
        }

        // Try convert to UTF-8 again
        if (toString == "" && toCharset != StringUtils::UTF8)
        {
            toCharset = StringUtils::UTF8;
            toString = StringUtils::convertString(rawString.c_str(), rawString.size(), fromCharset.c_str(), toCharset.c_str());

            if (toString.size() > 0)
            {
                result.charset = toCharset;
            }
        }

        if (toString.size())
            result.data = toString;
    }
#endif

    for (int j = 0; j < zx_result->getResultPoints()->size(); j++)
    {
        QBAR_POINT point;
        point.x = zx_result->getResultPoints()[j]->getX();
        point.y = zx_result->getResultPoints()[j]->getY();
        result.points.push_back(point);
    }

    result.reportMsg.qrcodeVersion = zx_result->getQRCodeVersion();
    result.reportMsg.pyramidLv = zx_result->getPyramidLv();

    result.reportMsg.ecLevel = zx_result->getEcLevel();
    result.reportMsg.charsetMode = zx_result->getChartsetMode();
    result.reportMsg.scale_list_ = zx_result->getScaleList();
    result.reportMsg.decode_scale_ = zx_result->getDecodeScale();
    result.reportMsg.detect_time_ = zx_result->getDetectTime();
    result.reportMsg.sr_time_ = zx_result->getSrTime();
    result.reportMsg.has_sr = zx_result->getHasSr();
    result.reportMsg.decode_time_ = zx_result->getDecodeTime();

    result.reportMsg.pre_detect_time_ = zx_result->getPreDetectTime();
    result.reportMsg.detect_infer_pre_time_ = zx_result->getDetectInferPreTime();
    result.reportMsg.detect_infer_time_ = zx_result->getDetectInferTime();
    result.reportMsg.detect_infer_after_time_ = zx_result->getDetectInferAfterTime();
    result.reportMsg.after_detect_time_ = zx_result->getAfterDetectTime();
    result.reportMsg.seg_time_ = zx_result->getSegeTime();
    result.reportMsg.has_seg = zx_result->getHasSeg();
    result.reportMsg.after_seg_time_ = zx_result->getAfterSegTime();
    result.reportMsg.decode_all_time_ = zx_result->getDecodeAllTime();
    result.reportMsg.has_decode = zx_result->getHasDecode();

    return result;
}

class ParallelDecode : public cv::ParallelLoopBody {
public:
    ParallelDecode(QBarDecoder* decoder, const Mat& srcImage, const std::vector<DetectInfo>& detect_results, 
                   std::vector<QBAR_RESULT>& results)
        : decoder(decoder), srcImage(srcImage), detect_results(detect_results), results(results) {}

    void operator()(const cv::Range& range) const CV_OVERRIDE {
        QBarDecoder local_decoder;
        local_decoder.SetReaders(decoder->readers_);

        for (int i = range.start; i < range.end; ++i) {
            const DetectInfo& detect_info = detect_results[i];
            Align aligner;
            Mat crop_image = decoder->cropObj(srcImage, detect_info, aligner);

            auto scale_list = decoder->getScaleList(crop_image.cols, crop_image.rows);
            QBAR_RESULT result;

            for (auto cur_scale : scale_list) {
                Mat scaled_img;
                {
                    std::lock_guard<std::mutex> lock(decoder->sr_mutex);
                    scaled_img = decoder->sr_->ProcessImageScale(crop_image, cur_scale, decoder->_init_sr_model_);
                }
                result = local_decoder.Decode(scaled_img); 

                if (result.typeID != 0) {
                    std::vector<Point2f> points_qr;
                    for (size_t j = 0; j < result.points.size(); ++j) {
                        Point2f point(result.points[j].x, result.points[j].y);
                        point /= cur_scale;
                        points_qr.push_back(point);
                    }
                    if (decoder->_init_sr_model_) {
                        points_qr = aligner.warpBack(points_qr);
                    }
                    for (size_t j = 0; j < points_qr.size(); ++j) {
                        result.points[j].x = points_qr[j].x;
                        result.points[j].y = points_qr[j].y;
                    }
                    break; 
                }
            }
            if (result.typeID != 0) {            
                results[i] = result;
            }
        }
    }

private:
    QBarDecoder* decoder;
    const Mat& srcImage;
    const std::vector<DetectInfo>& detect_results;
    std::vector<QBAR_RESULT>& results;
};

std::vector<QBAR_RESULT> QBarDecoder::Decode(Mat srcImage, std::vector<DetectInfo>& detect_results) {
    std::vector<QBAR_RESULT> results(detect_results.size());

    ParallelDecode parallelDecode(this, srcImage, detect_results, results);
    
    parallel_for_(Range(0, int(detect_results.size())), parallelDecode);
    
    return results;
}

void QBarDecoder::nms(std::vector<QBAR_RESULT>& results, float NMS_THRESH) {
    if (results.size() <= 1) return;
    std::unordered_map<int, std::vector<QBAR_RESULT>> class_map;

    for (const auto& result : results) {
        class_map[result.typeID].push_back(result);
    }

    std::vector<QBAR_RESULT> final_results;

    for (auto& pair : class_map) {
        auto& class_results = pair.second;
        
        // leftup: p1   rightdown: p3
        std::sort(class_results.begin(), class_results.end(), [](const QBAR_RESULT& a, const QBAR_RESULT& b) {
            int widthA = a.points[3].x - a.points[1].x + 1;
            int heightA = a.points[3].y - a.points[1].y + 1;
            int widthB = b.points[3].x - b.points[1].x + 1;
            int heightB = b.points[3].y - b.points[1].y + 1;
            return (widthA * heightA) > (widthB * heightB); 
        });

        std::vector<float> vArea(class_results.size());
        for (size_t i = 0; i < class_results.size(); ++i) {
            vArea[i] = (class_results[i].points[3].x - class_results[i].points[1].x + 1) * (class_results[i].points[3].y - class_results[i].points[1].y + 1);
        }

        for (size_t i = 0; i < class_results.size(); ++i) {
            final_results.push_back(class_results[i]);
            if (class_results[i].typeID == 0) continue; 
            // skip oned
            if (class_results[i].typeID != 6 && class_results[i].typeID != 11 && class_results[i].typeID != 12) continue; 

            for (size_t j = i + 1; j < class_results.size();) {
                float xx1 = std::max(class_results[i].points[1].x, class_results[j].points[1].x);
                float yy1 = std::max(class_results[i].points[1].y, class_results[j].points[1].y);
                float xx2 = std::min(class_results[i].points[3].x, class_results[j].points[3].x);
                float yy2 = std::min(class_results[i].points[3].y, class_results[j].points[3].y);

                float w = std::max(0.0f, xx2 - xx1 + 1);
                float h = std::max(0.0f, yy2 - yy1 + 1);
                float inter = w * h;

                float ovr = inter / (vArea[i] + vArea[j] - inter);
                float cover = inter / std::min(vArea[i], vArea[j]);

                if (ovr >= NMS_THRESH) {
                    class_results.erase(class_results.begin() + j);
                    vArea.erase(vArea.begin() + j);
                } else if (cover >= 0.96) {
                    if (vArea[i] > vArea[j]) {
                        class_results.erase(class_results.begin() + j);
                        vArea.erase(vArea.begin() + j);
                    } else {
                        class_results[i].points[1] = class_results[j].points[1];
                        class_results[i].points[3] = class_results[j].points[3];
                        class_results.erase(class_results.begin() + j);
                        vArea.erase(vArea.begin() + j);
                    }
                } else {
                    j++; 
                }
            }
        }
    }

    results = final_results;
}

void QBarDecoder::AddFormatsToDecodeHints(zxing::DecodeHints &hints) {
    if (readers_.count(QBAR_READER::ONED_BARCODE))
    {
        hints.addFormat(BarcodeFormat::CODE_25);
        hints.addFormat(BarcodeFormat::CODE_39);
        hints.addFormat(BarcodeFormat::CODE_93);
        hints.addFormat(BarcodeFormat::CODE_128);
        hints.addFormat(BarcodeFormat::ITF);
        hints.addFormat(BarcodeFormat::CODABAR);
        hints.addFormat(BarcodeFormat::UPC_A);
        hints.addFormat(BarcodeFormat::UPC_E);
        hints.addFormat(BarcodeFormat::EAN_8);
        hints.addFormat(BarcodeFormat::EAN_13);
        hints.addFormat(BarcodeFormat::RSS_14);
    }
    if (readers_.count(QBAR_READER::QRCODE))
    {
        hints.addFormat(BarcodeFormat::QR_CODE);
    }
    if (readers_.count(QBAR_READER::PDF417))
    {
        hints.addFormat(BarcodeFormat::PDF_417);
    }
    if (readers_.count(QBAR_READER::DATAMATRIX))
    {
        hints.addFormat(BarcodeFormat::DATA_MATRIX);
    }
}

int QBarDecoder::InitAIModel(QBAR_ML_MODE &ml_mode){
    detector_ = std::shared_ptr<QBarDetector>(new QBarDetector());
    int ret = detector_->Init(ml_mode.detection_model_path_);
    if(ret)
    {   
        return ret;
    }
    _init_detector_model_ = true;

    sr_ = std::shared_ptr<SuperScale>(new SuperScale());
    ret = sr_->Init(ml_mode.super_resolution_model_path_);
    if(ret)
    {   
        return ret;
    }
    _init_sr_model_ = true;

    return ret;
}

Mat QBarDecoder::cropObj(const Mat& img, const DetectInfo& bbox, Align& aligner) 
{
    auto point = Mat(4, 2, CV_32FC1);
    point.at<float>(0, 0) = CLIP(bbox.x,0,img.cols);
    point.at<float>(0, 1) = CLIP(bbox.y,0,img.rows);
    point.at<float>(1, 0) = CLIP(bbox.x + bbox.width,0,img.cols);
    point.at<float>(1, 1) = CLIP(bbox.y,0,img.rows);
    point.at<float>(2, 0) = CLIP(bbox.x + bbox.width,0,img.cols);
    point.at<float>(2, 1) = CLIP(bbox.y + bbox.height,0,img.rows);
    point.at<float>(3, 0) = CLIP(bbox.x,0,img.cols);
    point.at<float>(3, 1) = CLIP(bbox.y + bbox.height,0,img.rows);
    
    
    // make some padding to boost the qrcode details recall.
    float padding_w = 0.1f, padding_h = 0.1f;
    auto min_padding = 15;
    auto cropped = aligner.crop(img, point, padding_w, padding_h, min_padding);
    return cropped;
}

// empirical rules
std::vector<float> QBarDecoder::getScaleList(const int width, const int height) {
    if (width < 320 || height < 320) return {1.0, 2.0, 0.5};
    if (width < 640 && height < 640) return {1.0, 0.5};
    return {0.5, 1.0};
}
}  // namespace cv